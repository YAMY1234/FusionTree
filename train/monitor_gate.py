"""
门控监控模块：统计和分析门控权重分布
"""

import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GateMonitor:
    """
    门控统计监控器
    收集和分析各层门控权重的统计信息，用于模型裁剪决策
    """
    
    def __init__(self, num_layers: int, collect_detailed: bool = True):
        """
        Args:
            num_layers: 模型层数
            collect_detailed: 是否收集详细统计信息
        """
        self.num_layers = num_layers
        self.collect_detailed = collect_detailed
        
        # 基础统计信息
        self.layer_stats = [[] for _ in range(num_layers)]
        
        # 详细统计信息
        if collect_detailed:
            self.detailed_stats = defaultdict(list)
            self.token_type_stats = defaultdict(lambda: defaultdict(list))
        
        self.step_count = 0
        
    def update(
        self, 
        layer_idx: int, 
        gate_stats: Dict[str, Any],
        token_types: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None
    ):
        """
        更新统计信息
        
        Args:
            layer_idx: 层索引
            gate_stats: 门控统计信息
            token_types: token类型 [B, L]
            segment_ids: 段落ID [B, L]
        """
        if 'gate_weights' not in gate_stats:
            return
        
        gate_weights = gate_stats['gate_weights']  # [B, L, H]
        
        # 基础统计
        mean_gate = gate_weights.mean().item()
        std_gate = gate_weights.std().item()
        min_gate = gate_weights.min().item()
        max_gate = gate_weights.max().item()
        
        basic_stats = {
            'step': self.step_count,
            'mean': mean_gate,
            'std': std_gate,
            'min': min_gate,
            'max': max_gate
        }
        
        self.layer_stats[layer_idx].append(basic_stats)
        
        # 详细统计
        if self.collect_detailed:
            self._update_detailed_stats(layer_idx, gate_weights, gate_stats)
            
            # 按token类型统计
            if token_types is not None:
                self._update_token_type_stats(
                    layer_idx, gate_weights, token_types, segment_ids
                )
        
        self.step_count += 1
    
    def _update_detailed_stats(
        self, 
        layer_idx: int, 
        gate_weights: torch.Tensor,
        gate_stats: Dict[str, Any]
    ):
        """更新详细统计信息"""
        # 分位数统计
        flat_gates = gate_weights.flatten()
        percentiles = torch.quantile(
            flat_gates, 
            torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=flat_gates.device)
        )
        
        # 极端值比例
        extreme_low = (gate_weights < 0.1).float().mean().item()
        extreme_high = (gate_weights > 0.9).float().mean().item()
        
        # 熵计算
        eps = 1e-6
        p = torch.clamp(gate_weights, eps, 1 - eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        mean_entropy = entropy.mean().item()
        
        detailed = {
            'step': self.step_count,
            'layer': layer_idx,
            'percentiles': percentiles.cpu().tolist(),
            'extreme_low_ratio': extreme_low,
            'extreme_high_ratio': extreme_high,
            'entropy': mean_entropy,
            'mamba_norm': gate_stats.get('mamba_norm', 0.0),
            'attn_norm': gate_stats.get('attn_norm', 0.0)
        }
        
        self.detailed_stats[layer_idx].append(detailed)
    
    def _update_token_type_stats(
        self,
        layer_idx: int,
        gate_weights: torch.Tensor,
        token_types: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None
    ):
        """按token类型更新统计"""
        # 简化的token类型分类
        # 0: padding, 1: special tokens, 2: normal tokens
        for token_type in [0, 1, 2]:
            mask = (token_types == token_type)
            if mask.sum() > 0:
                masked_gates = gate_weights[mask]
                mean_gate = masked_gates.mean().item()
                
                self.token_type_stats[layer_idx][f'type_{token_type}'].append({
                    'step': self.step_count,
                    'mean': mean_gate,
                    'count': mask.sum().item()
                })
    
    def get_layer_summary(self, layer_idx: int) -> Dict[str, float]:
        """获取指定层的统计摘要"""
        if not self.layer_stats[layer_idx]:
            return {}
        
        stats = self.layer_stats[layer_idx]
        means = [s['mean'] for s in stats[-100:]]  # 最近100步
        
        return {
            'recent_mean': np.mean(means),
            'recent_std': np.std(means),
            'trend': np.polyfit(range(len(means)), means, 1)[0] if len(means) > 1 else 0,
            'total_steps': len(stats)
        }
    
    def export_prune_plan(
        self, 
        mamba_threshold_high: float = 0.8,
        attention_threshold_low: float = 0.2,
        min_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """
        导出裁剪计划
        
        Args:
            mamba_threshold_high: 偏向Mamba的阈值
            attention_threshold_low: 偏向Attention的阈值
            min_steps: 最少统计步数
            
        Returns:
            裁剪计划列表
        """
        plan = []
        
        for layer_idx in range(self.num_layers):
            summary = self.get_layer_summary(layer_idx)
            
            if summary.get('total_steps', 0) < min_steps:
                # 统计不足，保持混合
                decision = 'hybrid'
                confidence = 0.0
            else:
                mean_gate = summary['recent_mean']
                std_gate = summary['recent_std']
                
                # 决策逻辑
                if mean_gate > mamba_threshold_high and std_gate < 0.1:
                    decision = 'mamba'
                    confidence = min(1.0, (mean_gate - mamba_threshold_high) / 0.2)
                elif mean_gate < attention_threshold_low and std_gate < 0.1:
                    decision = 'attention'
                    confidence = min(1.0, (attention_threshold_low - mean_gate) / 0.2)
                else:
                    decision = 'hybrid'
                    confidence = 1.0 - abs(mean_gate - 0.5) * 2
            
            layer_plan = {
                'layer_idx': layer_idx,
                'decision': decision,
                'confidence': confidence,
                'gate_mean': summary.get('recent_mean', 0.5),
                'gate_std': summary.get('recent_std', 0.0),
                'trend': summary.get('trend', 0.0)
            }
            
            plan.append(layer_plan)
            logger.info(f"Layer {layer_idx}: {decision} (confidence: {confidence:.3f})")
        
        return plan
    
    def save_statistics(self, filepath: str):
        """保存统计信息到文件"""
        data = {
            'num_layers': self.num_layers,
            'total_steps': self.step_count,
            'layer_stats': self.layer_stats,
            'layer_summaries': [
                self.get_layer_summary(i) for i in range(self.num_layers)
            ]
        }
        
        if self.collect_detailed:
            data['detailed_stats'] = dict(self.detailed_stats)
            data['token_type_stats'] = dict(self.token_type_stats)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Gate statistics saved to {filepath}")
    
    def load_statistics(self, filepath: str):
        """从文件加载统计信息"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.num_layers = data['num_layers']
        self.step_count = data['total_steps']
        self.layer_stats = data['layer_stats']
        
        if 'detailed_stats' in data:
            self.detailed_stats = defaultdict(list, data['detailed_stats'])
            
        if 'token_type_stats' in data:
            self.token_type_stats = defaultdict(
                lambda: defaultdict(list), 
                data['token_type_stats']
            )
        
        logger.info(f"Gate statistics loaded from {filepath}")
    
    def reset(self):
        """重置所有统计信息"""
        self.layer_stats = [[] for _ in range(self.num_layers)]
        if self.collect_detailed:
            self.detailed_stats.clear()
            self.token_type_stats.clear()
        self.step_count = 0
        logger.info("Gate monitor reset")
    
    def plot_gate_distribution(self, layer_idx: int, save_path: Optional[str] = None):
        """绘制门控分布图（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.layer_stats[layer_idx]:
                logger.warning(f"No data for layer {layer_idx}")
                return
            
            stats = self.layer_stats[layer_idx]
            steps = [s['step'] for s in stats]
            means = [s['mean'] for s in stats]
            stds = [s['std'] for s in stats]
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(steps, means, label='Mean', alpha=0.8)
            plt.fill_between(
                steps, 
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.3, label='±1 std'
            )
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Balanced')
            plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Mamba bias')
            plt.axhline(y=0.2, color='b', linestyle='--', alpha=0.5, label='Attention bias')
            plt.xlabel('Training Steps')
            plt.ylabel('Gate Weight')
            plt.title(f'Layer {layer_idx} Gate Weight Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(steps, stds, color='orange', label='Standard Deviation')
            plt.xlabel('Training Steps')
            plt.ylabel('Gate Weight Std')
            plt.title(f'Layer {layer_idx} Gate Weight Stability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Gate distribution plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available, cannot plot gate distribution") 