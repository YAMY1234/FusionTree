"""
系统性能评估：内存使用、推理速度、吞吐量等
"""

import os
import sys
import torch
import time
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig

logger = logging.getLogger(__name__)


class SystemPerformanceEvaluator:
    """系统性能评估器"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            except Exception:
                # 回退到传统加载方式（用于包含自定义对象的检查点）
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            if 'config' in checkpoint:
                config = HybridLanguageModelConfig(**checkpoint['config']['model'])
                self.model = HybridLanguageModel(config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                config = HybridLanguageModelConfig()
                self.model = HybridLanguageModel(config)
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # 创建默认模型
            config = HybridLanguageModelConfig(
                vocab_size=1000,
                hidden_size=512,  # 较小的模型用于测试
                num_layers=8
            )
            self.model = HybridLanguageModel(config).to(self.device)
            self.model.eval()
            logger.warning("Using default model for testing")
    
    def benchmark_memory_usage(
        self, 
        batch_sizes: List[int] = [1, 2, 4, 8],
        sequence_lengths: List[int] = [1024, 2048, 4096, 8192]
    ) -> Dict[str, Any]:
        """评估内存使用情况"""
        results = {}
        
        for batch_size in tqdm(batch_sizes, desc="Memory benchmark - batch sizes"):
            for seq_len in tqdm(sequence_lengths, desc=f"Seq lengths (bs={batch_size})", leave=False):
                key = f"bs{batch_size}_len{seq_len}"
                
                try:
                    # 清空显存
                    if self.device.startswith('cuda') and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    
                    # 创建输入
                    input_ids = torch.randint(
                        0, 1000, (batch_size, seq_len), 
                        device=self.device
                    )
                    
                    # 前向传播
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                    
                    # 获取内存使用量
                    if self.device.startswith('cuda') and torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                    else:
                        memory_allocated = memory_reserved = peak_memory = None
                    
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'memory_allocated_gb': memory_allocated,
                        'memory_reserved_gb': memory_reserved,
                        'peak_memory_gb': peak_memory,
                        'status': 'success'
                    }
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        results[key] = {
                            'batch_size': batch_size,
                            'sequence_length': seq_len,
                            'status': 'OOM',
                            'error': str(e)
                        }
                    else:
                        results[key] = {
                            'batch_size': batch_size,
                            'sequence_length': seq_len,
                            'status': 'error',
                            'error': str(e)
                        }
                
                # 清理
                if 'input_ids' in locals():
                    del input_ids
                if 'outputs' in locals():
                    del outputs
                if self.device.startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
    
    def benchmark_inference_speed(
        self,
        batch_sizes: List[int] = [1, 2, 4],
        sequence_lengths: List[int] = [1024, 2048, 4096],
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """评估推理速度"""
        results = {}
        
        for batch_size in tqdm(batch_sizes, desc="Speed benchmark - batch sizes"):
            for seq_len in tqdm(sequence_lengths, desc=f"Seq lengths (bs={batch_size})", leave=False):
                key = f"bs{batch_size}_len{seq_len}"
                
                try:
                    # 创建输入
                    input_ids = torch.randint(
                        0, 1000, (batch_size, seq_len),
                        device=self.device
                    )
                    
                    # 预热
                    for _ in range(3):
                        with torch.no_grad():
                            _ = self.model(input_ids)
                    
                    if self.device.startswith('cuda') and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Prefill 测试
                    start_time = time.time()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            outputs = self.model(input_ids, use_cache=True)
                        if self.device.startswith('cuda') and torch.cuda.is_available():
                            torch.cuda.synchronize()
                    prefill_time = (time.time() - start_time) / num_runs
                    
                    # Decode 测试（如果支持缓存）
                    decode_time = None
                    decode_tokens_per_sec = None
                    
                    if outputs.get('past_key_values') is not None:
                        # 单token decode
                        decode_input = input_ids[:, -1:]
                        past_kv = outputs['past_key_values']
                        past_mamba = outputs.get('past_mamba_states')
                        
                        start_time = time.time()
                        for _ in range(num_runs):
                            with torch.no_grad():
                                _ = self.model(
                                    decode_input,
                                    past_key_values=past_kv,
                                    past_mamba_states=past_mamba,
                                    use_cache=True
                                )
                            torch.cuda.synchronize()
                        decode_time = (time.time() - start_time) / num_runs
                        decode_tokens_per_sec = batch_size / decode_time
                    
                    # 计算指标
                    prefill_tokens_per_sec = (batch_size * seq_len) / prefill_time
                    
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'prefill_time_sec': prefill_time,
                        'prefill_tokens_per_sec': prefill_tokens_per_sec,
                        'decode_time_sec': decode_time,
                        'decode_tokens_per_sec': decode_tokens_per_sec,
                        'status': 'success'
                    }
                    
                except Exception as e:
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'status': 'error',
                        'error': str(e)
                    }
                
                # 清理
                if 'input_ids' in locals():
                    del input_ids
                if 'outputs' in locals():
                    del outputs
                torch.cuda.empty_cache()
        
        return results
    
    def benchmark_generation_speed(
        self,
        prompt_lengths: List[int] = [512, 1024, 2048],
        generation_lengths: List[int] = [50, 100, 200],
        num_samples: int = 5
    ) -> Dict[str, Any]:
        """评估生成速度"""
        results = {}
        
        for prompt_len in tqdm(prompt_lengths, desc="Generation benchmark - prompt lengths"):
            for gen_len in tqdm(generation_lengths, desc=f"Gen lengths (prompt={prompt_len})", leave=False):
                key = f"prompt{prompt_len}_gen{gen_len}"
                
                try:
                    times = []
                    
                    for _ in range(num_samples):
                        # 创建输入
                        input_ids = torch.randint(
                            0, 1000, (1, prompt_len),
                            device=self.device
                        )
                        
                        # 生成
                        start_time = time.time()
                        with torch.no_grad():
                            generated = self.model.generate(
                                input_ids,
                                max_new_tokens=gen_len,
                                temperature=0.8,
                                do_sample=True
                            )
                        torch.cuda.synchronize()
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        
                        # 清理
                        del input_ids, generated
                        torch.cuda.empty_cache()
                    
                    # 计算统计指标
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    tokens_per_sec = gen_len / avg_time
                    
                    results[key] = {
                        'prompt_length': prompt_len,
                        'generation_length': gen_len,
                        'avg_time_sec': avg_time,
                        'std_time_sec': std_time,
                        'tokens_per_sec': tokens_per_sec,
                        'samples': num_samples,
                        'status': 'success'
                    }
                    
                except Exception as e:
                    results[key] = {
                        'prompt_length': prompt_len,
                        'generation_length': gen_len,
                        'status': 'error',
                        'error': str(e)
                    }
        
        return results


def evaluate_system_performance(
    model_path: str,
    output_dir: str = None,
    profile_memory: bool = True,
    profile_speed: bool = True,
    profile_generation: bool = True,
    batch_sizes: List[int] = None,
    sequence_lengths: List[int] = None
) -> Dict[str, Any]:
    """
    评估系统性能
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        profile_memory: 是否评估内存
        profile_speed: 是否评估速度
        profile_generation: 是否评估生成
        batch_sizes: 批大小列表
        sequence_lengths: 序列长度列表
        
    Returns:
        评估结果字典
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4]
    if sequence_lengths is None:
        sequence_lengths = [1024, 2048, 4096]
    
    # 创建评估器
    evaluator = SystemPerformanceEvaluator(model_path)
    
    results = {
        'model_path': model_path,
        'device': evaluator.device,
        'timestamp': time.time()
    }
    
    # 内存评估
    if profile_memory:
        logger.info("Evaluating memory usage...")
        memory_results = evaluator.benchmark_memory_usage(
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths
        )
        results['memory'] = memory_results
    
    # 速度评估
    if profile_speed:
        logger.info("Evaluating inference speed...")
        speed_results = evaluator.benchmark_inference_speed(
            batch_sizes=batch_sizes,
            sequence_lengths=sequence_lengths
        )
        results['speed'] = speed_results
    
    # 生成评估
    if profile_generation:
        logger.info("Evaluating generation speed...")
        generation_results = evaluator.benchmark_generation_speed()
        results['generation'] = generation_results
    
    # 计算摘要统计
    results['summary'] = _compute_summary(results)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'system_evaluation_results.json')
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_path}")
    
    return results


def _compute_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """计算摘要统计"""
    summary = {}
    
    # 内存摘要
    if 'memory' in results:
        memory_data = results['memory']
        successful_runs = [v for v in memory_data.values() if v.get('status') == 'success']
        
        if successful_runs:
            peak_memories = [run['peak_memory_gb'] for run in successful_runs if run.get('peak_memory_gb') is not None]
            if peak_memories:  # 确保列表不为空
            summary['memory'] = {
                'max_peak_memory_gb': max(peak_memories),
                'avg_peak_memory_gb': np.mean(peak_memories),
                'successful_runs': len(successful_runs),
                'total_runs': len(memory_data)
            }
            else:
                summary['memory'] = {
                    'max_peak_memory_gb': 0.0,
                    'avg_peak_memory_gb': 0.0,
                    'successful_runs': len(successful_runs),
                    'total_runs': len(memory_data)
                }
    
    # 速度摘要
    if 'speed' in results:
        speed_data = results['speed']
        successful_runs = [v for v in speed_data.values() if v.get('status') == 'success']
        
        if successful_runs:
            prefill_speeds = [run['prefill_tokens_per_sec'] for run in successful_runs]
            decode_speeds = [run['decode_tokens_per_sec'] for run in successful_runs 
                           if run['decode_tokens_per_sec'] is not None]
            
            summary['speed'] = {
                'max_prefill_tokens_per_sec': max(prefill_speeds),
                'avg_prefill_tokens_per_sec': np.mean(prefill_speeds),
                'max_decode_tokens_per_sec': max(decode_speeds) if decode_speeds else None,
                'avg_decode_tokens_per_sec': np.mean(decode_speeds) if decode_speeds else None,
                'successful_runs': len(successful_runs)
            }
    
    # 生成摘要
    if 'generation' in results:
        generation_data = results['generation']
        successful_runs = [v for v in generation_data.values() if v.get('status') == 'success']
        
        if successful_runs:
            generation_speeds = [run['tokens_per_sec'] for run in successful_runs]
            summary['generation'] = {
                'max_generation_tokens_per_sec': max(generation_speeds),
                'avg_generation_tokens_per_sec': np.mean(generation_speeds),
                'successful_runs': len(successful_runs)
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="FusionTree System Performance Evaluation")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--profile_memory', action='store_true', help='评估内存使用')
    parser.add_argument('--profile_speed', action='store_true', help='评估推理速度')
    parser.add_argument('--profile_generation', action='store_true', help='评估生成速度')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4', help='批大小（逗号分隔）')
    parser.add_argument('--sequence_lengths', type=str, default='1024,2048,4096', help='序列长度（逗号分隔）')
    
    args = parser.parse_args()
    
    # 解析参数
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    sequence_lengths = [int(x) for x in args.sequence_lengths.split(',')]
    
    # 如果没有指定任何profile，则全部开启
    if not any([args.profile_memory, args.profile_speed, args.profile_generation]):
        args.profile_memory = True
        args.profile_speed = True
        args.profile_generation = True
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行评估
    results = evaluate_system_performance(
        model_path=args.model_path,
        output_dir=args.output_dir,
        profile_memory=args.profile_memory,
        profile_speed=args.profile_speed,
        profile_generation=args.profile_generation,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths
    )
    
    # 打印摘要
    print("\n=== System Performance Results ===")
    summary = results.get('summary', {})
    
    for category, metrics in summary.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main() 