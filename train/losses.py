"""
损失函数模块：负载均衡、熵正则、蒸馏损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class HybridModelLoss(nn.Module):
    """
    混合模型的复合损失函数
    包含语言建模损失、负载均衡损失、熵正则化损失和蒸馏损失
    """
    
    def __init__(
        self,
        load_balance_coeff: float = 0.1,
        entropy_reg_coeff: float = 1e-4,
        distill_coeff: float = 0.0,
        distill_temperature: float = 4.0,
        gate_target: float = 0.5,
        gate_margin: float = 0.1
    ):
        super().__init__()
        self.load_balance_coeff = load_balance_coeff
        self.entropy_reg_coeff = entropy_reg_coeff
        self.distill_coeff = distill_coeff
        self.distill_temperature = distill_temperature
        self.gate_target = gate_target
        self.gate_margin = gate_margin
        
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gate_stats: Optional[list] = None,
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算复合损失
        
        Args:
            logits: 学生模型输出 [B, L, V]
            labels: 标签 [B, L]
            gate_stats: 门控统计信息列表
            teacher_logits: 教师模型输出 [B, L, V]
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 语言建模损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        lm_loss = self.lm_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        losses['lm_loss'] = lm_loss
        
        total_loss = lm_loss
        
        # 负载均衡和熵正则化损失
        if gate_stats:
            lb_loss = torch.tensor(0.0, device=logits.device)
            entropy_loss = torch.tensor(0.0, device=logits.device)
            
            for stats in gate_stats:
                if 'gate_weights' in stats:
                    gate_weights = stats['gate_weights']
                    lb_loss += self._load_balance_loss(gate_weights)
                    entropy_loss += self._entropy_regularization(gate_weights)
            
            losses['load_balance_loss'] = lb_loss
            losses['entropy_loss'] = entropy_loss
            
            total_loss += self.load_balance_coeff * lb_loss
            total_loss += self.entropy_reg_coeff * entropy_loss
        
        # 蒸馏损失
        if teacher_logits is not None and self.distill_coeff > 0:
            distill_loss = self._distillation_loss(
                logits[..., :-1, :], 
                teacher_logits[..., :-1, :]
            )
            losses['distill_loss'] = distill_loss
            total_loss += self.distill_coeff * distill_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _load_balance_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """负载均衡损失"""
        mean_gate = gate_weights.mean()
        lower_bound = self.gate_target - self.gate_margin
        upper_bound = self.gate_target + self.gate_margin
        
        loss = (F.relu(lower_bound - mean_gate) + 
                F.relu(mean_gate - upper_bound))
        return loss
    
    def _entropy_regularization(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """熵正则化损失"""
        eps = 1e-6
        p = torch.clamp(gate_weights, eps, 1 - eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return -entropy.mean()  # 负熵作为正则项
    
    def _distillation_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """知识蒸馏损失"""
        student_probs = F.log_softmax(
            student_logits / self.distill_temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.distill_temperature, dim=-1
        )
        
        kl_loss = F.kl_div(
            student_probs, teacher_probs, 
            reduction='batchmean'
        )
        
        return kl_loss * (self.distill_temperature ** 2) 