"""
HybridBlock: 并行双分支混合架构的核心模块
包含Mamba分支（语义主干）+ Attention分支（局部细节）+ 轻门控融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from .mamba_block import MambaBlock
from .local_global_attn import LocalGlobalAttention


class SRTE(nn.Module):
    """Shared Relative Time Encoding: 生成统一的相对时间/位置编码基底"""
    
    def __init__(self, hidden_size: int, max_len: int = 65536, encoding_type: str = "learnable"):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoding_type = encoding_type
        
        if encoding_type == "learnable":
            # 可学习的相对位置编码
            self.freqs = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)
        elif encoding_type == "sincos":
            # 固定的sin/cos位置编码
            self.register_buffer('freqs', self._create_sincos_encoding(max_len, hidden_size))
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def _create_sincos_encoding(self, max_len: int, hidden_size: int) -> torch.Tensor:
        """创建sin/cos位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(math.log(10000.0) / hidden_size))
        
        encoding = torch.zeros(1, max_len, hidden_size)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        return encoding
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: 序列长度
        Returns:
            相对时间编码 [1, seq_len, hidden_size]
        """
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        return self.freqs[:, :seq_len, :]


class AlignmentMLP(nn.Module):
    """对齐模块：融合前对两分支进行特征对齐"""
    
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size
            
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        # 添加残差连接帮助学习
        self.residual_proj = nn.Linear(2 * hidden_size, hidden_size)
        
    def forward(self, h_mamba: torch.Tensor, h_attn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_mamba: Mamba分支输出 [B, L, H]
            h_attn: Attention分支输出 [B, L, H]  
        Returns:
            对齐后的特征 [B, L, H]
        """
        concatenated = torch.cat([h_mamba, h_attn], dim=-1)
        aligned = self.net(concatenated)
        residual = self.residual_proj(concatenated)
        return aligned + residual


class GateLowRank(nn.Module):
    """逐通道门控（低秩因式分解以省参）"""
    
    def __init__(self, hidden_size: int, rank: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = min(rank, hidden_size // 2)  # 确保rank不会太大
        
        self.u = nn.Linear(hidden_size, self.rank, bias=False)
        self.d = nn.Linear(self.rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化偏向0.5
        nn.init.xavier_uniform_(self.u.weight)
        nn.init.xavier_uniform_(self.d.weight)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 输入特征 [B, L, H]
        Returns:
            门控权重 [B, L, H] ∈ (0,1)
        """
        intermediate = torch.tanh(self.u(z))
        intermediate = self.dropout(intermediate)
        gate_logits = self.d(intermediate)
        return torch.sigmoid(gate_logits)


class HybridBlock(nn.Module):
    """
    混合架构的核心模块：
    - 并行双分支：Mamba（长程语义）+ Attention（局部细节）
    - 轻门控融合：逐通道动态权重
    - 统一时间编码：SRTE为两分支提供对齐的时间基底
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 32,
        window_size: int = 1024,
        global_heads: int = 2,
        gate_rank: int = 256,
        drop_branch_prob: float = 0.0,
        srte_encoding: str = "learnable",
        use_alignment: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.global_heads = global_heads
        self.drop_branch_prob = drop_branch_prob
        self.use_alignment = use_alignment
        
        # 层归一化
        self.ln_input = nn.LayerNorm(hidden_size)
        
        # 两个并行分支
        self.mamba = MambaBlock(hidden_size)
        self.attention = LocalGlobalAttention(
            hidden_size,
            num_heads=num_heads,
            window_size=window_size, 
            global_heads=global_heads
        )
        
        # 统一时间编码
        self.srte = SRTE(hidden_size, encoding_type=srte_encoding)
        
        # 对齐模块
        if use_alignment:
            self.alignment = AlignmentMLP(hidden_size)
        
        # 门控融合
        self.gate = GateLowRank(hidden_size, rank=gate_rank)
        self.fusion_proj = nn.Linear(hidden_size, hidden_size)
        
        # 小型MLP
        self.small_mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(0.1)
        )
        
        # 输出层归一化
        self.ln_output = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        mamba_state: Optional[torch.Tensor] = None,
        training: bool = True,
        collect_gate_stats: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Args:
            hidden_states: 输入隐藏状态 [B, L, H]
            attention_mask: 注意力掩码 [B, L] 
            kv_cache: KV缓存字典
            mamba_state: Mamba状态
            training: 是否训练模式
            collect_gate_stats: 是否收集门控统计信息
            
        Returns:
            output: 输出隐藏状态 [B, L, H]
            new_kv_cache: 更新后的KV缓存
            new_mamba_state: 更新后的Mamba状态
            auxiliary_outputs: 辅助输出（门控统计等）
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 输入层归一化
        normalized_input = self.ln_input(hidden_states)
        
        # 获取统一时间编码
        time_encoding = self.srte(seq_len).to(hidden_states.dtype)  # [1, L, H]
        
        # 为分支添加时间编码（避免与RoPE冲突，只给Mamba分支）
        mamba_input = normalized_input + time_encoding
        attn_input = normalized_input  # Attention分支使用RoPE，不加SRTE
        
        # 并行计算两分支
        h_mamba, new_mamba_state = self.mamba(mamba_input, state=mamba_state)
        h_attn, new_kv_cache = self.attention(
            attn_input, 
            attention_mask=attention_mask, 
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        
        # BranchDropout正则化（仅训练时）
        if training and self.drop_branch_prob > 0:
            if torch.rand(1).item() < self.drop_branch_prob:
                if torch.rand(1).item() < 0.5:
                    h_mamba = torch.zeros_like(h_mamba)
                else:
                    h_attn = torch.zeros_like(h_attn)
        
        # 特征对齐
        if self.use_alignment:
            aligned_features = self.alignment(h_mamba, h_attn)
        else:
            aligned_features = (h_mamba + h_attn) / 2
        
        # 门控融合
        gate_weights = self.gate(aligned_features)  # [B, L, H]
        fused_features = gate_weights * h_mamba + (1 - gate_weights) * h_attn
        
        # 投影和小型MLP
        projected = self.fusion_proj(fused_features)
        mlp_output = self.small_mlp(aligned_features)
        
        # 残差连接和输出层归一化
        output = hidden_states + projected + mlp_output
        output = self.ln_output(output)
        
        # 收集辅助信息
        auxiliary_outputs = None
        if collect_gate_stats:
            auxiliary_outputs = {
                'gate_weights': gate_weights.detach(),
                'gate_mean': gate_weights.mean().item(),
                'gate_std': gate_weights.std().item(),
                'mamba_norm': h_mamba.norm().item(),
                'attn_norm': h_attn.norm().item()
            }
        
        return output, new_kv_cache, new_mamba_state, auxiliary_outputs


def load_balance_loss(gate_weights: torch.Tensor, target: float = 0.5, margin: float = 0.1) -> torch.Tensor:
    """
    负载均衡损失：约束门控权重的batch均值接近目标值
    
    Args:
        gate_weights: 门控权重 [B, L, H] ∈ (0,1)
        target: 目标均值
        margin: 允许的偏差范围
        
    Returns:
        负载均衡损失
    """
    mean_gate = gate_weights.mean()
    lower_bound = target - margin
    upper_bound = target + margin
    
    # Hinge loss到目标区间
    loss = F.relu(lower_bound - mean_gate) + F.relu(mean_gate - upper_bound)
    return loss


def gate_entropy_regularization(gate_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    门控熵正则化：惩罚极端的0/1门控，鼓励多样性
    
    Args:
        gate_weights: 门控权重 [B, L, H] ∈ (0,1)
        eps: 数值稳定性参数
        
    Returns:
        熵正则化损失（负熵，即惩罚低熵）
    """
    # 数值稳定性夹断
    p = torch.clamp(gate_weights, eps, 1 - eps)
    
    # 计算二元熵
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    
    # 返回负熵作为正则项（惩罚低熵/极端值）
    return -entropy.mean() 