"""
HybridBlock: 并行双分支混合架构的核心模块
包含Mamba分支（语义主干）+ Attention分支（局部细节）+ 轻门控融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math
from torch.utils.checkpoint import checkpoint

from .mamba_block import MambaBlock
from .local_global_attn import LocalGlobalAttention

def _assert_finite(name, x):
    """早期哨兵断言，方便定位 NaN/Inf 问题"""
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).float().mean().item()
        raise RuntimeError(f"[NaN/Inf] {name}: ratio={bad:.6f}, shape={tuple(x.shape)}")


class SRTE(nn.Module):
    """Shared Relative Time Encoding: 生成统一的相对时间/位置编码基底"""
    
    def __init__(self, hidden_size: int, max_len: int = 65536, encoding_type: str = "learnable", factorized_rank: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.encoding_type = encoding_type
        self.factorized_rank = factorized_rank
        
        if encoding_type == "learnable":
            if factorized_rank and factorized_rank < hidden_size:
                # 低秩分解：[L, r] + [r, H] 而不是 [L, H]
                self.lowrank = nn.Parameter(torch.randn(1, max_len, factorized_rank) * 0.02)
                self.proj = nn.Linear(factorized_rank, hidden_size, bias=False)
                print(f"SRTE using factorized learnable encoding: {max_len} x {factorized_rank} + {factorized_rank} x {hidden_size} = {max_len*factorized_rank + factorized_rank*hidden_size:,} params")
            else:
                # 传统全参数可学习编码
                self.freqs = nn.Parameter(torch.randn(1, max_len, hidden_size) * 0.02)
                print(f"SRTE using full learnable encoding: {max_len} x {hidden_size} = {max_len*hidden_size:,} params")
        elif encoding_type == "sincos":
            # 按需计算sin/cos：只缓存inv_freq，不预存整表
            import math
            inv_freq = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))
            self.register_buffer('inv_freq', inv_freq)
            print(f"SRTE using on-the-fly sincos encoding: {len(inv_freq)} inv_freq params")
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    

    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: 序列长度
        Returns:
            相对时间编码 [1, seq_len, hidden_size]
        """
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        if self.encoding_type == "learnable":
            if hasattr(self, 'lowrank'):
                # 低秩分解：先取lowrank部分，再投影到full space
                return self.proj(self.lowrank[:, :seq_len, :])
            else:
                # 传统learnable
                return self.freqs[:, :seq_len, :]
        else:
            # sincos按需生成
            device = self.inv_freq.device
            dtype = self.inv_freq.dtype
            t = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # [L, 1]
            freqs = t * self.inv_freq.unsqueeze(0)  # [L, H/2]
            emb = torch.zeros(seq_len, self.hidden_size, device=device, dtype=dtype)
            emb[:, 0::2] = torch.sin(freqs)
            emb[:, 1::2] = torch.cos(freqs)
            return emb.unsqueeze(0)  # [1, L, H]


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
        srte_max_len: int = 65536,
        srte_shared=None,
        srte_factorized_rank: int = 0,
        use_alignment: bool = True,
        shared_rope=None,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.global_heads = global_heads
        self.drop_branch_prob = drop_branch_prob
        self.use_alignment = use_alignment
        self.gradient_checkpointing = gradient_checkpointing
        
        # 层归一化
        self.ln_input = nn.LayerNorm(hidden_size)
        
        # 两个并行分支
        self.mamba = MambaBlock(hidden_size)
        self.attention = LocalGlobalAttention(
            hidden_size,
            num_heads=num_heads,
            window_size=window_size, 
            global_heads=global_heads,
            rope=shared_rope
        )
        
        # 统一时间编码
        if srte_shared is not None:
            # 使用共享的SRTE
            self.srte = srte_shared
        else:
            # 创建独立的SRTE
            self.srte = SRTE(
                hidden_size, 
                max_len=srte_max_len, 
                encoding_type=srte_encoding,
                factorized_rank=srte_factorized_rank
            )
        
        # 对齐模块
        if use_alignment:
            self.alignment = AlignmentMLP(hidden_size)
        
        # 门控融合
        self.gate = GateLowRank(hidden_size, rank=gate_rank)
        self.fusion_proj = nn.Linear(hidden_size, hidden_size)
        
        # 小型MLP - 使用2H而不是4H减少内存使用
        self.small_mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, hidden_size),
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
        if attention_mask is not None:
            normalized_input = normalized_input * attention_mask[:, :, None].to(normalized_input.dtype)
        
        # 🔍 早期哨兵检查
        _assert_finite("embeddings", hidden_states)
        _assert_finite("normalized_input", normalized_input)
        
        # 获取统一时间编码
        time_encoding = self.srte(seq_len).to(hidden_states.dtype)  # [1, L, H]
        
        # 为分支添加时间编码（避免与RoPE冲突，只给Mamba分支）
        mamba_input = normalized_input + time_encoding
        attn_input = normalized_input  # Attention分支使用RoPE，不加SRTE
        
        # 并行计算两分支，使用选择性梯度检查点
        if self.gradient_checkpointing and training and torch.is_grad_enabled():
            # 对Mamba分支使用梯度检查点（只checkpoint输出，state不参与）
            def mamba_forward(x):
                return self.mamba(x, state=mamba_state)[0]  # 只返回输出
            
            h_mamba = checkpoint(mamba_forward, mamba_input, use_reentrant=False)
            # state在训练时通常不需要，推理时会走else分支
            new_mamba_state = None
            
            # 对Attention分支使用梯度检查点（只checkpoint输出，cache不参与）
            def attn_forward(x):
                return self.attention(x, attention_mask=attention_mask, 
                                    kv_cache=kv_cache, use_cache=False)[0]  # 只返回输出
            
            h_attn = checkpoint(attn_forward, attn_input, use_reentrant=False)
            # cache在训练时通常不需要，推理时会走else分支
            new_kv_cache = None
        else:
            # 正常计算（推理模式或不使用checkpoint）
            h_mamba, new_mamba_state = self.mamba(mamba_input, state=mamba_state)
            h_attn, new_kv_cache = self.attention(
                attn_input, 
                attention_mask=attention_mask, 
                kv_cache=kv_cache,
                use_cache=use_cache
            )
        
        # BranchDropout正则化（仅训练时）- 确保分布式训练时所有rank一致
        if training and self.drop_branch_prob > 0:
            # 🔧 分布式安全的随机分支丢弃
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    # 分布式模式：rank0生成随机数，广播给所有rank确保一致性
                    if dist.get_rank() == 0:
                        drop_decision = torch.rand(2, device=h_mamba.device)  # [drop_prob, branch_choice]
                    else:
                        drop_decision = torch.zeros(2, device=h_mamba.device)
                    
                    # 广播随机数到所有rank
                    dist.broadcast(drop_decision, src=0)
                    
                    should_drop = drop_decision[0].item() < self.drop_branch_prob
                    drop_mamba = drop_decision[1].item() < 0.5
                else:
                    # 非分布式模式：正常随机
                    should_drop = torch.rand(1).item() < self.drop_branch_prob
                    drop_mamba = torch.rand(1).item() < 0.5
                    
            except ImportError:
                # 未安装分布式包：正常随机
                should_drop = torch.rand(1).item() < self.drop_branch_prob
                drop_mamba = torch.rand(1).item() < 0.5
                
            if should_drop:
                if drop_mamba:
                    h_mamba = torch.zeros_like(h_mamba)
                else:
                    h_attn = torch.zeros_like(h_attn)
        
        # 🔍 哨兵检查分支输出
        _assert_finite("mamba_out", h_mamba)
        _assert_finite("attn_out", h_attn)
        
        # 特征对齐
        if self.use_alignment:
            aligned_features = self.alignment(h_mamba, h_attn)
        else:
            aligned_features = (h_mamba + h_attn) / 2
        
        _assert_finite("aligned", aligned_features)
        
        # 门控融合
        gate_weights = self.gate(aligned_features)  # [B, L, H]
        _assert_finite("gate", gate_weights)
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
                'gate_weights': gate_weights,  # 保留梯度，不要detach
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