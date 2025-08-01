"""
LocalGlobalAttention: 局部+全局混合注意力机制
结合滑窗局部注意力和少量全局注意力头，平衡效率与性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""
    
    def __init__(self, dim: int, max_seq_len: int = 16384, base: float = 10000.0, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.base = base
        self.dtype = dtype
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 初始化空的缓存状态
        self.max_seq_len = 0
        self._build_cache(1, dtype, device='cpu')  # 延迟到第一次forward搬到GPU
        
    def _build_cache(self, seqlen: int, dtype: torch.dtype, device: torch.device):
        """按需构建缓存"""
        if (seqlen <= self.max_seq_len and 
            getattr(self, 'cos_cached', None) is not None and
            self.cos_cached.dtype == dtype and 
            self.cos_cached.device == device):
            return
            
        # 扩容策略：至少128，或者翻倍当前大小
        seqlen = max(seqlen, 128 if self.max_seq_len == 0 else self.max_seq_len * 2)
        
        t = torch.arange(seqlen, dtype=dtype, device=device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(dtype).to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        self.max_seq_len = seqlen
        
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转输入张量的一半维度"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用RoPE到查询和键
        
        Args:
            q: 查询张量 [B, H, L_q, D]
            k: 键张量 [B, H, L_k, D]
            q_pos: q的位置索引 [L_q]
            k_pos: k的位置索引 [L_k]
            
        Returns:
            旋转后的q和k
        """
        # 按需构建缓存
        max_pos = max(q_pos.max().item() + 1, k_pos.max().item() + 1)
        self._build_cache(max_pos, q.dtype, q.device)
        
        cos_q, sin_q = self._cos_sin_at(q_pos)  # [1, 1, L_q, D]
        cos_k, sin_k = self._cos_sin_at(k_pos)  # [1, 1, L_k, D]
        
        q_rotated = q * cos_q + self.rotate_half(q) * sin_q
        k_rotated = k * cos_k + self.rotate_half(k) * sin_k
        
        return q_rotated, k_rotated
    
    def _cos_sin_at(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定位置的cos/sin值"""
        cos = self.cos_cached[:, :, positions, :]
        sin = self.sin_cached[:, :, positions, :]
        return cos, sin


class SlidingWindowAttention(nn.Module):
    """滑窗注意力：只关注局部邻域"""
    
    def __init__(self, window_size: int, causal: bool = True):
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: [B, H, L, D]
            attention_mask: [B, L] 或 None
            
        Returns:
            output: [B, H, L, D]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 强制使用块化计算避免O(L²)内存开销
        return self._blockwise_attention(q, k, v, attention_mask)
    
    def _blockwise_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """块化滑窗注意力计算，使用SDPA优化"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = min(self.window_size, 1024)  # 块大小
        
        output = torch.zeros_like(q)
        
        for start_idx in range(0, seq_len, block_size):
            end_idx = min(start_idx + block_size, seq_len)
            
            # 当前块的查询
            q_block = q[:, :, start_idx:end_idx, :]  # [B,H,Lq,D]
            
            # 计算有效的键值范围（滑窗内）
            k_start = max(0, start_idx - self.window_size // 2)
            k_end = min(seq_len, end_idx + self.window_size // 2)
            
            # 因果约束：不能看到未来
            if self.causal:
                k_end = min(k_end, end_idx)
            
            k_block = k[:, :, k_start:k_end, :]  # [B,H,K,D]
            v_block = v[:, :, k_start:k_end, :]
            
            # —— 关键改动：同样采用"零化 K/V + 零化 pad 查询 + 输出乘回查询掩码" ——
            if attention_mask is not None:
                q_mask_blk = attention_mask[:, None, start_idx:end_idx, None].to(q_block.dtype)  # [B,1,Lq,1]
                kv_mask_blk = attention_mask[:, None, k_start:k_end, None].to(k_block.dtype)    # [B,1,K,1] 修复维度
                q_block = q_block * q_mask_blk
                k_block = k_block * kv_mask_blk
                v_block = v_block * kv_mask_blk

            output_block = F.scaled_dot_product_attention(
                q_block, k_block, v_block, attn_mask=None, is_causal=self.causal
            )
            output_block = torch.nan_to_num(output_block, nan=0.0, posinf=0.0, neginf=0.0)
            if attention_mask is not None:
                output_block = output_block * q_mask_blk
            output[:, :, start_idx:end_idx, :] = output_block
        
        return output
    
    def _standard_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """标准滑窗注意力（用于短序列）"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # 应用滑窗掩码
        if self.window_size < seq_len:
            window_mask = self._create_sliding_window_mask(seq_len, self.window_size, q.device)
            scores = scores.masked_fill(~window_mask, float('-inf'))
        
        # 应用因果掩码
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # 应用attention_mask
        if attention_mask is not None:
            expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(~expanded_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用到值
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def _create_sliding_window_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """创建滑窗掩码"""
        # 创建基础掩码矩阵
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            # 计算窗口范围
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            
            # 如果是因果注意力，不能看到未来
            if self.causal:
                end = min(end, i + 1)
            
            mask[i, start:end] = True
            
        return mask[None, None, :, :]  # [1, 1, L, L]


class GlobalAttention(nn.Module):
    """全局注意力：关注整个序列，使用SDPA优化"""
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用SDPA的全局注意力计算
        
        Args:
            q, k, v: [B, H, L, D]
            attention_mask: [B, L] 布尔掩码，True表示有效位置
            
        Returns:
            output: [B, H, L, D]
        """
        # —— 关键改动：用"零化"而非布尔 mask，避免 all-masked 触发 NaN ——
        if attention_mask is not None:
            q_mask = attention_mask[:, None, :, None].to(q.dtype)  # [B,1,L,1] 查询位置
            kv_mask = attention_mask[:, None, :, None].to(k.dtype) # [B,1,L,1] 键值位置（修复维度）
            q = q * q_mask
            k = k * kv_mask
            v = v * kv_mask

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        # 额外稳健性：把任何潜在 NaN 归零（理论上不会再出现）
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        if attention_mask is not None:
            output = output * q_mask  # 保证 pad 查询位的输出仍为 0
        return output


class LocalGlobalAttention(nn.Module):
    """
    混合注意力机制：
    - 大部分注意力头使用滑窗局部注意力（高效）
    - 少量注意力头使用全局注意力（保持长程建模能力）
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 32,
        window_size: int = 1024,
        global_heads: int = 2,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope: Optional[RotaryPositionalEmbedding] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.global_heads = min(global_heads, num_heads)
        self.local_heads = num_heads - self.global_heads
        
        if head_dim is None:
            head_dim = hidden_size // num_heads
        self.head_dim = head_dim
        
        # QKV投影
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        # 注意力机制
        self.sliding_attention = SlidingWindowAttention(window_size, causal=True)
        self.global_attention = GlobalAttention()
        
        # 位置编码
        if use_rope:
            if rope is not None:
                # 使用共享的RoPE实例
                self.rope = rope
            else:
                # 创建新的RoPE实例
                self.rope = RotaryPositionalEmbedding(head_dim, dtype=torch.bfloat16)
        else:
            self.rope = None
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Args:
            hidden_states: [B, L, H]
            attention_mask: [B, L] 布尔掩码，True表示有效位置
            kv_cache: KV缓存字典
            use_cache: 是否使用/更新缓存
            
        Returns:
            output: [B, L, H]
            new_kv_cache: 更新后的KV缓存
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV投影
        q = self.q_proj(hidden_states)  # [B, L, num_heads * head_dim]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 处理KV缓存（用于增量推理）
        if use_cache:
            if kv_cache is not None and 'k' in kv_cache and 'v' in kv_cache:
                # 拼接历史KV
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            new_kv_cache = {'k': k, 'v': v}
        else:
            new_kv_cache = None
        
        # 应用RoPE位置编码
        if self.rope is not None:
            # 计算位置索引
            if kv_cache is not None and use_cache and 'k' in kv_cache:
                # decode模式：q是新token，位置是past_length
                past_length = kv_cache['k'].size(2)
                q_pos = torch.tensor([past_length], device=q.device)
                k_pos = torch.arange(k.size(2), device=k.device)
            else:
                # prefill模式：q和k都是从0开始的完整序列
                seq_len = q.size(2)
                q_pos = torch.arange(seq_len, device=q.device)
                k_pos = torch.arange(seq_len, device=k.device)
            
            q, k = self.rope(q, k, q_pos, k_pos)
        
        # 分离局部和全局头
        if self.global_heads > 0:
            q_local, q_global = q[:, :-self.global_heads], q[:, -self.global_heads:]
            k_local, k_global = k[:, :-self.global_heads], k[:, -self.global_heads:]
            v_local, v_global = v[:, :-self.global_heads], v[:, -self.global_heads:]
        else:
            q_local, k_local, v_local = q, k, v
            q_global = k_global = v_global = None
        
        outputs = []
        
        # 局部注意力
        if self.local_heads > 0:
            local_output = self.sliding_attention(q_local, k_local, v_local, attention_mask)
            outputs.append(local_output)
        
        # 全局注意力
        if self.global_heads > 0:
            global_output = self.global_attention(q_global, k_global, v_global, attention_mask)
            outputs.append(global_output)
        
        # 合并所有头的输出
        if len(outputs) > 1:
            combined_output = torch.cat(outputs, dim=1)  # [B, H, L, D]
        else:
            combined_output = outputs[0]
        
        # 重塑回原始形状
        combined_output = combined_output.transpose(1, 2).contiguous()  # [B, L, H, D]
        combined_output = combined_output.view(batch_size, seq_len, -1)  # [B, L, H*D]
        
        # 输出投影和dropout
        output = self.out_proj(combined_output)
        output = self.dropout(output)
        
        return output, new_kv_cache


class PyramidalAttention(LocalGlobalAttention):
    """
    金字塔注意力：不同层使用不同的窗口大小
    底层用小窗口关注细节，高层用大窗口捕获长程依赖
    """
    
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        num_layers: int,
        base_window_size: int = 512,
        max_window_size: int = 2048,
        **kwargs
    ):
        # 计算当前层的窗口大小
        window_ratio = layer_idx / max(1, num_layers - 1)
        current_window_size = int(
            base_window_size + (max_window_size - base_window_size) * window_ratio
        )
        
        super().__init__(
            hidden_size=hidden_size,
            window_size=current_window_size,
            **kwargs
        )
        
        self.layer_idx = layer_idx
        self.base_window_size = base_window_size
        self.max_window_size = max_window_size


def create_attention_layer(
    hidden_size: int,
    layer_idx: int = 0,
    num_layers: int = 1,
    attention_type: str = "local_global",
    **kwargs
) -> nn.Module:
    """
    注意力层工厂函数
    
    Args:
        hidden_size: 隐藏层大小
        layer_idx: 当前层索引
        num_layers: 总层数
        attention_type: 注意力类型 ("local_global", "pyramidal")
        **kwargs: 其他参数
        
    Returns:
        注意力层实例
    """
    if attention_type == "local_global":
        return LocalGlobalAttention(hidden_size, **kwargs)
    elif attention_type == "pyramidal":
        return PyramidalAttention(
            hidden_size,
            layer_idx=layer_idx,
            num_layers=num_layers,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}") 