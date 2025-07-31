"""
MambaBlock: 状态空间模型分支，专注于长程语义依赖
这里提供接口定义和简化实现，可替换为高效的并行扫描内核
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MambaBlock(nn.Module):
    """
    Mamba状态空间模型块
    专门处理长程语义依赖，在序列建模中作为语义主干
    
    注意：这是一个简化的接口实现，实际部署时应替换为：
    - 优化的并行扫描内核
    - 第三方高效Mamba实现（如mamba-ssm）
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 64,
        conv_kernel_size: int = 4,
        expand_factor: int = 2,
        dt_rank: int = None,
        use_bias: bool = True,
        use_conv_bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.conv_kernel_size = conv_kernel_size
        self.expand_factor = expand_factor
        self.intermediate_size = hidden_size * expand_factor
        
        if dt_rank is None:
            dt_rank = math.ceil(hidden_size / 16)
        self.dt_rank = dt_rank
        
        # 输入投影
        self.in_proj = nn.Linear(hidden_size, self.intermediate_size * 2, bias=use_bias)
        
        # 卷积层（局部依赖）
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.intermediate_size,  # Depthwise convolution
            bias=use_conv_bias
        )
        
        # SSM参数投影
        self.x_proj = nn.Linear(self.intermediate_size, self.dt_rank + self.state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.intermediate_size, bias=True)
        
        # 状态空间参数
        # A: 状态转移矩阵 (D, N) - 初始化为负实数，但保持bfloat16兼容性
        A = torch.arange(1, self.state_size + 1, dtype=torch.float32).repeat(self.intermediate_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D: 跳跃连接参数
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        
        # 输出投影
        self.out_proj = nn.Linear(self.intermediate_size, hidden_size, bias=use_bias)
        
        # 激活函数
        self.activation = nn.SiLU()
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: 输入隐藏状态 [B, L, H]
            state: 前一步的SSM状态 [B, D, N] 或 None
            
        Returns:
            output: 输出隐藏状态 [B, L, H]
            new_state: 更新后的SSM状态 [B, D, N]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 输入投影和门控
        projected = self.in_proj(hidden_states)  # [B, L, 2*D]
        x_and_res = projected.chunk(2, dim=-1)
        x = x_and_res[0]  # [B, L, D]
        res = x_and_res[1]  # [B, L, D]
        
        # 应用激活到残差路径
        res = self.activation(res)
        
        # 1D卷积（需要转置到 [B, D, L]）
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv1d(x)  # [B, D, L]
        
        # 移除padding（如果有的话）
        if self.conv_kernel_size > 1:
            x = x[:, :, :seq_len]
        
        x = x.transpose(1, 2)  # [B, L, D]
        x = self.activation(x)
        
        # SSM参数投影
        ssm_params = self.x_proj(x)  # [B, L, dt_rank + 2*N]
        dt, B, C = torch.split(
            ssm_params, 
            [self.dt_rank, self.state_size, self.state_size], 
            dim=-1
        )
        
        # 计算时间步长
        dt = self.dt_proj(dt)  # [B, L, D]
        dt = F.softplus(dt + self.dt_proj.bias)
        
        # 获取状态转移矩阵A，确保类型匹配
        A = -torch.exp(self.A_log)  # [D, N] 保持与输入相同的dtype
        
        # 简化的SSM计算（实际应该用并行扫描算法）
        y = self._ssm_scan(x, dt, A, B, C, self.D, state)
        
        # 门控融合
        y = y * res
        
        # 输出投影
        output = self.out_proj(y)
        
        # 计算新状态（简化版，实际部署时需要优化）
        new_state = self._compute_final_state(x, dt, A, B, C, state)
        
        return output, new_state
    
    def _ssm_scan(
        self, 
        x: torch.Tensor,
        dt: torch.Tensor, 
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        简化的SSM扫描实现
        实际部署时应替换为高效的并行扫描内核
        
        Args:
            x: 输入序列 [B, L, D]
            dt: 时间步长 [B, L, D]  
            A: 状态转移矩阵 [D, N]
            B: 输入矩阵 [B, L, N]
            C: 输出矩阵 [B, L, N]
            D: 跳跃连接 [D]
            initial_state: 初始状态 [B, D, N]
            
        Returns:
            输出序列 [B, L, D]
        """
        batch_size, seq_len, d_model = x.shape
        n_state = A.shape[1]
        
        # 确保所有张量类型一致
        target_dtype = x.dtype
        A = A.to(target_dtype)
        B = B.to(target_dtype)
        C = C.to(target_dtype)
        D = D.to(target_dtype)
        dt = dt.to(target_dtype)
        
        if initial_state is None:
            h = torch.zeros(batch_size, d_model, n_state, device=x.device, dtype=target_dtype)
        else:
            h = initial_state.to(target_dtype)
        
        outputs = []
        
        for i in range(seq_len):
            # 当前时间步的输入
            x_i = x[:, i, :]  # [B, D]
            dt_i = dt[:, i, :]  # [B, D]
            B_i = B[:, i, :]  # [B, N]
            C_i = C[:, i, :]  # [B, N]
            
            # 离散化
            dA = torch.exp(dt_i.unsqueeze(-1) * A.unsqueeze(0))  # [B, D, N]
            dB = dt_i.unsqueeze(-1) * B_i.unsqueeze(1)  # [B, D, N]
            
            # 状态更新
            h = h * dA + dB * x_i.unsqueeze(-1)  # [B, D, N]
            
            # 输出计算 - 确保所有张量类型一致
            y_i = torch.einsum('bdn,bn->bd', h, C_i) + x_i * D
            outputs.append(y_i)
        
        return torch.stack(outputs, dim=1)  # [B, L, D]
    
    def _compute_final_state(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor, 
        B: torch.Tensor,
        C: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算最终状态（用于增量推理）
        这是一个简化实现，实际使用时需要优化
        """
        batch_size, seq_len, d_model = x.shape
        n_state = A.shape[1]
        
        # 确保类型一致
        target_dtype = x.dtype
        A = A.to(target_dtype)
        B = B.to(target_dtype)
        dt = dt.to(target_dtype)
        
        if initial_state is None:
            h = torch.zeros(batch_size, d_model, n_state, device=x.device, dtype=target_dtype)
        else:
            h = initial_state.to(target_dtype)
            
        # 简化：只返回最后时间步的状态
        # 实际实现应该使用高效的并行算法
        for i in range(seq_len):
            x_i = x[:, i, :]
            dt_i = dt[:, i, :]
            B_i = B[:, i, :]
            
            dA = torch.exp(dt_i.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_i.unsqueeze(-1) * B_i.unsqueeze(1)
            
            h = h * dA + dB * x_i.unsqueeze(-1)
        
        return h


class SimplifiedMambaBlock(nn.Module):
    """
    更简化的Mamba实现，用于快速原型验证
    当没有专门的SSM内核时可以使用这个版本
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 简化为RNN-like结构作为占位符
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # 门控机制
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """简化的前向传播"""
        # 使用GRU作为状态空间模型的代理
        output, new_state = self.rnn(hidden_states, state)
        
        # 门控
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        gated_output = gate * output + (1 - gate) * hidden_states
        
        # 输出投影
        final_output = self.out_proj(gated_output)
        
        return final_output, new_state


# TODO: 生产环境应该替换为以下高效实现之一：
# 1. mamba-ssm库的官方实现
# 2. 自定义CUDA内核的并行扫描算法
# 3. Triton实现的优化版本

def create_mamba_block(hidden_size: int, use_simplified: bool = False) -> nn.Module:
    """
    工厂函数：根据环境选择Mamba实现
    
    Args:
        hidden_size: 隐藏层大小
        use_simplified: 是否使用简化版本
        
    Returns:
        MambaBlock实例
    """
    if use_simplified:
        return SimplifiedMambaBlock(hidden_size)
    else:
        return MambaBlock(hidden_size) 