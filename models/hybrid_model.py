"""
HybridLanguageModel: 完整的混合架构语言模型
包含词嵌入、多层HybridBlock堆叠、语言建模头等组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import json

from .hybrid_block import HybridBlock, load_balance_loss, gate_entropy_regularization
from .local_global_attn import create_attention_layer


class HybridLanguageModelConfig:
    """混合模型配置类"""
    
    def __init__(
        self,
        vocab_size: int = 50432,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        window_size: int = 1024,
        global_heads: int = 2,
        gate_rank: int = 256,
        max_position_embeddings: int = 65536,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.1,
        drop_branch_prob: float = 0.1,
        attention_type: str = "local_global",  # "local_global" or "pyramidal"
        srte_encoding: str = "learnable",  # "learnable" or "sincos"
        # SRTE优化选项
        srte_share_across_layers: bool = True,  # 是否在层间共享SRTE
        srte_factorized_rank: int = 0,  # 低秩分解rank，0表示不使用
        use_alignment: bool = True,
        tie_word_embeddings: bool = False,
        # 训练相关
        load_balance_coeff: float = 0.1,
        entropy_reg_coeff: float = 1e-4,
        # 推理相关
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.window_size = window_size
        self.global_heads = global_heads
        self.gate_rank = gate_rank
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.drop_branch_prob = drop_branch_prob
        self.attention_type = attention_type
        self.srte_encoding = srte_encoding
        self.srte_share_across_layers = srte_share_across_layers
        self.srte_factorized_rank = srte_factorized_rank
        self.use_alignment = use_alignment
        self.tie_word_embeddings = tie_word_embeddings
        self.load_balance_coeff = load_balance_coeff
        self.entropy_reg_coeff = entropy_reg_coeff
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridLanguageModelConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory: str):
        """保存配置到目录"""
        import os
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class HybridLanguageModel(nn.Module):
    """
    混合架构语言模型
    
    结合Mamba（长程语义）和Attention（局部细节）的并行双分支架构
    通过轻门控机制动态融合两分支输出
    """
    
    def __init__(self, config: HybridLanguageModelConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # 创建共享SRTE（如果配置启用）
        self.shared_srte = None
        if config.srte_share_across_layers:
            from .hybrid_block import SRTE
            self.shared_srte = SRTE(
                config.hidden_size, 
                max_len=config.max_position_embeddings, 
                encoding_type=config.srte_encoding,
                factorized_rank=config.srte_factorized_rank
            )
            print(f"Created shared SRTE for {config.num_layers} layers")
        
        # 混合块堆叠
        self.layers = nn.ModuleList([
            HybridBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                window_size=self._get_layer_window_size(i),
                global_heads=config.global_heads,
                gate_rank=config.gate_rank,
                drop_branch_prob=config.drop_branch_prob,
                srte_encoding=config.srte_encoding,
                srte_max_len=config.max_position_embeddings,
                srte_shared=self.shared_srte,
                srte_factorized_rank=config.srte_factorized_rank,
                use_alignment=config.use_alignment
            )
            for i in range(config.num_layers)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 语言建模头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重绑定
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _get_layer_window_size(self, layer_idx: int) -> int:
        """获取指定层的窗口大小（支持金字塔注意力）"""
        if self.config.attention_type == "pyramidal":
            base_size = 512
            max_size = self.config.window_size
            ratio = layer_idx / max(1, self.config.num_layers - 1)
            return int(base_size + (max_size - base_size) * ratio)
        else:
            return self.config.window_size
    
    def _init_weights(self, module: nn.Module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """设置输入嵌入层"""
        self.embed_tokens = value
    
    def get_output_embeddings(self) -> nn.Linear:
        """获取输出嵌入层"""
        return self.lm_head
    
    def set_output_embeddings(self, value: nn.Linear):
        """设置输出嵌入层"""
        self.lm_head = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Dict[str, torch.Tensor]]] = None,
        past_mamba_states: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
        collect_gate_stats: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [B, L]
            attention_mask: 注意力掩码 [B, L]
            past_key_values: 历史KV缓存列表
            labels: 训练标签 [B, L]
            use_cache: 是否使用缓存
            return_dict: 是否返回字典格式
            collect_gate_stats: 是否收集门控统计信息
            
        Returns:
            模型输出字典
        """
        if use_cache is None:
            use_cache = self.config.use_cache and not self.training
        
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # 处理注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)
        
        # 初始化缓存和状态
        if past_key_values is None:
            past_key_values = [None] * self.config.num_layers
        if past_mamba_states is None:
            past_mamba_states = [None] * self.config.num_layers
        
        present_key_values = []
        present_mamba_states = []
        all_gate_stats = []
        
        # 通过各层
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                kv_cache=past_key_values[i],
                mamba_state=past_mamba_states[i],
                training=self.training,
                collect_gate_stats=collect_gate_stats,
                use_cache=use_cache
            )
            
            hidden_states = layer_outputs[0]
            present_kv = layer_outputs[1]
            new_mamba_state = layer_outputs[2]
            aux_outputs = layer_outputs[3]
            
            if use_cache:
                present_key_values.append(present_kv)
                present_mamba_states.append(new_mamba_state)
            
            if aux_outputs is not None:
                all_gate_stats.append(aux_outputs)
        
        # 最终层归一化
        hidden_states = self.norm(hidden_states)
        
        # 计算logits
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        aux_losses = {}
        
        if labels is not None:
            # 语言建模损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 展平用于交叉熵计算
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 只计算语言建模损失，正则化损失在train/losses.py中统一处理
            loss = lm_loss
            aux_losses['lm_loss'] = lm_loss
        
        # 构建输出
        if not return_dict:
            outputs = (logits,)
            if use_cache:
                outputs += (present_key_values,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
        
        result = {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values if use_cache else None,
            'past_mamba_states': present_mamba_states if use_cache else None,
            'aux_losses': aux_losses,
            'gate_stats': all_gate_stats if collect_gate_stats else None
        }
        
        return result
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
        past_mamba_states: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """为生成准备输入"""
        if past_key_values is not None:
            # 只保留最后一个token
            input_ids = input_ids[:, -1:]
        
        # 注意：保持完整 attention_mask 以正确屏蔽历史 padding
        # if attention_mask is not None and past_key_values is not None:
        #     attention_mask = attention_mask[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "past_mamba_states": past_mamba_states,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        简单的贪心/采样生成
        
        Args:
            input_ids: 输入序列 [B, L]
            max_length: 最大总长度
            max_new_tokens: 最大新生成token数
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            do_sample: 是否采样
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
            attention_mask: 注意力掩码
            
        Returns:
            生成的序列 [B, L']
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 确定生成长度
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        
        # 初始化
        generated = input_ids.clone()
        past_key_values = None
        past_mamba_states = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_length - input_ids.shape[1]):
            # 准备输入
            model_inputs = self.prepare_inputs_for_generation(
                generated, past_key_values, past_mamba_states, attention_mask
            )
            
            # 前向传播
            outputs = self(**model_inputs)
            logits = outputs['logits']
            past_key_values = outputs['past_key_values']
            past_mamba_states = outputs['past_mamba_states']
            
            # 获取下一个token的logits
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 获取下一个token
            if do_sample:
                next_tokens = self._sample_next_token(
                    next_token_logits, top_k, top_p
                )
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 如果已经结束，使用pad token
            next_tokens = next_tokens.masked_fill(finished, pad_token_id)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            # 检查是否结束
            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)
                if finished.all():
                    break
        
        return generated
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """采样下一个token"""
        if top_k is not None:
            # top-k采样
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p is not None:
            # nucleus采样
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_tokens


def create_hybrid_model(
    vocab_size: int = 50432,
    hidden_size: int = 4096,
    num_layers: int = 32,
    **kwargs
) -> HybridLanguageModel:
    """
    创建混合模型的工厂函数
    
    Args:
        vocab_size: 词汇表大小
        hidden_size: 隐藏层大小
        num_layers: 层数
        **kwargs: 其他配置参数
        
    Returns:
        HybridLanguageModel实例
    """
    config = HybridLanguageModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    )
    
    return HybridLanguageModel(config) 