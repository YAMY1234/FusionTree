#!/usr/bin/env python3
"""
模型参数量统计脚本
用于分析HybridLanguageModel各模块的参数分布
"""

import torch
from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig

def profile_model_weights(config_dict=None):
    """分析模型参数量分布"""
    
    # 默认配置（当前的重参数配置）
    if config_dict is None:
        config_dict = {
            'vocab_size': 50432,
            'hidden_size': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'window_size': 1024,
            'global_heads': 2,
            'gate_rank': 256,
            'max_position_embeddings': 65536,
            'srte_encoding': "learnable",
            'srte_share_across_layers': True,  # 默认启用共享
            'srte_factorized_rank': 0,  # 默认不使用低秩
            'tie_word_embeddings': False
        }
    
    print("=== 模型配置 ===")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print()
    
    # 创建模型
    cfg = HybridLanguageModelConfig(**config_dict)
    model = HybridLanguageModel(cfg)
    
    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print("=== 参数量统计 ===")
    print(f"总参数量: {total_params/1e9:.3f}B ({total_params:,})")
    print()
    
    # 各模块参数量
    embed_params = model.embed_tokens.weight.numel()
    lm_head_params = model.lm_head.weight.numel()
    norm_params = sum(p.numel() for p in model.norm.parameters())
    
    print("=== 主要组件 ===")
    print(f"词嵌入 (embed_tokens): {embed_params/1e6:.1f}M ({embed_params:,})")
    print(f"语言建模头 (lm_head): {lm_head_params/1e6:.1f}M ({lm_head_params:,})")
    print(f"最终LayerNorm: {norm_params/1e6:.3f}M ({norm_params:,})")
    print()
    
    # 逐层分析
    layer_params = []
    srte_total = 0
    mamba_total = 0
    attn_total = 0
    gate_total = 0
    mlp_total = 0
    alignment_total = 0
    
    # 先检查是否有共享SRTE
    if hasattr(model, 'shared_srte') and model.shared_srte is not None:
        srte_total = sum(p.numel() for p in model.shared_srte.parameters())
        print(f"Found shared SRTE with {srte_total:,} parameters")
    
    for i, layer in enumerate(model.layers):
        layer_param_count = sum(p.numel() for p in layer.parameters())
        layer_params.append(layer_param_count)
        
        # 统计各子模块
        srte_params = sum(p.numel() for n, p in layer.named_parameters() if n.startswith('srte'))
        mamba_params = sum(p.numel() for n, p in layer.named_parameters() if n.startswith('mamba'))
        attn_params = sum(p.numel() for n, p in layer.named_parameters() if n.startswith('attention'))
        gate_params = sum(p.numel() for n, p in layer.named_parameters() if n.startswith('gate'))
        mlp_params = sum(p.numel() for n, p in layer.named_parameters() if n.startswith('small_mlp'))
        alignment_params = sum(p.numel() for n, p in layer.named_parameters() if n.startswith('alignment'))
        
        # 只有非共享模式才累加各层SRTE
        if not (hasattr(model, 'shared_srte') and model.shared_srte is not None):
            srte_total += srte_params
        mamba_total += mamba_params
        attn_total += attn_params
        gate_total += gate_params
        mlp_total += mlp_params
        alignment_total += alignment_params
        
        if i < 3 or i >= cfg.num_layers - 3:  # 显示前3层和后3层
            print(f"第 {i:2d} 层: {layer_param_count/1e6:6.1f}M | "
                  f"SRTE: {srte_params/1e6:5.1f}M | "
                  f"Mamba: {mamba_params/1e6:5.1f}M | "
                  f"Attn: {attn_params/1e6:5.1f}M | "
                  f"Gate: {gate_params/1e6:4.1f}M | "
                  f"MLP: {mlp_params/1e6:5.1f}M")
    
    print(f"...")
    print()
    
    print("=== 各子模块汇总 ===")
    print(f"SRTE 总计    : {srte_total/1e9:6.3f}B ({srte_total/total_params*100:5.1f}%)")
    print(f"Mamba 总计   : {mamba_total/1e9:6.3f}B ({mamba_total/total_params*100:5.1f}%)")
    print(f"Attention 总计: {attn_total/1e9:6.3f}B ({attn_total/total_params*100:5.1f}%)")
    print(f"AlignmentMLP : {alignment_total/1e9:6.3f}B ({alignment_total/total_params*100:5.1f}%)")
    print(f"Gate 总计    : {gate_total/1e9:6.3f}B ({gate_total/total_params*100:5.1f}%)")
    print(f"Small MLP 总计: {mlp_total/1e9:6.3f}B ({mlp_total/total_params*100:5.1f}%)")
    print()
    
    # SRTE详细分析
    print("=== SRTE 详细分析 ===")
    if cfg.srte_encoding == "learnable":
        single_srte_params = cfg.max_position_embeddings * cfg.hidden_size
        print(f"单层SRTE参数量: {cfg.max_position_embeddings} × {cfg.hidden_size} = {single_srte_params/1e6:.1f}M")
        print(f"共 {cfg.num_layers} 层 × {single_srte_params/1e6:.1f}M = {srte_total/1e9:.3f}B")
        print(f"SRTE占总参数量的 {srte_total/total_params*100:.1f}%")
    else:
        print(f"sincos编码模式，SRTE参数量: {srte_total/1e6:.3f}M")
    print()
    
    return {
        'total_params': total_params,
        'srte_params': srte_total,
        'mamba_params': mamba_total,
        'attn_params': attn_total,
        'alignment_params': alignment_total,
        'gate_params': gate_total,
        'mlp_params': mlp_total,
        'embed_params': embed_params,
        'lm_head_params': lm_head_params
    }

def compare_configs():
    """比较不同配置下的参数量"""
    
    print("=" * 80)
    print("配置对比分析")
    print("=" * 80)
    
    # 原始配置
    print("\n【当前配置 - 重参数版本】")
    current_stats = profile_model_weights()
    
    # 优化配置1：共享SRTE + sincos
    print("\n【优化配置1 - 共享SRTE + sincos】")
    optimized1_config = {
        'vocab_size': 50432,
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'window_size': 1024,
        'global_heads': 2,
        'gate_rank': 256,
        'max_position_embeddings': 32768,  # 减小
        'srte_encoding': "sincos",
        'tie_word_embeddings': True,  # 绑定权重
        'srte_share_across_layers': True,  # 需要新增
        'srte_factorized_rank': 0
    }
    print("注意：这需要代码修改支持")
    
    # 优化配置2：低秩SRTE
    print("\n【优化配置2 - 低秩learnable SRTE】")
    optimized2_config = {
        'vocab_size': 50432,
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'window_size': 1024,
        'global_heads': 2,
        'gate_rank': 256,
        'max_position_embeddings': 32768,
        'srte_encoding': "learnable",
        'tie_word_embeddings': True,
        'srte_share_across_layers': True,
        'srte_factorized_rank': 128  # 低秩分解
    }
    print("注意：这需要代码修改支持")
    
    # 计算节省
    current_total = current_stats['total_params']
    current_srte = current_stats['srte_params']
    
    print(f"\n=== 预估参数节省 ===")
    print(f"当前总参数: {current_total/1e9:.3f}B")
    print(f"当前SRTE参数: {current_srte/1e9:.3f}B ({current_srte/current_total*100:.1f}%)")
    print()
    
    # 共享SRTE节省
    shared_srte_savings = current_srte - (current_srte / 32)  # 只保留1份
    print(f"共享SRTE后节省: {shared_srte_savings/1e9:.3f}B")
    print(f"新总参数: {(current_total - shared_srte_savings)/1e9:.3f}B")
    print()
    
    # 低秩分解额外节省
    if optimized2_config['srte_factorized_rank'] > 0:
        rank = optimized2_config['srte_factorized_rank']
        max_len = optimized2_config['max_position_embeddings']
        hidden_size = optimized2_config['hidden_size']
        
        original_single_srte = max_len * hidden_size
        factorized_single_srte = max_len * rank + rank * hidden_size
        
        print(f"低秩分解(r={rank}):")
        print(f"  原始单SRTE: {max_len} × {hidden_size} = {original_single_srte/1e6:.1f}M")
        print(f"  低秩单SRTE: {max_len} × {rank} + {rank} × {hidden_size} = {factorized_single_srte/1e6:.1f}M")
        print(f"  压缩比: {factorized_single_srte/original_single_srte:.3f}x")
        
        total_factorized_savings = shared_srte_savings + (original_single_srte - factorized_single_srte) / 1e6
        print(f"  总节省: {total_factorized_savings/1e3:.3f}B")
        print(f"  最终总参数: {(current_total - total_factorized_savings*1e6)/1e9:.3f}B")

if __name__ == "__main__":
    # 运行当前配置分析
    profile_model_weights()
    
    # 运行配置对比
    compare_configs() 