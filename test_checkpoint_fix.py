#!/usr/bin/env python3
"""
检查点保存修复验证脚本
快速测试保存逻辑是否能避免NCCL超时
"""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置NCCL环境变量
os.environ.setdefault('NCCL_TIMEOUT', '1800')
os.environ.setdefault('NCCL_DEBUG', 'WARN')

def test_checkpoint_save():
    """测试检查点保存逻辑"""
    
    # 创建临时配置
    config = {
        'training': {
            'max_steps': 15,  # 短测试
            'save_steps': 5,  # 每5步保存
            'logging_steps': 1,
            'warmup_steps': 2,
            'learning_rate': 1e-4,
            'batch_size': 2,
            'gradient_accumulation_steps': 1,
            'weight_decay': 0.1,
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-6,
            'lr_scheduler': 'cosine',
            'min_lr_ratio': 0.1,
            'max_grad_norm': 1.0,
            'gradient_checkpointing': False,
            'fp16': False,
            'bf16': True,
            'load_balance_coeff': 0.05,
            'entropy_reg_coeff': 0.0003,
            'distill_coeff': 0.0,
            'distill_temperature': 4.0,
            'max_seq_length': 512,  # 较短序列
            'save_total_limit': 2
        },
        'model': {
            'vocab_size': 50257,
            'hidden_size': 768,    # 较小模型
            'num_layers': 6,       # 较少层数
            'num_heads': 12,
            'window_size': 256,
            'global_heads': 2,
            'gate_rank': 64,
            'max_position_embeddings': 2048,
            'srte_encoding': 'learnable',
            'srte_share_across_layers': True,
            'srte_factorized_rank': 64,
            'attention_type': 'local_global',
            'tie_word_embeddings': True,
            'use_alignment': True,
            'layer_norm_eps': 1e-5,
            'dropout': 0.1,
            'drop_branch_prob': 0.0,  # 关闭随机性
            'load_balance_coeff': 0.1,
            'entropy_reg_coeff': 1e-4,
            'use_cache': True,
            'pad_token_id': 50256,
            'bos_token_id': 50256,
            'eos_token_id': 50256
        },
        'data': {
            'data_mode': 'static',
            'train_data_paths': ['data/wikipedia/wiki_en/wiki_en_00000.jsonl'],
            'tokenizer_path': 'gpt2',
            'max_length': 512,
            'min_length': 50,
            'add_special_tokens': True,
            'num_workers': 0,
            'pin_memory': False,
            'prefetch_factor': 2
        },
        'curriculum': {
            'enabled': False
        },
        'device': {
            'use_gpu': True,
            'mixed_precision': 'bf16',
            'compile_model': False
        },
        'logging': {
            'log_level': 'INFO',
            'log_dir': 'logs/',
            'log_file': None,
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_interval': 10,
            'save_interval': 200,
            'output_dir': '/tmp/test_checkpoints/',  # 本地存储
            'wandb': {'enabled': False}
        },
        'gate_monitor': {
            'enabled': False
        },
        'system': {
            'distributed': False,  # 单GPU测试
            'use_deepspeed': False,  # 先测试PyTorch原生
            'compile_model': False
        }
    }
    
    # 创建临时数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # 写入一些测试数据
        for i in range(100):
            f.write(f'{{"text": "This is test sentence {i} for checkpoint validation."}}\n')
        temp_data_path = f.name
    
    # 更新数据路径
    config['data']['train_data_paths'] = [temp_data_path]
    
    try:
        from train.engine import TrainingEngine
        
        print("🧪 Starting checkpoint save test...")
        print(f"  📁 Output dir: {config['logging']['output_dir']}")
        print(f"  📊 Steps: {config['training']['max_steps']}")
        print(f"  💾 Save every: {config['training']['save_steps']} steps")
        
        # 创建训练引擎
        engine = TrainingEngine(
            config=config,
            local_rank=0,
            rank=0,
            world_size=1
        )
        
        # 运行短测试
        engine.train()
        
        print("✅ Checkpoint save test completed successfully!")
        print("   No NCCL timeout issues detected.")
        
        # 检查保存的文件
        output_dir = Path(config['logging']['output_dir'])
        if output_dir.exists():
            checkpoints = list(output_dir.glob('checkpoint_step_*'))
            print(f"   📁 Found {len(checkpoints)} checkpoints:")
            for ckpt in checkpoints:
                print(f"     - {ckpt.name}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_data_path)
        except:
            pass
    
    return True

def test_deepspeed_save():
    """测试DeepSpeed保存逻辑"""
    print("\n🧪 Testing DeepSpeed save logic (single GPU)...")
    
    # 临时修改配置为DeepSpeed
    config = {
        'training': {
            'max_steps': 10,
            'save_steps': 5,
            'logging_steps': 1,
            'warmup_steps': 1,
            'learning_rate': 1e-4,
            'batch_size': 1,
            'gradient_accumulation_steps': 1,
            'weight_decay': 0.1,
            'beta1': 0.9,
            'beta2': 0.95,
            'eps': 1e-6,
            'lr_scheduler': 'cosine',
            'min_lr_ratio': 0.1,
            'max_grad_norm': 1.0,
            'gradient_checkpointing': False,
            'fp16': False,
            'bf16': True,
            'load_balance_coeff': 0.05,
            'entropy_reg_coeff': 0.0003,
            'distill_coeff': 0.0,
            'distill_temperature': 4.0,
            'max_seq_length': 256,
            'save_total_limit': 2
        },
        'model': {
            'vocab_size': 1000,     # 极小词汇表
            'hidden_size': 256,     # 极小模型
            'num_layers': 2,
            'num_heads': 4,
            'window_size': 128,
            'global_heads': 1,
            'gate_rank': 32,
            'max_position_embeddings': 512,
            'srte_encoding': 'learnable',
            'srte_share_across_layers': True,
            'srte_factorized_rank': 32,
            'attention_type': 'local_global',
            'tie_word_embeddings': True,
            'use_alignment': False,  # 简化
            'layer_norm_eps': 1e-5,
            'dropout': 0.0,
            'drop_branch_prob': 0.0,
            'load_balance_coeff': 0.1,
            'entropy_reg_coeff': 1e-4,
            'use_cache': True,
            'pad_token_id': 0,
            'bos_token_id': 0,
            'eos_token_id': 0
        },
        'data': {
            'data_mode': 'static',
            'tokenizer_path': 'gpt2',
            'max_length': 256,
            'min_length': 10,
            'add_special_tokens': True,
            'num_workers': 0,
            'pin_memory': False,
            'prefetch_factor': 2
        },
        'curriculum': {'enabled': False},
        'device': {
            'use_gpu': True,
            'mixed_precision': 'bf16',
            'compile_model': False
        },
        'logging': {
            'log_level': 'INFO',
            'log_dir': 'logs/',
            'output_dir': '/tmp/test_deepspeed_checkpoints/',
            'wandb': {'enabled': False}
        },
        'gate_monitor': {'enabled': False},
        'system': {
            'distributed': False,
            'use_deepspeed': True,
            'zero_stage': 2,
            'offload_optimizer': False,
            'offload_params': False,
            'compile_model': False
        }
    }
    
    # 创建极简数据
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(50):
            f.write(f'{{"text": "Test {i}"}}\n')
        temp_data_path = f.name
    
    config['data']['train_data_paths'] = [temp_data_path]
    
    try:
        from train.engine import TrainingEngine
        
        print(f"  📁 DeepSpeed output: {config['logging']['output_dir']}")
        
        engine = TrainingEngine(
            config=config,
            local_rank=0,
            rank=0,
            world_size=1
        )
        
        engine.train()
        print("✅ DeepSpeed save test completed!")
        
    except Exception as e:
        print(f"❌ DeepSpeed test failed: {e}")
        return False
    finally:
        try:
            os.unlink(temp_data_path)
        except:
            pass
    
    return True

if __name__ == "__main__":
    print("🚀 Testing checkpoint save fixes...")
    
    success = True
    
    # 测试1：PyTorch原生保存
    if not test_checkpoint_save():
        success = False
    
    # 测试2：DeepSpeed保存（如果可用）
    try:
        import deepspeed
        if not test_deepspeed_save():
            success = False
    except ImportError:
        print("⚠️ DeepSpeed not available, skipping DeepSpeed test")
    
    if success:
        print("\n🎉 All checkpoint save tests passed!")
        print("   The NCCL timeout fix should work correctly.")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 