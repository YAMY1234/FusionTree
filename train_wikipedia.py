#!/usr/bin/env python3
"""
Wikipedia数据集训练脚本
使用真实Wikipedia数据进行混合模型预训练
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
import torch
import torch.distributed as dist

# 🚀 防止NCCL超时的环境变量设置（在import任何分布式相关模块前设置）
os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30分钟超时（默认10分钟）
os.environ.setdefault('NCCL_DEBUG', 'WARN')    # 调试模式改为WARN，减少日志噪音
os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '0')  # 生产环境关闭阻塞等待
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')  # 异步错误处理
os.environ.setdefault('OMP_NUM_THREADS', '1')   # 避免CPU过度订阅
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')  # 异步CUDA启动

# 项目路径设置
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入训练相关模块
from train.engine import TrainingEngine
from utils.config import load_config
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Wikipedia数据集训练")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--num_gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--master_port', type=str, default='29500', help='分布式训练端口')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    return parser.parse_args()

def setup_distributed(num_gpus: int, master_port: str):
    """设置分布式训练环境"""
    if num_gpus > 1:
        # 设置分布式相关环境变量
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', master_port)
        
        # 🚀 增加分布式超时时间防止保存检查点时超时
        from datetime import timedelta
        timeout = timedelta(seconds=1800)  # 30分钟
        
        # 初始化分布式环境
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                timeout=timeout,  # 防止NCCL超时
                world_size=num_gpus,
                rank=int(os.environ.get('LOCAL_RANK', 0))
            )
            
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        
        return local_rank, dist.get_rank(), dist.get_world_size()
    else:
        return 0, 0, 1

def main():
    args = parse_args()
    
    # 🚀 设置更详细的NCCL调试信息（仅在调试模式）
    if args.debug:
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'  # 调试时开启阻塞等待
        print("🔧 Debug mode enabled: NCCL_DEBUG=INFO, BLOCKING_WAIT=1")
    
    # 设置分布式环境
    local_rank, rank, world_size = setup_distributed(args.num_gpus, args.master_port)
    
    # 设置日志（只在rank0设置文件日志）
    logger = setup_logger(
        name="wikipedia_train",
        level=logging.DEBUG if args.debug else logging.INFO,
        log_to_file=(rank == 0)
    )
    
    if rank == 0:
        logger.info("🚀 Starting Wikipedia training...")
        logger.info(f"  📊 GPUs: {args.num_gpus}")
        logger.info(f"  📁 Config: {args.config}")
        logger.info(f"  🌐 World size: {world_size}, Rank: {rank}")
        logger.info(f"  🔧 NCCL timeout: {os.environ.get('NCCL_TIMEOUT', 'default')}s")
        
        # 检查保存路径是否为网络盘（简单启发式检查）
        config = load_config(args.config)
        output_dir = config['logging']['output_dir']
        if any(indicator in output_dir.lower() for indicator in ['nfs', 'ceph', 'hdfs', 'gpfs']):
            logger.warning(f"⚠️ Checkpoint path appears to be network storage: {output_dir}")
            logger.warning("   Consider using local storage (/tmp, /local) for better I/O performance")
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 创建训练引擎
        engine = TrainingEngine(
            config=config,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size
        )
        
        # 开始训练
        engine.train(resume_path=args.resume)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if args.debug:
            import traceback
            logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()
        
        if rank == 0:
            logger.info("✅ Training script completed")

if __name__ == "__main__":
    main() 