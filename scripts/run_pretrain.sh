#!/bin/bash

# FusionTree 预训练脚本
# 使用方法: bash scripts/run_pretrain.sh [config_file] [num_gpus]

set -e

# 默认配置
CONFIG_FILE=${1:-"configs/pretrain_v1.yaml"}
NUM_GPUS=${2:-8}
MASTER_PORT=${3:-29500}

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== FusionTree 预训练启动 ==="
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS"
echo "项目根目录: $PROJECT_ROOT"
echo "============================="

# 检查配置文件
if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# DeepSpeed内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

# 创建输出目录
mkdir -p "$PROJECT_ROOT/checkpoints"
mkdir -p "$PROJECT_ROOT/logs"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/pretrain_${TIMESTAMP}.log"

echo "日志文件: $LOG_FILE"

# 启动训练
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "启动分布式训练 (${NUM_GPUS} GPUs)..."
    
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train/engine.py \
        --config "$CONFIG_FILE" \
        --distributed \
        2>&1 | tee "$LOG_FILE"
else
    echo "启动单GPU训练..."
    
    python train/engine.py \
        --config "$CONFIG_FILE" \
        2>&1 | tee "$LOG_FILE"
fi

echo "训练完成！日志保存在: $LOG_FILE" 