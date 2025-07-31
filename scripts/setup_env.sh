#!/bin/bash

# 设置 OpenMP 线程数（根据 CPU 核心数调整）
export OMP_NUM_THREADS=4

# 设置 Triton 缓存目录到本地非 NFS 路径
export TRITON_CACHE_DIR=/tmp/triton_cache_$(whoami)
mkdir -p $TRITON_CACHE_DIR

# 禁用 CUDA expandable_segments 警告
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# 减少 DeepSpeed 日志冗余
export DEEPSPEED_LOG_LEVEL=WARNING

# 设置 NCCL 参数以优化分布式通信
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# 设置 CUDA 可见设备（如果需要）
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Environment variables set for optimized training:"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "DEEPSPEED_LOG_LEVEL=$DEEPSPEED_LOG_LEVEL" 