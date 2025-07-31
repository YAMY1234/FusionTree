#!/bin/bash

# DeepSpeed安装和配置脚本

echo "=== 安装和配置DeepSpeed ==="

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

# 安装DeepSpeed
echo "正在安装DeepSpeed..."
pip install deepspeed

# 验证安装
echo "验证DeepSpeed安装..."
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# 检查CUDA和编译环境
echo "检查编译环境..."
ds_report

echo "=== DeepSpeed设置完成 ==="
echo "建议的环境变量已设置："
echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "export DS_BUILD_CPU_ADAM=1"  
echo "export DS_BUILD_UTILS=1" 