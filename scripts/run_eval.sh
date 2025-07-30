#!/bin/bash

# FusionTree 评估脚本
# 使用方法: bash scripts/run_eval.sh [model_path] [eval_config] [tasks]

set -e

# 默认配置
MODEL_PATH=${1:-"checkpoints/v1/best_model"}
EVAL_CONFIG=${2:-"configs/eval_config.yaml"}
TASKS=${3:-"all"}

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== FusionTree 模型评估 ==="
echo "模型路径: $MODEL_PATH"
echo "评估配置: $EVAL_CONFIG"
echo "评估任务: $TASKS"
echo "========================="

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p "$PROJECT_ROOT/eval_results"
mkdir -p "$PROJECT_ROOT/logs"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="$PROJECT_ROOT/eval_results/eval_${TIMESTAMP}"
LOG_FILE="$PROJECT_ROOT/logs/eval_${TIMESTAMP}.log"

mkdir -p "$RESULT_DIR"

echo "结果目录: $RESULT_DIR"
echo "日志文件: $LOG_FILE"

# 运行评估
echo "开始评估..."

python eval/eval_llm.py \
    --model_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --output_dir "$RESULT_DIR" \
    --save_predictions \
    --verbose \
    2>&1 | tee "$LOG_FILE"

# 运行系统性能评估
echo "开始系统性能评估..."

python eval/eval_system.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$RESULT_DIR" \
    --profile_memory \
    --profile_speed \
    --batch_sizes "1,2,4,8" \
    --sequence_lengths "1024,4096,8192,16384,32768" \
    2>&1 | tee -a "$LOG_FILE"

echo "评估完成！"
echo "结果保存在: $RESULT_DIR"
echo "日志保存在: $LOG_FILE"

# 生成报告摘要
echo "生成评估报告..."
python -c "
import json
import os
import glob

result_dir = '$RESULT_DIR'
files = glob.glob(os.path.join(result_dir, '*.json'))

print('\n=== 评估结果摘要 ===')
for file in files:
    if 'results' in os.path.basename(file):
        with open(file, 'r') as f:
            data = json.load(f)
            print(f'\n{os.path.basename(file)}:')
            for task, metrics in data.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f'  {task}.{metric}: {value:.4f}')
                        else:
                            print(f'  {task}.{metric}: {value}')
print('==================')
" 