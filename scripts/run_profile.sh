#!/bin/bash

# FusionTree 性能分析脚本
# 使用方法: bash scripts/run_profile.sh [model_path] [profile_type]

set -e

# 默认配置
MODEL_PATH=${1:-"checkpoints/v1/best_model"}
PROFILE_TYPE=${2:-"all"}  # "memory", "speed", "flops", "all"

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== FusionTree 性能分析 ==="
echo "模型路径: $MODEL_PATH"
echo "分析类型: $PROFILE_TYPE"
echo "========================="

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
mkdir -p "$PROJECT_ROOT/profile_results"
mkdir -p "$PROJECT_ROOT/logs"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="$PROJECT_ROOT/profile_results/profile_${TIMESTAMP}"
LOG_FILE="$PROJECT_ROOT/logs/profile_${TIMESTAMP}.log"

mkdir -p "$RESULT_DIR"

echo "结果目录: $RESULT_DIR"
echo "日志文件: $LOG_FILE"

# 内存分析
if [ "$PROFILE_TYPE" = "memory" ] || [ "$PROFILE_TYPE" = "all" ]; then
    echo "开始内存分析..."
    
    python -c "
import torch
import torch.profiler
import sys
import os
sys.path.append('$PROJECT_ROOT')

from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig

# 加载模型
config = HybridLanguageModelConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridLanguageModel(config).to(device)
model.eval()

# 内存分析
batch_sizes = [1, 2, 4, 8]
seq_lengths = [1024, 4096, 8192, 16384, 32768]

results = {}
for bs in batch_sizes:
    for seq_len in seq_lengths:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            input_ids = torch.randint(0, 1000, (bs, seq_len), device='cuda')
            
            with torch.no_grad():
                outputs = model(input_ids)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            results[f'bs{bs}_len{seq_len}'] = {
                'memory_gb': memory_used,
                'batch_size': bs,
                'sequence_length': seq_len
            }
            
            print(f'Batch={bs}, SeqLen={seq_len}: {memory_used:.2f} GB')
            
        except RuntimeError as e:
            print(f'OOM: Batch={bs}, SeqLen={seq_len}')
            results[f'bs{bs}_len{seq_len}'] = {'error': 'OOM'}

import json
with open('$RESULT_DIR/memory_profile.json', 'w') as f:
    json.dump(results, f, indent=2)
" 2>&1 | tee -a "$LOG_FILE"

fi

# 速度分析
if [ "$PROFILE_TYPE" = "speed" ] || [ "$PROFILE_TYPE" = "all" ]; then
    echo "开始速度分析..."
    
    python -c "
import torch
import time
import sys
import os
sys.path.append('$PROJECT_ROOT')

from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig

# 加载模型
config = HybridLanguageModelConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridLanguageModel(config).to(device)
model.eval()

def benchmark_speed(model, batch_size, seq_length, num_runs=10):
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    
    # 预热
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    
    # 测试 prefill
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = model(input_ids)
        torch.cuda.synchronize()
    prefill_time = (time.time() - start_time) / num_runs
    
    # 测试 decode (单token)
    past_kv = outputs.get('past_key_values')
    decode_input = input_ids[:, -1:] 
    
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(decode_input, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()
    decode_time = (time.time() - start_time) / num_runs
    
    return {
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'prefill_tokens_per_sec': batch_size * seq_length / prefill_time,
        'decode_tokens_per_sec': batch_size / decode_time
    }

# 测试不同配置
configs = [
    (1, 1024), (1, 4096), (1, 8192), (1, 16384),
    (2, 1024), (2, 4096), (4, 1024)
]

results = {}
for bs, seq_len in configs:
    try:
        print(f'Testing batch_size={bs}, seq_length={seq_len}...')
        result = benchmark_speed(model, bs, seq_len)
        results[f'bs{bs}_len{seq_len}'] = result
        
        print(f'  Prefill: {result[\"prefill_tokens_per_sec\"]:.1f} tokens/s')
        print(f'  Decode: {result[\"decode_tokens_per_sec\"]:.1f} tokens/s')
        
    except Exception as e:
        print(f'Error testing bs={bs}, seq_len={seq_len}: {e}')
        results[f'bs{bs}_len{seq_len}'] = {'error': str(e)}

import json
with open('$RESULT_DIR/speed_profile.json', 'w') as f:
    json.dump(results, f, indent=2)
" 2>&1 | tee -a "$LOG_FILE"

fi

# FLOPS分析
if [ "$PROFILE_TYPE" = "flops" ] || [ "$PROFILE_TYPE" = "all" ]; then
    echo "开始FLOPS分析..."
    
    # 需要安装 thop 或 ptflops
    pip install thop ptflops --quiet 2>/dev/null || echo "Warning: thop/ptflops not installed"
    
    python -c "
import torch
import sys
import os
sys.path.append('$PROJECT_ROOT')

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print('Warning: thop not available, skipping FLOPS analysis')

if THOP_AVAILABLE:
    from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig
    
    config = HybridLanguageModelConfig(
        hidden_size=1024,  # 较小模型用于FLOPS分析
        num_layers=12,
        num_heads=16
    )
    model = HybridLanguageModel(config)
    
    # 分析不同序列长度的FLOPS
    seq_lengths = [1024, 4096, 8192]
    results = {}
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, 1000, (1, seq_len))
        
        try:
            flops, params = profile(model, inputs=(input_ids,))
            results[f'seq_len_{seq_len}'] = {
                'flops': flops,
                'params': params,
                'flops_per_token': flops / seq_len
            }
            print(f'Seq Length {seq_len}: {flops/1e9:.2f} GFLOPs, {params/1e6:.2f}M params')
            
        except Exception as e:
            print(f'Error analyzing seq_len={seq_len}: {e}')
            results[f'seq_len_{seq_len}'] = {'error': str(e)}
    
    import json
    with open('$RESULT_DIR/flops_profile.json', 'w') as f:
        json.dump(results, f, indent=2)
" 2>&1 | tee -a "$LOG_FILE"

fi

# 生成综合报告
echo "生成综合报告..."
python -c "
import json
import os
import glob

result_dir = '$RESULT_DIR'
files = glob.glob(os.path.join(result_dir, '*.json'))

print('\n=== 性能分析报告 ===')

for file in files:
    profile_type = os.path.basename(file).replace('.json', '')
    print(f'\n{profile_type.upper()}:')
    
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            
        if 'memory' in profile_type:
            max_memory = 0
            for key, value in data.items():
                if 'memory_gb' in value:
                    max_memory = max(max_memory, value['memory_gb'])
            print(f'  最大内存使用: {max_memory:.2f} GB')
            
        elif 'speed' in profile_type:
            best_prefill = 0
            best_decode = 0
            for key, value in data.items():
                if 'prefill_tokens_per_sec' in value:
                    best_prefill = max(best_prefill, value['prefill_tokens_per_sec'])
                if 'decode_tokens_per_sec' in value:
                    best_decode = max(best_decode, value['decode_tokens_per_sec'])
            print(f'  最佳Prefill速度: {best_prefill:.1f} tokens/s')
            print(f'  最佳Decode速度: {best_decode:.1f} tokens/s')
            
        elif 'flops' in profile_type:
            for key, value in data.items():
                if 'flops' in value:
                    print(f'  {key}: {value[\"flops\"]/1e9:.2f} GFLOPs')
                    
    except Exception as e:
        print(f'  错误读取 {file}: {e}')

print('\n==================')
print(f'详细结果保存在: {result_dir}')
"

echo "性能分析完成！"
echo "结果保存在: $RESULT_DIR"
echo "日志保存在: $LOG_FILE" 