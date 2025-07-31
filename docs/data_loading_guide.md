# 🚀 FusionTree 数据加载系统使用指南

FusionTree支持三种数据加载模式，满足不同规模和场景的训练需求：

## 📋 模式对比

| 模式 | 适用场景 | 内存占用 | 启动速度 | 数据规模 | 复杂度 |
|------|----------|----------|----------|----------|--------|
| **static** | 调试、小数据集 | 高 | 慢 | <1GB | 低 |
| **lazy** | 本地大文件 | 低 | 快 | 1-100GB | 中 |
| **hf_streaming** | 生产训练 | 极低 | 极快 | 无限 | 中 |

## 🎯 使用场景选择

### 🔧 Static模式：传统加载（调试优先）
**适用场景：**
- 调试模型和训练流程
- 小数据集（<1GB）
- 需要使用curriculum learning
- 数据预处理复杂（needle-in-haystack等）

**优点：**
- 向后兼容
- 支持所有现有功能
- 数据完全可控

**缺点：**
- 内存占用大
- 启动时间长
- 不适合大数据集

**配置示例：**
```yaml
data:
  data_mode: static
  train_data_paths:
    - "data/small_dataset/*.jsonl"
  concat_docs: true
  enable_packing: true
  num_workers: 4
```

### 💾 Lazy模式：本地大文件（内存优化）
**适用场景：**
- 本地JSONL文件（1-100GB）
- 内存受限环境
- 快速测试大数据集

**优点：**
- 内存占用极低
- 快速启动
- 支持通配符路径

**缺点：**
- 仅支持JSONL格式
- 不支持复杂预处理
- 随机性有限

**配置示例：**
```yaml
data:
  data_mode: lazy
  train_data_paths:
    - "data/wikipedia/wiki_en/*.jsonl"
  max_samples_per_file: 10000  # 限制每文件样本数
  min_length: 50
  num_workers: 0  # 推荐0避免文件访问冲突
```

### 🔥 HF Streaming模式：工业界最佳实践
**适用场景：**
- 生产环境训练
- 多数据源混合
- 超大规模数据集
- 边下边训

**优点：**
- 内存占用极小
- 支持无限数据规模
- 多源按权重混合
- Token级高效打包
- 分布式友好
- 无需本地存储

**缺点：**
- 需要网络连接
- 依赖HuggingFace Hub

## 🚀 HF Streaming模式详解（推荐）

### 预设配置
使用预定义的数据源组合：
```yaml
data:
  data_mode: hf_streaming
  hf_preset: wikipedia_only    # 简单开始
  # 或者: english_mix, multilingual_mix
```

### 自定义多源混合
```yaml
data:
  data_mode: hf_streaming
  hf_streaming_sources:
    # 英文维基百科 - 基础权重1.0
    - name: "wikimedia/wikipedia"
      config: "20231101.en"
      split: "train"
      text_key: "text"
      weight: 1.0
    
    # OpenWebText - 更高权重1.5
    - name: "Skylion007/openwebtext"
      split: "train" 
      text_key: "text"
      weight: 1.5
    
    # C4英文 - 补充权重0.8
    - name: "c4"
      config: "en"
      split: "train"
      text_key: "text"
      weight: 0.8
```

### 高级优化参数
```yaml
data:
  data_mode: hf_streaming
  # ... sources配置 ...
  
  # Token级打包（强烈推荐）
  concat_docs: true
  document_separator: "\n\n"
  
  # 流式优化
  shuffle_buffer_size: 20000  # 更大=更随机，但占用更多内存
  min_length: 100             # 过滤短文本
  
  # 性能调优
  num_workers: 0              # 流式推荐0
  pin_memory: true
  trust_remote_code: false    # 安全考虑
```

## 🛠️ 实际使用示例

### 1. 快速验证（Wikipedia单源）
```bash
# 使用预设的Wikipedia数据
python train_model.py configs/pretrain_hf_streaming.yaml 2
```

### 2. 多语言大规模训练
```bash  
# 多数据源混合训练
python train_model.py configs/pretrain_multilingual_mix.yaml 4
```

### 3. 本地文件测试
```bash
# 使用本地JSONL文件
python train_model.py configs/pretrain_wikipedia.yaml 2
```

## ⚡ 性能对比

**初始化时间对比（80万样本）：**
- Static模式：~15分钟（需要预处理所有数据）
- Lazy模式：~2秒（只建立索引）
- HF Streaming：~5秒（连接数据源）

**内存占用对比（80万样本，1024 token/样本）：**
- Static模式：~12GB（所有数据在内存）
- Lazy模式：~100MB（只有索引和缓冲）
- HF Streaming：~50MB（只有流式缓冲）

**训练吞吐量：**
- 三种模式在训练时的GPU利用率基本相同
- HF Streaming的token级打包能提供最高的序列利用率

## 🔧 迁移指南

### 从Static模式迁移到HF Streaming：

1. **确定数据源：**
   - 本地JSONL → 上传到HF Hub 或使用公开数据集
   - 自定义格式 → 转换为标准HF数据集格式

2. **更新配置：**
   ```yaml
   # 旧配置
   data:
     train_data_paths: ["data/my_data/*.jsonl"]
     concat_docs: true
   
   # 新配置  
   data:
     data_mode: hf_streaming
     hf_streaming_sources:
       - name: "my_username/my_dataset"
         split: "train"
         text_key: "text"
         weight: 1.0
     concat_docs: true
   ```

3. **验证训练：**
   - 先用小步数测试（max_steps: 100）
   - 检查loss变化趋势
   - 确认GPU利用率正常

## 🚨 注意事项

### HF Streaming模式：
- 首次使用时需要下载数据（会有网络延迟）
- 建议在训练环境先测试网络连接
- 某些数据集可能需要`trust_remote_code: true`

### Lazy模式：
- 推荐`num_workers: 0`避免多进程文件冲突
- 文件较多时建立索引可能需要几分钟
- 支持`glob`通配符但不支持递归搜索

### Static模式：
- 大数据集会导致内存溢出
- 初始化时间与数据量成正比
- 保留为向后兼容和特殊需求

## 📈 最佳实践

1. **开发调试**：使用Static模式，小数据集
2. **规模测试**：使用Lazy模式，本地大文件
3. **生产训练**：使用HF Streaming模式，多源混合
4. **超大规模**：HF Streaming + 更大的`shuffle_buffer_size`
5. **多语言**：HF Streaming + 合适的分词器和权重配置

选择合适的模式能显著提升训练效率和开发体验！ 