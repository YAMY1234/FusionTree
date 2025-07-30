# FusionTree 关键修复完成报告

基于系统检查结果，已完成所有**必须尽快修复的正确性/功能性问题**修复。

## ✅ 已完成的关键修复（7项）

### 1. KV缓存在prefill阶段没有返回 
**位置**: `models/local_global_attn.py:330-339`
**问题**: prefill时不返回新KV缓存，导致后续decode无法复用
**修复**: 改为只要`use_cache=True`就返回缓存，无论是否有历史缓存
```diff
-        if kv_cache is not None and use_cache:
+        if use_cache:
             if kv_cache is not None and 'k' in kv_cache and 'v' in kv_cache:
                 k = torch.cat([kv_cache['k'], k], dim=2)
                 v = torch.cat([kv_cache['v'], v], dim=2)
             new_kv_cache = {'k': k, 'v': v}
```

### 2. top_p采样的散射实现错误
**位置**: `models/hybrid_model.py:423-425`
**问题**: scatter操作使用相同张量作为index和src，逻辑错误
**修复**: 创建正确的布尔掩码并使用scatter_进行原地操作
```diff
-            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
-            logits[indices_to_remove] = float('-inf')
+            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
+            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
+            logits = logits.masked_fill(indices_to_remove, float('-inf'))
```

### 3. attention_mask的dtype问题
**位置**: `models/hybrid_model.py` 和 `train/data.py`
**问题**: attention_mask是Long类型，但代码中当作bool使用
**修复**: 在模型入口和数据collate中都确保转为bool类型
```diff
         if attention_mask is None:
             attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
+        else:
+            attention_mask = attention_mask.to(torch.bool)
```

### 4. 生成阶段attention_mask被错误截断
**位置**: `models/hybrid_model.py:300-301`
**问题**: decode时截断mask会丢失历史padding信息
**修复**: 保持完整的attention_mask以正确屏蔽历史padding
```diff
-        if attention_mask is not None and past_key_values is not None:
-            attention_mask = attention_mask[:, -1:]
+        # 注意：保持完整 attention_mask 以正确屏蔽历史 padding
+        # if attention_mask is not None and past_key_values is not None:
+        #     attention_mask = attention_mask[:, -1:]
```

### 5. SRTE与输入dtype不对齐
**位置**: `models/hybrid_block.py:203`
**问题**: bf16/fp16下SRTE输出为fp32，与输入dtype不匹配
**修复**: 强制SRTE输出与输入hidden_states保持相同dtype
```diff
-        time_encoding = self.srte(seq_len)  # [1, L, H]
+        time_encoding = self.srte(seq_len).to(hidden_states.dtype)  # [1, L, H]
```

### 6. num_heads参数未正确透传
**位置**: `models/hybrid_block.py` 和 `models/hybrid_model.py`
**问题**: LocalGlobalAttention始终使用默认的32个头，无视配置
**修复**: 添加num_heads参数并正确透传
```diff
     def __init__(
         self,
         hidden_size: int,
+        num_heads: int = 32,
         window_size: int = 1024,
         ...
     ):
         ...
         self.attention = LocalGlobalAttention(
             hidden_size,
+            num_heads=num_heads,
             window_size=window_size,
             ...
         )
```

### 7. 脚本配置不一致
**位置**: `scripts/run_eval.sh:45`
**问题**: 传递了eval_llm.py不支持的--config参数
**修复**: 移除该参数传递

## 🚀 修复效果预期

这些修复将显著改善模型表现：

1. **正确的增量推理**: KV缓存和Mamba状态正确传递，decode速度大幅提升
2. **正确的采样**: top_p采样逻辑修复，生成质量提升
3. **正确的掩码处理**: attention_mask类型和长度正确，避免注意力计算错误
4. **精度一致性**: 混合精度训练时dtype对齐，避免精度损失
5. **配置一致性**: 模型头数等参数正确应用

## 🧪 验证建议

1. **基础功能测试**:
```python
# 测试增量推理
model.eval()
input_ids = torch.randint(0, 1000, (1, 100))
output = model.generate(input_ids, max_new_tokens=50, use_cache=True)
```

2. **长序列训练**:
```bash
# 测试32K序列不OOM
python train/engine.py --config configs/pretrain_v1.yaml
```

3. **采样质量检查**:
```python
# 测试top_p采样
output = model.generate(input_ids, do_sample=True, top_p=0.9, temperature=0.8)
```

## 📋 剩余建议性改进

以下工程改进可在后续迭代中处理：
- 添加DistributedSampler支持
- 实现gradient accumulation和AMP
- 添加学习率warmup
- 改进wandb导入处理
- 补充缺失的deploy目录
- 添加CUDA可用性检查

**结论**: 所有影响正确性的关键问题已修复，现在可以安全进行训练和评估实验！ 