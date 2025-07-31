"""
动态加载数据集：按需读取和tokenize，避免内存爆炸
"""

import torch
from torch.utils.data import Dataset
import json
import glob
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class LazyJSONLDataset(Dataset):
    """
    动态加载JSONL数据集
    - 初始化时只建立索引，不加载数据
    - __getitem__时实时读取和tokenize
    - 内存使用量极小，支持无限大数据集
    """
    
    def __init__(
        self, 
        data_paths: List[str],
        tokenizer,
        max_length: int = 1024,
        min_length: int = 10,
        add_special_tokens: bool = True,
        text_key: str = "text",
        max_samples_per_file: int = None  # 限制每个文件的样本数，用于测试
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.add_special_tokens = add_special_tokens
        self.text_key = text_key
        self.max_samples_per_file = max_samples_per_file
        
        # 🔧 只建立索引，不加载数据
        self.sample_index = self._build_index(data_paths)
        
        logger.info(f"LazyJSONLDataset initialized with {len(self.sample_index)} samples")
    
    def _build_index(self, data_paths: List[str]) -> List[Tuple[str, int]]:
        """
        建立样本索引：[(file_path, line_num), ...]
        只记录位置，不读取内容
        """
        sample_index = []
        
        for path_pattern in data_paths:
            # 展开通配符
            expanded_paths = glob.glob(str(path_pattern))
            
            if not expanded_paths:
                logger.warning(f"No files found for pattern: {path_pattern}")
                continue
            
            logger.info(f"Found {len(expanded_paths)} files for pattern: {path_pattern}")
            
            for file_path in expanded_paths:
                try:
                    # 快速扫描文件，只计算行数
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = 0
                        for line_num, line in enumerate(f, 1):
                            if line.strip():  # 跳过空行
                                sample_index.append((file_path, line_num))
                                line_count += 1
                                
                                # 限制样本数（用于测试）
                                if (self.max_samples_per_file and 
                                    line_count >= self.max_samples_per_file):
                                    break
                    
                    logger.info(f"Indexed {line_count} samples from {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    continue
        
        return sample_index
    
    def __len__(self) -> int:
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        动态读取和tokenize单个样本
        """
        file_path, line_num = self.sample_index[idx]
        
        # 读取指定行
        text = self._read_line(file_path, line_num)
        
        if not text or len(text) < self.min_length:
            # 返回一个最小样本避免错误
            text = "This is a placeholder text."
        
        # 实时tokenize
        return self._tokenize_text(text)
    
    def _read_line(self, file_path: str, line_num: int) -> str:
        """
        读取文件的指定行
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for current_line_num, line in enumerate(f, 1):
                    if current_line_num == line_num:
                        if line.strip():
                            doc = json.loads(line)
                            return doc.get(self.text_key, "")
                        break
        except Exception as e:
            logger.warning(f"Error reading {file_path}:{line_num}: {e}")
        
        return ""
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        对文本进行tokenize
        """
        # Tokenize文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding.get('attention_mask', torch.ones_like(input_ids)).squeeze(0)
        
        # 确保长度一致
        if len(input_ids) != len(attention_mask):
            min_len = min(len(input_ids), len(attention_mask))
            input_ids = input_ids[:min_len]
            attention_mask = attention_mask[:min_len]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask.bool(),
            'labels': input_ids.clone()  # 自回归语言模型
        }


def create_lazy_data_loader(
    data_paths: List[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    num_workers: int = 0,
    shuffle: bool = True,
    distributed: bool = False,
    max_samples_per_file: int = None,
    **kwargs
):
    """
    创建动态加载的数据加载器
    """
    from torch.utils.data import DataLoader
    from functools import partial
    
    # 创建lazy dataset
    dataset = LazyJSONLDataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples_per_file=max_samples_per_file,
        **kwargs
    )
    
    # 分布式采样器
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        shuffle = False
    
    # Collate function for dynamic batching
    def lazy_collate_fn(batch):
        # 找到batch中的最大长度
        max_len = max(item['input_ids'].size(0) for item in batch)
        max_len = min(max_len, max_length)  # 不超过模型最大长度
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            # 截断或填充到统一长度
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
            elif len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])  # -100忽略loss
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels)
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=lazy_collate_fn,
        drop_last=True,
        pin_memory=True
    ) 