"""
流式数据系统：支持HuggingFace数据集的边下边训
- 多数据源按权重混合采样
- Token级高效打包
- 分布式友好
- 内存占用极小
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import random
import bisect
import logging
from typing import List, Dict, Any, Optional, Iterator
from functools import partial
import itertools
import warnings

logger = logging.getLogger(__name__)


class HFStreamingIterableDataset(IterableDataset):
    """
    HuggingFace流式可迭代数据集
    
    核心特性：
    - 支持多个HF数据集按权重混合
    - Token级打包，最大化序列利用率
    - 分布式训练时自动切片
    - 无限流式，避免epoch概念的数据重复
    - 内存占用极小（只有buffer）
    """
    
    def __init__(
        self,
        sources: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 1024,
        min_length: int = 10,
        seed: int = 42,
        buffer_size: int = 10000,
        concat_docs: bool = True,
        document_separator: str = "\n\n",
        add_special_tokens: bool = True,
        shuffle_buffer_size: int = 10000,
        trust_remote_code: bool = False
    ):
        """
        Args:
            sources: 数据源配置列表，每个元素包含：
                - name: HF数据集名称 (如 "wikipedia")
                - config: 配置名称 (如 "20220301.en") 
                - split: 数据切片 (如 "train")
                - text_key: 文本字段名 (如 "text")
                - weight: 采样权重 (如 1.0)
                - streaming: 是否流式加载 (默认True)
            tokenizer: 分词器
            max_length: 最大序列长度
            min_length: 最小文本长度过滤
            seed: 随机种子
            buffer_size: Token打包缓冲区大小
            concat_docs: 是否启用文档拼接和token级打包
            document_separator: 文档分隔符
            add_special_tokens: 是否添加特殊token
            shuffle_buffer_size: HF数据集shuffle缓冲区大小
            trust_remote_code: 是否信任远程代码
        """
        self.sources = sources
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.seed = seed
        self.buffer_size = buffer_size
        self.concat_docs = concat_docs
        self.document_separator = document_separator
        self.add_special_tokens = add_special_tokens
        self.shuffle_buffer_size = shuffle_buffer_size
        self.trust_remote_code = trust_remote_code
        
        # 分布式设置
        self.rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        
        # 验证配置
        self._validate_sources()
        
        logger.info(f"HFStreamingIterableDataset initialized:")
        logger.info(f"  - {len(sources)} data sources")
        logger.info(f"  - max_length: {max_length}")
        logger.info(f"  - concat_docs: {concat_docs}")
        logger.info(f"  - rank: {self.rank}/{self.world_size}")
    
    def _validate_sources(self):
        """验证数据源配置"""
        required_keys = ['name']
        for i, source in enumerate(self.sources):
            for key in required_keys:
                if key not in source:
                    raise ValueError(f"Source {i} missing required key: {key}")
            
            # 设置默认值
            source.setdefault('config', None)
            source.setdefault('split', 'train')
            source.setdefault('text_key', 'text')
            source.setdefault('weight', 1.0)
            source.setdefault('streaming', True)
    
    def _load_datasets(self) -> List[Any]:
        """加载并配置所有数据集"""
        datasets = []
        
        for source in self.sources:
            try:
                logger.info(f"Loading dataset: {source['name']}")
                
                # 加载数据集
                dataset = load_dataset(
                    source['name'],
                    source['config'],
                    split=source['split'],
                    streaming=source['streaming'],
                    trust_remote_code=self.trust_remote_code
                )
                
                # 流式模式下的优化
                if source['streaming']:
                    # 分布式切片
                    if self.world_size > 1:
                        dataset = dataset.shard(
                            num_shards=self.world_size, 
                            index=self.rank
                        )
                    
                    # 随机打乱
                    dataset = dataset.shuffle(
                        seed=self.seed + self.rank,
                        buffer_size=self.shuffle_buffer_size
                    )
                
                datasets.append(dataset)
                logger.info(f"✅ Loaded dataset: {source['name']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load dataset {source['name']}: {e}")
                # 可以选择跳过或终止
                raise
        
        return datasets
    
    def _create_weighted_sampler(self, datasets: List[Any]) -> Iterator[tuple]:
        """创建按权重的数据源采样器"""
        weights = [source['weight'] for source in self.sources]
        
        # 构建累积分布函数(CDF)
        total_weight = sum(weights)
        cdf = []
        acc = 0
        for w in weights:
            acc += w / total_weight
            cdf.append(acc)
        
        # 创建数据集迭代器
        iterators = [iter(ds) for ds in datasets]
        rng = random.Random(self.seed + self.rank)
        
        while True:
            # 按权重选择数据源
            r = rng.random()
            source_idx = bisect.bisect_left(cdf, r)
            source_idx = min(source_idx, len(iterators) - 1)
            
            try:
                example = next(iterators[source_idx])
                yield source_idx, example
            except StopIteration:
                # 重新创建迭代器（流式数据集可以无限迭代）
                iterators[source_idx] = iter(datasets[source_idx])
                example = next(iterators[source_idx])
                yield source_idx, example
    
    def _extract_text(self, source_idx: int, example: Dict[str, Any]) -> str:
        """从样本中提取文本"""
        source = self.sources[source_idx]
        text_key = source['text_key']
        
        # 支持嵌套键名（如 "data.text"）
        text = example
        for key in text_key.split('.'):
            if isinstance(text, dict):
                text = text.get(key, "")
            else:
                text = ""
                break
        
        return str(text) if text else ""
    
    def _tokenize_single(self, text: str) -> Dict[str, torch.Tensor]:
        """单文档tokenize"""
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
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask.bool(),
            'labels': input_ids.clone()
        }
    
    def _packed_tokenize_generator(self, sampler: Iterator[tuple]) -> Iterator[Dict[str, torch.Tensor]]:
        """Token级打包生成器"""
        token_buffer = []
        sep_tokens = self.tokenizer.encode(
            self.document_separator, 
            add_special_tokens=False
        ) if self.document_separator else []
        
        for source_idx, example in sampler:
            text = self._extract_text(source_idx, example)
            
            # 过滤太短的文本
            if not text or len(text) < self.min_length:
                continue
            
            # 编码文本
            try:
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,  # 在打包时统一处理特殊token
                    truncation=False  # 不在单文档级截断
                )
                
                if not tokens:
                    continue
                
                # 添加到缓冲区
                if token_buffer and sep_tokens:
                    token_buffer.extend(sep_tokens)
                token_buffer.extend(tokens)
                
                # 当缓冲区够大时，输出序列
                while len(token_buffer) >= self.max_length:
                    # 提取一个完整序列
                    sequence = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]
                    
                    # 添加特殊token
                    if self.add_special_tokens and hasattr(self.tokenizer, 'bos_token_id'):
                        if self.tokenizer.bos_token_id is not None:
                            sequence = [self.tokenizer.bos_token_id] + sequence[:-1]
                    
                    input_ids = torch.tensor(sequence, dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    labels = input_ids.clone()
                    
                    yield {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': labels
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to tokenize text: {e}")
                continue
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """主迭代器"""
        # 加载数据集
        datasets = self._load_datasets()
        
        # 创建加权采样器
        sampler = self._create_weighted_sampler(datasets)
        
        if self.concat_docs:
            # Token级打包模式
            yield from self._packed_tokenize_generator(sampler)
        else:
            # 单文档模式
            for source_idx, example in sampler:
                text = self._extract_text(source_idx, example)
                
                if not text or len(text) < self.min_length:
                    continue
                
                try:
                    yield self._tokenize_single(text)
                except Exception as e:
                    logger.warning(f"Failed to process example: {e}")
                    continue


def create_streaming_data_loader(
    sources: List[Dict[str, Any]],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    创建流式数据加载器
    
    Args:
        sources: 数据源配置列表
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: worker进程数（IterableDataset推荐0）
        seed: 随机种子
        pin_memory: 是否使用pin_memory
        **dataset_kwargs: 传递给HFStreamingIterableDataset的额外参数
    
    Returns:
        DataLoader实例
    """
    
    # 创建流式数据集
    dataset = HFStreamingIterableDataset(
        sources=sources,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed,
        **dataset_kwargs
    )
    
    def streaming_collate_fn(batch):
        """流式数据的collate函数"""
        if not batch:
            return None
        
        # 找到批次中的最大长度
        max_len = max(item['input_ids'].size(0) for item in batch)
        max_len = min(max_len, max_length)  # 不超过设定的最大长度
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            # 处理长度
            if len(input_ids) > max_len:
                # 截断
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
            elif len(input_ids) < max_len:
                # 填充
                pad_len = max_len - len(input_ids)
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = tokenizer.eos_token_id
                
                input_ids = torch.cat([
                    input_ids, 
                    torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_len, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=labels.dtype)  # -100忽略loss计算
                ])
            
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
        shuffle=False,  # IterableDataset不需要shuffle
        sampler=None,   # IterableDataset不能使用DistributedSampler
        num_workers=num_workers,
        collate_fn=streaming_collate_fn,
        drop_last=True,
        pin_memory=pin_memory
    )


# 预定义的数据源配置
COMMON_DATASETS = {
    "wikipedia_en": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "text_key": "text",
        "weight": 1.0
    },
    "wikipedia_zh": {
        "name": "wikimedia/wikipedia", 
        "config": "20231101.zh",
        "split": "train",
        "text_key": "text",
        "weight": 0.5
    },
    "openwebtext": {
        "name": "Skylion007/openwebtext",
        "split": "train",
        "text_key": "text",
        "weight": 1.5
    },
    "c4": {
        "name": "c4",
        "config": "en",
        "split": "train",
        "text_key": "text", 
        "weight": 1.0
    },
    "redpajama": {
        "name": "togethercomputer/RedPajama-Data-1T",
        "config": "default",
        "split": "train",
        "text_key": "text",
        "weight": 1.0
    }
}


def get_preset_sources(preset_name: str) -> List[Dict[str, Any]]:
    """获取预设的数据源配置"""
    presets = {
        "english_mix": [
            COMMON_DATASETS["wikipedia_en"],
            COMMON_DATASETS["openwebtext"],
            COMMON_DATASETS["c4"]
        ],
        "multilingual_mix": [
            COMMON_DATASETS["wikipedia_en"],
            COMMON_DATASETS["wikipedia_zh"],
            COMMON_DATASETS["openwebtext"]
        ],
        "wikipedia_only": [
            COMMON_DATASETS["wikipedia_en"]
        ]
    }
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return presets[preset_name] 