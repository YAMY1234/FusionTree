"""
数据处理模块：长文拼接、mask、段落标注
支持长上下文训练的数据预处理和加载
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Any, Tuple
import json
import random
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LongContextDataset(Dataset):
    """
    长上下文数据集
    支持多种数据格式和长度课程学习
    """
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        max_length: int = 32768,
        min_length: int = 512,
        concat_docs: bool = True,
        add_special_tokens: bool = True,
        document_separator: str = "\n\n",
        enable_packing: bool = True,
        needle_in_haystack_prob: float = 0.0,
        structured_task_prob: float = 0.0
    ):
        """
        Args:
            data_paths: 数据文件路径列表
            tokenizer: 分词器
            max_length: 最大序列长度
            min_length: 最小序列长度
            concat_docs: 是否拼接多个文档
            add_special_tokens: 是否添加特殊token
            document_separator: 文档分隔符
            enable_packing: 是否启用序列打包
            needle_in_haystack_prob: needle-in-haystack任务概率
            structured_task_prob: 结构化任务概率
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.concat_docs = concat_docs
        self.add_special_tokens = add_special_tokens
        self.document_separator = document_separator
        self.enable_packing = enable_packing
        self.needle_in_haystack_prob = needle_in_haystack_prob
        self.structured_task_prob = structured_task_prob
        
        # 加载和预处理数据
        self.documents = self._load_documents(data_paths)
        self.examples = self._prepare_examples()
        
        logger.info(f"Loaded {len(self.documents)} documents, created {len(self.examples)} examples")
    
    def _load_documents(self, data_paths: List[str]) -> List[Dict[str, Any]]:
        """加载文档数据"""
        documents = []
        
        for path in data_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Data path {path} does not exist, skipping")
                continue
            
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            doc = json.loads(line)
                            documents.append(doc)
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        documents.extend(data)
                    else:
                        documents.append(data)
            else:
                # 纯文本文件
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 按段落分割
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    for para in paragraphs:
                        documents.append({'text': para, 'source': str(path)})
        
        return documents
    
    def _prepare_examples(self) -> List[Dict[str, Any]]:
        """准备训练样本"""
        examples = []
        
        if self.enable_packing:
            examples = self._pack_documents()
        else:
            # 简单处理：每个文档单独作为一个样本
            for doc in self.documents:
                text = doc.get('text', '')
                if len(text) < 50:  # 过滤太短的文档
                    continue
                
                example = self._tokenize_text(text)
                if example['input_ids'].size(0) >= self.min_length:
                    examples.append(example)
        
        return examples
    
    def _pack_documents(self) -> List[Dict[str, Any]]:
        """将多个文档打包到一个样本中"""
        examples = []
        current_text = ""
        current_docs = []
        
        for doc in self.documents:
            text = doc.get('text', '')
            if len(text) < 20:
                continue
            
            # 尝试添加当前文档
            test_text = current_text
            if test_text:
                test_text += self.document_separator
            test_text += text
            
            # 检查是否超长
            test_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
            
            if len(test_tokens) <= self.max_length:
                # 可以添加
                current_text = test_text
                current_docs.append(doc)
            else:
                # 超长了，保存当前序列并开始新的
                if current_text and len(current_docs) > 0:
                    example = self._create_packed_example(current_text, current_docs)
                    if example:
                        examples.append(example)
                
                # 开始新序列
                current_text = text
                current_docs = [doc]
        
        # 处理最后一个序列
        if current_text and len(current_docs) > 0:
            example = self._create_packed_example(current_text, current_docs)
            if example:
                examples.append(example)
        
        return examples
    
    def _create_packed_example(self, text: str, docs: List[Dict]) -> Optional[Dict[str, Any]]:
        """创建打包样本"""
        # 随机决定是否应用特殊任务
        if random.random() < self.needle_in_haystack_prob:
            text = self._add_needle_in_haystack(text)
        elif random.random() < self.structured_task_prob:
            text = self._add_structured_task(text, docs)
        
        example = self._tokenize_text(text)
        
        if example['input_ids'].size(0) < self.min_length:
            return None
        
        # 添加元信息
        example['num_docs'] = len(docs)
        example['doc_sources'] = [doc.get('source', 'unknown') for doc in docs]
        
        return example
    
    def _add_needle_in_haystack(self, text: str) -> str:
        """添加needle-in-haystack任务"""
        # 生成随机"针"
        needle_facts = [
            "The magic number is 42.",
            "The secret code is FUSION2024.",
            "The hidden treasure is buried at coordinates 123.456, 789.012.",
            "The password is: OpenSesame123!",
            "The answer to the ultimate question is forty-two."
        ]
        
        needle = random.choice(needle_facts)
        
        # 随机插入位置（不要太靠前或太靠后）
        sentences = text.split('.')
        if len(sentences) > 10:
            insert_pos = random.randint(len(sentences) // 4, 3 * len(sentences) // 4)
            sentences.insert(insert_pos, f" {needle}")
            text = '.'.join(sentences)
        
        # 在末尾添加问题
        question = f"\n\nQuestion: What is the magic number mentioned in the text above? Answer:"
        text += question
        
        return text
    
    def _add_structured_task(self, text: str, docs: List[Dict]) -> str:
        """添加结构化任务（指代解析、句法标注等）"""
        tasks = ['coreference', 'named_entity', 'sentiment']
        task = random.choice(tasks)
        
        if task == 'coreference':
            # 简单的指代解析任务
            text += "\n\nTask: Identify all pronouns in the text and their referents."
        elif task == 'named_entity':
            # 命名实体识别
            text += "\n\nTask: List all named entities (persons, locations, organizations) mentioned in the text."
        elif task == 'sentiment':
            # 情感分析
            text += "\n\nTask: Analyze the overall sentiment of each paragraph in the text."
        
        return text
    
    def _tokenize_text(self, text: str) -> Dict[str, Any]:
        """tokenize文本"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding.get('attention_mask', torch.ones_like(input_ids)).squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # 语言建模任务
        }
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
    
    def set_max_length(self, max_length: int):
        """动态调整最大长度（用于长度课程学习）"""
        if max_length != self.max_length:
            self.max_length = max_length
            logger.info(f"Updated max_length to {max_length}, regenerating examples...")
            self.examples = self._prepare_examples()


class LengthCurriculumDataset(LongContextDataset):
    """
    支持长度课程学习的数据集
    逐步增加序列长度：4K→8K→16K→32K
    """
    
    def __init__(self, curriculum_schedule: List[Tuple[int, int]], **kwargs):
        """
        Args:
            curriculum_schedule: [(length, steps), ...] 长度课程计划
        """
        self.curriculum_schedule = curriculum_schedule
        self.current_stage = 0
        self.steps_in_stage = 0
        
        # 从最小长度开始
        initial_length = curriculum_schedule[0][0]
        super().__init__(max_length=initial_length, **kwargs)
    
    def step(self) -> bool:
        """
        执行一步训练，检查是否需要更新长度
        
        Returns:
            是否更新了长度
        """
        self.steps_in_stage += 1
        
        if self.current_stage < len(self.curriculum_schedule) - 1:
            current_length, current_steps = self.curriculum_schedule[self.current_stage]
            
            if self.steps_in_stage >= current_steps:
                # 进入下一阶段
                self.current_stage += 1
                self.steps_in_stage = 0
                new_length = self.curriculum_schedule[self.current_stage][0]
                
                logger.info(f"Curriculum: advancing to stage {self.current_stage}, length {new_length}")
                self.set_max_length(new_length)
                return True
        
        return False


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    数据批处理函数
    处理不同长度的序列，进行padding
    """
    # 获取最大长度
    max_length = max(item['input_ids'].size(0) for item in batch)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        labels = item['labels']
        
        # 右padding
        pad_length = max_length - input_ids.size(0)
        if pad_length > 0:
            input_ids = F.pad(input_ids, (0, pad_length), value=0)
            attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
            labels = F.pad(labels, (0, pad_length), value=-100)  # -100会被ignore
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
    
    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask).to(torch.bool),
        'labels': torch.stack(batch_labels)
    }


def create_data_loader(
    data_paths: List[str],
    tokenizer,
    batch_size: int = 4,
    max_length: int = 32768,
    num_workers: int = 0,
    shuffle: bool = True,
    curriculum_schedule: Optional[List[Tuple[int, int]]] = None,
    **dataset_kwargs
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_paths: 数据文件路径
        tokenizer: 分词器
        batch_size: 批大小
        max_length: 最大序列长度
        num_workers: 工作进程数
        shuffle: 是否打乱
        curriculum_schedule: 长度课程计划
        **dataset_kwargs: 其他数据集参数
        
    Returns:
        DataLoader实例
    """
    if curriculum_schedule is not None:
        dataset = LengthCurriculumDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            max_length=max_length,
            curriculum_schedule=curriculum_schedule,
            **dataset_kwargs
        )
    else:
        dataset = LongContextDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            max_length=max_length,
            **dataset_kwargs
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def create_evaluation_datasets(
    eval_data_paths: Dict[str, List[str]],
    tokenizer,
    max_length: int = 32768,
    **kwargs
) -> Dict[str, Dataset]:
    """
    创建评估数据集
    
    Args:
        eval_data_paths: {"task_name": [data_paths]} 格式的评估数据
        tokenizer: 分词器
        max_length: 最大长度
        **kwargs: 其他参数
        
    Returns:
        任务名到数据集的映射
    """
    eval_datasets = {}
    
    for task_name, paths in eval_data_paths.items():
        dataset = LongContextDataset(
            data_paths=paths,
            tokenizer=tokenizer,
            max_length=max_length,
            concat_docs=False,  # 评估时不拼接文档
            enable_packing=False,  # 评估时不打包
            **kwargs
        )
        eval_datasets[task_name] = dataset
    
    return eval_datasets


# 预定义的长度课程计划
CURRICULUM_SCHEDULES = {
    'standard': [
        (4096, 5000),    # 4K, 5K steps
        (8192, 5000),    # 8K, 5K steps  
        (16384, 5000),   # 16K, 5K steps
        (32768, 10000),  # 32K, 10K steps
    ],
    'aggressive': [
        (2048, 2000),
        (4096, 3000),
        (8192, 3000),
        (16384, 4000),
        (32768, 8000),
    ],
    'conservative': [
        (4096, 10000),
        (8192, 10000),
        (16384, 10000),
        (32768, 20000),
        (65536, 10000),  # 最终到64K
    ]
} 