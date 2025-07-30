"""
语言模型评估：PPL、长文QA、needle-in-haystack等任务
"""

import os
import sys
import torch
import torch.nn.functional as F
import argparse
import json
import logging
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig

logger = logging.getLogger(__name__)


class LanguageModelEvaluator:
    """语言模型评估器"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 尝试加载检查点
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if 'config' in checkpoint:
                # 从检查点配置创建模型
                config = HybridLanguageModelConfig(**checkpoint['config']['model'])
                self.model = HybridLanguageModel(config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接加载模型状态
                config = HybridLanguageModelConfig()
                self.model = HybridLanguageModel(config)
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # 创建默认模型用于测试
            config = HybridLanguageModelConfig(
                vocab_size=1000,
                hidden_size=768,
                num_layers=12
            )
            self.model = HybridLanguageModel(config).to(self.device)
            self.model.eval()
            logger.warning("Using default model for testing")
        
        # 创建简单的tokenizer
        self._create_tokenizer()
    
    def _create_tokenizer(self):
        """创建简单的tokenizer"""
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                self.pad_token_id = 0
                self.bos_token_id = 1
                self.eos_token_id = 2
            
            def encode(self, text: str, max_length: int = None) -> List[int]:
                # 字符级编码
                tokens = [ord(c) % self.vocab_size for c in text]
                if max_length and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                return tokens
            
            def decode(self, tokens: List[int]) -> str:
                return ''.join(chr(t + ord('a')) for t in tokens[:100])
        
        self.tokenizer = SimpleTokenizer()
    
    def evaluate_perplexity(self, texts: List[str], max_length: int = 1024) -> Dict[str, float]:
        """评估困惑度"""
        total_loss = 0.0
        total_tokens = 0
        
        for text in tqdm(texts, desc="Computing perplexity"):
            # Tokenize
            tokens = self.tokenizer.encode(text, max_length)
            if len(tokens) < 2:
                continue
            
            input_ids = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs['loss']
                
                total_loss += loss.item() * len(tokens)
                total_tokens += len(tokens)
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens
        }
    
    def evaluate_needle_in_haystack(self, haystack_length: int = 8192, num_samples: int = 10) -> Dict[str, float]:
        """评估needle-in-haystack任务"""
        correct = 0
        total = 0
        
        for _ in tqdm(range(num_samples), desc="Needle-in-haystack evaluation"):
            # 生成haystack文本
            haystack = "The quick brown fox jumps over the lazy dog. " * (haystack_length // 50)
            
            # 插入needle
            needle = "The magic number is 42."
            insert_pos = len(haystack) // 2
            text_with_needle = haystack[:insert_pos] + needle + haystack[insert_pos:]
            
            # 添加问题
            question = "\n\nQuestion: What is the magic number? Answer:"
            full_text = text_with_needle + question
            
            # Tokenize并生成
            tokens = self.tokenizer.encode(full_text, max_length=haystack_length)
            input_ids = torch.tensor([tokens], device=self.device)
            
            try:
                with torch.no_grad():
                    # 生成回答
                    generated = self.model.generate(
                        input_ids,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=False
                    )
                    
                    # 检查是否包含正确答案
                    answer_tokens = generated[0, len(tokens):].tolist()
                    answer_text = self.tokenizer.decode(answer_tokens)
                    
                    if "42" in answer_text:
                        correct += 1
                    
                    total += 1
                    
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                total += 1
        
        accuracy = correct / max(total, 1)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_code_completion(self, code_samples: List[str]) -> Dict[str, float]:
        """评估代码补全任务"""
        total_score = 0.0
        total_samples = 0
        
        for code in tqdm(code_samples, desc="Code completion evaluation"):
            # 取前50%作为prompt，后50%作为target
            split_pos = len(code) // 2
            prompt = code[:split_pos]
            target = code[split_pos:]
            
            # Tokenize
            prompt_tokens = self.tokenizer.encode(prompt)
            target_tokens = self.tokenizer.encode(target)
            
            if len(prompt_tokens) == 0 or len(target_tokens) == 0:
                continue
            
            input_ids = torch.tensor([prompt_tokens], device=self.device)
            
            try:
                with torch.no_grad():
                    # 生成补全
                    generated = self.model.generate(
                        input_ids,
                        max_new_tokens=len(target_tokens),
                        temperature=0.1,
                        do_sample=False
                    )
                    
                    # 计算相似度得分（简化版）
                    generated_tokens = generated[0, len(prompt_tokens):].tolist()
                    
                    # 计算token级别的准确率
                    min_len = min(len(generated_tokens), len(target_tokens))
                    matches = sum(1 for i in range(min_len) 
                                if generated_tokens[i] == target_tokens[i])
                    
                    score = matches / max(len(target_tokens), 1)
                    total_score += score
                    total_samples += 1
                    
            except Exception as e:
                logger.warning(f"Code completion failed: {e}")
                total_samples += 1
        
        avg_score = total_score / max(total_samples, 1)
        
        return {
            'completion_score': avg_score,
            'total_samples': total_samples
        }


def evaluate_language_model(
    model_path: str,
    tasks: List[str] = None,
    output_dir: str = None,
    save_predictions: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    评估语言模型性能
    
    Args:
        model_path: 模型路径
        tasks: 评估任务列表
        output_dir: 输出目录
        save_predictions: 是否保存预测结果
        verbose: 是否详细输出
        
    Returns:
        评估结果字典
    """
    if tasks is None:
        tasks = ['perplexity', 'needle_in_haystack', 'code_completion']
    
    # 设置日志级别
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    # 创建评估器
    evaluator = LanguageModelEvaluator(model_path)
    
    results = {}
    
    # 困惑度评估
    if 'perplexity' in tasks:
        logger.info("Evaluating perplexity...")
        # 使用示例文本
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
        ] * 10  # 重复几次增加数据量
        
        ppl_results = evaluator.evaluate_perplexity(sample_texts)
        results['perplexity'] = ppl_results
        
        if verbose:
            logger.info(f"Perplexity: {ppl_results['perplexity']:.2f}")
    
    # Needle-in-haystack评估
    if 'needle_in_haystack' in tasks:
        logger.info("Evaluating needle-in-haystack...")
        needle_results = evaluator.evaluate_needle_in_haystack(
            haystack_length=4096,
            num_samples=5
        )
        results['needle_in_haystack'] = needle_results
        
        if verbose:
            logger.info(f"Needle accuracy: {needle_results['accuracy']:.3f}")
    
    # 代码补全评估
    if 'code_completion' in tasks:
        logger.info("Evaluating code completion...")
        sample_codes = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right",
            "import torch\nimport torch.nn as nn\n\nclass LinearModel(nn.Module):\n    def __init__(self, input_size, output_size):\n        super().__init__()\n        self.linear = nn.Linear(input_size, output_size)",
        ]
        
        code_results = evaluator.evaluate_code_completion(sample_codes)
        results['code_completion'] = code_results
        
        if verbose:
            logger.info(f"Code completion score: {code_results['completion_score']:.3f}")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, 'llm_evaluation_results.json')
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="FusionTree Language Model Evaluation")
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--tasks', type=str, default='all', help='评估任务')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--save_predictions', action='store_true', help='保存预测结果')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 解析任务
    if args.tasks == 'all':
        tasks = ['perplexity', 'needle_in_haystack', 'code_completion']
    else:
        tasks = args.tasks.split(',')
    
    # 运行评估
    results = evaluate_language_model(
        model_path=args.model_path,
        tasks=tasks,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        verbose=args.verbose
    )
    
    # 打印摘要
    print("\n=== Evaluation Results ===")
    for task, result in results.items():
        print(f"\n{task.upper()}:")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main() 