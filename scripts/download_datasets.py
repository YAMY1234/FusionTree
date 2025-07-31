#!/usr/bin/env python3
"""
数据集下载和转换脚本
支持从HuggingFace下载数据集并转换为JSONL格式
"""

import os
import json
import argparse
from typing import Optional, Dict, Any
from datasets import load_dataset
from tqdm import tqdm


def dump_jsonl(dataset, out_dir: str, shard_size: int = 50000, prefix: str = "shard"):
    """
    将数据集转储为JSONL分片文件
    
    Args:
        dataset: HuggingFace数据集对象
        out_dir: 输出目录
        shard_size: 每个分片的样本数
        prefix: 分片文件前缀
    """
    os.makedirs(out_dir, exist_ok=True)
    
    writer = None
    sample_count = 0
    shard_id = 0
    total_samples = 0
    skipped_samples = 0
    
    print(f"开始转换数据集到 {out_dir}")
    
    # 获取数据集长度（如果可能）
    try:
        total_length = len(dataset)
        progress_bar = tqdm(total=total_length, desc="转换进度")
    except:
        # 流式数据集可能没有长度
        progress_bar = tqdm(desc="转换进度")
    
    for example in dataset:
        # 提取文本内容
        text = ""
        if "text" in example:
            text = example["text"]
        elif "content" in example:
            text = example["content"]
        elif "article" in example:
            text = example["article"]
        else:
            # 尝试找到最长的字符串字段
            text_fields = []
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    text_fields.append((key, len(value), value))
            if text_fields:
                text = max(text_fields, key=lambda x: x[1])[2]
        
        # 过滤太短的文本
        if not text or len(text.strip()) < 50:
            skipped_samples += 1
            progress_bar.update(1)
            continue
        
        # 创建新的分片文件
        if writer is None:
            shard_path = os.path.join(out_dir, f"{prefix}_{shard_id:05d}.jsonl")
            writer = open(shard_path, 'w', encoding='utf-8')
            print(f"创建分片: {shard_path}")
        
        # 写入JSONL格式
        json_obj = {"text": text.strip()}
        writer.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        sample_count += 1
        total_samples += 1
        
        # 达到分片大小，关闭当前文件
        if sample_count >= shard_size:
            writer.close()
            writer = None
            sample_count = 0
            shard_id += 1
        
        progress_bar.update(1)
    
    # 关闭最后一个文件
    if writer is not None:
        writer.close()
    
    progress_bar.close()
    
    print(f"转换完成!")
    print(f"  总样本数: {total_samples}")
    print(f"  跳过样本: {skipped_samples}")
    print(f"  分片数量: {shard_id + (1 if sample_count > 0 else 0)}")
    print(f"  输出目录: {out_dir}")


def download_wikipedia(lang: str = "en", output_dir: str = "data/wikipedia", 
                      shard_size: int = 50000, date: str = "20220301"):
    """下载Wikipedia数据集"""
    print(f"下载Wikipedia ({lang}) 数据集...")
    
    # 尝试多种方式加载Wikipedia数据集
    dataset = None
    dataset_configs = [
        # 新格式：直接使用语言代码
        ("wikimedia/wikipedia", f"{date}.{lang}"),
        # 备选：使用简化配置名
        ("wikipedia", lang),
        # 备选：尝试其他常见格式
        ("wikipedia", f"20220301.{lang}"),
    ]
    
    for dataset_name, config_name in dataset_configs:
        try:
            print(f"尝试加载 {dataset_name} with config {config_name}")
            dataset = load_dataset(dataset_name, config_name, split="train")
            print(f"✅ 成功加载 {dataset_name} {config_name}")
            break
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            continue
    
    # 如果都失败了，尝试一些测试数据集
    if dataset is None:
        print("正式Wikipedia数据集加载失败，尝试使用替代数据集...")
        try:
            # 尝试使用更小的、容易获取的文本数据集
            print("尝试加载 wikitext-103-v1 数据集...")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
            print("✅ 成功加载 wikitext-103-v1 作为替代")
        except Exception as e:
            print(f"替代数据集也失败: {e}")
            return False
    
    if dataset is None:
        print("❌ 所有数据集加载尝试都失败")
        return False
    
    # 转换为JSONL
    lang_output_dir = os.path.join(output_dir, f"wiki_{lang}")
    dump_jsonl(dataset, lang_output_dir, shard_size, f"wiki_{lang}")
    
    return True


def download_openwebtext(output_dir: str = "data/openwebtext", shard_size: int = 50000):
    """下载OpenWebText数据集"""
    print("下载OpenWebText数据集...")
    
    try:
        dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
        print("成功加载OpenWebText")
    except Exception as e:
        print(f"加载失败: {e}")
        return False
    
    dump_jsonl(dataset, output_dir, shard_size, "openwebtext")
    return True


def download_custom_dataset(dataset_name: str, config_name: Optional[str] = None,
                          split: str = "train", output_dir: str = None,
                          shard_size: int = 50000):
    """下载自定义数据集"""
    print(f"下载自定义数据集: {dataset_name}")
    
    if output_dir is None:
        output_dir = f"data/{dataset_name.replace('/', '_')}"
    
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        print(f"成功加载 {dataset_name}")
    except Exception as e:
        print(f"加载失败: {e}")
        return False
    
    prefix = dataset_name.replace('/', '_')
    if config_name:
        prefix += f"_{config_name}"
    
    dump_jsonl(dataset, output_dir, shard_size, prefix)
    return True


def main():
    parser = argparse.ArgumentParser(description="下载和转换训练数据集")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["wikipedia", "openwebtext", "custom"],
                       help="数据集类型")
    
    # Wikipedia选项
    parser.add_argument("--lang", type=str, default="en",
                       help="Wikipedia语言代码 (default: en)")
    parser.add_argument("--date", type=str, default="20220301",
                       help="Wikipedia版本日期 (default: 20220301)")
    
    # 自定义数据集选项
    parser.add_argument("--dataset-name", type=str,
                       help="自定义数据集名称")
    parser.add_argument("--config-name", type=str,
                       help="数据集配置名称")
    parser.add_argument("--split", type=str, default="train",
                       help="数据集切分 (default: train)")
    
    # 通用选项
    parser.add_argument("--output-dir", type=str, default="data",
                       help="输出根目录 (default: data)")
    parser.add_argument("--shard-size", type=int, default=50000,
                       help="每个分片的样本数 (default: 50000)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = False
    
    if args.dataset == "wikipedia":
        output_dir = os.path.join(args.output_dir, "wikipedia")
        success = download_wikipedia(args.lang, output_dir, args.shard_size, args.date)
        
    elif args.dataset == "openwebtext":
        output_dir = os.path.join(args.output_dir, "openwebtext")
        success = download_openwebtext(output_dir, args.shard_size)
        
    elif args.dataset == "custom":
        if not args.dataset_name:
            print("错误: 自定义数据集需要 --dataset-name 参数")
            return
        
        output_dir = os.path.join(args.output_dir, args.dataset_name.replace('/', '_'))
        success = download_custom_dataset(
            args.dataset_name, args.config_name, args.split,
            output_dir, args.shard_size
        )
    
    if success:
        print(f"\n✅ 数据集下载和转换成功!")
        print(f"数据位置: {output_dir}")
        print(f"\n接下来:")
        print(f"1. 修改配置文件中的 data.train_data_paths")
        print(f"2. 设置 concat_docs: true 和 enable_packing: true")
        print(f"3. 开始训练!")
    else:
        print("❌ 数据集下载失败")


if __name__ == "__main__":
    main() 