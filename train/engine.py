"""
训练引擎：混合模型的训练循环和优化
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import logging
from typing import Dict, Any, Optional
import wandb
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_model import HybridLanguageModel, HybridLanguageModelConfig
from train.data import create_data_loader, CURRICULUM_SCHEDULES
from train.losses import HybridModelLoss
from train.monitor_gate import GateMonitor

logger = logging.getLogger(__name__)


class TrainingEngine:
    """混合模型训练引擎"""
    
    def __init__(self, config_path: str, distributed: bool = False):
        self.config_path = config_path
        self.distributed = distributed
        self.rank = 0
        self.world_size = 1
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化分布式
        if distributed:
            self._init_distributed()
        
        # 设置设备
        self.device = torch.device(f'cuda:{self.rank}')
        torch.cuda.set_device(self.device)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self._init_model()
        self._init_tokenizer()
        self._init_data()
        self._init_optimizer()
        self._init_loss()
        self._init_monitor()
        
        # 初始化wandb
        if self.config['logging']['wandb']['enabled'] and self.rank == 0:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                name=self.config['logging']['wandb']['name'],
                tags=self.config['logging']['wandb']['tags'],
                config=self.config
            )
    
    def _init_distributed(self):
        """初始化分布式训练"""
        dist.init_process_group(backend='nccl')
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
    def _setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config['logging']['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _init_model(self):
        """初始化模型"""
        model_config = HybridLanguageModelConfig(**self.config['model'])
        self.model = HybridLanguageModel(model_config)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 分布式包装
        if self.distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.rank],
                find_unused_parameters=self.config['system']['find_unused_parameters']
            )
        
        # 编译模型（如果启用）
        if self.config['system']['compile_model']:
            self.model = torch.compile(self.model)
            
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
    
    def _init_tokenizer(self):
        """初始化分词器"""
        # 这里使用占位符，实际应该加载对应的分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            # 如果加载失败，创建一个简单的占位符
            class DummyTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.bos_token_id = 1
                    self.eos_token_id = 2
                    
                def encode(self, text, add_special_tokens=False):
                    # 简单的字符级编码
                    return [ord(c) % 1000 for c in text[:100]]
                
                def __call__(self, text, **kwargs):
                    input_ids = torch.tensor([self.encode(text)])
                    return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
            
            self.tokenizer = DummyTokenizer()
            logger.warning("Using dummy tokenizer for testing")
    
    def _init_data(self):
        """初始化数据加载器"""
        data_config = self.config['data']
        training_config = self.config['training']
        
        # 获取课程学习计划
        curriculum_schedule = None
        if self.config['curriculum']['enabled']:
            schedule_name = self.config['curriculum']['schedule']
            if schedule_name in CURRICULUM_SCHEDULES:
                curriculum_schedule = CURRICULUM_SCHEDULES[schedule_name]
            else:
                curriculum_schedule = self.config['curriculum']['custom_schedule']
        
        # 创建训练数据加载器
        self.train_loader = create_data_loader(
            data_paths=data_config['train_data_paths'],
            tokenizer=self.tokenizer,
            batch_size=training_config['batch_size'],
            max_length=4096,  # 从课程最小长度开始
            num_workers=data_config['num_workers'],
            shuffle=True,
            curriculum_schedule=curriculum_schedule,
            **{k: v for k, v in data_config.items() 
               if k not in ['train_data_paths', 'eval_data_paths', 'num_workers']}
        )
        
        logger.info(f"Training data loader created with {len(self.train_loader)} batches")
    
    def _init_optimizer(self):
        """初始化优化器"""
        training_config = self.config['training']
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            betas=(training_config['beta1'], training_config['beta2']),
            eps=training_config['eps'],
            weight_decay=training_config['weight_decay']
        )
        
        # 学习率调度器
        if training_config['lr_scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['max_steps'],
                eta_min=training_config['learning_rate'] * training_config['min_lr_ratio']
            )
        else:
            self.scheduler = None
    
    def _init_loss(self):
        """初始化损失函数"""
        training_config = self.config['training']
        
        self.loss_fn = HybridModelLoss(
            load_balance_coeff=training_config['load_balance_coeff'],
            entropy_reg_coeff=training_config['entropy_reg_coeff'],
            distill_coeff=training_config['distill_coeff'],
            distill_temperature=training_config['distill_temperature']
        )
    
    def _init_monitor(self):
        """初始化门控监控器"""
        gate_config = self.config['gate_monitor']
        if gate_config['enabled']:
            self.gate_monitor = GateMonitor(
                num_layers=self.config['model']['num_layers'],
                collect_detailed=gate_config['collect_detailed']
            )
        else:
            self.gate_monitor = None
    
    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 移动数据到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            collect_gate_stats=True,
            return_dict=True
        )
        
        # 计算损失
        loss_dict = self.loss_fn(
            logits=outputs['logits'],
            labels=labels,
            gate_stats=outputs.get('gate_stats', None)
        )
        
        loss = loss_dict['total_loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config['training']['max_grad_norm']
        )
        
        # 优化器步骤
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        # 更新门控监控
        if self.gate_monitor is not None and outputs.get('gate_stats'):
            for layer_idx, gate_stats in enumerate(outputs['gate_stats']):
                self.gate_monitor.update(layer_idx, gate_stats)
        
        # 返回损失统计
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def train(self):
        """主训练循环"""
        training_config = self.config['training']
        logging_config = self.config['logging']
        
        step = 0
        epoch = 0
        
        logger.info("Starting training...")
        
        while step < training_config['max_steps']:
            epoch += 1
            
            # 检查课程学习是否需要更新
            if hasattr(self.train_loader.dataset, 'step'):
                length_updated = self.train_loader.dataset.step()
                if length_updated:
                    logger.info(f"Curriculum updated at step {step}")
            
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch}",
                disable=self.rank != 0
            )
            
            for batch in progress_bar:
                step += 1
                
                # 训练步骤
                loss_dict = self.train_step(batch, step)
                
                # 日志记录
                if step % logging_config['log_interval'] == 0 and self.rank == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    log_msg = f"Step {step}/{training_config['max_steps']}, "
                    log_msg += f"LR: {lr:.2e}, "
                    log_msg += f"Loss: {loss_dict['total_loss']:.4f}"
                    
                    if 'load_balance_loss' in loss_dict:
                        log_msg += f", LB: {loss_dict['load_balance_loss']:.4f}"
                    if 'entropy_loss' in loss_dict:
                        log_msg += f", Ent: {loss_dict['entropy_loss']:.4f}"
                    
                    logger.info(log_msg)
                    
                    # 更新进度条
                    progress_bar.set_postfix(loss=loss_dict['total_loss'])
                    
                    # wandb记录
                    if wandb.run is not None:
                        wandb.log({
                            'train/loss': loss_dict['total_loss'],
                            'train/lm_loss': loss_dict.get('lm_loss', 0),
                            'train/lb_loss': loss_dict.get('load_balance_loss', 0),
                            'train/entropy_loss': loss_dict.get('entropy_loss', 0),
                            'train/learning_rate': lr,
                            'train/step': step
                        })
                
                # 保存检查点
                if step % logging_config['save_interval'] == 0 and self.rank == 0:
                    self._save_checkpoint(step)
                
                # 门控监控保存
                if (self.gate_monitor is not None and 
                    step % self.config['gate_monitor']['save_interval'] == 0 and 
                    self.rank == 0):
                    save_path = self.config['gate_monitor']['save_path']
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.gate_monitor.save_statistics(save_path)
                
                if step >= training_config['max_steps']:
                    break
        
        # 训练结束后保存最终模型和门控统计
        if self.rank == 0:
            self._save_checkpoint(step, is_final=True)
            
            if self.gate_monitor is not None:
                # 导出裁剪计划
                prune_plan = self.gate_monitor.export_prune_plan(
                    mamba_threshold_high=self.config['gate_monitor']['mamba_threshold_high'],
                    attention_threshold_low=self.config['gate_monitor']['attention_threshold_low'],
                    min_steps=self.config['gate_monitor']['min_steps_for_pruning']
                )
                
                plan_path = os.path.join(
                    self.config['logging']['output_dir'], 
                    'prune_plan.json'
                )
                import json
                with open(plan_path, 'w', encoding='utf-8') as f:
                    json.dump(prune_plan, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Pruning plan saved to {plan_path}")
        
        logger.info("Training completed!")
    
    def _save_checkpoint(self, step: int, is_final: bool = False):
        """保存检查点"""
        output_dir = self.config['logging']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取模型状态（处理DDP包装）
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        
        checkpoint = {
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存检查点
        if is_final:
            save_path = os.path.join(output_dir, 'final_model.pt')
        else:
            save_path = os.path.join(output_dir, f'checkpoint_step_{step}.pt')
        
        torch.save(checkpoint, save_path)
        
        # 保存最佳模型链接
        best_path = os.path.join(output_dir, 'best_model.pt')
        if not os.path.exists(best_path) or is_final:
            if os.path.exists(best_path):
                os.remove(best_path)
            os.symlink(os.path.basename(save_path), best_path)
        
        logger.info(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FusionTree Training")
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--distributed', action='store_true', help='是否使用分布式训练')
    
    args = parser.parse_args()
    
    # 创建训练引擎
    engine = TrainingEngine(args.config, args.distributed)
    
    # 开始训练
    engine.train()


if __name__ == "__main__":
    main() 