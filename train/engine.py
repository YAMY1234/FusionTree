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
import json  # 🔧 添加json导入

# 尝试导入DeepSpeed
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available, falling back to FSDP")

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
        self._init_deepspeed()  # 在优化器之后初始化DeepSpeed
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
        
        # 检查是否使用DeepSpeed
        if DEEPSPEED_AVAILABLE and self.config['system'].get('use_deepspeed', True):
            # 使用DeepSpeed时不直接移动模型到设备
            self.model = HybridLanguageModel(model_config)
            self.use_deepspeed = True
            logger.info("Using DeepSpeed for memory optimization")
        else:
            # 降级到FSDP或标准DDP
            self.model = HybridLanguageModel(model_config)
            
            # 分批移动模型参数到设备以避免OOM
            logger.info("Moving model to device in chunks to avoid OOM...")
            self._move_model_to_device_safely()
            
            # 分布式包装
            if self.distributed:
                self.model = DDP(
                    self.model, 
                    device_ids=[self.rank],
                    find_unused_parameters=self.config['system']['find_unused_parameters']
                )
            
            self.use_deepspeed = False
        
        # 编译模型（如果启用）
        if self.config['system']['compile_model']:
            self.model = torch.compile(self.model)
            
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
    
    def _move_model_to_device_safely(self):
        """分批移动模型参数到设备以避免OOM"""
        # 清空缓存
        torch.cuda.empty_cache()
        
        # 逐层移动模型
        for name, module in self.model.named_children():
            logger.info(f"Moving {name} to device...")
            module.to(self.device)
            torch.cuda.empty_cache()  # 每次移动后清空缓存
    
    def _init_tokenizer(self):
        """初始化分词器"""
        # 这里使用占位符，实际应该加载对应的分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 设置model_max_length以避免长度警告，使用我们模型的window_size
            model_config = self.config.get('model', {})
            model_max_length = model_config.get('window_size', 1024)
            self.tokenizer.model_max_length = model_max_length
            logger.info(f"Set tokenizer max_length to {model_max_length}")
            
        except:
            # 如果加载失败，创建一个简单的占位符
            model_config = self.config.get('model', {})
            window_size = model_config.get('window_size', 1024)
            
            class DummyTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.bos_token_id = 1
                    self.eos_token_id = 2
                    self.model_max_length = window_size
                    
                def encode(self, text, add_special_tokens=False, max_length=None, truncation=False):
                    # 简单的字符级编码
                    tokens = [ord(c) % 1000 for c in text[:100]]
                    if max_length and truncation and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    return tokens
                
                def __call__(self, text, max_length=None, truncation=False, **kwargs):
                    tokens = self.encode(text, max_length=max_length, truncation=truncation)
                    input_ids = torch.tensor([tokens])
                    return {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
            
            self.tokenizer = DummyTokenizer()
            logger.warning("Using dummy tokenizer for testing")
    
    def _init_data(self):
        """初始化数据加载器"""
        data_config = self.config['data']
        training_config = self.config['training']
        model_config = self.config['model']
        
        # 获取课程学习计划
        curriculum_schedule = None
        if self.config['curriculum']['enabled']:
            schedule_name = self.config['curriculum']['schedule']
            if schedule_name in CURRICULUM_SCHEDULES:
                curriculum_schedule = CURRICULUM_SCHEDULES[schedule_name]
            else:
                curriculum_schedule = self.config['curriculum']['custom_schedule']
        
        # 使用模型的window_size作为最大序列长度，确保与模型配置一致
        max_seq_length = model_config.get('window_size', 1024)
        logger.info(f"Setting data max_length to {max_seq_length} (from model.window_size)")
        
        # 创建训练数据加载器
        logger.info(f"[DEBUG] Creating train dataloader with:")
        logger.info(f"  - data_paths: {data_config['train_data_paths']}")
        logger.info(f"  - batch_size: {training_config['batch_size']}")
        logger.info(f"  - max_length: {max_seq_length}")
        logger.info(f"  - distributed: {self.distributed}")
        
        # 🚀 可扩展的数据加载系统
        data_mode = data_config.get('data_mode', 'static')  # static|lazy|hf_streaming
        
        logger.info(f"Data loading mode: {data_mode}")
        
        if data_mode == 'hf_streaming':
            # 🔥 HuggingFace流式加载 - 工业界最佳实践
            from train.streaming_data import create_streaming_data_loader, get_preset_sources
            
            # 获取数据源配置
            if 'hf_streaming_sources' in data_config:
                sources = data_config['hf_streaming_sources']
            elif 'hf_preset' in data_config:
                sources = get_preset_sources(data_config['hf_preset'])
                logger.info(f"Using preset '{data_config['hf_preset']}' with {len(sources)} sources")
            else:
                raise ValueError("hf_streaming mode requires 'hf_streaming_sources' or 'hf_preset' in config")
            
            logger.info("Using HFStreamingIterableDataset for maximum efficiency")
            self.train_loader = create_streaming_data_loader(
                sources=sources,
                tokenizer=self.tokenizer,
                batch_size=training_config['batch_size'],
                max_length=max_seq_length,
                num_workers=data_config.get('num_workers', 0),  # 流式推荐0
                seed=data_config.get('seed', 42),
                pin_memory=data_config.get('pin_memory', True),
                concat_docs=data_config.get('concat_docs', True),
                min_length=data_config.get('min_length', 10),
                add_special_tokens=data_config.get('add_special_tokens', True),
                document_separator=data_config.get('document_separator', '\n\n'),
                shuffle_buffer_size=data_config.get('shuffle_buffer_size', 10000),
                trust_remote_code=data_config.get('trust_remote_code', False)
            )
            
        elif data_mode == 'lazy':
            # 💾 Lazy JSONL加载 - 适合本地大文件
            from train.lazy_data import create_lazy_data_loader
            logger.info("Using LazyJSONLDataset for memory-efficient loading")
            
            self.train_loader = create_lazy_data_loader(
                data_paths=data_config['train_data_paths'],
                tokenizer=self.tokenizer,
                batch_size=training_config['batch_size'],
                max_length=max_seq_length,
                num_workers=data_config.get('num_workers', 0),
                shuffle=True,
                distributed=self.distributed,
                max_samples_per_file=data_config.get('max_samples_per_file', None),
                min_length=data_config.get('min_length', 10),
                add_special_tokens=data_config.get('add_special_tokens', True)
            )
            
        elif data_mode == 'static':
            # 📁 传统静态加载 - 适合小数据集和调试
            logger.info("Using traditional LongContextDataset (static loading)")
            self.train_loader = create_data_loader(
                data_paths=data_config['train_data_paths'],
                tokenizer=self.tokenizer,
                batch_size=training_config['batch_size'],
                max_length=max_seq_length,  # 使用模型window_size
                num_workers=data_config['num_workers'],
                shuffle=True,
                distributed=self.distributed,  # 传递分布式标志
                curriculum_schedule=curriculum_schedule,
                **{k: v for k, v in data_config.items() 
                   if k not in ['train_data_paths', 'eval_data_paths', 'num_workers', 'max_length', 'tokenizer_path', 'data_mode']}
            )
            
        else:
            raise ValueError(f"Unknown data_mode: {data_mode}. Supported: static|lazy|hf_streaming")
        
        # 关键DEBUG信息
        if hasattr(self.train_loader, 'dataset'):
            dataset_len = len(self.train_loader.dataset)
            logger.info(f"[DEBUG] len(train_dataset) = {dataset_len}")
        else:
            logger.info(f"[DEBUG] train_loader has no dataset attribute")
            
        dataloader_len = len(self.train_loader)
        logger.info(f"[DEBUG] batch_size = {training_config['batch_size']}")
        logger.info(f"[DEBUG] len(train_dataloader) = {dataloader_len}")
        
        if dataloader_len == 0:
            raise RuntimeError(
                f"Empty train_dataloader detected! Dataset length: {dataset_len if 'dataset_len' in locals() else 'unknown'}, "
                f"batch_size: {training_config['batch_size']}. 请检查数据路径/格式/过滤与packing配置。"
            )
        
        logger.info(f"Training data loader created with {len(self.train_loader)} batches")
    
    def _init_optimizer(self):
        """初始化优化器"""
        training_config = self.config['training']
        
        # 创建优化器（无论是否使用DeepSpeed都需要先创建）
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
            
        if hasattr(self, 'use_deepspeed') and self.use_deepspeed:
            logger.info("Optimizer will be initialized by DeepSpeed")
    
    def _init_deepspeed(self):
        """初始化DeepSpeed（如果启用）"""
        if not (hasattr(self, 'use_deepspeed') and self.use_deepspeed):
            return
        
        if not DEEPSPEED_AVAILABLE:
            logger.error("DeepSpeed not available but use_deepspeed=True")
            return
        
        training_config = self.config['training']
        
        # ✅ 从配置读取ZeRO stage而不是硬编码
        system_config = self.config['system']
        zero_stage = int(system_config.get('zero_stage', 2))
        offload_optimizer = system_config.get('offload_optimizer', False)
        offload_params = system_config.get('offload_params', False)
        
        logger.info(f"DeepSpeed ZeRO stage: {zero_stage}, offload_optimizer: {offload_optimizer}, offload_params: {offload_params}")
        
        # 构建zero_optimization配置
        zero_config = {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
        }
        
        # 根据stage添加相应配置
        if zero_stage == 3:
            zero_config.update({
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto", 
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            })
            
        # 配置offload（注意ZeRO-2不支持参数offload）
        if offload_optimizer:
            zero_config["offload_optimizer"] = {"device": "cpu"}
        else:
            zero_config["offload_optimizer"] = {"device": "none"}
            
        if zero_stage == 3 and offload_params:
            zero_config["offload_param"] = {"device": "cpu"}
        else:
            zero_config["offload_param"] = {"device": "none"}
        
        # 设置DeepSpeed配置 - 完全禁用自定义优化器避免GCC编译问题
        ds_config = {
            "train_batch_size": training_config['batch_size'] * training_config['gradient_accumulation_steps'] * self.world_size,
            "train_micro_batch_size_per_gpu": training_config['batch_size'],
            "gradient_accumulation_steps": training_config['gradient_accumulation_steps'],
            # 移除optimizer配置，强制使用我们传入的PyTorch原生优化器
            # "optimizer": {...},  # 注释掉避免FusedAdam编译
            
            # ✅ 让DeepSpeed管理学习率调度，避免调用顺序问题
            "scheduler": {
                "type": "WarmupCosineLR",
                "params": {
                    "total_num_steps": training_config['max_steps'],
                    "warmup_num_steps": training_config['warmup_steps']
                }
            },
            
            "gradient_clipping": training_config['max_grad_norm'],
            "fp16": {"enabled": training_config.get('fp16', False)},
            "bf16": {"enabled": training_config.get('bf16', True)},
            "zero_optimization": zero_config,
            # ✅ 允许使用自定义PyTorch优化器与ZeRO-Offload
            "zero_force_ds_cpu_optimizer": False
        }
        
        # 初始化DeepSpeed - 传入我们的PyTorch原生优化器，让DS管理scheduler
        logger.info("Initializing DeepSpeed engine...")
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,  # 使用我们已创建的PyTorch AdamW
            lr_scheduler=None,  # ✅ 让DeepSpeed自己创建和管理scheduler
            config=ds_config,
            dist_init_required=False  # 已经初始化了分布式
        )
        
        logger.info("DeepSpeed engine initialized successfully")
    
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
        if self.rank == 0 and step <= 5:  # 前5步显示详细debug
            print(f"  🔧 [DEBUG] Step {step}: Setting model to train mode...")
        self.model.train()
        
        if self.rank == 0 and step <= 5:
            print(f"  📊 [DEBUG] Step {step}: Moving batch data to device...")
        # 移动数据到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        if self.rank == 0 and step <= 5:
            print(f"  🧠 [DEBUG] Step {step}: Starting model forward pass (shape: {input_ids.shape})...")
        
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            collect_gate_stats=True,
            return_dict=True
        )
        
        if self.rank == 0 and step <= 5:
            print(f"  ✅ [DEBUG] Step {step}: Forward pass completed!")
            print(f"  📉 [DEBUG] Step {step}: Computing loss...")
        
        # 计算损失
        loss_dict = self.loss_fn(
            logits=outputs['logits'],
            labels=labels,
            gate_stats=outputs.get('gate_stats', None)
        )
        
        loss = loss_dict['total_loss']
        
        if self.rank == 0 and step <= 5:
            print(f"  🔙 [DEBUG] Step {step}: Starting backward pass...")
        
        # ✅ 关键修改：DeepSpeed 与 非 DeepSpeed 走不同路径
        use_deepspeed = self.config['system'].get('use_deepspeed', False)
        
        if use_deepspeed:
            if self.rank == 0 and step <= 5:
                print(f"  🔥 [DEBUG] Step {step}: Using DeepSpeed backward...")
            
            # DeepSpeed路径：backward & step 都交给 DeepSpeed
            self.model.backward(loss)
            
            if self.rank == 0 and step <= 5:
                print(f"  🚀 [DEBUG] Step {step}: DeepSpeed model step...")
            
            self.model.step()
            
            # 返回是否真正更新了参数
            did_update = self.model.is_gradient_accumulation_boundary()
            
            if self.rank == 0 and step <= 5:
                print(f"  ✅ [DEBUG] Step {step}: DeepSpeed step completed, did_update: {did_update}")
        
        else:
            if self.rank == 0 and step <= 5:
                print(f"  🔥 [DEBUG] Step {step}: Using PyTorch backward...")
            
            # 纯 PyTorch 路径
            loss.backward()
            
            if self.rank == 0 and step <= 5:
                print(f"  ✂️ [DEBUG] Step {step}: Clipping gradients...")
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            
            if self.rank == 0 and step <= 5:
                print(f"  🚀 [DEBUG] Step {step}: PyTorch optimizer step...")
            
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            did_update = True
            
            if self.rank == 0 and step <= 5:
                print(f"  ✅ [DEBUG] Step {step}: PyTorch step completed")
        
        # 更新门控监控
        if self.gate_monitor is not None and outputs.get('gate_stats'):
            for layer_idx, gate_stats in enumerate(outputs['gate_stats']):
                self.gate_monitor.update(layer_idx, gate_stats)
        
        # 返回损失统计和是否更新了参数
        result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        result['did_update'] = did_update
        return result
    
    def train(self):
        """主训练循环"""
        training_config = self.config['training']
        logging_config = self.config['logging']
        
        step = 0
        epoch = 0
        training_complete = False  # 🔧 添加全局停训标志
        
        logger.info("Starting training...")
        
        while step < training_config['max_steps'] and not training_complete:
            epoch += 1
            
            # 设置分布式采样器的epoch（关键：确保各rank数据一致性）
            if (self.distributed and 
                hasattr(self.train_loader, 'sampler') and 
                hasattr(self.train_loader.sampler, 'set_epoch')):
                self.train_loader.sampler.set_epoch(epoch)
            
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
            
            for batch_idx, batch in enumerate(progress_bar):
                if self.rank == 0 and step < 5:  # 前5步显示详细debug
                    print(f"📦 [DEBUG] Starting batch {batch_idx+1}, current step: {step}")
                
                # 训练步骤
                loss_dict = self.train_step(batch, step + 1)
                
                if self.rank == 0 and step < 5:
                    print(f"📦 [DEBUG] train_step completed for batch {batch_idx+1}")
                
                # ✅ 只有在真正更新参数时才增加step
                did_update = loss_dict.pop('did_update', True)
                
                if did_update:
                    step += 1
                    if self.rank == 0:
                        print(f"✅ Step {step}: Loss={loss_dict.get('total_loss', 'N/A'):.4f}")
                else:
                    if self.rank == 0 and step < 5:  # 只在前几步显示累积信息
                        print(f"⏳ Accumulating gradients (step will be {step+1})")
                
                if self.rank == 0 and step < 5:
                    print(f"📦 [DEBUG] Finished processing batch {batch_idx+1}, moving to next...")
                
                # 日志记录（只在真正更新时）
                log_interval = logging_config.get('log_interval', 10)
                if did_update and step % log_interval == 0 and self.rank == 0:
                    # ✅ 获取学习率 - 兼容DeepSpeed和PyTorch
                    use_deepspeed = self.config['system'].get('use_deepspeed', False)
                    if use_deepspeed:
                        lr = self.model.get_lr()[0] if hasattr(self.model, 'get_lr') else self.optimizer.param_groups[0]['lr']
                    else:
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
                
                # 保存检查点（只在真正更新时）
                save_interval = logging_config.get('save_interval', 100)
                if did_update and step % save_interval == 0 and self.rank == 0:
                    self._save_checkpoint(step)
                
                # 门控监控保存（只在真正更新时）
                if (did_update and self.gate_monitor is not None and 
                    step % self.config['gate_monitor']['save_interval'] == 0 and 
                    self.rank == 0):
                    save_path = self.config['gate_monitor']['save_path']
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    self.gate_monitor.save_statistics(save_path)
                
                # 🔧 分布式安全的停训机制：确保所有rank同时收到停训信号
                if did_update:  # 只在完成完整优化步骤后判断
                    if self.distributed:
                        # 创建停训标志张量
                        stop_flag = torch.tensor([0], device=self.device)
                        
                        # rank0判断是否达到max_steps
                        if self.rank == 0 and step >= training_config['max_steps']:
                            stop_flag.fill_(1)
                            
                        # 广播停训信号到所有rank
                        torch.distributed.broadcast(stop_flag, src=0)
                        
                                                # 所有rank根据广播信号决定是否停训
                        if stop_flag.item() == 1:
                            if self.rank == 0:
                                print(f"🔍 [DEBUG] Reached max_steps ({training_config['max_steps']}), broadcasting stop signal")
                            training_complete = True
                            break
                    else:
                        # 非分布式模式：直接判断
                        if step >= training_config['max_steps']:
                            training_complete = True
                            break
            
            # 🔧 确保tqdm进度条正确关闭
            if hasattr(progress_bar, 'close'):
                progress_bar.close()
        
        # 训练结束后保存最终模型和门控统计
        # 🔧 关键修复：所有rank都必须调用DeepSpeed保存，避免barrier死锁
        self._save_checkpoint(step, is_final=True)
        
        # 只有rank0处理gate monitor和其他文件操作
        if self.rank == 0:
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
        
        # 🔧 显式同步确保所有rank完成保存后再打印完成信息
        if hasattr(self, 'distributed') and self.distributed:
            torch.distributed.barrier()
            
        if self.rank == 0:
            logger.info("Training completed!")
    
    def _save_checkpoint(self, step: int, is_final: bool = False):
        """保存检查点"""
        output_dir = self.config['logging']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        use_deepspeed = self.config['system'].get('use_deepspeed', False)
        
        if use_deepspeed:
            # ✅ DeepSpeed保存方式 - 所有rank都必须调用save_checkpoint
            tag = "final" if is_final else f"step_{step}"
            
            if self.rank == 0:
                logger.info(f"Saving DeepSpeed checkpoint: {tag}")
            
            # 🔧 关键：所有rank都必须调用这个函数（内部有barrier）
            self.model.save_checkpoint(output_dir, tag=tag)
            
            # 🔧 只有rank0保存额外的配置文件，避免文件冲突
            if self.rank == 0:
                config_path = os.path.join(output_dir, tag, "config.json")
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                logger.info(f"✅ DeepSpeed checkpoint saved: {output_dir}/{tag}")
                
        else:
            # ✅ PyTorch原生保存方式
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
            
            logger.info(f"Saving PyTorch checkpoint: {save_path}")
            torch.save(checkpoint, save_path)
        
        # 🔧 保存最佳模型链接（只在非DeepSpeed模式或rank0）
        if use_deepspeed:
            # DeepSpeed模式：定义保存路径用于链接创建
            tag = "final" if is_final else f"step_{step}"
            save_path = os.path.join(output_dir, tag)
            
            # 只有rank0创建最佳模型链接，避免冲突
            if self.rank == 0:
                best_path = os.path.join(output_dir, 'best_model')
                if not os.path.exists(best_path) or is_final:
                    if os.path.exists(best_path):
                        os.remove(best_path) 
                    os.symlink(tag, best_path)  # 链接到checkpoint目录
        else:
            # PyTorch模式：正常创建链接
            best_path = os.path.join(output_dir, 'best_model.pt')
            if not os.path.exists(best_path) or is_final:
                if os.path.exists(best_path):
                    os.remove(best_path)
                os.symlink(os.path.basename(save_path), best_path)
            
        # 只在rank0记录保存完成日志
        if self.rank == 0:
            if use_deepspeed:
                logger.info(f"✅ DeepSpeed checkpoint and links created: {save_path}")
            else:
                logger.info(f"✅ PyTorch checkpoint saved: {save_path}")


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