import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Callable
import time


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_steps: int = 1000,
        grad_clip: float = 1.0,
        device: str = 'cuda',
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        save_dir: Optional[str] = None,
        callback: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.grad_clip = grad_clip
        self.num_epochs = num_epochs
        self.use_amp = use_amp and device == 'cuda'
        self.amp_dtype = amp_dtype
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.callback = callback
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        for pg in self.optimizer.param_groups:
            pg['initial_lr'] = pg['lr']
        
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.1,
        )
        
        self.warmup_steps = warmup_steps
        self.global_step = 0
        
        self.scaler = GradScaler() if self.use_amp else None
    
    def train(self):
        """完整训练"""
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                
                if self.global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {self.global_step} | Loss: {metrics['loss']:.4f} | "
                          f"LR: {lr:.2e} | Acc: {metrics.get('accuracy', 0):.4f}")
                
                if self.val_dataloader and self.global_step % self.eval_interval == 0:
                    val_metrics = self.validate()
                    print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                          f"Val Acc: {val_metrics.get('accuracy', 0):.4f}")
                
                if self.save_dir and self.global_step % self.save_interval == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"\nEpoch {epoch+1}/{self.num_epochs} | "
                  f"Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s\n")
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """单步训练"""
        prefix = batch['prefix'].to(self.device)
        target = batch['target'].to(self.device)
        prefix_mask = batch.get('prefix_mask')
        if prefix_mask is not None:
            prefix_mask = prefix_mask.to(self.device)
        
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['initial_lr'] * lr_scale
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                outputs = self.model(prefix, target, prefix_mask=prefix_mask)
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(prefix, target, prefix_mask=prefix_mask)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        if self.global_step >= self.warmup_steps:
            self.scheduler.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in outputs.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_metrics = {}
        count = 0
        
        for batch in self.val_dataloader:
            prefix = batch['prefix'].to(self.device)
            target = batch['target'].to(self.device)
            prefix_mask = batch.get('prefix_mask')
            if prefix_mask is not None:
                prefix_mask = prefix_mask.to(self.device)
            
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(prefix, target, prefix_mask=prefix_mask)
            else:
                outputs = self.model(prefix, target, prefix_mask=prefix_mask)
            
            for k, v in outputs.items():
                if torch.is_tensor(v):
                    v = v.item()
                total_metrics[k] = total_metrics.get(k, 0) + v
            count += 1
        
        self.model.train()
        return {k: v / count for k, v in total_metrics.items()}
    
    def save_checkpoint(self, path: Optional[str] = None):
        """保存检查点"""
        if path is None:
            path = f"{self.save_dir}/checkpoint_{self.global_step}.pt"
        
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        ckpt = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        if self.scaler and ckpt.get('scaler'):
            self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt['global_step']


def train_autoencoder(
    autoencoder: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cuda',
    use_amp: bool = True,
    save_dir: str = "checkpoints",
    resume_epoch: int = 0,
) -> nn.Module:
    """
    阶段1：预训练 Autoencoder（固定学习率，无调度器）
    """
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)
    
    autoencoder = autoencoder.to(device)
    
    # 加载 checkpoint
    if resume_epoch > 0:
        ckpt_path = f"{save_dir}/autoencoder_epoch{resume_epoch}.pt"
        autoencoder.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"✓ Resumed AE from epoch {resume_epoch}")
    
    autoencoder.train()
    
    optimizer = AdamW(autoencoder.parameters(), lr=lr)
    scaler = GradScaler() if use_amp and device == 'cuda' else None
    
    for epoch in range(resume_epoch, num_epochs):
        total_loss = 0
        total_acc = 0
        autoencoder.train()
        
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast(dtype=torch.bfloat16):
                    outputs = autoencoder(tokens)
                    loss = outputs['loss']
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = autoencoder(tokens)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            total_acc += outputs['accuracy'].item()
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        print(f"AE Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        
        # 保存
        torch.save(
            autoencoder.state_dict(),
            f"{save_dir}/autoencoder_epoch{epoch+1}.pt"
        )
    
    return autoencoder