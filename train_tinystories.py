#!/usr/bin/env python3
"""
TinyStories 训练脚本
- 3K 字节回退词表
- 共享 embedding
- Cloudflare 隧道监控
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time

from model import LatentPlanLM, LatentPlanLMConfig
from data import get_dataloaders, get_ae_dataloader
from trainer import train_autoencoder
from monitor import MetricsLogger, start_monitor


def get_config_3k():
    """3K 词表优化配置"""
    return LatentPlanLMConfig(
        vocab_size=3000,
        chunk_size=8,
        latent_dim=256,
        embed_dim=384,
        
        ae_hidden_dim=256,
        ae_num_layers=2,
        ae_num_heads=4,
        
        encoder_hidden_dim=384,
        encoder_num_layers=6,
        encoder_num_heads=6,
        
        planner_hidden_dim=256,
        planner_num_layers=4,
        planner_num_heads=4,
        
        decoder_hidden_dim=384,
        decoder_num_layers=4,
        decoder_num_heads=6,
        num_iterations=3,
        
        dropout=0.1,
        use_checkpoint=True,
        share_embed=True,
        tie_output=True,
    )


def get_tiny_config():
    """快速测试配置"""
    return LatentPlanLMConfig(
        vocab_size=3000,
        chunk_size=4,
        latent_dim=128,
        embed_dim=256,
        
        ae_hidden_dim=192,
        ae_num_layers=1,
        ae_num_heads=4,
        
        encoder_hidden_dim=256,
        encoder_num_layers=2,
        encoder_num_heads=4,
        
        planner_hidden_dim=192,
        planner_num_layers=2,
        planner_num_heads=4,
        
        decoder_hidden_dim=256,
        decoder_num_layers=2,
        decoder_num_heads=4,
        num_iterations=2,
        
        dropout=0.1,
        use_checkpoint=False,
        share_embed=True,
        tie_output=True,
    )


class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, logger, config, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.logger = logger
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], betas=(0.9, 0.95))
        
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]
        
        self.scaler = GradScaler() if config["use_amp"] else None
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def train(self):
        warmup_steps = self.config["warmup_steps"]
        
        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step < warmup_steps:
                    lr_scale = (self.global_step + 1) / warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg["initial_lr"] * lr_scale
                
                metrics = self.train_step(batch)
                
                if self.global_step % self.config["log_interval"] == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.log(step=self.global_step, train_loss=metrics["loss"], train_acc=metrics["accuracy"], lr=lr, loss_planner=metrics.get("loss_planner"), loss_decoder=metrics.get("loss_decoder"), epoch=epoch, total_epochs=self.config["num_epochs"])
                    print(f"Step {self.global_step:>6} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.3f} | LR: {lr:.2e}")
                
                if self.global_step % self.config["eval_interval"] == 0 and self.global_step > 0:
                    val_metrics = self.validate()
                    self.logger.log(step=self.global_step, val_loss=val_metrics["loss"], val_acc=val_metrics["accuracy"])
                    print(f"         Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.3f}")
                    
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.save_checkpoint("best_model.pt")
                        print("         ✓ Best model saved!")
                
                if self.global_step % self.config["save_interval"] == 0 and self.global_step > 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
                
                self.global_step += 1
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']} completed in {time.time() - epoch_start:.1f}s")
            print(f"{'='*50}\n")
        
        self.logger.set_status("completed")
    
    def train_step(self, batch):
        prefix = batch["prefix"].to(self.device)
        target = batch["target"].to(self.device)
        prefix_mask = batch.get("prefix_mask")
        if prefix_mask is not None:
            prefix_mask = prefix_mask.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(prefix, target, prefix_mask=prefix_mask)
                loss = outputs["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(prefix, target, prefix_mask=prefix_mask)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in outputs.items()}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_acc, count = 0, 0, 0
        
        for batch in self.val_loader:
            prefix = batch["prefix"].to(self.device)
            target = batch["target"].to(self.device)
            prefix_mask = batch.get("prefix_mask")
            if prefix_mask is not None:
                prefix_mask = prefix_mask.to(self.device)
            
            if self.scaler:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(prefix, target, prefix_mask=prefix_mask)
            else:
                outputs = self.model(prefix, target, prefix_mask=prefix_mask)
            
            total_loss += outputs["loss"].item()
            total_acc += outputs["accuracy"].item()
            count += 1
            if count >= 50:
                break
        
        self.model.train()
        return {"loss": total_loss / count, "accuracy": total_acc / count}
    
    def save_checkpoint(self, filename):
        save_dir = Path(self.config.get("save_dir", "checkpoints"))
        save_dir.mkdir(exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict() if self.scaler else None, "global_step": self.global_step, "best_val_loss": self.best_val_loss}, save_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="3k", choices=["3k", "tiny"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_3k")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_tunnel", action="store_true")
    parser.add_argument("--skip_ae_pretrain", action="store_true")
    parser.add_argument("--skip_tokenizer", action="store_true")
    parser.add_argument("--ae_epochs", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
    # 监控
    print("\n📊 Starting monitor...")
    monitor = start_monitor(log_dir=args.log_dir, port=args.port, use_tunnel=not args.no_tunnel)
    logger = MetricsLogger(args.log_dir)
    logger.set_status("initializing")
    
    # 训练 tokenizer
    if not args.skip_tokenizer and not Path(args.tokenizer_path).exists():
        print("\n🔤 Building tokenizer...")
        from build_tokenizer import build_byte_level_bpe
        build_byte_level_bpe(vocab_size=3000, save_path=args.tokenizer_path)
    
    # 配置
    if args.config == "3k":
        model_config = get_config_3k()
    else:
        model_config = get_tiny_config()
    
    train_config = {"lr": args.lr, "weight_decay": 0.01, "num_epochs": args.num_epochs, "warmup_steps": args.warmup_steps, "grad_clip": args.grad_clip, "log_interval": args.log_interval, "eval_interval": args.eval_interval, "save_interval": args.save_interval, "save_dir": args.save_dir, "use_amp": device == "cuda"}
    
    # 数据
    print("\n📚 Loading data...")
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=args.batch_size, max_prefix_len=256, max_target_len=256, chunk_size=model_config.chunk_size, num_workers=2, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples, tokenizer_path=args.tokenizer_path)
    
    model_config.vocab_size = tokenizer.vocab_size
    
    # 模型
    print("\n🏗️  Creating model...")
    model = LatentPlanLM(model_config)
    
    params = model.count_parameters()
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Shared embedding: {params['shared_embed']:,}")
    print(f"   Autoencoder: {params['autoencoder']:,}")
    print(f"   Encoder: {params['encoder']:,}")
    print(f"   Planner: {params['planner']:,}")
    print(f"   Decoder: {params['decoder']:,}")
    
    # AE 预训练
    if not args.skip_ae_pretrain and args.resume is None:
        print("\n🔧 Phase 1: Pretraining Autoencoder...")
        logger.set_status("ae_pretrain")
        ae_loader, _ = get_ae_dataloader(batch_size=64, chunk_size=model_config.chunk_size, num_workers=2, max_samples=50000, tokenizer_path=args.tokenizer_path)
        train_autoencoder(model.autoencoder, ae_loader, num_epochs=args.ae_epochs, lr=1e-3, device=device, use_amp=train_config["use_amp"])
        model.freeze_autoencoder()
    
    # 恢复
    if args.resume:
        print(f"\n📂 Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
    
    # 训练
    print("\n🚀 Phase 2: Training Full Model...")
    logger.set_status("training")
    
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, tokenizer=tokenizer, logger=logger, config=train_config, device=device)
    
    if args.resume and "optimizer" in ckpt:
        trainer.global_step = ckpt.get("global_step", 0)
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        if trainer.scaler and ckpt.get("scaler"):
            trainer.scaler.load_state_dict(ckpt["scaler"])
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        logger.set_status("interrupted")
        trainer.save_checkpoint("interrupted.pt")
    
    trainer.save_checkpoint("final_model.pt")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()