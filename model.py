"""Latent Plan Language Model - 合并版"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from autoencoder import TokenChunkAutoencoder
from world_model import WorldModel
from decoder import IterativeNARDecoder


@dataclass
class LatentPlanLMConfig:
    vocab_size: int = 3000
    chunk_size: int = 8
    latent_dim: int = 256
    embed_dim: int = 384
    
    # Autoencoder
    ae_hidden_dim: int = 256
    ae_num_layers: int = 2
    ae_num_heads: int = 4
    
    # World Model (8层)
    wm_hidden_dim: int = 384
    wm_num_layers: int = 8
    wm_num_heads: int = 6
    
    # Decoder
    decoder_hidden_dim: int = 384
    decoder_num_layers: int = 4
    decoder_num_heads: int = 6
    num_iterations: int = 3
    
    dropout: float = 0.1
    use_checkpoint: bool = True
    eos_loss_weight: float = 0.1
    share_embed: bool = True
    tie_output: bool = True


class LatentPlanLM(nn.Module):
    def __init__(self, config: LatentPlanLMConfig):
        super().__init__()
        self.config = config
        self.chunk_size = config.chunk_size
        
        # 共享 Embedding
        if config.share_embed:
            self.shared_token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
            if config.tie_output:
                self.shared_output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
                self.shared_output_proj.weight = self.shared_token_embed.weight
            else:
                self.shared_output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        else:
            self.shared_token_embed = None
            self.shared_output_proj = None
        
        # Autoencoder
        self.autoencoder = TokenChunkAutoencoder(
            vocab_size=config.vocab_size,
            chunk_size=config.chunk_size,
            latent_dim=config.latent_dim,
            hidden_dim=config.ae_hidden_dim,
            num_layers=config.ae_num_layers,
            num_heads=config.ae_num_heads,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            shared_token_embed=self.shared_token_embed,
            shared_output_proj=self.shared_output_proj,
        )
        
        # World Model
        self.world_model = WorldModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.wm_hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.wm_num_layers,
            num_heads=config.wm_num_heads,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            shared_token_embed=self.shared_token_embed,
        )
        
        # Decoder
        self.decoder = IterativeNARDecoder(
            vocab_size=config.vocab_size,
            latent_dim=config.latent_dim,
            hidden_dim=config.decoder_hidden_dim,
            context_dim=config.wm_hidden_dim,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            chunk_size=config.chunk_size,
            num_iterations=config.num_iterations,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            shared_token_embed=self.shared_token_embed,
            shared_output_proj=self.shared_output_proj,
        )
    
    def forward(
        self,
        prefix_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        noise_scale: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        B = prefix_tokens.shape[0]
        target_tokens = self._pad_to_chunk(target_tokens)
    
        # 编码目标（frozen AE）
        with torch.no_grad():
            z_target = self.autoencoder.encode_sequence(target_tokens)
    
        # World Model forward
        wm_out = self.world_model(prefix_tokens, z_target, prefix_mask)
        z_pred = wm_out['z_pred']
        eos_logits = wm_out['eos_logits']
        prefix_hidden = wm_out['prefix_hidden']
    
        # World Model Loss
        M = z_target.shape[1]
        eos_vector = self.world_model.eos_vector.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        z_target_eos = torch.cat([z_target, eos_vector], dim=1)
    
        loss_wm_z = F.mse_loss(z_pred, z_target_eos)
    
        eos_labels = torch.zeros(B, M + 1, device=z_target.device)
        eos_labels[:, -1] = 1.0
        loss_wm_eos = F.binary_cross_entropy_with_logits(eos_logits, eos_labels)
    
        loss_wm = loss_wm_z + self.config.eos_loss_weight * loss_wm_eos
    
        # Decoder forward
        z_noisy = z_target + torch.randn_like(z_target) * noise_scale
        ctx_kvs = self.decoder.precompute_context_kv(prefix_hidden)
        logits = self.decoder(
            z_plan=z_noisy,
            z_context=None,
            context_mask=prefix_mask,
            context_kvs=ctx_kvs,
        )
    
        loss_decoder = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            target_tokens.view(-1),
        )
    
        # Total loss
        loss = loss_wm + loss_decoder
    
        # Metrics
        with torch.no_grad():
            accuracy = (logits.argmax(-1) == target_tokens).float().mean()
    
        return {
            'loss': loss,
            'loss_wm': loss_wm,
            'loss_wm_z': loss_wm_z,
            'loss_wm_eos': loss_wm_eos,
            'loss_decoder': loss_decoder,
            'accuracy': accuracy,
        }
    
    @torch.no_grad()
    def generate(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        max_new_chunks: int = 128,
        eos_threshold: float = 0.5,
        num_iterations: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if prefix_tokens.dim() == 1:
            prefix_tokens = prefix_tokens.unsqueeze(0)
            if prefix_mask is not None:
                prefix_mask = prefix_mask.unsqueeze(0)
        
        z_plan, eos_probs, prefix_hidden = self.world_model.generate(
            prefix_tokens, prefix_mask, max_new_chunks, eos_threshold
        )
        z_plan = self._truncate_at_eos(z_plan, eos_probs, eos_threshold)
        
        ctx_kvs = self.decoder.precompute_context_kv(prefix_hidden)
        tokens = self.decoder.generate(
            z_plan, context_mask=prefix_mask, context_kvs=ctx_kvs,
            num_iterations=num_iterations, temperature=temperature, top_k=top_k, top_p=top_p
        )
        return tokens
    
    def _pad_to_chunk(self, tokens):
        B, T = tokens.shape
        r = T % self.chunk_size
        if r == 0:
            return tokens
        pad = torch.full((B, self.chunk_size - r), self.config.vocab_size - 1, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, pad], dim=1)
    
    def _truncate_at_eos(self, z, eos_probs, threshold):
        """在第一个 EOS 处截断"""
        B, M = eos_probs.shape
        device = z.device
    
        is_eos = eos_probs > threshold
        has_eos = is_eos.any(dim=1)
    
        # 有 EOS 取第一个位置，没有则取 M（全保留）
        first_eos = torch.where(
            has_eos,
            is_eos.long().argmax(dim=1),
            torch.full((B,), M, device=device, dtype=torch.long)
        )
     # 至少保留 1 个 chunk
        max_len = int(torch.clamp(first_eos, min=1).max().item())
        return z[:, :max_len]
    
    def freeze_autoencoder(self):
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        if self.shared_token_embed is not None:
            self.shared_token_embed.requires_grad_(True)
        if self.shared_output_proj is not None:
            for p in self.shared_output_proj.parameters():
                p.requires_grad = True
    
    def count_parameters(self) -> Dict[str, int]:
        def cnt(m): return sum(p.numel() for p in m.parameters())
        return {
            'total': cnt(self),
            'shared_embed': cnt(self.shared_token_embed) if self.shared_token_embed else 0,
            'autoencoder': cnt(self.autoencoder),
            'world_model': cnt(self.world_model),
            'decoder': cnt(self.decoder),
        }