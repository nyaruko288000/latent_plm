import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from autoencoder import TokenChunkAutoencoder
from encoder import ContextEncoder
from planner import LatentARPlanner
from decoder import IterativeNARDecoder


@dataclass
class LatentPlanLMConfig:
    """模型配置"""
    
    vocab_size: int = 3000  # 默认使用小词表
    chunk_size: int = 8
    latent_dim: int = 192
    
    # 共享 embedding 维度（所有组件使用相同维度）
    embed_dim: int = 384
    
    # Autoencoder
    ae_hidden_dim: int = 256
    ae_num_layers: int = 2
    ae_num_heads: int = 4
    
    # Encoder
    encoder_hidden_dim: int = 384
    encoder_num_layers: int = 6
    encoder_num_heads: int = 6
    
    # Planner
    planner_hidden_dim: int = 256
    planner_num_layers: int = 4
    planner_num_heads: int = 4
    
    # Decoder
    decoder_hidden_dim: int = 384
    decoder_num_layers: int = 4
    decoder_num_heads: int = 6
    num_iterations: int = 3
    
    dropout: float = 0.1
    use_checkpoint: bool = True
    eos_loss_weight: float = 0.1
    
    # 共享选项
    share_embed: bool = True  # 是否共享 embedding
    tie_output: bool = True   # 是否绑定 output 和 embedding
    
    def __post_init__(self):
        assert self.encoder_hidden_dim % self.encoder_num_heads == 0
        assert self.planner_hidden_dim % self.planner_num_heads == 0
        assert self.decoder_hidden_dim % self.decoder_num_heads == 0


class LatentPlanLM(nn.Module):
    """
    Latent Plan Language Model
    
    共享 embedding 架构
    """
    
    def __init__(self, config: LatentPlanLMConfig):
        super().__init__()
        
        self.config = config
        self.chunk_size = config.chunk_size
        
        # ========== 共享 Token Embedding ==========
        if config.share_embed:
            self.shared_token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
            
            # 共享输出投影（绑定权重）
            if config.tie_output:
                self.shared_output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
                self.shared_output_proj.weight = self.shared_token_embed.weight
            else:
                self.shared_output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        else:
            self.shared_token_embed = None
            self.shared_output_proj = None
        
        # ========== Autoencoder ==========
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
        
        # ========== Encoder ==========
        self.encoder = ContextEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
            shared_token_embed=self.shared_token_embed,
        )
        
        # ========== Planner ==========
        self.planner = LatentARPlanner(
            latent_dim=config.latent_dim,
            hidden_dim=config.planner_hidden_dim,
            context_dim=config.encoder_hidden_dim,
            num_layers=config.planner_num_layers,
            num_heads=config.planner_num_heads,
            dropout=config.dropout,
            use_checkpoint=config.use_checkpoint,
        )
        
        # ========== Decoder ==========
        self.decoder = IterativeNARDecoder(
            vocab_size=config.vocab_size,
            latent_dim=config.latent_dim,
            hidden_dim=config.decoder_hidden_dim,
            context_dim=config.encoder_hidden_dim,
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
        
        z_context = self.encoder(prefix_tokens, attention_mask=prefix_mask)
        
        planner_context_kvs = self.planner.precompute_context_kv(z_context)
        decoder_context_kvs = self.decoder.precompute_context_kv(z_context)
        
        target_tokens = self._pad_to_chunk(target_tokens)
        
        with torch.no_grad():
            z_target = self.autoencoder.encode_sequence(target_tokens)
        
        planner_out = self.planner(z_context, z_target, context_mask=prefix_mask, context_kvs=planner_context_kvs)
        z_pred = planner_out['z_pred']
        eos_logits = planner_out['eos_logits']
        
        eos_vector = self.planner.eos_vector.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        z_target_with_eos = torch.cat([z_target, eos_vector], dim=1)
        
        loss_planner_z = F.mse_loss(z_pred, z_target_with_eos)
        
        M = z_target.shape[1]
        eos_labels = torch.zeros(B, M + 1, device=z_target.device)
        eos_labels[:, -1] = 1.0
        loss_planner_eos = F.binary_cross_entropy_with_logits(eos_logits, eos_labels)
        
        loss_planner = loss_planner_z + self.config.eos_loss_weight * loss_planner_eos
        
        z_noisy = z_target + torch.randn_like(z_target) * noise_scale
        
        logits = self.decoder(z_noisy, z_context, context_mask=prefix_mask, context_kvs=decoder_context_kvs)
        loss_decoder = F.cross_entropy(logits.view(-1, self.config.vocab_size), target_tokens.view(-1))
        
        loss = loss_planner + loss_decoder
        
        with torch.no_grad():
            preds = logits.argmax(-1)
            accuracy = (preds == target_tokens).float().mean()
        
        return {
            'loss': loss,
            'loss_planner': loss_planner,
            'loss_planner_z': loss_planner_z,
            'loss_planner_eos': loss_planner_eos,
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
        planner_temperature: float = 0.0,
    ) -> torch.Tensor:
        if prefix_tokens.dim() == 1:
            prefix_tokens = prefix_tokens.unsqueeze(0)
            if prefix_mask is not None:
                prefix_mask = prefix_mask.unsqueeze(0)
        
        z_context = self.encoder(prefix_tokens, attention_mask=prefix_mask)
        decoder_context_kvs = self.decoder.precompute_context_kv(z_context)
        
        z_plan, eos_probs = self.planner.generate(z_context, context_mask=prefix_mask, max_length=max_new_chunks, eos_threshold=eos_threshold, temperature=planner_temperature)
        z_plan, valid_mask = self._process_eos(z_plan, eos_probs, eos_threshold)
        
        tokens = self.decoder.generate(z_plan, z_context=None, context_mask=prefix_mask, context_kvs=decoder_context_kvs, num_iterations=num_iterations, temperature=temperature, top_k=top_k, top_p=top_p)
        
        return tokens
    
    def _pad_to_chunk(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        remainder = T % self.chunk_size
        if remainder == 0:
            return tokens
        pad_len = self.chunk_size - remainder
        padding = torch.full((B, pad_len), self.config.vocab_size - 1, dtype=tokens.dtype, device=tokens.device)
        return torch.cat([tokens, padding], dim=1)
    
    def _process_eos(self, z_plan: torch.Tensor, eos_probs: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
        B, M, _ = z_plan.shape
        device = z_plan.device
        
        is_eos = eos_probs > threshold
        has_eos = is_eos.any(dim=1)
        
        first_eos_idx = torch.where(has_eos, is_eos.long().argmax(dim=1), torch.tensor(M, device=device).expand(B))
        
        positions = torch.arange(M, device=device).unsqueeze(0)
        valid_mask = positions < first_eos_idx.unsqueeze(1)
        valid_mask[:, 0] = True
        
        valid_lengths = valid_mask.sum(dim=1)
        max_valid_len = int(valid_lengths.max().item())
        
        z_plan = z_plan[:, :max_valid_len]
        valid_mask = valid_mask[:, :max_valid_len]
        
        return z_plan, valid_mask
    
    def freeze_autoencoder(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        # 但保持共享 embedding 可训练
        if self.shared_token_embed is not None:
            self.shared_token_embed.requires_grad_(True)
        if self.shared_output_proj is not None:
            for param in self.shared_output_proj.parameters():
                param.requires_grad = True
    
    def unfreeze_autoencoder(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = True
    
    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        shared_embed_params = count(self.shared_token_embed) if self.shared_token_embed else 0
        shared_output_params = count(self.shared_output_proj) if self.shared_output_proj else 0
        
        # 扣除共享部分的重复计算
        total = count(self)
        
        return {
            'total': total,
            'shared_embed': shared_embed_params,
            'shared_output': shared_output_params if not self.config.tie_output else 0,  # tie_output 时不额外计算
            'autoencoder': count(self.autoencoder) - shared_embed_params,
            'encoder': count(self.encoder) - shared_embed_params,
            'planner': count(self.planner),
            'decoder': count(self.decoder) - shared_embed_params,
        }