import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch.utils.checkpoint import checkpoint

from .components import RMSNorm, Attention, MLP


class AutoencoderBlock(nn.Module):
    """Autoencoder Transformer Block"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_causal=False,
            use_rope=False,
        )
        
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = MLP(hidden_dim, ffn_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x)
        x = residual + x
        
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class TokenChunkAutoencoder(nn.Module):
    """
    Token Chunk Autoencoder
    
    支持共享 embedding（可选）
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        chunk_size: int = 8,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        kl_weight: float = 0.001,
        free_bits: float = 2.0,
        use_checkpoint: bool = False,
        # 共享 embedding
        shared_token_embed: Optional[nn.Embedding] = None,
        shared_output_proj: Optional[nn.Linear] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.free_bits = free_bits
        self.use_checkpoint = use_checkpoint
        
        # ========== Token Embedding ==========
        if shared_token_embed is not None:
            self.token_embed = shared_token_embed
            embed_dim = shared_token_embed.embedding_dim
            if embed_dim != hidden_dim:
                self.embed_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
            else:
                self.embed_proj = nn.Identity()
            self._shared_embed = True
        else:
            self.token_embed = nn.Embedding(vocab_size, hidden_dim)
            self.embed_proj = nn.Identity()
            self._shared_embed = False
        
        # 位置编码
        self.pos_embed = nn.Embedding(chunk_size, hidden_dim)
        
        # ========== Encoder ==========
        self.encoder_layers = nn.ModuleList([
            AutoencoderBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = RMSNorm(hidden_dim)
        
        self.compress = nn.Sequential(
            nn.Linear(chunk_size * hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # ========== Decoder ==========
        self.decompress = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, chunk_size * hidden_dim),
        )
        
        self.decoder_layers = nn.ModuleList([
            AutoencoderBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_norm = RMSNorm(hidden_dim)
        
        # ========== Output ==========
        if shared_output_proj is not None:
            self.output_proj = shared_output_proj
            self._shared_output = True
        else:
            self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
            self._shared_output = False
            
            # 可选：绑定 embedding 权重
            if not self._shared_embed:
                self.output_proj.weight = self.token_embed.weight
        
        # 如果输出维度不匹配，需要投影
        if shared_output_proj is not None:
            out_features = shared_output_proj.in_features
            if out_features != hidden_dim:
                self.output_hidden_proj = nn.Linear(hidden_dim, out_features, bias=False)
            else:
                self.output_hidden_proj = nn.Identity()
        else:
            self.output_hidden_proj = nn.Identity()
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not (self._shared_embed and module is self.token_embed):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(
        self,
        tokens: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = tokens.shape
        
        if tokens.dim() == 3:
            B, M, K = tokens.shape
            tokens = tokens.view(B * M, K)
        else:
            B, K = tokens.shape
            M = None
        
        assert K == self.chunk_size
        
        positions = torch.arange(K, device=tokens.device)
        h = self.token_embed(tokens)
        h = self.embed_proj(h)
        h = h + self.pos_embed(positions)
        
        for layer in self.encoder_layers:
            if self.use_checkpoint and self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        
        h = self.encoder_norm(h)
        h = h.view(-1, K * self.hidden_dim)
        h = self.compress(h)
        
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        
        if M is not None:
            z = z.view(B, M, -1)
            mu = mu.view(B, M, -1)
            logvar = logvar.view(B, M, -1)
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        original_shape = z.shape
        
        if z.dim() == 3:
            B, M, D = z.shape
            z = z.view(B * M, D)
        else:
            B, D = z.shape
            M = None
        
        h = self.decompress(z)
        h = h.view(-1, self.chunk_size, self.hidden_dim)
        
        positions = torch.arange(self.chunk_size, device=z.device)
        h = h + self.pos_embed(positions)
        
        for layer in self.decoder_layers:
            if self.use_checkpoint and self.training:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        
        h = self.decoder_norm(h)
        h = self.output_hidden_proj(h)
        logits = self.output_proj(h)
        
        if M is not None:
            logits = logits.view(B, M, self.chunk_size, -1)
        
        return logits
    
    def forward(
        self,
        tokens: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(tokens)
        logits = self.decode(z)
        
        if not return_loss:
            return {'z': z, 'logits': logits, 'mu': mu, 'logvar': logvar}
        
        recon_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            tokens.view(-1)
        )
        
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_per_dim.mean()
        
        kl_total_per_sample = kl_per_dim.sum(dim=-1)
        free_bits_threshold = self.free_bits * self.latent_dim
        kl_above_threshold = F.relu(kl_total_per_sample - free_bits_threshold)
        kl_loss_adjusted = kl_above_threshold.mean() / self.latent_dim
        
        total_loss = recon_loss + self.kl_weight * kl_loss_adjusted
        
        with torch.no_grad():
            preds = logits.argmax(-1)
            accuracy = (preds == tokens).float().mean()
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_loss_adjusted': kl_loss_adjusted,
            'accuracy': accuracy,
            'z': z,
            'mu': mu,
            'logvar': logvar,
        }
    
    def encode_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        assert T % self.chunk_size == 0
        
        M = T // self.chunk_size
        chunks = tokens.view(B, M, self.chunk_size)
        z, _, _ = self.encode(chunks, deterministic=True)
        
        return z
    
    def decode_sequence(
        self,
        z: torch.Tensor,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        logits = self.decode(z)
        
        if temperature <= 0:
            tokens = logits.argmax(-1)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)),
                num_samples=1
            ).view(logits.shape[:-1])
        
        B, M, K = tokens.shape
        return tokens.view(B, M * K)