import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from components import RMSNorm, RotaryEmbedding, Attention, SwiGLU


class EncoderBlock(nn.Module):
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
            use_rope=True,
        )
        
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, attention_mask=attention_mask)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


class ContextEncoder(nn.Module):
    """
    Context Encoder
    
    支持共享 token embedding
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        # 共享 embedding
        shared_token_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 8 / 3)
            ffn_dim = ((ffn_dim + 63) // 64) * 64
        
        # Token embedding
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
        
        self.embed_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not (self._shared_embed and module is self.token_embed):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.token_embed(tokens)
        h = self.embed_proj(h)
        h = self.embed_dropout(h)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask.float()) * torch.finfo(h.dtype).min
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = checkpoint(layer, h, attention_mask, use_reentrant=False)
            else:
                h = layer(h, attention_mask)
        
        h = self.norm(h)
        return h