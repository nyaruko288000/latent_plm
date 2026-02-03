"""
World Model = Encoder + Planner (合并)

Prefix-LM 风格:
- Prefix tokens: 双向注意力
- Latent sequence: 因果注意力 + 可看所有 prefix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from torch.utils.checkpoint import checkpoint

from components import RMSNorm, RotaryEmbedding, SwiGLU, apply_rotary_pos_emb


class WorldModelBlock(nn.Module):
    """World Model 的 Transformer Block"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.attn_norm = RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim)
        
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        
        h = self.attn_norm(x)
        
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = self.rotary(q, seq_len=position_offset + T)
        cos = cos[position_offset:position_offset + T]
        sin = sin[position_offset:position_offset + T]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_kv = (k, v) if use_cache else None
        
        # Attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = self.o_proj(attn_out)
        
        h = x + self.dropout(attn_out)
        h = h + self.dropout(self.ffn(self.ffn_norm(h)))
        
        return h, present_kv


class WorldModel(nn.Module):
    """
    World Model = Encoder + Planner (Prefix-LM)
    
    - Prefix 部分：双向注意力，理解上下文
    - Latent 部分：因果注意力 + 看 prefix，规划潜向量
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 384,
        latent_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 6,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        use_checkpoint: bool = False,
        shared_token_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 8 / 3)
            ffn_dim = ((ffn_dim + 63) // 64) * 64
        
        # Token Embedding
        if shared_token_embed is not None:
            self.token_embed = shared_token_embed
            embed_dim = shared_token_embed.embedding_dim
            self.token_proj = nn.Linear(embed_dim, hidden_dim, bias=False) if embed_dim != hidden_dim else nn.Identity()
        else:
            self.token_embed = nn.Embedding(vocab_size, hidden_dim)
            self.token_proj = nn.Identity()
        
        # Latent Embedding
        self.latent_proj = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.start_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        # 类型编码: 0=prefix, 1=latent
        self.type_embed = nn.Embedding(2, hidden_dim)
        
        # Transformer
        self.layers = nn.ModuleList([
            WorldModelBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        
        # Output Heads
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.eos_head = nn.Linear(hidden_dim, 1)
        self.eos_vector = nn.Parameter(torch.randn(latent_dim) * 0.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _build_attn_mask(
        self,
        prefix_len: int,
        latent_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prefix-LM mask:
        - prefix: 双向 (互相可见)
        - latent: 因果 + 可看所有 prefix
        """
        total = prefix_len + latent_len
        mask = torch.zeros(total, total, device=device, dtype=dtype)
        
        # prefix 不能看 latent
        mask[:prefix_len, prefix_len:] = torch.finfo(dtype).min
        
        # latent 内部因果
        if latent_len > 1:
            latent_causal = torch.triu(
                torch.full((latent_len, latent_len), torch.finfo(dtype).min, device=device, dtype=dtype),
                diagonal=1
            )
            mask[prefix_len:, prefix_len:] = latent_causal
        
        return mask
    
    def forward(
        self,
        prefix_tokens: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练模式 (Teacher forcing)
        
        Args:
            prefix_tokens: [B, N]
            z_target: [B, M, latent_dim]
            prefix_mask: [B, N], 1=valid, 0=padding
        
        Returns:
            z_pred, eos_logits, prefix_hidden
        """
        B, N = prefix_tokens.shape
        device = prefix_tokens.device
        dtype = next(self.parameters()).dtype
        
        # Embed prefix
        h_prefix = self.token_embed(prefix_tokens)
        h_prefix = self.token_proj(h_prefix)
        h_prefix = h_prefix + self.type_embed.weight[0]
        
        # Embed latent
        start = self.start_token.view(1, 1, -1).expand(B, 1, -1)
        if z_target is not None:
            h_latent = torch.cat([start, self.latent_proj(z_target)], dim=1)
        else:
            h_latent = start
        
        L = h_latent.shape[1]
        h_latent = h_latent + self.type_embed.weight[1]
        
        # 拼接
        h = torch.cat([h_prefix, h_latent], dim=1)
        
        # Attention mask
        attn_mask = self._build_attn_mask(N, L, device, h.dtype)
        
        # Prefix padding mask
        if prefix_mask is not None:
            pad_mask = (1.0 - prefix_mask.float()) * torch.finfo(h.dtype).min
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1).clone()
            attn_mask[:, :, :N] = attn_mask[:, :, :N] + pad_mask.unsqueeze(1)
        
        # Transformer
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h, _ = checkpoint(lambda x, m: layer(x, attn_mask=m), h, attn_mask, use_reentrant=False)
            else:
                h, _ = layer(h, attn_mask=attn_mask)
        
        h = self.norm(h)
        
        # Output
        return {
            'z_pred': self.latent_head(h[:, N:]),
            'eos_logits': self.eos_head(h[:, N:]).squeeze(-1),
            'prefix_hidden': h[:, :N],
        }
    
    @torch.no_grad()
    def generate(
        self,
        prefix_tokens: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        eos_threshold: float = 0.5,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自回归生成潜向量序列"""
        B, N = prefix_tokens.shape
        device = prefix_tokens.device
        dtype = next(self.parameters()).dtype
        
        # Prefill: prefix + start
        h_prefix = self.token_embed(prefix_tokens)
        h_prefix = self.token_proj(h_prefix)
        h_prefix = h_prefix + self.type_embed.weight[0]
        
        start = self.start_token.view(1, 1, -1).expand(B, 1, -1) + self.type_embed.weight[1]
        h = torch.cat([h_prefix, start], dim=1)
        
        # Prefill mask
        prefill_mask = self._build_attn_mask(N, 1, device, dtype)
        if prefix_mask is not None:
            pad_mask = (1.0 - prefix_mask.float()) * torch.finfo(dtype).min
            prefill_mask = prefill_mask.unsqueeze(0).expand(B, -1, -1).clone()
            prefill_mask[:, :, :N] = prefill_mask[:, :, :N] + pad_mask.unsqueeze(1)
        
        # Prefill forward
        past_kvs = [None] * self.num_layers
        for i, layer in enumerate(self.layers):
            h, past_kvs[i] = layer(h, attn_mask=prefill_mask, use_cache=True)
        h = self.norm(h)
        
        prefix_hidden = h[:, :N]
        h_last = h[:, -1:]
        
        # Autoregressive
        generated_z, eos_probs = [], []
        
        for step in range(max_length):
            z_next = self.latent_head(h_last)
            eos_prob = torch.sigmoid(self.eos_head(h_last)).squeeze(-1)
            
            if temperature > 0:
                z_next = z_next + torch.randn_like(z_next) * temperature * 0.1
            
            generated_z.append(z_next)
            eos_probs.append(eos_prob)
            
            if (eos_prob.squeeze(-1) > eos_threshold).all():
                break
            
            # Next step
            h_next = self.latent_proj(z_next) + self.type_embed.weight[1]
            for i, layer in enumerate(self.layers):
                h_next, past_kvs[i] = layer(
                    h_next, past_kv=past_kvs[i], use_cache=True, position_offset=N + 1 + step
                )
            h_last = self.norm(h_next)
        
        return torch.cat(generated_z, dim=1), torch.cat(eos_probs, dim=1), prefix_hidden