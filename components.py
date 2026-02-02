import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# ============================================================
# 兼容性：RMSNorm
# ============================================================

try:
    from torch.nn import RMSNorm
except ImportError:
    class RMSNorm(nn.Module):
        """兼容 PyTorch < 2.4"""
        def __init__(self, normalized_shape: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x * rms * self.weight


# ============================================================
# RoPE：支持动态扩展
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding
    支持动态长度扩展，适配 KV Cache
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        self._cached_seq_len = 0
        self.register_buffer('cos_cached', None, persistent=False)
        self.register_buffer('sin_cached', None, persistent=False)
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._cached_seq_len and self.cos_cached is not None:
            return
        
        self._cached_seq_len = max(seq_len, self.max_seq_len)
        
        t = torch.arange(self._cached_seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(device, dtype))
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-2]
        
        self._update_cache(seq_len, x.device, x.dtype)
        
        if position_ids is not None:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed


# ============================================================
# Attention：支持 Flash Attention + KV Cache
# ============================================================

class Attention(nn.Module):
    """
    通用 Attention 层
    - 支持 Flash Attention (SDPA)
    - 支持 KV Cache
    - 支持 RoPE
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        is_causal: bool = False,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        
        self.dropout = dropout
        
        if use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, base=rope_base)
        else:
            self.rotary_emb = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        hidden_states: [B, T, D]
        past_key_value: (K, V) 各 [B, H, S, head_dim]
        
        返回: (output, present_key_value)
        """
        B, T, _ = hidden_states.shape
        
        # ✅ KV Cache 使用时的断言
        if past_key_value is not None and self.is_causal and T > 1:
            raise ValueError(
                "KV cache with causal attention only supports single token generation (T=1). "
                f"Got T={T}. For prefill, don't use past_key_value."
            )
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_rope and self.rotary_emb is not None:
            seq_len = position_offset + T
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
            cos = cos[position_offset:position_offset + T]
            sin = sin[position_offset:position_offset + T]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        present_key_value = (k, v) if use_cache else None
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal and past_key_value is None,
            scale=self.scale,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class CrossAttention(nn.Module):
    """
    Cross Attention
    - 支持预计算 Context KV
    - 支持 Context Mask
    - 支持 Flash Attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.context_dim = context_dim or hidden_dim
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.context_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.context_dim, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)
        
        self.dropout = dropout
    
    def compute_kv(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预计算 Context 的 K 和 V
        
        Args:
            context: [B, N, context_dim]
        
        Returns:
            (K, V) 各 [B, H, N, head_dim]
        """
        B, N, _ = context.shape
        
        k = self.k_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        return k, v
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hidden_dim]
            context: [B, N, context_dim]（如果没有预计算 KV）
            context_kv: (K, V) 预计算的 KV
            context_mask: [B, N]，1 表示有效，0 表示 padding
        
        Returns:
            [B, T, hidden_dim]
        """
        B, T, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if context_kv is not None:
            k, v = context_kv
        elif context is not None:
            k, v = self.compute_kv(context)
        else:
            raise ValueError("Must provide either context or context_kv")
        
        # ✅ 处理 context mask
        attn_mask = None
        if context_mask is not None:
            # context_mask: [B, N], 1=valid, 0=padding
            # -> [B, 1, 1, N] for broadcasting with [B, H, T, N]
            attn_mask = context_mask.unsqueeze(1).unsqueeze(2).to(dtype=q.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(q.dtype).min
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_output)


# ============================================================
# FFN
# ============================================================

class SwiGLU(nn.Module):
    """SwiGLU FFN"""
    
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    """标准 MLP with GELU"""
    
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))