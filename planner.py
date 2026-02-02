import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from functools import partial
from torch.utils.checkpoint import checkpoint

from .components import RMSNorm, RotaryEmbedding, Attention, CrossAttention, SwiGLU


class PlannerBlock(nn.Module):
    """Planner 的 Transformer Block"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        context_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn_norm = RMSNorm(hidden_dim)
        self.self_attn = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            is_causal=True,
            use_rope=True,
        )
        
        self.cross_attn_norm = RMSNorm(hidden_dim)
        self.cross_attn = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            context_dim=context_dim,
            dropout=dropout,
        )
        
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim, ffn_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        context_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [B, T, hidden_dim]
            context: [B, N, context_dim]
            context_kv: 预计算的 (K, V)
            context_mask: [B, N]，1=valid, 0=padding
            past_key_value: Self-attention 的 KV cache
            use_cache: 是否返回 KV cache
            position_offset: RoPE 位置偏移
        
        Returns:
            (output, present_self_kv)
            - output: [B, T, hidden_dim]
            - present_self_kv: Optional[Tuple[K, V]] 用于 KV cache
        """
        # Self-Attention
        residual = x
        x = self.self_attn_norm(x)
        x, present_self_kv = self.self_attn(
            x,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        x = residual + self.dropout(x)
        
        # Cross-Attention
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(
            x, 
            context=context, 
            context_kv=context_kv,
            context_mask=context_mask,
        )
        x = residual + self.dropout(x)
        
        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x, present_self_kv


def _checkpoint_forward(
    layer: PlannerBlock,
    x: torch.Tensor,
    context_kv: Tuple[torch.Tensor, torch.Tensor],
    context_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, None]:
    """Checkpoint wrapper，避免位置参数顺序问题"""
    out, _ = layer(
        x,
        context=None,
        context_kv=context_kv,
        context_mask=context_mask,
        past_key_value=None,
        use_cache=False,
        position_offset=0,
    )
    return out, None


class LatentARPlanner(nn.Module):
    """
    潜空间自回归规划器
    
    改进：
    - KV Cache 支持
    - Context KV 预计算
    - Context Mask 支持
    - 可学习 EOS + 分类头
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        context_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 8 / 3)
            ffn_dim = ((ffn_dim + 63) // 64) * 64
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.start_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        self.layers = nn.ModuleList([
            PlannerBlock(hidden_dim, num_heads, ffn_dim, context_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
        
        self.output_head = nn.Sequential(
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
    
    def precompute_context_kv(
        self, 
        context: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """预计算所有层的 context KV"""
        context_kvs = []
        for layer in self.layers:
            kv = layer.cross_attn.compute_kv(context)
            context_kvs.append(kv)
        return context_kvs
    
    def forward(
        self,
        z_context: torch.Tensor,
        z_target: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        context_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练模式：Teacher forcing
        
        Args:
            z_context: [B, N, context_dim]
            z_target: [B, M, latent_dim]
            context_mask: [B, N]，1=valid, 0=padding
            context_kvs: 预计算的 context KV（可选）
        
        Returns:
            {'z_pred': [B, M+1, latent_dim], 'eos_logits': [B, M+1]}
        """
        B = z_context.shape[0]
        
        if context_kvs is None:
            context_kvs = self.precompute_context_kv(z_context)
        
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        
        if z_target is not None:
            z_input = self.input_proj(z_target)
            h = torch.cat([start, z_input], dim=1)
        else:
            h = start
        
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                h, _ = checkpoint(
                    _checkpoint_forward,
                    layer, h, context_kvs[i], context_mask,
                    use_reentrant=False
                )
            else:
                h, _ = layer(
                    h, 
                    context_kv=context_kvs[i],
                    context_mask=context_mask,
                )
        
        h = self.norm(h)
        
        z_pred = self.output_head(h)
        eos_logits = self.eos_head(h).squeeze(-1)
        
        return {
            'z_pred': z_pred,
            'eos_logits': eos_logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        z_context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        eos_threshold: float = 0.5,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自回归生成
        
        Args:
            z_context: [B, N, context_dim]
            context_mask: [B, N]
            max_length: 最大生成长度
            eos_threshold: EOS 概率阈值
            temperature: 采样温度
        
        Returns:
            (z_sequence, eos_probs)
            - z_sequence: [B, M, latent_dim]
            - eos_probs: [B, M]
        """
        B = z_context.shape[0]
        device = z_context.device
        
        context_kvs = self.precompute_context_kv(z_context)
        past_kvs = [None] * self.num_layers
        
        h = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        
        generated_z = []
        eos_probs = []
        
        for step in range(max_length):
            h_out = h
            new_past_kvs = []
            
            for i, layer in enumerate(self.layers):
                h_out, present_kv = layer(
                    h_out,
                    context_kv=context_kvs[i],
                    context_mask=context_mask,
                    past_key_value=past_kvs[i],
                    use_cache=True,
                    position_offset=step,
                )
                new_past_kvs.append(present_kv)
            
            past_kvs = new_past_kvs
            h_out = self.norm(h_out)
            
            z_next = self.output_head(h_out[:, -1:])
            eos_prob = torch.sigmoid(self.eos_head(h_out[:, -1:]).squeeze(-1))
            
            if temperature > 0:
                z_next = z_next + torch.randn_like(z_next) * temperature * 0.1
            
            generated_z.append(z_next)
            eos_probs.append(eos_prob)
            
            if (eos_prob > eos_threshold).all():
                break
            
            h = self.input_proj(z_next)
        
        z_sequence = torch.cat(generated_z, dim=1)
        eos_probs = torch.cat(eos_probs, dim=1)
        
        return z_sequence, eos_probs