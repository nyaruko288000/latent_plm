import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from functools import partial
from torch.utils.checkpoint import checkpoint

from .components import RMSNorm, Attention, CrossAttention, SwiGLU


class DecoderBlock(nn.Module):
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
            is_causal=False,
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
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x)
        x = residual + self.dropout(x)
        
        if context is not None or context_kv is not None:
            residual = x
            x = self.cross_attn_norm(x)
            x = self.cross_attn(x, context=context, context_kv=context_kv, context_mask=context_mask)
            x = residual + self.dropout(x)
        
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        
        return x


def _checkpoint_forward(layer, x, context_kv, context_mask):
    return layer(x, context=None, context_kv=context_kv, context_mask=context_mask)


class IterativeNARDecoder(nn.Module):
    """
    Iterative NAR Decoder
    
    支持共享 token embedding
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        latent_dim: int = 256,
        hidden_dim: int = 768,
        context_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_dim: Optional[int] = None,
        chunk_size: int = 8,
        num_iterations: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        # 共享 embedding
        shared_token_embed: Optional[nn.Embedding] = None,
        shared_output_proj: Optional[nn.Linear] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.num_iterations = num_iterations
        self.use_checkpoint = use_checkpoint
        
        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 8 / 3)
            ffn_dim = ((ffn_dim + 63) // 64) * 64
        
        # 上采样
        self.z_upsample = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * hidden_dim),
        )
        
        # Token embedding（共享）
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
        
        # 迭代编码
        self.iter_embed = nn.Embedding(num_iterations, hidden_dim)
        self.register_buffer('iter_indices', torch.arange(num_iterations))
        
        # Transformer
        self.layers = nn.ModuleList([
            DecoderBlock(hidden_dim, num_heads, ffn_dim, context_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
        
        # 输出（共享）
        if shared_output_proj is not None:
            self.output_proj = shared_output_proj
            out_dim = shared_output_proj.in_features
            if out_dim != hidden_dim:
                self.output_hidden_proj = nn.Linear(hidden_dim, out_dim, bias=False)
            else:
                self.output_hidden_proj = nn.Identity()
            self._shared_output = True
        else:
            self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.output_hidden_proj = nn.Identity()
            self._shared_output = False
        
        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if not (self._shared_embed and module is self.token_embed):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def precompute_context_kv(self, context: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        context_kvs = []
        for layer in self.layers:
            kv = layer.cross_attn.compute_kv(context)
            context_kvs.append(kv)
        return context_kvs
    
    def forward(
        self,
        z_plan: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        context_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_iterations: Optional[int] = None,
        return_all_iterations: bool = False,
        early_stop: bool = True,
    ) -> torch.Tensor:
        if num_iterations is None:
            num_iterations = self.num_iterations
        
        B, M, _ = z_plan.shape
        T = M * self.chunk_size
        
        if context_kvs is None and z_context is not None:
            context_kvs = self.precompute_context_kv(z_context)
        
        h = self.z_upsample(z_plan)
        h = h.view(B, T, self.hidden_dim)
        z_representation = h.clone()
        
        all_logits = []
        prev_tokens = None
        
        for it in range(num_iterations):
            iter_emb = self.iter_embed(self.iter_indices[it])
            h_input = h + iter_emb
            
            for i, layer in enumerate(self.layers):
                ctx_kv = context_kvs[i] if context_kvs is not None else None
                
                if self.use_checkpoint and self.training:
                    h_input = checkpoint(_checkpoint_forward, layer, h_input, ctx_kv, context_mask, use_reentrant=False)
                else:
                    h_input = layer(h_input, context_kv=ctx_kv, context_mask=context_mask)
            
            h_output = self.norm(h_input)
            h_for_output = self.output_hidden_proj(h_output)
            logits = self.output_proj(h_for_output)
            
            if return_all_iterations:
                all_logits.append(logits)
            
            if early_stop and not self.training and it < num_iterations - 1:
                current_tokens = logits.argmax(-1)
                if prev_tokens is not None and (current_tokens == prev_tokens).all():
                    break
                prev_tokens = current_tokens
            
            if it < num_iterations - 1:
                tokens = logits.argmax(-1)
                token_emb = self.token_embed(tokens)
                token_emb = self.embed_proj(token_emb)
                
                gate = self.fusion_gate(torch.cat([z_representation, token_emb], dim=-1))
                h = gate * z_representation + (1 - gate) * token_emb
        
        if return_all_iterations:
            return torch.stack(all_logits, dim=0)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        z_plan: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        context_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_iterations: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        logits = self.forward(z_plan, z_context, context_mask=context_mask, context_kvs=context_kvs, num_iterations=num_iterations, early_stop=True)
        
        if temperature <= 0:
            return logits.argmax(-1)
        
        logits = logits / temperature
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float('inf')
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.shape[:-1])
        
        return tokens