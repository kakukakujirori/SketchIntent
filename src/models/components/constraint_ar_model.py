import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.components.primitive_model import PrimitiveModel
from src.models.components import dit
from src.sketch_handlers.constants import ConstraintToken


class CausalCrossBlock(nn.Module):
    """Transformer block with causal self-attention (RoPE) + cross-attention.

    Architecture per block:
        1. Pre-norm → Self-attention with RoPE + causal mask → Residual
        2. Pre-norm → Cross-attention (Q from target, K/V from memory) → Residual
        3. Pre-norm → FFN → Residual
    """
    def __init__(self, dim, n_heads, feedforward_dim=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        # Self-attention
        self.norm1 = dit.LayerNorm(dim)
        self.self_attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_attn_out = nn.Linear(dim, dim, bias=False)

        # Cross-attention
        self.norm_cross = dit.LayerNorm(dim)
        self.cross_attn_q = nn.Linear(dim, dim, bias=False)
        self.cross_attn_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_attn_out = nn.Linear(dim, dim, bias=False)

        # FFN
        ff_dim = feedforward_dim or dim * 4
        self.norm2 = dit.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_dim, dim, bias=True))

    def forward(self, x, memory, rotary_cos_sin):
        batch_size, seq_len = x.shape[:2]

        # 1. Self-attention with RoPE + causal mask
        x_skip = x
        x = self.norm1(x)
        qkv = self.self_attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d',
                         three=3, h=self.n_heads)

        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = dit.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, S, D]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = self.self_attn_out(x)
        x = F.dropout(x, p=self.dropout, training=self.training) + x_skip

        # 2. Cross-attention (Q from target, K/V from memory)
        x_skip = x
        x_normed = self.norm_cross(x)
        q = self.cross_attn_q(x_normed)
        kv = self.cross_attn_kv(memory)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.n_heads)
        kv = rearrange(kv, 'b s (two h d) -> b s two h d',
                       two=2, h=self.n_heads)
        k_c = kv[:, :, 0].transpose(1, 2)
        v_c = kv[:, :, 1].transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k_c, v_c,
            dropout_p=self.dropout if self.training else 0.)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = self.cross_attn_out(x)
        x = F.dropout(x, p=self.dropout, training=self.training) + x_skip

        # 3. FFN
        x_skip = x
        x = self.mlp(self.norm2(x))
        x = F.dropout(x, p=self.dropout, training=self.training) + x_skip

        return x


class ConstraintARModel(torch.nn.Module):
    """Autoregressive generative model of constraints conditioned on primitives.

    Primitives are first embedded via a non-masked PrimitiveModel.
    Constraints are sampled autoregressively using a causal transformer decoder.
    """
    def __init__(self,
                 num_bins: int,
                 embed_dim: int,
                 feedforward_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0):
        super().__init__()
        self.constraint_token_num = len(ConstraintToken)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Primitive model (for dynamic embeddings)
        self.prim_model = PrimitiveModel(num_bins, embed_dim, num_heads, num_layers, dropout)

        # Value embeddings (only fixed ones)
        num_val_embeddings = self.constraint_token_num
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim, padding_idx=ConstraintToken.Pad)

        # Rotary Position Embedding (replaces nn.Embedding-based absolute pos encoding)
        self.rotary_emb = dit.Rotary(self.head_dim)

        # Causal decoder blocks with cross-attention
        self.blocks = nn.ModuleList([
            CausalCrossBlock(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm (pre-norm transformers require a trailing norm before
        # the output projection; without it the residual stream is unbounded).
        self.final_norm = dit.LayerNorm(embed_dim)

    def compute_memory(self, src: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the decoder memory (primitive + special-token embeddings).

        This is the part of the forward pass that depends only on the primitives,
        NOT on the constraint tokens.  Call once and pass to forward(memory=...)
        to avoid redundant prim_model calls during autoregressive generation.

        Returns
        -------
        memory : torch.Tensor  [B, N_special + L_prim, embed_dim]
        """
        prim_embeddings = self.prim_model(src)  # [B, L_prim, E]
        batch_size = prim_embeddings.shape[0]
        fixed_tokens = np.tile(range(self.constraint_token_num), (batch_size, 1))
        fixed_tokens = torch.tensor(fixed_tokens, device=prim_embeddings.device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)  # [B, N_special, E]
        return torch.cat([fixed_val_embeddings, prim_embeddings], dim=1)  # [B, N_special+L_prim, E]

    def _embed_tokens(self, src: dict[str, torch.Tensor], memory: torch.Tensor | None = None):
        # Compute (or reuse) primitive + special-token embeddings
        if memory is None:
            prim_embeddings = self.compute_memory(src)
        else:
            prim_embeddings = memory  # reuse precomputed memory

        # Gather embeddings for constraint tokens
        c_val_indices = src['c_val'].unsqueeze(-1).expand(-1, -1, self.embed_dim)
        val_embeddings = torch.gather(prim_embeddings, 1, c_val_indices)

        # No absolute position embeddings — RoPE is applied in attention layers
        return val_embeddings, prim_embeddings

    def forward(self, src: dict[str, torch.Tensor], memory: torch.Tensor | None = None):
        """Forward pass.

        Parameters
        ----------
        src : dict
            Must contain 'c_val' (and primitive keys when memory is None).
        memory : torch.Tensor | None
            Precomputed decoder memory [B, N_special+L_prim, E] from
            compute_memory().  Pass to skip redundant prim_model re-computation
            during autoregressive generation.
        """
        # Embed
        c_embeddings, p_embeddings = self._embed_tokens(src, memory=memory)

        # Compute rotary embeddings
        rotary_cos_sin = self.rotary_emb(c_embeddings)

        # Run through causal decoder blocks with cross-attention to primitives
        x = c_embeddings
        for block in self.blocks:
            x = block(x, p_embeddings, rotary_cos_sin)
        x = self.final_norm(x)

        # Pointer dot-product
        p_embeddings = p_embeddings.transpose(1, 2)
        output = torch.matmul(x, p_embeddings)
        return output

    # ------------------------------------------------------------------ #
    #  KV-cache API for efficient autoregressive generation               #
    # ------------------------------------------------------------------ #

    def _rotary_at_pos(self, pos: int, device, dtype):
        """Compute RoPE cos/sin for a single position.

        Returns cos, sin each of shape [1, head_dim].
        """
        inv_freq = self.rotary_emb.inv_freq
        t = torch.tensor([pos], device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [1, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [1, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def embed_token_at_pos(
        self,
        memory: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a single batch of tokens (no position encoding needed — RoPE
        is applied inside the decoder blocks).

        Parameters
        ----------
        memory   : [B, M, E]  — output of compute_memory()
        token_ids: [B]        — integer indices into memory

        Returns
        -------
        [B, 1, E]
        """
        E = self.embed_dim
        idx = token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, E)  # [B, 1, E]
        return memory.gather(1, idx)                                   # [B, 1, E]

    def compute_cross_kv_cache(
        self,
        memory: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Project memory into K, V for every cross-attention layer.

        Call **once** before the AR loop.  The same cache is reused at every
        generation step (memory does not change across steps).

        Parameters
        ----------
        memory : [B, M, E]

        Returns
        -------
        list of (k, v) per layer, each shaped [B, H, M, d_head]
        """
        B, M, E = memory.shape
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            kv = block.cross_attn_kv(memory)              # [B, M, 2*E]
            kv = kv.view(B, M, 2, self.num_heads, self.head_dim)
            kv = kv.permute(0, 3, 1, 2, 4)                # [B, H, M, 2, d]
            k = kv[:, :, :, 0, :].contiguous()             # [B, H, M, d]
            v = kv[:, :, :, 1, :].contiguous()
            caches.append((k, v))
        return caches

    def init_kv_cache(
        self,
        batch_size: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-allocate self-attention KV cache buffers for all layers.

        Returns a list of (k_buf, v_buf) per layer, each [B, H, max_len, d_head].
        Buffers are written in-place inside decoder_step_cached().
        """
        H = self.num_heads
        d = self.head_dim
        return [
            (
                torch.zeros(batch_size, H, max_len, d, device=device, dtype=dtype),
                torch.zeros(batch_size, H, max_len, d, device=device, dtype=dtype),
            )
            for _ in self.blocks
        ]

    def decoder_step_cached(
        self,
        tgt: torch.Tensor,
        pos: int,
        self_kv: list[tuple[torch.Tensor, torch.Tensor]],
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Run one AR step through all decoder blocks with KV cache.

        self_kv buffers are **mutated in-place** (k_buf[:, :, pos, :] is written).

        Parameters
        ----------
        tgt      : [B, 1, E]  — embedding of the new token at position *pos*
        pos      : int        — current sequence position (write index)
        self_kv  : list of (k_buf, v_buf) per layer, each [B, H, max_len, d_head]
        cross_kv : list of (k, v) per layer from compute_cross_kv_cache

        Returns
        -------
        [B, 1, E]  — decoder output for this position
        """
        B, _, E = tgt.shape
        H = self.num_heads
        d = self.head_dim

        # Compute RoPE cos/sin for this position
        cos, sin = self._rotary_at_pos(pos, tgt.device, tgt.dtype)
        # cos, sin: [1, head_dim]

        for i, block in enumerate(self.blocks):
            # ── Self-attention (with pre-allocated KV cache + RoPE) ──── #
            normed = block.norm1(tgt)
            qkv = block.self_attn_qkv(normed)          # [B, 1, 3E]
            q, k_new, v_new = qkv.split(E, dim=-1)     # each [B, 1, E]

            # Reshape to [B, H, 1, d]
            q     = q.view(B, 1, H, d).transpose(1, 2)
            k_new = k_new.view(B, 1, H, d).transpose(1, 2)
            v_new = v_new.view(B, 1, H, d).transpose(1, 2)

            # Apply RoPE to Q and K  (cos/sin: [1, D] -> [1, 1, 1, D])
            cos_exp = cos.unsqueeze(0).unsqueeze(0)     # [1, 1, 1, D]
            sin_exp = sin.unsqueeze(0).unsqueeze(0)
            q     = q * cos_exp + dit.rotate_half(q) * sin_exp
            k_new = k_new * cos_exp + dit.rotate_half(k_new) * sin_exp

            # Write new K, V into pre-allocated buffer at position *pos*
            k_buf, v_buf = self_kv[i]
            k_buf[:, :, pos, :] = k_new.squeeze(2)
            v_buf[:, :, pos, :] = v_new.squeeze(2)

            # Read the accumulated cache up to (and including) *pos*
            k_all = k_buf[:, :, :pos + 1, :]           # [B, H, pos+1, d]
            v_all = v_buf[:, :, :pos + 1, :]

            attn_dp = block.dropout if self.training else 0.0
            sa_out = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=attn_dp)
            sa_out = sa_out.transpose(1, 2).contiguous().view(B, 1, E)
            sa_out = block.self_attn_out(sa_out)

            tgt = F.dropout(sa_out, p=block.dropout, training=self.training) + tgt

            # ── Cross-attention (reuses precomputed KV from memory) ──── #
            cross_k, cross_v = cross_kv[i]   # [B, H, M, d]

            normed = block.norm_cross(tgt)
            q_c = block.cross_attn_q(normed)
            q_c = q_c.view(B, 1, H, d).transpose(1, 2)  # [B, H, 1, d]

            ca_dp = block.dropout if self.training else 0.0
            ca_out = F.scaled_dot_product_attention(q_c, cross_k, cross_v, dropout_p=ca_dp)
            ca_out = ca_out.transpose(1, 2).contiguous().view(B, 1, E)
            ca_out = block.cross_attn_out(ca_out)

            tgt = F.dropout(ca_out, p=block.dropout, training=self.training) + tgt

            # ── Feed-forward ─────────────────────────────────────────── #
            ff_out = block.mlp(block.norm2(tgt))
            tgt = F.dropout(ff_out, p=block.dropout, training=self.training) + tgt

        tgt = self.final_norm(tgt)
        return tgt  # [B, 1, E]