import numpy as np
import torch
import torch.nn.functional as F

from src.models.components.primitive_model import PrimitiveModel
from src.sketch_handlers.constants import ConstraintToken


class ConstraintARModel(torch.nn.Module):
    """Autoregressive generative model of constraints conditioned on primitives.

    Primitives are first embedded via a non-masked PrimitiveModel.
    Constraints are sampled autoregressively using a causal transformer decoder.
    """
    def __init__(self,
                 num_bins: int,
                 max_token_length: int,
                 embed_dim: int,
                 feedforward_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0):
        super().__init__()
        self.constraint_token_num = len(ConstraintToken)
        # Primitive model (for dynamic embeddings)
        self.prim_model = PrimitiveModel(num_bins, max_token_length, embed_dim, num_heads, num_layers, dropout)
        # Value embeddings (only fixed ones)
        num_val_embeddings = self.constraint_token_num
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim, padding_idx=ConstraintToken.Pad)
        # Position embeddings
        num_pos_embeddings = max_token_length
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim, padding_idx=ConstraintToken.Pad)
        # Transformer decoder
        decoder_layers = torch.nn.TransformerDecoderLayer(embed_dim, num_heads, feedforward_dim, dropout=dropout, batch_first=True)
        self.trans_decoder = torch.nn.TransformerDecoder(decoder_layers, num_layers)
        self.embed_dim = embed_dim

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
        batch_size, seq_len = src['c_val'].shape[:2]
        c_val_indices = src['c_val'].unsqueeze(-1).expand(-1, -1, self.embed_dim)
        val_embeddings = torch.gather(prim_embeddings, 1, c_val_indices)

        # Generate position indices internally (0, 1, 2, ..., seq_len-1)
        pos_indices = torch.arange(seq_len, device=src['c_val'].device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(pos_indices)
        embeddings = val_embeddings + pos_embeddings
        return embeddings, prim_embeddings

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
        # Pass to transformer with causal mask
        seq_len = c_embeddings.shape[1]
        tgt_mask = c_embeddings.new_full((seq_len, seq_len), fill_value=float('-inf')).triu_(diagonal=1)
        output = self.trans_decoder(c_embeddings, p_embeddings, tgt_mask)
        # Pointer dot-product
        p_embeddings = p_embeddings.transpose(1, 2)
        output = torch.matmul(output, p_embeddings)
        return output

    # ------------------------------------------------------------------ #
    #  KV-cache API for efficient autoregressive generation               #
    # ------------------------------------------------------------------ #

    def embed_token_at_pos(
        self,
        memory: torch.Tensor,
        token_ids: torch.Tensor,
        pos: int,
    ) -> torch.Tensor:
        """Embed a single batch of tokens at position *pos*.

        Parameters
        ----------
        memory   : [B, M, E]  — output of compute_memory()
        token_ids: [B]        — integer indices into memory
        pos      : int        — position in the sequence

        Returns
        -------
        [B, 1, E]
        """
        E = self.embed_dim
        idx = token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, E)  # [B, 1, E]
        val_emb = memory.gather(1, idx)                                # [B, 1, E]
        pos_idx = torch.full(
            (token_ids.shape[0], 1), pos,
            device=memory.device, dtype=torch.long,
        )
        pos_emb = self.pos_embed(pos_idx)                             # [B, 1, E]
        return val_emb + pos_emb

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
        for layer in self.trans_decoder.layers:
            ca = layer.multihead_attn
            H  = ca.num_heads
            d  = E // H
            w, b = ca.in_proj_weight, ca.in_proj_bias
            # K and V projections applied to memory (Q projection is for the tgt)
            k = F.linear(memory, w[E:2*E], b[E:2*E] if b is not None else None)  # [B, M, E]
            v = F.linear(memory, w[2*E:],  b[2*E:]  if b is not None else None)  # [B, M, E]
            # Reshape to [B, H, M, d] — format expected by SDPA
            k = k.view(B, M, H, d).transpose(1, 2)  # [B, H, M, d]
            v = v.view(B, M, H, d).transpose(1, 2)
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

        Pre-allocation avoids repeated torch.cat() calls (O(L²) memory copies)
        that would occur with a naive append-each-step approach.
        """
        layer0 = self.trans_decoder.layers[0]
        H = layer0.self_attn.num_heads
        d = self.embed_dim // H
        return [
            (
                torch.zeros(batch_size, H, max_len, d, device=device, dtype=dtype),
                torch.zeros(batch_size, H, max_len, d, device=device, dtype=dtype),
            )
            for _ in self.trans_decoder.layers
        ]

    def decoder_step_cached(
        self,
        tgt: torch.Tensor,
        pos: int,
        self_kv: list[tuple[torch.Tensor, torch.Tensor]],
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Run one AR step through all decoder layers with KV cache.

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

        for i, layer in enumerate(self.trans_decoder.layers):
            H = layer.self_attn.num_heads
            d = E // H

            # ── Self-attention (with pre-allocated KV cache) ─────────── #
            sa = layer.self_attn

            # Fused QKV projection: one GEMM for all three projections
            qkv    = F.linear(tgt, sa.in_proj_weight, sa.in_proj_bias)  # [B, 1, 3E]
            q, k_new, v_new = qkv.split(E, dim=-1)                      # each [B, 1, E]

            # Reshape to SDPA format
            q     = q.view(B, 1, H, d).transpose(1, 2)   # [B, H, 1, d]
            k_new = k_new.view(B, H, d)                   # [B, H, d]
            v_new = v_new.view(B, H, d)                   # [B, H, d]

            # Write new K, V into pre-allocated buffer at position *pos*
            k_buf, v_buf = self_kv[i]
            k_buf[:, :, pos, :] = k_new
            v_buf[:, :, pos, :] = v_new

            # Read the accumulated cache up to (and including) *pos*
            k_all = k_buf[:, :, :pos + 1, :]  # [B, H, pos+1, d]
            v_all = v_buf[:, :, :pos + 1, :]

            # SDPA — automatically uses Flash Attention on CUDA + float16/bfloat16
            attn_dp = sa.dropout if self.training else 0.0
            sa_out = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=attn_dp)
            sa_out = sa_out.transpose(1, 2).contiguous().view(B, 1, E)
            sa_out = F.linear(sa_out, sa.out_proj.weight, sa.out_proj.bias)

            tgt = layer.norm1(tgt + layer.dropout1(sa_out))  # dropout1: SA residual

            # ── Cross-attention (reuses precomputed KV from memory) ───── #
            ca = layer.multihead_attn
            cross_k, cross_v = cross_kv[i]   # [B, H, M, d]

            # Only Q is computed for the new token; K/V come from cache
            q_c = F.linear(
                tgt,
                ca.in_proj_weight[:E],
                ca.in_proj_bias[:E] if ca.in_proj_bias is not None else None,
            )
            q_c = q_c.view(B, 1, H, d).transpose(1, 2)  # [B, H, 1, d]

            ca_dp  = ca.dropout if self.training else 0.0
            ca_out = F.scaled_dot_product_attention(q_c, cross_k, cross_v, dropout_p=ca_dp)
            ca_out = ca_out.transpose(1, 2).contiguous().view(B, 1, E)
            ca_out = F.linear(ca_out, ca.out_proj.weight, ca.out_proj.bias)

            tgt = layer.norm2(tgt + layer.dropout2(ca_out))  # dropout2: CA residual

            # ── Feed-forward ──────────────────────────────────────────── #
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt))))
            tgt = layer.norm3(tgt + layer.dropout3(ff_out))  # dropout3: FFN residual

        return tgt  # [B, 1, E]