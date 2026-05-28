import torch
import torch.nn.functional as F

from src.models.components import dit
from src.models.components.primitive_model import PrimitiveModel
from src.sketch_handlers.constants import ConstraintToken


def _apply_rotary_to_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rotary_cos_sin
    cos = cos[:, :, 0].to(q.dtype)
    sin = sin[:, :, 0].to(q.dtype)
    return (
        q * cos + dit.rotate_half(q) * sin,
        k * cos + dit.rotate_half(k) * sin,
    )


class RoPECausalDecoderLayer(torch.nn.TransformerDecoderLayer):
    """TransformerDecoderLayer with RoPE applied only to causal self-attention."""

    def _rope_self_attn(
        self,
        tgt: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, L, E = tgt.shape
        H = self.self_attn.num_heads
        d = E // H

        qkv = F.linear(tgt, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        qkv = qkv.view(B, L, 3, H, d)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        with torch.amp.autocast('cuda', enabled=False):
            q, k = _apply_rotary_to_qk(q, k, rotary_cos_sin)

        q = q.transpose(1, 2)  # [B, H, L, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.self_attn.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        return F.linear(
            attn_out,
            self.self_attn.out_proj.weight,
            self.self_attn.out_proj.bias,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = tgt
        x = self.norm1(x + self.dropout1(self._rope_self_attn(x, rotary_cos_sin)))
        x = self.norm2(x + self._mha_block(x, memory, None, None, False))
        x = self.norm3(x + self._ff_block(x))
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
        if embed_dim % num_heads != 0 or (embed_dim // num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.constraint_token_num = len(ConstraintToken)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Primitive model (for dynamic embeddings)
        self.prim_model = PrimitiveModel(num_bins, embed_dim, num_heads, num_layers, dropout)
        # Value embeddings (only fixed ones)
        num_val_embeddings = self.constraint_token_num
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim, padding_idx=ConstraintToken.Pad)
        # RoPE replaces nn.Embedding-based absolute position encoding.
        self.rotary_emb = dit.Rotary(self.head_dim)
        # Transformer decoder
        self.layers = torch.nn.ModuleList([
            RoPECausalDecoderLayer(
                embed_dim,
                num_heads,
                feedforward_dim,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

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
        fixed_tokens = torch.arange(self.constraint_token_num, device=prim_embeddings.device)
        fixed_tokens = fixed_tokens.unsqueeze(0).expand(batch_size, -1)
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
        # Pass to transformer with causal self-attention.
        rotary_cos_sin = self.rotary_emb(c_embeddings)
        output = c_embeddings
        for layer in self.layers:
            output = layer(output, p_embeddings, rotary_cos_sin)
        # Pointer dot-product
        p_embeddings = p_embeddings.transpose(1, 2)
        output = torch.matmul(output, p_embeddings)
        return output

    # ------------------------------------------------------------------ #
    #  KV-cache API for efficient autoregressive generation               #
    # ------------------------------------------------------------------ #

    def embed_token(
        self,
        memory: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed a single batch of tokens.

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
        return memory.gather(1, idx)                                    # [B, 1, E]

    def _rotary_at_pos(
        self,
        pos: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch RoPE cos/sin for a single cached decoding position."""
        if isinstance(pos, torch.Tensor):
            pos_idx = pos.reshape(1).to(
                device=self.rotary_emb.cos_cached.device,
                dtype=torch.long,
            )
            cos = self.rotary_emb.cos_cached.index_select(1, pos_idx)
            sin = self.rotary_emb.sin_cached.index_select(1, pos_idx)
        else:
            cos = self.rotary_emb.cos_cached[:, pos:pos + 1]
            sin = self.rotary_emb.sin_cached[:, pos:pos + 1]
        return cos, sin

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
        for layer in self.layers:
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
        layer0 = self.layers[0]
        H = layer0.self_attn.num_heads
        d = self.embed_dim // H
        return [
            (
                torch.zeros(batch_size, H, max_len, d, device=device, dtype=dtype),
                torch.zeros(batch_size, H, max_len, d, device=device, dtype=dtype),
            )
            for _ in self.layers
        ]

    def decoder_step_cached(
        self,
        tgt: torch.Tensor,
        pos: int | torch.Tensor,
        self_kv: list[tuple[torch.Tensor, torch.Tensor]],
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]],
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one AR step through all decoder layers with KV cache.

        self_kv buffers are **mutated in-place** (k_buf[:, :, pos, :] is written).

        Parameters
        ----------
        tgt      : [B, 1, E]  — embedding of the new token at position *pos*
        pos      : int | Tensor — current sequence position (write index).
                   Pass a scalar tensor for torch.compile compatibility.
        self_kv  : list of (k_buf, v_buf) per layer, each [B, H, max_len, d_head]
        cross_kv : list of (k, v) per layer from compute_cross_kv_cache
        causal_mask  : [1, 1, 1, max_len] attention mask (0 = attend, -inf = ignore).
                   If None, falls back to dynamic slicing (eager-mode only).

        Returns
        -------
        [B, 1, E]  — decoder output for this position
        """
        B, _, E = tgt.shape

        for i, layer in enumerate(self.layers):
            H = layer.self_attn.num_heads
            d = E // H

            # ── Self-attention (with pre-allocated KV cache) ─────────── #
            sa = layer.self_attn

            # Fused QKV projection: one GEMM for all three projections
            qkv = F.linear(tgt, sa.in_proj_weight, sa.in_proj_bias)  # [B, 1, 3E]
            qkv = qkv.view(B, 1, 3, H, d)
            q = qkv[:, :, 0]
            k_new = qkv[:, :, 1]
            v_new = qkv[:, :, 2]

            with torch.amp.autocast('cuda', enabled=False):
                cos, sin = self._rotary_at_pos(pos)
                q, k_new = _apply_rotary_to_qk(q, k_new, (cos, sin))

            # Reshape to SDPA format
            q     = q.transpose(1, 2)  # [B, H, 1, d]
            k_new = k_new.transpose(1, 2)
            v_new = v_new.transpose(1, 2)

            # Write new K, V into pre-allocated buffer at position *pos*
            # When pos is a scalar tensor, dynamo traces it symbolically
            # (no per-value int guard → single compiled graph for all steps).
            k_buf, v_buf = self_kv[i]
            if isinstance(pos, torch.Tensor):
                pos_idx = pos.reshape(1)
                k_buf.index_copy_(2, pos_idx, k_new)
                v_buf.index_copy_(2, pos_idx, v_new)
            else:
                k_buf[:, :, pos:pos + 1, :] = k_new
                v_buf[:, :, pos:pos + 1, :] = v_new

            # SDPA — automatically uses Flash Attention on CUDA + float16/bfloat16
            attn_dp = sa.dropout if self.training else 0.0
            if causal_mask is not None:
                # Fixed-shape path: full buffer + mask → static tensor shapes
                sa_out = F.scaled_dot_product_attention(
                    q, k_buf, v_buf, attn_mask=causal_mask, dropout_p=attn_dp,
                )
            else:
                # Dynamic-slice path: eager-mode fallback
                p = pos.item() if isinstance(pos, torch.Tensor) else pos
                k_all = k_buf[:, :, :p + 1, :]
                v_all = v_buf[:, :, :p + 1, :]
                sa_out = F.scaled_dot_product_attention(
                    q, k_all, v_all, dropout_p=attn_dp,
                )
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

    # ------------------------------------------------------------------ #
    #  Compiled AR step — single call combining embed + decode + logits   #
    # ------------------------------------------------------------------ #

    def generate_one_step(
        self,
        memory: torch.Tensor,
        token_ids: torch.Tensor,
        pos: torch.Tensor,
        self_kv: list[tuple[torch.Tensor, torch.Tensor]],
        cross_kv: list[tuple[torch.Tensor, torch.Tensor]],
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One AR step: embed → decode → pointer logits.

        Designed to be wrapped by torch.compile as a single compilation unit.
        All tensor shapes are static (KV cache is full-length, masked by causal_mask).
        ``pos`` is a scalar **tensor** (not int) to avoid per-step recompilation.

        Parameters
        ----------
        memory    : [B, M, E]  — output of compute_memory()
        token_ids : [B]        — token indices into memory
        pos       : Tensor     — scalar tensor, current position in the sequence
        self_kv   : KV cache (mutated in-place)
        cross_kv  : precomputed cross-attention KV
        causal_mask   : [1, 1, 1, max_len]  — causal mask for this step

        Returns
        -------
        next_token_logits : [B, M]  — unnormalised logits over memory tokens
        """
        new_emb = self.embed_token(memory, token_ids)
        dec_out = self.decoder_step_cached(new_emb, pos, self_kv, cross_kv, causal_mask)
        # Pointer dot-product: [B, 1, E] @ [B, E, M] → [B, M]
        logits = torch.matmul(dec_out, memory.transpose(1, 2)).squeeze(1)
        return logits

    @staticmethod
    def build_causal_masks(
        max_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Pre-build causal attention masks for all AR positions.

        Returns
        -------
        masks : [max_len, 1, 1, max_len]
            ``masks[pos]`` is the mask for step *pos*: 0 at positions 0..pos,
            -inf at positions pos+1..max_len-1.  Shape of each slice is
            [1, 1, max_len] — broadcast-friendly with SDPA's
            ``[B, H, 1, max_len]``.
        """
        return torch.triu(
            torch.full((max_len, max_len), float('-inf'), device=device, dtype=dtype),
            diagonal=1,
        ).unsqueeze(1).unsqueeze(1)  # [max_len, 1, 1, max_len]
