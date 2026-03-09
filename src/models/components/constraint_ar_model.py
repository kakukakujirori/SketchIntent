import numpy as np
import torch

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

    def _embed_tokens(self, src: dict[str, torch.Tensor]):
        # Embed primitives
        prim_embeddings = self.prim_model(src)
        # Prepend fixed val embeddings (constraint types)
        batch_size, seq_len = src['c_val'].shape[:2]
        fixed_tokens = np.tile(range(self.constraint_token_num), (batch_size, 1))
        fixed_tokens = torch.tensor(fixed_tokens).to(src['c_val'].device)
        fixed_val_embeddings = self.val_embed(fixed_tokens)
        prim_embeddings = torch.cat([fixed_val_embeddings, prim_embeddings], 1)

        # Gather embeddings for constraint tokens
        c_val_indices = src['c_val'].unsqueeze(-1).expand(-1, -1, self.embed_dim)
        val_embeddings = torch.gather(prim_embeddings, 1, c_val_indices)

        # Generate position indices internally (0, 1, 2, ..., seq_len-1)
        pos_indices = torch.arange(seq_len, device=src['c_val'].device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.pos_embed(pos_indices)
        embeddings = val_embeddings + pos_embeddings
        return embeddings, prim_embeddings

    def forward(self, src: dict[str, torch.Tensor]):
        # Embed
        c_embeddings, p_embeddings = self._embed_tokens(src)
        # Pass to transformer with causal mask
        seq_len = c_embeddings.shape[1]
        tgt_mask = c_embeddings.new_full((seq_len, seq_len), fill_value=float('-inf')).triu_(diagonal=1)
        output = self.trans_decoder(c_embeddings, p_embeddings, tgt_mask)
        # Pointer dot-product
        p_embeddings = p_embeddings.transpose(1, 2)
        output = torch.matmul(output, p_embeddings)
        return output