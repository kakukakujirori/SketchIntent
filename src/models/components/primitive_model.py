import torch

from src.models.components import dit
from src.sketch_handlers.constants import PrimitiveToken, PrimitiveAttrToken, PRIMITIVE_VAL_OFFSET


class PrimitiveModel(torch.nn.Module):
    def __init__(
        self,
        num_bins: int,
        max_token_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0,
    ):
        super().__init__()

        # Embeddings
        # Value embeddings
        num_val_embeddings = PRIMITIVE_VAL_OFFSET + num_bins + 2  # 2 -> isConstruction
        self.val_embed = torch.nn.Embedding(num_val_embeddings, embed_dim, padding_idx=PrimitiveToken.Pad)

        # Coordinate embeddings (use PrimitiveParamToken size for semantic param types)
        num_attr_embeddings = len(PrimitiveAttrToken)
        self.attr_embed = torch.nn.Embedding(num_attr_embeddings, embed_dim, padding_idx=PrimitiveAttrToken.Pad)

        # Position embeddings
        # NOTE: num_pos_embedding is redundantly large, but it's fine
        num_pos_embeddings = max_token_length // 2
        self.pos_embed = torch.nn.Embedding(num_pos_embeddings, embed_dim, padding_idx=PrimitiveToken.Pad)

        # Rotary
        self.rotary_emb = dit.Rotary(embed_dim // num_heads)

        # Transformer Blocks
        self.blocks = torch.nn.ModuleList([
            dit.DDiTPlainBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])

        # Final Norm
        self.norm = dit.LayerNorm(embed_dim)

    def forward(self, src: dict[str, torch.Tensor]) -> torch.Tensor:
        # Embed
        val_embeddings = self.val_embed(src['p_val'])
        attr_embeddings = self.attr_embed(src['p_attr'])
        pos_embeddings = self.pos_embed(src['p_pos'])
        x = val_embeddings + attr_embeddings + pos_embeddings

        # Rotary
        rotary_cos_sin = self.rotary_emb(x)

        # Blocks
        # Using bfloat16 for blocks as per DiT design
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin)
            x = self.norm(x)

        return x