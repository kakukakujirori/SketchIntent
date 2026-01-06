import re

import torch
from einops import rearrange


class TokenMasking:
    def __init__(self, mask_id: int):
        self.mask_id = mask_id

    def __call__(
        self,
        tokens: torch.Tensor,
        mask_prob: float | torch.Tensor,
        token_prefix_len: int = 0,
        strategy: str = "tokenwise",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        if len(tokens.shape) != 2:
            raise ValueError(f"tokens shape must be [B, L], but got {tokens.shape}")

        if not isinstance(mask_prob, torch.Tensor):
            mask_prob = torch.tensor([mask_prob])

        # factory
        if strategy == "tokenwise":
            return self.tokenwise_masking(tokens, mask_prob, token_prefix_len)
        elif re.fullmatch(r"^\d+-tokenwise$", strategy):  # e.g., 4-tokenwise
            block_len = int(strategy.split("-")[0])
            return self.blockwise_masking(tokens, mask_prob, block_len, token_prefix_len)
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

    def tokenwise_masking(
        self,
        tokens: torch.Tensor,
        mask_prob: torch.Tensor,
        token_prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(tokens.shape) == 2
        mask = torch.rand_like(tokens, dtype=torch.float32) < mask_prob.reshape(-1, 1)
        tokens_masked = torch.where(mask, self.mask_id, tokens)
        mask[:, :token_prefix_len] = False
        return tokens_masked, mask

    def blockwise_masking(
        self,
        tokens: torch.Tensor,
        mask_prob: torch.Tensor,
        block_len: int,
        token_prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(tokens.shape) == 2
        # reshape
        _, token_num = tokens.shape
        block_num = (token_num - token_prefix_len) // block_len
        token_mid_block = rearrange(
            tokens[:, token_prefix_len : token_prefix_len + block_num * block_len],
            "b (c_num c_tokens) -> b c_num c_tokens",
            c_tokens=block_len)
        # masking
        b, c_num, c_tokens = token_mid_block.shape
        mask = torch.rand((b, c_num, 1), device=mask_prob.device) < mask_prob.reshape(-1, 1, 1)
        token_mid_block_masked = torch.where(mask, self.mask_id, token_mid_block)
        # flatten
        tokens_masked = tokens.clone()
        tokens_masked[:, token_prefix_len : token_prefix_len + block_num * block_len] = token_mid_block_masked.reshape(b, -1)
        mask = tokens_masked == self.mask_id
        return tokens_masked, mask
