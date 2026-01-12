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
        mask_start_pos: int = 1,
        prefix_len: int = 1,
        strategy: str = "tokenwise",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sanity check
        if len(tokens.shape) != 2:
            raise ValueError(f"tokens shape must be [B, L], but got {tokens.shape}")
        if mask_start_pos < 0:
            raise ValueError(f"mask_start_pos must be non-negative, but got {mask_start_pos}")
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, but got {prefix_len}")

        if not isinstance(mask_prob, torch.Tensor):
            mask_prob = torch.tensor([mask_prob])

        # factory
        if strategy == "tokenwise":
            return self.tokenwise_masking(tokens, mask_prob, mask_start_pos, prefix_len)
        elif re.fullmatch(r"^\d+-tokenwise-all$", strategy):  # e.g., 4-tokenwise-all
            block_len = int(strategy.split("-")[0])
            return self.blockwise_masking_all(tokens, mask_prob, mask_start_pos, block_len, prefix_len)
        elif re.fullmatch(r"^\d+-tokenwise-causal$", strategy):  # e.g., 4-tokenwise-causal
            block_len = int(strategy.split("-")[0])
            return self.blockwise_masking_causal(tokens, mask_prob, mask_start_pos, block_len, prefix_len)
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

    def tokenwise_masking(
        self,
        tokens: torch.Tensor,
        mask_prob: torch.Tensor,
        mask_start_pos: int,
        prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(tokens.shape) == 2
        assert mask_start_pos >= 0
        assert prefix_len >= 0
        mask = torch.rand_like(tokens, dtype=torch.float32) < mask_prob.reshape(-1, 1)
        mask[:, :mask_start_pos] = False
        tokens_masked = torch.where(mask, self.mask_id, tokens)
        return tokens_masked, mask

    def blockwise_masking_all(
        self,
        tokens: torch.Tensor,
        mask_prob: torch.Tensor,
        mask_start_pos: int,
        block_len: int,
        prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(tokens.shape) == 2
        assert mask_start_pos >= 0
        assert block_len > 0
        assert prefix_len >= 0
        # reshape
        _, token_num = tokens.shape
        block_num = (token_num - prefix_len) // block_len
        token_mid_block = rearrange(
            tokens[:, prefix_len : prefix_len + block_num * block_len],
            "b (c_num c_tokens) -> b c_num c_tokens",
            c_tokens=block_len)
        # masking
        b, c_num, c_tokens = token_mid_block.shape
        mask = torch.rand((b, c_num, 1), device=mask_prob.device) < mask_prob.reshape(-1, 1, 1)
        token_mid_block_masked = torch.where(mask, self.mask_id, token_mid_block)
        # flatten
        tokens_masked = tokens.clone()
        tokens_masked[:, prefix_len : prefix_len + block_num * block_len] = token_mid_block_masked.reshape(b, -1)

        # care about prefix masking
        if mask_start_pos < prefix_len:
            # prefix masking
            prefix_mask = torch.rand((b, prefix_len - mask_start_pos), device=tokens.device) < mask_prob.reshape(-1, 1)
            tokens_masked[:, mask_start_pos:prefix_len] = torch.where(prefix_mask, self.mask_id, tokens[:, mask_start_pos:prefix_len])
        else:
            # unmask before mask_start_pos
            tokens_masked[:, :mask_start_pos] = tokens[:, :mask_start_pos]

        mask = tokens_masked == self.mask_id
        return tokens_masked, mask

    def blockwise_masking_causal(
        self,
        tokens: torch.Tensor,
        mask_prob: torch.Tensor,
        mask_start_pos: int,
        block_len: int,
        prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(tokens.shape) == 2
        assert mask_start_pos >= 0
        assert block_len > 0
        assert prefix_len >= 0

        # ensure that tensors are on the same device
        mask_prob = mask_prob.to(tokens.device)

        # reshape
        _, token_num = tokens.shape
        block_num = (token_num - prefix_len) // block_len
        token_mid_block = rearrange(
            tokens[:, prefix_len : prefix_len + block_num * block_len],
            "b (c_num c_tokens) -> b c_num c_tokens",
            c_tokens=block_len)

        b, c_num, _ = token_mid_block.shape

        # ------------------------------------------------------------------
        # Linear Shifted Uniform Strategy
        # 幅5.0の一様分布を、pに応じて [-4, 1] から [4, 9] までスライドさせる手法
        # ------------------------------------------------------------------

        # 1. Uniform Noise [0, 1)
        u = torch.rand((b, c_num), device=tokens.device)

        # 2. Mask Prob expansion
        p = mask_prob.reshape(-1, 1).expand(b, c_num)

        # 3. Continuous Count Calculation
        # Width = 5.0 (to cover 0,1,2,3,4 uniformly)
        # Shift = 8p - 4
        #   p=0   -> Range [-4, 1) -> floor -> {-4..0} -> clamp -> {0} (All 0)
        #   p=0.5 -> Range [0, 5)  -> floor -> {0..4}  -> clamp -> {0..4} (Uniform)
        #   p=1   -> Range [4, 9)  -> floor -> {4..8}  -> clamp -> {4} (All 4)

        width = float(block_len) + 1.0  # 5.0
        shift = (2.0 * width - 2.0) * p - (width - 1.0) # 8p - 4

        continuous_count = u * width + shift

        # 4. Discretization
        num_masked = torch.floor(continuous_count).clamp(0, block_len)

        # ------------------------------------------------------------------

        # Generate Mask (Right-aligned)
        mask_start_idx = (block_len - num_masked).reshape(b, c_num, 1)

        indices = torch.arange(block_len, device=tokens.device).expand_as(token_mid_block)
        token_mid_block_masked = torch.where(mask_start_idx <= indices, self.mask_id, token_mid_block)

        # flatten
        tokens_masked = tokens.clone()
        tokens_masked[:, prefix_len : prefix_len + block_num * block_len] = token_mid_block_masked.reshape(b, -1)

        # care about prefix masking
        if mask_start_pos < prefix_len:
            # prefix masking
            prefix_mask = torch.rand((b, prefix_len - mask_start_pos), device=tokens.device) < mask_prob.reshape(-1, 1)
            tokens_masked[:, mask_start_pos:prefix_len] = torch.where(prefix_mask, self.mask_id, tokens[:, mask_start_pos:prefix_len])
        else:
            # unmask before mask_start_pos
            tokens_masked[:, :mask_start_pos] = tokens[:, :mask_start_pos]

        mask = tokens_masked == self.mask_id

        return tokens_masked, mask