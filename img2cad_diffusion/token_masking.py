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
        elif re.fullmatch(r"^\d+-tokenwise-all$", strategy):  # e.g., 4-tokenwise-all
            block_len = int(strategy.split("-")[0])
            return self.blockwise_masking_all(tokens, mask_prob, block_len, token_prefix_len)
        elif re.fullmatch(r"^\d+-tokenwise-causal$", strategy):  # e.g., 4-tokenwise-causal
            block_len = int(strategy.split("-")[0])
            return self.blockwise_masking_causal(tokens, mask_prob, block_len, token_prefix_len)
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
        mask[:, :token_prefix_len] = False
        tokens_masked = torch.where(mask, self.mask_id, tokens)
        return tokens_masked, mask

    def blockwise_masking_all(
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

    def blockwise_masking_causal(
        self,
        tokens: torch.Tensor,
        mask_prob: torch.Tensor,
        block_len: int,
        token_prefix_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(tokens.shape) == 2

        mask_prob = mask_prob.to(tokens.device) # [B]

        # reshape
        _, token_num = tokens.shape
        block_num = (token_num - token_prefix_len) // block_len
        token_mid_block = rearrange(
            tokens[:, token_prefix_len : token_prefix_len + block_num * block_len],
            "b (c_num c_tokens) -> b c_num c_tokens",
            c_tokens=block_len)

        b, c_num, _ = token_mid_block.shape

        # ------------------------------------------------------------------
        # "Shifted Uniform" Strategy
        # 目標: 平均が 4p となり、かつ分布が可能な限り平坦(Uniform)になるようにマスク個数を決める
        # ------------------------------------------------------------------

        # 1. 各ブロックに一様乱数を生成 [B, c_num]
        u = torch.rand((b, c_num), device=tokens.device)

        # 2. mask_prob を [B, c_num] に拡張
        p = mask_prob.reshape(-1, 1).expand(b, c_num)

        # 3. 連続空間でのマスク個数を計算
        # 幅5.0の一様分布を作成し、その中心を「目標個数(block_len * p)」に合わせる
        # block_len=4 の場合、理想的なUniform分布の中心は 2.0
        # したがって、(4p - 2.0) だけシフトさせるのが基本だが、
        # floor(U[0,5]) は {0,1,2,3,4} を均等に出すため、中心は 2.5 ではなく 2.0 相当の挙動になる。
        # ここではシンプルに「期待値 4p を狙う」式にする。

        # continuous_count ~ Uniform[4p - 2.5, 4p + 2.5]
        # これにより、p=0.5 のとき U[-0.5, 4.5] -> floor -> {-1, 0, 1, 2, 3, 4} -> clamp -> {0, 1, 2, 3, 4}
        width = float(block_len) + 1.0  # 5.0
        center = float(block_len) * p   # 4p

        # U[0, 1] * 5.0 -> U[0, 5] (平均2.5)
        # これを center (平均4p) に移動させるには: + (center - 2.5)
        continuous_count = u * width + (center - width / 2.0)

        # 4. 整数化とクリッピング
        # floorを取ることで、確率質量を整数ビンに落とす
        num_masked = torch.floor(continuous_count).clamp(0, block_len)

        # ------------------------------------------------------------------

        # マスク生成ロジック (右詰め)
        # num_masked=1 -> 右端1つ(idx=3)マスク -> indices >= 3
        mask_start_idx = (block_len - num_masked).reshape(b, c_num, 1)
        indices = torch.arange(block_len, device=tokens.device).expand_as(token_mid_block)
        token_mid_block_masked = torch.where(mask_start_idx <= indices, self.mask_id, token_mid_block)

        # flatten
        tokens_masked = tokens.clone()
        tokens_masked[:, token_prefix_len : token_prefix_len + block_num * block_len] = token_mid_block_masked.reshape(b, -1)
        mask = tokens_masked == self.mask_id
        return tokens_masked, mask
