"""GTMatchReward — bonus for predicted constraint blocks that match a GT block.

Greedy one-to-one block matching: each GT block can be consumed by at most one
predicted block (first-match in scan order).  This prevents farming the same
GT block N times via duplicates.

Both predicted and GT token sequences are canonicalised by ``sort_intra_refs``
before comparison.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.sketch_handlers.constants import (
    CONSTRAINT_VAL_OFFSET,
    ConstraintToken,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


@dataclass
class GTMatchAsyncHandle:
    future: Future[RewardOutput]


class GTMatchReward(BaseReward):
    """Per-block GT-match bonus placed at each matched block's tail token."""

    name = "gt_match"
    is_async = True

    def __init__(
        self,
        context: SketchContext,
        coeff: float = 1.0,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(weight=weight, enabled=enabled)
        self.coeff = float(coeff)
        self.context = context
        self.tokenizer = SketchTokenizer(context)
        self._block_size = int(context.align_constraint_token_length_to)
        self._prefix_len = 1  # START
        self._executor: ThreadPoolExecutor | None = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self._executor = None

    def _sort_intra_refs(self, cons_tokens: np.ndarray) -> np.ndarray:
        """Canonicalise refs within each constraint block (copy of
        SolverReward.sort_intra_refs to keep this reward self-contained)."""
        if len(cons_tokens) == 0:
            return cons_tokens
        assert cons_tokens.ndim == 1
        bs = self._block_size
        if bs <= 0:
            raise NotImplementedError("block_size must be > 0")

        if cons_tokens[0] == ConstraintToken.Start:
            prefix = cons_tokens[0:1]
            body = cons_tokens[1:].copy()
        else:
            prefix = np.array([], dtype=cons_tokens.dtype)
            body = cons_tokens.copy()

        cons_len = body.shape[0] // bs
        body_r = body[: cons_len * bs].reshape(cons_len, bs)
        for i in range(cons_len):
            cons_id, *refs = body_r[i].tolist()
            if cons_id in ConstraintToken._value2member_map_:
                cons_type = ConstraintToken(cons_id).name
                handler = self.tokenizer.context.get_constraint_handler(cons_type)
                if handler is not None:
                    true_refs = tuple(r for r in refs if r >= CONSTRAINT_VAL_OFFSET)
                    pad_refs = tuple(r for r in refs if r < CONSTRAINT_VAL_OFFSET)
                    sorted_true = handler._sort_ref_tokens(true_refs)
                    body_r[i] = np.array([cons_id, *sorted_true, *pad_refs])
        return np.concatenate([prefix, body_r.flatten()])

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        if ctx.gt_tokens is None:
            raise ValueError("GTMatchReward requires ctx.gt_tokens (GT constraint sequence)")
        generated = ctx.cons_tokens.detach().cpu()
        gt = ctx.gt_tokens.detach().cpu()
        BK, L = generated.shape
        bs = self._block_size
        prefix = self._prefix_len
        N = (L - prefix) // bs
        body_end = prefix + N * bs

        gen_np = generated.numpy()
        gt_np = gt.numpy()
        gen_sorted_np = np.stack(
            [self._sort_intra_refs(gen_np[k]) for k in range(BK)], axis=0
        )
        gt_sorted_np = np.stack(
            [self._sort_intra_refs(gt_np[k]) for k in range(BK)], axis=0
        )
        gen_sorted = torch.from_numpy(gen_sorted_np)
        gt_sorted = torch.from_numpy(gt_sorted_np)

        gen_body = gen_sorted[:, prefix:body_end].reshape(BK, N, bs)
        gt_body = gt_sorted[:, prefix:body_end].reshape(BK, N, bs)
        M = gt_body.shape[1]

        pad_id = int(ConstraintToken.Pad)
        stop_id = int(ConstraintToken.Stop)
        gen_invalid = ((gen_body == pad_id) | (gen_body == stop_id)).any(dim=-1)
        gt_invalid = ((gt_body == pad_id) | (gt_body == stop_id)).any(dim=-1)

        gen_exp = gen_body.unsqueeze(2)  # (BK, N, 1, bs)
        gt_exp = gt_body.unsqueeze(1)    # (BK, 1, M, bs)
        block_match = (gen_exp == gt_exp).all(dim=-1)
        block_match = block_match & (~gt_invalid.unsqueeze(1))
        block_match = block_match & (~gen_invalid.unsqueeze(2))

        used = torch.zeros(BK, M, dtype=torch.bool)
        matched = torch.zeros(BK, N, dtype=torch.bool)
        row_idx = torch.arange(BK)
        for n in range(N):
            cand = block_match[:, n, :] & ~used
            has_match = cand.any(dim=-1)
            first_idx = cand.int().argmax(dim=-1)
            matched[:, n] = has_match
            used[row_idx[has_match], first_idx[has_match]] = True

        scores = torch.zeros(BK, L, dtype=torch.float32)
        if matched.any():
            block_tail_positions = (
                torch.arange(N) * bs + (bs - 1) + prefix
            )
            reward_to_add = matched.float() * self.coeff
            scores.scatter_add_(
                dim=1,
                index=block_tail_positions.unsqueeze(0).expand(BK, -1),
                src=reward_to_add,
            )

        metrics = {
            "gt_match_block_count": float(matched.sum().item()),
            "gt_match_block_rate": float(matched.float().mean().item()),
        }
        return RewardOutput(scores=scores, metrics=metrics)

    def submit_async(self, ctx: RewardContext) -> GTMatchAsyncHandle:
        """Run GT matching in a background thread while GPU logprobs compute."""
        return GTMatchAsyncHandle(self._get_executor().submit(self.compute, ctx))

    def collect_results(self, handle: GTMatchAsyncHandle) -> RewardOutput:
        return handle.future.result()
