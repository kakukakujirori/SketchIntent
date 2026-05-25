"""DimParsimonyReward — penalty on the number of dimensional constraints.

Single home for the dim-count parsimony term that previously lived in both
``SolverReward`` (``penalty_num_dimensional_constraints``) and
``RelStructReward`` (``dim_penalty``).  Modes mirror the legacy
``RelStructReward.dim_penalty_mode`` values.
"""
from __future__ import annotations

import torch

from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.sketch_handlers.constants import (
    ConstraintToken,
    DIMENSIONAL_CONSTRAINT_TOKEN_IDS,
)
from src.sketch_handlers.context import SketchContext


_DIM_PENALTY_MODES = (
    "count",
    "ratio",
    "per_primitive",
    "excess_over_gt",
    "excess_over_gt_per_primitive",
)
_GATE_MODES = ("always", "converged", "fc")


class DimParsimonyReward(BaseReward):
    """Negative reward proportional to the number of dimensional constraints."""

    name = "dim_parsimony"

    def __init__(
        self,
        context: SketchContext,
        coeff: float = 0.05,
        mode: str = "excess_over_gt_per_primitive",
        excess_margin: int = 0,
        gate: str = "converged",
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(weight=weight, enabled=enabled)
        if mode not in _DIM_PENALTY_MODES:
            raise ValueError(
                f"mode must be one of {_DIM_PENALTY_MODES}, got {mode!r}"
            )
        if excess_margin < 0:
            raise ValueError(f"excess_margin must be >= 0, got {excess_margin}")
        if gate not in _GATE_MODES:
            raise ValueError(f"gate must be one of {_GATE_MODES}, got {gate!r}")
        self.coeff = float(coeff)
        self.mode = mode
        self.excess_margin = int(excess_margin)
        self.gate = gate
        self._block_size = int(context.align_constraint_token_length_to)
        self._prefix_len = 1
        # Buffer of dimensional token IDs for vectorised counting.
        self.register_buffer(
            "_dim_token_ids",
            torch.tensor(DIMENSIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )

    def _gate_pass(self, status: str | None) -> bool:
        if self.gate == "always" or status is None:
            return True
        if self.gate == "converged":
            return status in ("FC", "Under")
        if self.gate == "fc":
            return status == "FC"
        return False

    def _penalty_value(
        self,
        n_dims: int,
        n_total: int,
        n_primitives: int,
        n_gt_dims: int,
    ) -> float:
        if self.mode == "count":
            return float(n_dims)
        if self.mode == "ratio":
            return float(n_dims) / float(max(1, n_total))
        if self.mode == "per_primitive":
            return float(n_dims) / float(max(1, n_primitives))
        if self.mode == "excess_over_gt":
            return float(max(0, n_dims - n_gt_dims - self.excess_margin))
        # excess_over_gt_per_primitive
        excess = max(0, n_dims - n_gt_dims - self.excess_margin)
        return float(excess) / float(max(1, n_primitives))

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        cons_tokens = ctx.cons_tokens
        gt_tokens = ctx.gt_tokens
        sketches = ctx.sketch_original_list
        per_sample_status = ctx.per_sample_status
        bs = self._block_size
        prefix = self._prefix_len
        B, L = cons_tokens.shape

        # Per-sample dim counts via head-of-block tokens (one row per constraint).
        # cons_id is at the first position of each block.
        N = (L - prefix) // bs
        head_idx = prefix + torch.arange(N, device=cons_tokens.device) * bs
        heads = cons_tokens[:, head_idx]                                   # (B, N)
        dim_mask = (heads.unsqueeze(-1) == self._dim_token_ids.to(heads.device)).any(dim=-1)
        stop_mask = (heads == int(ConstraintToken.Stop)).cumsum(dim=-1) >= 1
        valid_mask = ~stop_mask & (heads != int(ConstraintToken.Pad))
        n_dims_pred = (dim_mask & valid_mask).sum(dim=-1).tolist()        # list[int], len B
        n_total_pred = valid_mask.sum(dim=-1).tolist()

        # GT dim counts (used by excess_over_gt modes).
        if gt_tokens is not None:
            gt_heads = gt_tokens[:, head_idx]
            gt_dim_mask = (gt_heads.unsqueeze(-1) == self._dim_token_ids.to(gt_heads.device)).any(dim=-1)
            gt_stop = (gt_heads == int(ConstraintToken.Stop)).cumsum(dim=-1) >= 1
            gt_valid = ~gt_stop & (gt_heads != int(ConstraintToken.Pad))
            n_dims_gt = (gt_dim_mask & gt_valid).sum(dim=-1).tolist()
        else:
            n_dims_gt = [0] * B

        # Find STOP positions (placement target for the sequence-level penalty).
        eq_stop = (cons_tokens == int(ConstraintToken.Stop)).int()
        any_stop = eq_stop.any(dim=-1)
        first_stop = torch.where(
            any_stop,
            eq_stop.argmax(dim=-1),
            torch.full((B,), L - 1, device=cons_tokens.device, dtype=torch.long),
        )

        scores = torch.zeros((B, L), dtype=torch.float32)
        penalty_values: list[float] = []
        applied = 0
        for i in range(B):
            status = per_sample_status[i] if per_sample_status is not None else None
            n_primitives = len(sketches[i].entities) if sketches[i] is not None else 1
            val = self._penalty_value(
                n_dims_pred[i],
                n_total_pred[i],
                n_primitives,
                n_dims_gt[i],
            )
            penalty_values.append(val)
            if not self._gate_pass(status):
                continue
            scores[i, int(first_stop[i].item())] += -self.coeff * val
            applied += 1

        metrics: dict[str, float | int] = {
            "dim_parsimony_n_dims_mean": float(sum(n_dims_pred) / max(1, B)),
            "dim_parsimony_n_gt_dims_mean": float(sum(n_dims_gt) / max(1, B)),
            "dim_parsimony_value_mean": float(sum(penalty_values) / max(1, B)),
            "dim_parsimony_applied_n": applied,
        }
        return RewardOutput(scores=scores, metrics=metrics)
