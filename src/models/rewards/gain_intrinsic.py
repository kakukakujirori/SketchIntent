"""GainIntrinsicReward — bonus for attainability of intrinsic-size probes.

For each primitive whose intrinsic size is a natural driving parameter
(Line → Length, Arc/Circle → Radius), build a synthetic probe and measure
how reachable that probe is via the predicted edit response.  Lower gain
means better attainability; the reward is a linear bonus on (cap - gain).
"""
from __future__ import annotations

import math
import statistics

import numpy as np
import torch

from src.metrics.edit_response_distance import (
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.models.rewards.utils.probes import build_intrinsic_probes
from src.models.rewards.utils.sketch_builder import build_pred_sketch
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


def _compute_gain_intrinsic_value(pred_native, gt_sketch, context, eps):
    """Mean per-probe gain magnitude (lower = better attainability)."""
    probes = build_intrinsic_probes(gt_sketch)
    if not probes:
        return 0.0, 0
    gain_sum = 0.0
    count = 0
    for probe in probes:
        try:
            S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps)
            if S_pr is None:
                continue
            Jq = constraint_jacobian(probe, pred_native.state, context)
            if Jq is None:
                continue
            k_q = max(1, Jq.shape[0])
            eye = torch.eye(Jq.shape[0], dtype=Jq.dtype, device=Jq.device)
            gain_residual = Jq @ S_pr - eye
            gain_ms = float(torch.linalg.norm(gain_residual) ** 2) / k_q
            gain_sum += math.sqrt(max(gain_ms, 0.0))
            count += 1
        except Exception:
            continue
    return (gain_sum / count, count) if count > 0 else (0.0, 0)


def _mean(xs):
    return float(statistics.fmean(xs)) if xs else 0.0


class GainIntrinsicReward(BaseReward):
    """Linear bonus on intrinsic-probe attainability (FC gate by default)."""

    name = "gain_intrinsic"

    def __init__(
        self,
        context: SketchContext,
        coeff: float = 3.0,
        cap: float = 0.5,
        eps: float = 1e-9,
        gate: str = "fc",
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(weight=weight, enabled=enabled)
        if gate not in ("always", "converged", "fc"):
            raise ValueError(f"gate must be one of always/converged/fc, got {gate!r}")
        if cap <= 0.0:
            raise ValueError(f"cap must be > 0, got {cap}")
        self.context = context
        self.coeff = float(coeff)
        self.cap = float(cap)
        self.eps = float(eps)
        self.gate = gate
        self.tokenizer = SketchTokenizer(context)

    def _gate_pass(self, status: str | None) -> bool:
        if self.gate == "always" or status is None:
            return True
        if self.gate == "converged":
            return status in ("FC", "Under")
        return status == "FC"

    def _bonus(self, value: float) -> float:
        clipped = min(max(value, 0.0), self.cap)
        return self.coeff * (1.0 - clipped / self.cap)

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        prim_tokens = ctx.prim_tokens
        cons_tokens = ctx.cons_tokens
        sketches = ctx.sketch_original_list
        per_sample_solver_status = ctx.per_sample_solver_status
        B, L = cons_tokens.shape

        scores = torch.zeros((B, L), dtype=torch.float32)
        prim_np = prim_tokens.detach().cpu().numpy()
        cons_np = cons_tokens.detach().cpu().numpy()

        gain_vals: list[float] = []

        for i in range(B):
            status = per_sample_solver_status[i] if per_sample_solver_status is not None else None
            if not self._gate_pass(status):
                continue

            cons_i = cons_np[i].copy()
            stop_pos = np.where(cons_i == int(ConstraintToken.Stop))[0]
            first_stop_idx = int(stop_pos[0]) if len(stop_pos) > 0 else (L - 1)
            if 0 <= first_stop_idx < L - 1:
                cons_i[first_stop_idx + 1:] = int(ConstraintToken.Pad)

            gt_sketch = sketches[i]
            try:
                pred_sketch = build_pred_sketch(self.tokenizer, cons_i, prim_np[i], gt_sketch)
                pred_native = compute_native_response(pred_sketch, self.context, eps=self.eps)
            except Exception:
                continue

            gain, _ = _compute_gain_intrinsic_value(
                pred_native, gt_sketch, self.context, self.eps
            )
            gain_vals.append(gain)
            scores[i, first_stop_idx] += self._bonus(gain)

        metrics: dict[str, float | int] = {
            "gain_intrinsic_mean": _mean(gain_vals),
        }
        return RewardOutput(scores=scores, metrics=metrics)
