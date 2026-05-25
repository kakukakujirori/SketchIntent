"""RewardComposer — single home for reward aggregation.

The composer builds the per-batch ``RewardContext`` once, dispatches async
rewards (currently only ``SolverReward``) before the GPU forward pass, and
gathers sync rewards afterwards.  Per_sample_status from the solver is
written into the context so that downstream rewards can gate on it.

Per-reward weights, individual clipping, and a final clip on the aggregated
tensor are all configurable.  Metrics from each ``RewardOutput`` are
flattened into a single dict; reward modules are expected to namespace
their own metric keys (``solver_*``, ``relstruct_*``, ``gain_intrinsic_*``…).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch

import sketchgraphs.data as datalib

from src.models.rewards.base import BaseReward, RewardContext, RewardOutput


class RewardComposer(torch.nn.Module):
    """Orchestrates a list of ``BaseReward`` modules.

    Args:
        rewards: ordered list of rewards. ``SolverReward`` (``is_async=True``)
            should appear first so its ``per_sample_status`` is available
            to subsequent rewards.
        clip_total: optional ``(lo, hi)`` clamp applied after aggregation.
        clip_per_reward: optional ``{name: (lo, hi)}`` per-reward clamp on
            ``RewardOutput.scores`` before the weighted sum.
    """

    def __init__(
        self,
        rewards: list[BaseReward],
        clip_total: tuple[float, float] | None = None,
        clip_per_reward: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        names = [r.name for r in rewards]
        if len(set(names)) != len(names):
            raise ValueError(f"reward names must be unique, got {names}")
        # Order async first so per_sample_status propagates correctly.
        async_rs = [r for r in rewards if r.is_async]
        sync_rs = [r for r in rewards if not r.is_async]
        self.rewards = torch.nn.ModuleList(async_rs + sync_rs)
        self.clip_total = clip_total
        self.clip_per_reward = clip_per_reward or {}

    # ------------------------------------------------------------------ #
    #  Phase A — build context                                           #
    # ------------------------------------------------------------------ #
    def build_context(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch],
        gt_tokens: torch.Tensor | None = None,
    ) -> RewardContext:
        return RewardContext(
            prim_tokens=prim_tokens,
            cons_tokens=cons_tokens,
            sketch_original_list=sketch_original_list,
            gt_tokens=gt_tokens,
        )

    # ------------------------------------------------------------------ #
    #  Phase B — kick off async rewards                                  #
    # ------------------------------------------------------------------ #
    def submit_async(self, ctx: RewardContext) -> dict[str, Any]:
        """Kick off any async rewards. Call before the GPU forward pass."""
        handles: dict[str, Any] = {}
        for r in self.rewards:
            if not r.enabled or not r.is_async:
                continue
            # SolverReward keeps its legacy positional signature.
            handles[r.name] = r.submit_async(
                ctx.prim_tokens, ctx.cons_tokens, ctx.sketch_original_list
            )
        return handles

    # ------------------------------------------------------------------ #
    #  Phase C — gather all rewards and aggregate                        #
    # ------------------------------------------------------------------ #
    def collect(
        self,
        handles: dict[str, Any],
        ctx: RewardContext,
    ) -> tuple[torch.Tensor, dict[str, RewardOutput]]:
        outputs: dict[str, RewardOutput] = {}
        total: torch.Tensor | None = None

        for r in self.rewards:
            if not r.enabled:
                continue
            if r.is_async:
                # SolverReward returns (scores_cpu, metrics, per_sample_status).
                handle = handles.get(r.name)
                scores_cpu, metrics, status = r.collect_results(handle)
                out = RewardOutput(
                    scores=scores_cpu, metrics=metrics, per_sample_status=status
                )
            else:
                out = r.compute(ctx)
            outputs[r.name] = out

            # Propagate per_sample_status for downstream gating.
            if out.per_sample_status is not None:
                ctx.per_sample_status = out.per_sample_status

            s = out.scores
            if r.name in self.clip_per_reward:
                lo, hi = self.clip_per_reward[r.name]
                s = s.clamp(lo, hi)
            weighted = r.weight * s
            total = weighted if total is None else total + weighted

        if total is None:
            total = torch.zeros_like(ctx.cons_tokens, dtype=torch.float32)
        if self.clip_total is not None:
            total = total.clamp(*self.clip_total)
        return total, outputs

    def flatten_metrics(
        self,
        outputs: dict[str, RewardOutput],
    ) -> dict[str, float | int | Counter]:
        """Flatten metrics from all rewards into a single dict.

        Reward modules already namespace their own keys (``solver_``,
        ``relstruct_``…), so this just unions them without further prefixing.
        """
        merged: dict[str, Any] = {}
        for out in outputs.values():
            merged.update(out.metrics)
        return merged
