"""RewardComposer — single home for reward aggregation.

The composer builds the per-batch ``RewardContext`` once, dispatches async
rewards before the GPU forward pass, and gathers sync rewards afterwards.
``per_sample_solver_status`` from ``SolverReward`` is written into the context
so that downstream rewards can gate on it.

Per-reward weights, individual clipping, and a final clip on the aggregated
tensor are all configurable.  Metrics from each ``RewardOutput`` are
flattened into a single dict; reward modules are expected to namespace
their own metric keys (``solver_*``, ``relstruct_*``, ``gain_intrinsic_*``…).
"""
from __future__ import annotations

from typing import Any

import torch

import sketchgraphs.data as datalib

from src.models.rewards.base import BaseReward, RewardContext, RewardOutput


class RewardComposer(torch.nn.Module):
    """Orchestrates a list of ``BaseReward`` modules.

    Args:
        rewards: ordered list of rewards. Async rewards are collected before
            sync rewards; only ``SolverReward`` may populate
            ``per_sample_solver_status`` for subsequent rewards.
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
        # Order async first so solver status is available to subsequent sync rewards.
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
            handles[r.name] = r.submit_async(ctx)
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
                handle = handles.get(r.name)
                out = r.collect_results(handle)
                if not isinstance(out, RewardOutput):
                    raise TypeError(
                        f"async reward {r.name!r} collect_results() must return RewardOutput"
                    )
            else:
                out = r.compute(ctx)
            outputs[r.name] = out

            # Propagate solver status for downstream gating.
            if out.per_sample_solver_status is not None:
                if r.name != "solver":
                    raise ValueError(
                        f"reward {r.name!r} populated per_sample_solver_status; "
                        "only SolverReward may do so"
                    )
                ctx.per_sample_solver_status = out.per_sample_solver_status

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
    ) -> dict[str, Any]:
        """Flatten scalar metrics from all rewards into a single dict.

        Reward modules already namespace their own keys (``solver_``,
        ``relstruct_``…), so this just unions them without further prefixing.
        Non-scalar metric values (e.g. raw ``Counter`` histograms) are filtered
        out.
        """
        merged: dict[str, Any] = {}
        for out in outputs.values():
            for k, v in out.metrics.items():
                if isinstance(v, (int, float)):
                    merged[k] = v
                elif isinstance(v, torch.Tensor) and v.ndim == 0:
                    merged[k] = v
        return merged
