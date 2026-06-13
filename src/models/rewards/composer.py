"""RewardComposer — single home for reward aggregation.

The composer builds the per-batch ``RewardContext`` once, dispatches async
rewards before the GPU forward pass, and gathers sync rewards afterwards.
``per_sample_solver_status`` from ``SolverReward`` is written into the context
so that downstream rewards can gate on it.

Per-reward weights, individual clipping, and a final clip on the aggregated
tensor are all configurable.  Metrics from each ``RewardOutput`` are
flattened into a single dict; reward modules are expected to namespace
their own metric keys (``solver_*``, ``intent_*``, ``gain_intrinsic_*``…).
"""
from __future__ import annotations

from typing import Any

import torch

import sketchgraphs.data as datalib

from src.metrics.constraint_stats import (
    constraint_head_counts,
    constraint_type_counts,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.sketch_handlers.constants import (
    ANGLE_FAMILY_CONSTRAINT_TOKEN_IDS,
    CONSTRAINT_TYPE_TOKEN_IDS,
    ConstraintToken,
    DIMENSIONAL_CONSTRAINT_TOKEN_IDS,
    DISTANCE_FAMILY_CONSTRAINT_TOKEN_IDS,
    KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS,
)
from src.sketch_handlers.context import SketchContext

# Reserved key under which RewardValComposer.compute() returns the non-scalar
# per-type constraint-mix payload for the W&B Table / bar chart.  The validation
# loop peels off any ``_``-prefixed key before scalar logging.
CONSTRAINT_MIX_KEY = "_constraint_mix"


def _propagate_context_outputs(
    out: RewardOutput,
    ctx: RewardContext,
    producer_name: str,
) -> None:
    """Copy explicit producer outputs into the shared reward context."""
    # Propagate solver status for downstream gating.
    if out.per_sample_solver_status is not None:
        if producer_name != "solver":
            raise ValueError(
                f"reward {producer_name!r} populated per_sample_solver_status; "
                "only SolverReward may do so"
            )
        ctx.per_sample_solver_status = out.per_sample_solver_status
    # Propagate continuous closure progress for gate="dof" consumers.
    if out.per_sample_dof_reduction_rate is not None:
        if producer_name != "solver":
            raise ValueError(
                f"reward {producer_name!r} populated per_sample_dof_reduction_rate; "
                "only SolverReward may do so"
            )
        ctx.per_sample_dof_reduction_rate = out.per_sample_dof_reduction_rate


class RewardComposer(torch.nn.Module):
    """Orchestrates a list of ``BaseReward`` modules.

    Args:
        rewards: ordered list of rewards. Async rewards are collected before
            sync rewards; ``SolverReward`` may populate
            ``per_sample_solver_status`` for subsequent rewards, and async
            producer rewards may populate their own explicit caches.
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
        group_sample_size: int | None = None,
    ) -> RewardContext:
        return RewardContext(
            prim_tokens=prim_tokens,
            cons_tokens=cons_tokens,
            sketch_original_list=sketch_original_list,
            gt_tokens=gt_tokens,
            group_sample_size=group_sample_size,
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

            _propagate_context_outputs(out, ctx, r.name)

            s = out.scores
            if r.name in self.clip_per_reward:
                lo, hi = self.clip_per_reward[r.name]
                s = s.clamp(lo, hi)
            weighted = r.weight * s

            # Per-reward logging
            n_seq = max(1, int(s.shape[0]))
            out.metrics[f"reward_contrib_{r.name}_seq_mean"] = (
                float(weighted.sum().item()) / n_seq
            )
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
        ``intent_``…), so this just unions them without further prefixing.
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
        # Total reward contribution = sum of the per-reward contributions, so the
        # breakdown and the total always agree by construction.
        contribs = [
            float(v)
            for k, v in merged.items()
            if k.startswith("reward_contrib_") and k.endswith("_seq_mean")
        ]
        if contribs:
            merged["reward_contrib_total_seq_mean"] = sum(contribs)
        return merged


class RewardValComposer(torch.nn.Module):
    """Validation-time metric composer — the eval-side sibling of RewardComposer.

    Holds a configurable list of ``BaseReward`` modules run purely for their
    metrics — typically the SAME reward classes the trainer uses (``SolverReward``
    for ``solver_*`` rates, ``ResponseMetricEvaluator`` for ``intent_*_angle``
    incl. the canonical ``response_full_angle``), so eval never reimplements a reward.
    Each module's ``compute(ctx)`` runs over a shared context and its scalar
    ``metrics`` are merged; there is no weighting, no aggregated score, and no
    ``reward_contrib`` (those are training concerns owned by RewardComposer).
    Keeping evaluation in its own composer means ``validation_gen/*`` never
    duplicates the ``train/*`` reward diagnostics.

    In addition, the composer always computes the relational constraint mix
    (``eval_rel_ratio_mean``) itself — it is a trivial token-level quantity that
    no reward produces, so it does not warrant its own module.

    ``per_sample_solver_status`` and ``per_sample_dof_reduction_rate`` produced
    by a solver module are propagated into the context so later modules may gate
    on them if they choose.
    """

    def __init__(self, metrics: list[BaseReward], context: Any):
        super().__init__()
        names = [m.name for m in metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"val-metric names must be unique, got {names}")
        # Async modules (e.g. SolverReward) first so solver status is available
        # to any later module that wants to gate on it.
        async_ms = [m for m in metrics if m.is_async]
        sync_ms = [m for m in metrics if not m.is_async]
        self.metrics = torch.nn.ModuleList(async_ms + sync_ms)
        # For the built-in constraint-mix metric.
        self._block_size = int(context.align_constraint_token_length_to)
        self._prefix_len = 1
        self.register_buffer(
            "_dim_token_ids",
            torch.tensor(DIMENSIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )
        # Token-id groups for the family-ratio scalars (key relations, distance
        # family, angle family) and the full per-type constraint-mix chart.
        self.register_buffer(
            "_key_rel_token_ids",
            torch.tensor(KEY_RELATIONAL_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_distance_family_token_ids",
            torch.tensor(DISTANCE_FAMILY_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_angle_family_token_ids",
            torch.tensor(ANGLE_FAMILY_CONSTRAINT_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_all_type_token_ids",
            torch.tensor(CONSTRAINT_TYPE_TOKEN_IDS, dtype=torch.long),
            persistent=False,
        )
        # Parallel name / family labels for the chart payload (CPU-side, static).
        dim_id_set = set(int(t) for t in DIMENSIONAL_CONSTRAINT_TOKEN_IDS)
        self._all_type_names = [
            ConstraintToken(int(t)).name for t in CONSTRAINT_TYPE_TOKEN_IDS
        ]
        self._all_type_families = [
            "dimensional" if int(t) in dim_id_set else "relational"
            for t in CONSTRAINT_TYPE_TOKEN_IDS
        ]

    @torch.no_grad()
    def compute(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch],
        gt_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run every enabled module and return their merged scalar metrics."""
        ctx = RewardContext(
            prim_tokens=prim_tokens,
            cons_tokens=cons_tokens,
            sketch_original_list=sketch_original_list,
            gt_tokens=gt_tokens,
        )
        merged: dict[str, Any] = {}
        for m in self.metrics:
            if not m.enabled:
                continue
            out = m.compute(ctx)
            _propagate_context_outputs(out, ctx, m.name)
            for k, v in out.metrics.items():
                if isinstance(v, (int, float)):
                    merged[k] = float(v)
                elif isinstance(v, torch.Tensor) and v.ndim == 0:
                    merged[k] = float(v.item())

        # Built-in: relational fraction of the generated set (no reward emits it).
        n_rel, _n_dim, n_total = constraint_type_counts(
            cons_tokens, self._block_size, self._dim_token_ids,
            prefix_len=self._prefix_len,
        )
        n_total_f = n_total.clamp(min=1).float()
        merged["eval_rel_ratio_mean"] = float(
            (n_rel.float() / n_total_f).mean()
        )

        # Family ratios: which structural relations / dimensional substitutes
        # dominate the generated set.  Each is mean over samples of
        # (family head count) / n_total.  type-wise breakdown goes to the chart
        # below rather than a wall of parallel scalars.
        def _family_ratio(token_ids: torch.Tensor) -> float:
            fam = constraint_head_counts(
                cons_tokens, self._block_size, token_ids,
                prefix_len=self._prefix_len,
            ).sum(dim=1)                                          # (B,)
            return float((fam.float() / n_total_f).mean())

        merged["eval_key_rel_ratio_mean"] = _family_ratio(self._key_rel_token_ids)
        merged["eval_distance_family_ratio_mean"] = _family_ratio(
            self._distance_family_token_ids
        )
        merged["eval_angle_family_ratio_mean"] = _family_ratio(
            self._angle_family_token_ids
        )

        # Non-scalar payload for the per-type constraint-mix Table / bar chart.
        # The validation loop accumulates this across batches and logs it on
        # rank 0; it is keyed with a leading underscore so scalar logging skips
        # it.  ``per_type_counts`` is summed over the batch.
        per_type_counts = constraint_head_counts(
            cons_tokens, self._block_size, self._all_type_token_ids,
            prefix_len=self._prefix_len,
        ).sum(dim=0)                                             # (n_types,)
        merged[CONSTRAINT_MIX_KEY] = {
            "names": self._all_type_names,
            "families": self._all_type_families,
            "counts": per_type_counts.detach().cpu().tolist(),
            "n_total": int(n_total.sum().item()),
            "n_sketches": int(cons_tokens.shape[0]),
        }
        return merged
