"""Abstract base class and shared types for RL rewards.

Each concrete reward inherits ``BaseReward`` and implements ``compute(ctx)``.
The ``RewardComposer`` (introduced in a later phase) is responsible for
building the shared ``RewardContext`` once per batch and invoking each
reward in order.

Async-backed rewards set ``is_async = True`` and provide ``submit_async`` /
``collect_results`` methods.  The composer dispatches async work via those
methods directly to avoid colliding with the abstract sync interface.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

import sketchgraphs.data as datalib

if TYPE_CHECKING:
    from src.metrics.edit_response_distance import NativeResponse
    from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


@dataclass
class RewardOutput:
    """Per-reward return value.

    Attributes:
        scores: (B, L) float tensor placed at token positions (typically the
            first STOP per sample for sequence-level rewards, or per-constraint
            tail positions for granular rewards).
        metrics: scalar logging values, optionally namespaced by the reward's
            ``name`` by the ``RewardComposer``.
        per_sample_solver_status: only ``SolverReward`` populates this;
            downstream rewards read it from the ``RewardContext`` for gating.
        per_sample_dof_reduction_rate: only ``SolverReward`` populates this;
            ``(orig_dof - DoF) / (orig_dof - fully_constrained_dof)`` clamped to
            ``[0, 1]`` (0 for invalid/unsolved samples).  Consumed via the
            ``RewardContext`` by the continuous ``gate="dof"`` weighting.
    """
    scores: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)
    per_sample_solver_status: list[str] | None = None
    per_sample_dof_reduction_rate: list[float] | None = None


@dataclass
class RewardContext:
    """Per-batch shared inputs passed to every reward in a ``RewardComposer``.

    The composer is responsible for constructing this object once per batch
    and feeding it to each reward.  Mutable fields such as
    ``per_sample_solver_status`` is written by ``SolverReward`` and consumed
    by subsequent rewards.  ``native_cache`` exists so that the same Pred / GT
    ``NativeResponse`` is computed at most once even when several rewards need
    it.
    """
    prim_tokens: torch.Tensor                                   # (B, L_prim)
    cons_tokens: torch.Tensor                                   # (B, L_cons) — generated
    sketch_original_list: list[datalib.Sketch]
    gt_tokens: torch.Tensor | None = None                       # (B, L_cons) — GT, for GT-match
    per_sample_solver_status: list[str] | None = None           # written by SolverReward
    per_sample_dof_reduction_rate: list[float] | None = None     # written by SolverReward (gate="dof")
    pred_sketches: list[datalib.Sketch | None] | None = None    # built on demand
    native_cache: dict[tuple[str, int], "NativeResponse"] = field(default_factory=dict)
    # Cache for pairwise-distance Jacobian J_phi.  Keyed by ``(id(sketch),
    # dtype)`` so the same sketch object reuses its J_phi across (a) the
    # multiple subspace computations within one sample and (b) the K rollouts
    # of one RLOO prompt group, where ``gt_sketch`` is the same Python object.
    j_phi_cache: dict[tuple[int, torch.dtype], torch.Tensor] = field(default_factory=dict)
    # Cache for GT-side subspace results that only depend on ``gt_sketch``:
    # motion_full subspace U_W_G, motion_relational subspace U_gt at gt's own
    # state, and the orthonormal range of J_R(gt_rels).T used by the leak
    # term.  Each entry is keyed by ``(id(gt_sketch), dtype, kind)`` so an
    # RLOO group with K rollouts hits the cache K-1 times after the first
    # rollout populates it.  Correctness relies on
    # ``pred_native.state.q ≡ gt_native.state.q`` (guaranteed by
    # ``overwrite_primitives_with_original`` deepcopying gt entities into
    # pred), so J_R(gt_rels) is state-independent in practice.
    gt_subspace_cache: dict[tuple[int, torch.dtype, str], object] = field(default_factory=dict)
    # Producer/consumer cache for response Htheta metrics.  The concrete values
    # are ``HThetaMetrics | None`` from ``response_metric_reward.py``; keep the
    # type loose here to avoid a reward-module import cycle.
    h_theta_metrics: dict[int, Any] = field(default_factory=dict)
    # Producer/consumer cache for PREFIX-level response metrics.  Keyed by
    # batch index, each value is ``list[PrefixBlockMetric] | None`` (one entry
    # per completed constraint block) from ``response_metric_reward.py``.  The
    # ``PrefixResponseMetricEvaluator`` producer does the expensive per-prefix
    # sketch/native rebuild once; ``PrefixMotionAngleReward`` and
    # ``PrefixLeakAngleReward`` consume it.  Loose type to avoid an import cycle.
    prefix_metrics: dict[int, Any] = field(default_factory=dict)
    sketch_tokenizer: "SketchTokenizer | None" = None
    # RLOO group size K: the batch is laid out as B contiguous blocks of K
    # rollouts (sample index = b*K + k), matching the (B, K) regrouping used for
    # the advantage baseline in ``constraint_ar_module_rloo``.  Rewards that need
    # the group structure (e.g. the shuffled-intent negative control) read it
    # here.  ``None`` outside RLOO (e.g. the validation composer).
    group_sample_size: int | None = None


class BaseReward(torch.nn.Module, abc.ABC):
    """Abstract base class for RL rewards.

    Subclasses set the class-level ``name`` (used for metrics prefixing and
    config lookups) and implement ``compute(ctx)``.  Async-backed rewards
    additionally set ``is_async = True`` and keep their own ``submit_async`` /
    ``collect_results`` methods; the composer dispatches to those directly.
    """

    name: str = ""              # subclass must set
    is_async: bool = False

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__()
        self.weight = float(weight)
        self.enabled = bool(enabled)

    @abc.abstractmethod
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """Compute scores + metrics for one batch."""

    def submit_async(self, ctx: RewardContext) -> Any:
        """Kick off async reward computation and return an opaque handle."""
        raise NotImplementedError(
            f"{type(self).__name__} has is_async=True but does not implement submit_async()"
        )

    def collect_results(self, handle: Any) -> RewardOutput:
        """Collect async reward computation from a handle."""
        raise NotImplementedError(
            f"{type(self).__name__} has is_async=True but does not implement collect_results()"
        )
