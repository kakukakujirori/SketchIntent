"""Abstract base class and shared types for RL rewards.

Each concrete reward inherits ``BaseReward`` and implements ``compute(ctx)``.
The ``RewardComposer`` (introduced in a later phase) is responsible for
building the shared ``RewardContext`` once per batch and invoking each
reward in order.

Solver-backed rewards override ``submit_async`` / ``collect_results`` to
keep the async CPU pipeline; sync rewards inherit the default no-op
implementations.
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
        per_sample_status: only ``SolverReward`` populates this; downstream
            rewards read it from the ``RewardContext`` for gating.
    """
    scores: torch.Tensor
    metrics: dict[str, Any] = field(default_factory=dict)
    per_sample_status: list[str] | None = None


@dataclass
class RewardContext:
    """Per-batch shared inputs passed to every reward in a ``RewardComposer``.

    The composer is responsible for constructing this object once per batch
    and feeding it to each reward.  Mutable fields such as
    ``per_sample_status`` are written by ``SolverReward`` and consumed by
    subsequent rewards.  ``native_cache`` exists so that the same Pred / GT
    ``NativeResponse`` is computed at most once even when several rewards
    need it.
    """
    prim_tokens: torch.Tensor                                   # (B, L_prim)
    cons_tokens: torch.Tensor                                   # (B, L_cons) — generated
    sketch_original_list: list[datalib.Sketch]
    gt_tokens: torch.Tensor | None = None                       # (B, L_cons) — GT, for GT-match
    per_sample_status: list[str] | None = None                  # written by SolverReward
    pred_sketches: list[datalib.Sketch | None] | None = None    # built on demand
    native_cache: dict[tuple[str, int], "NativeResponse"] = field(default_factory=dict)
    sketch_tokenizer: "SketchTokenizer | None" = None


class BaseReward(torch.nn.Module, abc.ABC):
    """Abstract base class for RL rewards.

    Subclasses set the class-level ``name`` (used for metrics prefixing and
    config lookups) and implement ``compute(ctx)``.  Async-backed rewards
    (currently only ``SolverReward``) override ``submit_async`` /
    ``collect_results``; the default sync flow simply invokes ``compute``.
    """

    name: str = ""              # subclass must set
    is_async: bool = False      # SolverReward only

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__()
        self.weight = float(weight)
        self.enabled = bool(enabled)

    @abc.abstractmethod
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """Compute scores + metrics for one batch."""

    # ------------------------------------------------------------------
    # Async integration — overridden by SolverReward.  For sync rewards the
    # composer is free to call ``compute`` directly, but exposing the 3-phase
    # API uniformly lets the composer treat every reward identically.
    # ------------------------------------------------------------------
    def submit_async(self, ctx: RewardContext) -> Any:
        return None

    def collect_results(self, handle: Any, ctx: RewardContext) -> RewardOutput:
        return self.compute(ctx)
