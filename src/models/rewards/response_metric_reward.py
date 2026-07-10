"""Response metric rewards — Htheta / relation-intent signals.

This file intentionally keeps the experimental reward in one place, but it is
organized around three boundaries:

1. metric definitions for Htheta / relation intent,
2. per-sample reward scoring, and
3. batch-level logging aggregation.

Reward configs use the two-axis naming convention:

``<component>_<scope>_angle``

where ``component`` is one of:

- ``motion``: allowed first-order motion/articulation subspace, the successor
  of the v3 gain term;
- ``leak``: relation-violation direction, the successor of the v3 leak term;
- ``response``: the combined metric used as a scalar reward/diagnostic.

and ``scope`` is one of:

- ``full``: the canonical response metric that includes dimensional drivers via
  ``range(J_phi R_C)``;
- ``relational``: the reward-oriented variant that replaces dimensional-driver
  placement with relation-only structure.

The metric names are:

``response_full_angle``       -> canonical Htheta response angle
``response_relational_angle`` -> reward surrogate that combines relation-only
                                 motion structure and relation leak
``response_relational_weighted_angle`` -> linear blend of the previous two with a
                                 config-set mixing ratio (``leak_weight``)
``motion_full_angle``         -> full motion-subspace angle (uses R_C)
``motion_relational_angle``   -> mode-normalized relation-only motion angle
``leak_relational_angle``     -> relation-leak angle

The most important distinction is evaluation metric vs training reward.
``response_full_angle`` is the canonical Htheta scalar from the v4 analysis:
full motion-subspace mismatch plus angular relation leak.  The default training
signal, ``response_relational_angle``, replaces the dimensional-driver-aware
motion term with relation-only motion structure.  That avoids rewarding a model
for matching one annotator's particular dimensional-constraint placement while
still encouraging relational design intent.

The probe-conditioned response-SVD family lives in
``probe_response_svd_reward.py``.  It shares the same first-order response map
but is a supplement diagnostic, not part of the default intent reward.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import math
import multiprocessing as mp
import statistics
import warnings
from typing import Any

import numpy as np
import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _is_relational,
    compute_native_response,
    pairwise_distance_observable,
)
from src.metrics.motion_subspace import (
    leak_angle_from_overlap as _leak_angle_from_overlap,
    motion_full_subspace as _motion_full_subspace,
    motion_relational_subspace as _motion_relational_subspace,
    orthonormal_range as _orthonormal_range,
    padded_subspace_angle as _padded_subspace_angle,
    principal_cos2_sum as _principal_cos2_sum,
    stack_relation_jacobians_at as _stack_relation_jacobians_at,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.models.rewards.utils.sketch_builder import (
    build_pred_sketch,
    first_stop_and_pad,
    get_or_build_pred_native,
    get_or_compute_gt_motion_full_subspace,
    get_or_compute_gt_motion_relational_subspace,
    get_or_compute_gt_native,
    get_or_compute_gt_relation_nullspace,
    get_or_compute_j_phi,
)
from src.sketch_handlers.constants import (
    ConstraintToken,
    DIMENSIONAL_CONSTRAINT_TOKEN_IDS,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


logger = logging.getLogger(__name__)


H_THETA_REWARD_METRICS = {
    "response_full_angle",
    "response_relational_angle",
    "motion_full_angle",
    "motion_relational_angle",
    "leak_relational_angle",
    # Weighted variant of response_relational_angle: the linear blend
    #   (1 - leak_weight) * motion_relational_angle
    #   + leak_weight * leak_relational_angle.
    # Unlike the fixed-RMS response_relational_angle, the mixing ratio is a
    # config knob (ResponseIntentReward.leak_weight): lowering leak_weight
    # shrinks the payoff of the "emit no dims -> leak=0" under-game that the
    # 1:1 RMS over-rewards under amplified coefficients (scaffold10 collapse).
    "response_relational_weighted_angle",
}


MOTION_RELATIONAL_ANGLE_MODES = {
    "symmetric",
    "pred_subset_gt",
    "mixed_subset",
}


_GATE_MODES = ("always", "converged", "fc")
# Consumer rewards additionally support the continuous ``dof`` mode: instead of a
# 0/1 status gate, the bonus is scaled by the per-sample solver dof_reduction_rate
# (0 at no/invalid closure -> 1 at fully constrained).  This smoothly ramps a
# component (e.g. leak) in as a sketch closes, removing the hard FC cliff
# (over-dim rush) and the no-dims degenerate minimum (under-collapse) at once.
_CONSUMER_GATE_MODES = _GATE_MODES + ("dof",)


# response_*_angle combines two unit-angle components (motion + leak), so the
# raw Pythagorean sum saturates at sqrt(2).  We divide by sqrt(2) so that the
# combined response angle also lives on the [0, 1] scale, matching the
# component metrics and letting the same ``cap`` config apply to every metric
# in H_THETA_REWARD_METRICS.  Geometrically this is the RMS of the two
# component angles rather than their Euclidean sum.
_RESPONSE_ANGLE_NORM = math.sqrt(2.0)


def _validate_h_theta_metric_name(name: str) -> str:
    """Validate and return a config-facing Htheta metric name."""
    if name not in H_THETA_REWARD_METRICS:
        valid = sorted(H_THETA_REWARD_METRICS)
        raise ValueError(
            f"h_angle_reward_metric must be one of {valid}, got {name!r}"
        )
    return name


def _mean(xs: Sequence[float | int]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _median(xs: Sequence[float | int]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _linear_bonus(value: float, cap: float, coeff: float) -> float:
    """Convert a low-is-good angle/metric value into a clipped reward bonus."""
    if cap <= 0.0:
        return 0.0
    clipped = min(max(value, 0.0), cap)
    return coeff * (1.0 - clipped / cap)


def _status_passes_gate(status: str | None, gate: str) -> bool:
    if gate == "always" or status is None:
        return True
    if gate == "converged":
        return status in ("FC", "Under")
    if gate == "fc":
        return status == "FC"
    return False


def _shuffle_scores_within_group(scores: torch.Tensor, ctx: RewardContext) -> torch.Tensor:
    """Permute per-sample STOP rewards within each contiguous RLOO group."""
    K = ctx.group_sample_size
    B, L = scores.shape
    if not K or K < 2 or B % K != 0:
        return scores

    scalar = scores.sum(dim=1)
    cons = ctx.cons_tokens.detach().to("cpu")
    eq_stop = (cons == int(ConstraintToken.Stop)).int()
    any_stop = eq_stop.any(dim=-1)
    first_stop = torch.where(
        any_stop,
        eq_stop.argmax(dim=-1),
        torch.full((B,), L - 1, dtype=torch.long),
    )

    perm = torch.arange(B)
    for g in range(B // K):
        base = g * K
        perm[base : base + K] = base + torch.randperm(K)

    new_scores = torch.zeros_like(scores)
    new_scores[torch.arange(B), first_stop] = scalar[perm]
    return new_scores


def _motion_relational_angle_scale(
    mode: str,
    dim_pred: int,
    dim_gt: int,
    reverse_weight: float,
) -> float:
    """Return the mode-specific maximum of the raw motion-relational angle.

    ``_motion_relational_angle`` returns a raw comparison score whose ceiling
    depends on the comparison mode.  ``motion_relational_angle`` (the public,
    normalized metric) divides by that ceiling so the relation-only motion
    component is on a [0, 1] angle-like scale before it is combined with leak.

    For ``symmetric`` this is the maximum for the current dimension pair.  It
    equals ``sqrt(2)`` when ``dim_pred == dim_gt`` but is smaller for unequal
    dimensions, so orthogonal unequal-dimensional subspaces still map to 1.
    """
    if reverse_weight < 0.0:
        raise ValueError(
            "reverse_weight must be non-negative for motion-relational "
            f"scaling, got {reverse_weight}"
        )
    if mode == "symmetric":
        d_norm = float(max(dim_pred, dim_gt, 1))
        return math.sqrt(max((dim_pred + dim_gt) / d_norm, 1e-12))
    if mode == "mixed_subset":
        return math.sqrt(max(1.0 + reverse_weight, 1e-12))
    return 1.0


def _normalized_motion_relational_angle(
    raw_value: float,
    mode: str,
    dim_pred: int,
    dim_gt: int,
    reverse_weight: float,
) -> float:
    """Normalize the raw motion-relational angle into the public [0, 1] form."""
    scale = _motion_relational_angle_scale(mode, dim_pred, dim_gt, reverse_weight)
    return min(max(raw_value / scale, 0.0), 1.0)


@dataclass(frozen=True)
class HThetaMetrics:
    """Hθ family values for one Pred/GT pair.

    Fields:
        response_full_angle: Canonical Htheta scalar — the RMS of
            ``motion_full_angle`` and ``leak_relational_angle``.  Equal to
            ``sqrt((motion_full_angle^2 + leak_relational_angle^2) / 2)`` so
            that the combined scalar shares the [0, 1] range of its components.
        response_relational_angle: Reward-oriented scalar — the RMS of
            ``motion_relational_angle`` and ``leak_relational_angle``.  Same
            ``/ sqrt(2)`` normalization as ``response_full_angle``.
        motion_full_angle: Full motion-subspace mismatch (uses ``range(J_phi
            R_C)``) in [0, 1].
        motion_relational_angle: Relation-only motion-subspace mismatch,
            normalized by the mode-specific raw maximum so that orthogonal
            subspaces map to 1 even for unequal dimensions.
        leak_relational_angle: Fraction of Pred's allowed motion that enters
            GT relation-forbidden directions, in [0, 1].
        motion_relational_pred_subset_gt_angle / motion_relational_gt_subset_pred_angle:
            Asymmetric inclusion angles of the relation-only motion subspaces,
            useful for diagnosing which side over- or under-constrains.
        motion_relational_dim_pred / motion_relational_dim_gt: Dimensions of
            the Pred and GT relation-only motion subspaces in observable space.
        motion_full_dim_pred / motion_full_dim_gt: Dimensions of the Pred and
            GT full motion subspaces ``range(J_phi R_C)`` in observable space.
    """

    response_full_angle: float
    response_relational_angle: float
    motion_full_angle: float
    motion_relational_angle: float
    leak_relational_angle: float
    motion_relational_pred_subset_gt_angle: float
    motion_relational_gt_subset_pred_angle: float
    motion_relational_dim_pred: int
    motion_relational_dim_gt: int
    motion_full_dim_pred: int
    motion_full_dim_gt: int

    def value(self, metric_name: str) -> float:
        """Return a metric selected by its config-facing name."""
        metric_name = _validate_h_theta_metric_name(metric_name)
        return float(self.to_dict()[metric_name])

    def to_dict(self) -> dict[str, float | int]:
        """Expose the public response-intent metric names."""
        return {
            "response_full_angle": self.response_full_angle,
            "response_relational_angle": self.response_relational_angle,
            "motion_full_angle": self.motion_full_angle,
            "motion_relational_angle": self.motion_relational_angle,
            "leak_relational_angle": self.leak_relational_angle,
            "motion_relational_pred_subset_gt_angle": (
                self.motion_relational_pred_subset_gt_angle
            ),
            "motion_relational_gt_subset_pred_angle": (
                self.motion_relational_gt_subset_pred_angle
            ),
            "motion_relational_dim_pred": self.motion_relational_dim_pred,
            "motion_relational_dim_gt": self.motion_relational_dim_gt,
            "motion_full_dim_pred": self.motion_full_dim_pred,
            "motion_full_dim_gt": self.motion_full_dim_gt,
        }


@dataclass
class ResponseIntentBatchStats:
    """Mutable aggregation buffer for batch-level reward logging.

    The reward path needs scalar lists because training logs are batch means /
    medians, while the per-sample path wants typed metric objects.  Keeping this
    bookkeeping in one dataclass prevents the main forward loop from becoming a
    wall of parallel append calls.
    """

    reward_vals: list[float] = field(default_factory=list)
    response_full_angle_vals: list[float] = field(default_factory=list)
    response_relational_angle_vals: list[float] = field(default_factory=list)
    motion_full_angle_vals: list[float] = field(default_factory=list)
    motion_relational_angle_vals: list[float] = field(default_factory=list)
    leak_relational_angle_vals: list[float] = field(default_factory=list)
    motion_relational_pred_subset_gt_angle_vals: list[float] = field(default_factory=list)
    motion_relational_gt_subset_pred_angle_vals: list[float] = field(default_factory=list)
    motion_relational_dim_pred_vals: list[int] = field(default_factory=list)
    motion_relational_dim_gt_vals: list[int] = field(default_factory=list)
    motion_full_dim_pred_vals: list[int] = field(default_factory=list)
    motion_full_dim_gt_vals: list[int] = field(default_factory=list)

    def extend(self, other: "ResponseIntentBatchStats") -> None:
        """Merge another per-worker stats buffer into this one."""
        self.reward_vals.extend(other.reward_vals)
        self.response_full_angle_vals.extend(other.response_full_angle_vals)
        self.response_relational_angle_vals.extend(
            other.response_relational_angle_vals
        )
        self.motion_full_angle_vals.extend(other.motion_full_angle_vals)
        self.motion_relational_angle_vals.extend(
            other.motion_relational_angle_vals
        )
        self.leak_relational_angle_vals.extend(other.leak_relational_angle_vals)
        self.motion_relational_pred_subset_gt_angle_vals.extend(
            other.motion_relational_pred_subset_gt_angle_vals
        )
        self.motion_relational_gt_subset_pred_angle_vals.extend(
            other.motion_relational_gt_subset_pred_angle_vals
        )
        self.motion_relational_dim_pred_vals.extend(
            other.motion_relational_dim_pred_vals
        )
        self.motion_relational_dim_gt_vals.extend(
            other.motion_relational_dim_gt_vals
        )
        self.motion_full_dim_pred_vals.extend(other.motion_full_dim_pred_vals)
        self.motion_full_dim_gt_vals.extend(other.motion_full_dim_gt_vals)

    def add_h_theta(self, metrics: HThetaMetrics) -> None:
        """Record one successful Htheta/relation-intent computation."""
        self.response_full_angle_vals.append(metrics.response_full_angle)
        self.response_relational_angle_vals.append(metrics.response_relational_angle)
        self.motion_full_angle_vals.append(metrics.motion_full_angle)
        self.motion_relational_angle_vals.append(metrics.motion_relational_angle)
        self.leak_relational_angle_vals.append(metrics.leak_relational_angle)
        self.motion_relational_pred_subset_gt_angle_vals.append(
            metrics.motion_relational_pred_subset_gt_angle
        )
        self.motion_relational_gt_subset_pred_angle_vals.append(
            metrics.motion_relational_gt_subset_pred_angle
        )
        self.motion_relational_dim_pred_vals.append(
            metrics.motion_relational_dim_pred
        )
        self.motion_relational_dim_gt_vals.append(metrics.motion_relational_dim_gt)
        self.motion_full_dim_pred_vals.append(metrics.motion_full_dim_pred)
        self.motion_full_dim_gt_vals.append(metrics.motion_full_dim_gt)

    def to_metrics_dict(self) -> dict[str, float | int]:
        """Return flat scalar metrics under ``intent_*`` keys.

        The composer flattens reward metrics without further prefixing, so
        each key already encodes its owning reward (``intent_``) and the
        metric name without redundant repetition.
        """
        return {
            "intent_reward_mean": _mean(self.reward_vals),
            "intent_reward_median": _median(self.reward_vals),
            "intent_response_full_angle_mean": _mean(
                self.response_full_angle_vals
            ),
            "intent_response_full_angle_median": _median(
                self.response_full_angle_vals
            ),
            "intent_response_relational_angle_mean": _mean(
                self.response_relational_angle_vals
            ),
            "intent_response_relational_angle_median": _median(
                self.response_relational_angle_vals
            ),
            "intent_motion_full_angle_mean": _mean(self.motion_full_angle_vals),
            "intent_motion_relational_angle_mean": _mean(
                self.motion_relational_angle_vals
            ),
            "intent_leak_relational_angle_mean": _mean(
                self.leak_relational_angle_vals
            ),
            "intent_motion_relational_pred_subset_gt_angle_mean": _mean(
                self.motion_relational_pred_subset_gt_angle_vals
            ),
            "intent_motion_relational_gt_subset_pred_angle_mean": _mean(
                self.motion_relational_gt_subset_pred_angle_vals
            ),
            "intent_motion_relational_dim_pred_mean": _mean(
                self.motion_relational_dim_pred_vals
            ),
            "intent_motion_relational_dim_gt_mean": _mean(
                self.motion_relational_dim_gt_vals
            ),
            "intent_motion_full_dim_pred_mean": _mean(
                self.motion_full_dim_pred_vals
            ),
            "intent_motion_full_dim_gt_mean": _mean(self.motion_full_dim_gt_vals),
        }


@dataclass(frozen=True)
class SubspaceAngleResult:
    """Padded principal-angle comparison between two orthonormal subspaces.

    Used as the uniform return type for ``_motion_full_angle``,
    ``_motion_relational_angle``, and ``_leak_relational_angle`` so all three
    helpers share one calling convention.

    Fields:
        angle: Primary angle scalar in [0, 1].  Squaring at the caller is
            trivial, so the redundant ``angle_sq`` field is intentionally not
            stored; use ``result.angle ** 2`` when you need the squared form.
        dim_pred / dim_gt: Subspace dimensions involved in the comparison.
            For motion-vs-motion helpers both are motion-subspace dims; for
            relation-leak ``dim_pred`` is the Pred motion-subspace dim and
            ``dim_gt`` is the rank of the GT relation row space.
        pred_subset_gt_angle / gt_subset_pred_angle: Asymmetric subset-angle
            decomposition.  Populated only by ``_motion_relational_angle``
            (the only helper whose mode menu exposes asymmetric variants);
            ``None`` for the symmetric / single-side helpers, where the
            decomposition is not a meaningful diagnostic.
    """

    angle: float
    dim_pred: int
    dim_gt: int
    pred_subset_gt_angle: float | None = None
    gt_subset_pred_angle: float | None = None


def _motion_full_angle(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_native: NativeResponse,
    eps: float,
    *,
    J_phi_pred: torch.Tensor | None = None,
    J_phi_gt: torch.Tensor | None = None,
    gt_motion_full: tuple[torch.Tensor, int] | None = None,
) -> SubspaceAngleResult:
    """Padded principal-angle metric between full motion subspaces.

    ``R_C`` maps native dimensional-parameter changes to primitive motion.  Its
    raw primitive coordinates mix positions, radii, angles, etc., so canonical
    Htheta compares the span after projection by ``J_phi`` into the common
    pairwise-distance observable coordinate space:

        W_C = range(J_phi R_C)

    This is the "full" motion subspace — it includes the contribution of every
    column of ``R_C`` (dimensional and relational null-space alike).  The
    relation-only counterpart (``_motion_relational_angle``) uses ``null(J_R)``
    instead.

    Missing dimensions are treated as 90-degree principal angles by normalizing
    with ``max(dim_P, dim_G)``.  This keeps dimension-dropout failures visible
    while putting equal-dimension angular mismatch on a [0, 1] scale.
    """
    U_W_P, dim_P_full = _motion_full_subspace(
        pred_sketch, pred_native, eps, J_phi=J_phi_pred
    )
    if gt_motion_full is not None:
        U_W_G, dim_G_full = gt_motion_full
    else:
        U_W_G, dim_G_full = _motion_full_subspace(
            gt_sketch, gt_native, eps, J_phi=J_phi_gt
        )

    # Both sides have no observable motion (fully constrained with no
    # dimensional drivers) — their zero subspaces trivially agree.
    if dim_P_full == 0 and dim_G_full == 0:
        return SubspaceAngleResult(angle=0.0, dim_pred=0, dim_gt=0)

    cos2_sum = _principal_cos2_sum(U_W_P, U_W_G)
    return SubspaceAngleResult(
        angle=_padded_subspace_angle(cos2_sum, dim_P_full, dim_G_full),
        dim_pred=int(dim_P_full),
        dim_gt=int(dim_G_full),
    )


def _motion_relational_angle(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    context: SketchContext,
    eps: float,
    *,
    angle_mode: str = "symmetric",
    reverse_weight: float = 0.5,
    J_phi_pred: torch.Tensor | None = None,
    J_phi_gt: torch.Tensor | None = None,
    gt_native: NativeResponse | None = None,
    gt_motion_relational: tuple[torch.Tensor, int] | None = None,
) -> SubspaceAngleResult:
    """Compare relation-only motion subspaces in observable space.

    The reward-oriented sibling of ``_motion_full_angle``: dimensional drivers
    are deliberately ignored and only the primitive motions allowed by the
    sketch's relational constraints are compared.  Each subspace is evaluated
    at its own native state (pred at ``pred_native.state``, GT at
    ``gt_native.state``) and mapped through ``J_phi`` into the shared
    pairwise-distance observable space, so the principal-angle SVD lives in a
    common coordinate system even when GT carries extra construction geometry
    (a larger primitive parameter vector).  When ``gt_native`` is omitted we
    fall back to ``pred_native.state`` for backward compatibility; this is
    numerically identical whenever pred and GT share the same primitives (the
    geometry is overwritten with the original), but mismatches and errors out
    when GT added construction primitives, which is why the GT state is
    preferred.

    The returned ``angle`` is the *raw* motion-relational angle whose ceiling
    depends on the comparison mode; higher-level callers normalize it into
    [0, 1] via ``_normalized_motion_relational_angle`` before combining with
    the leak term.

    Args:
        pred_sketch / gt_sketch: Sketches to compare.
        pred_native: NativeResponse providing the configuration at which
            Jacobians are evaluated.
        context: Sketch tokenization / solver context.
        eps: Numerical rank threshold for SVD and null-space computations.
        angle_mode: How the two inclusions are combined into a single raw
            angle scalar.  ``symmetric`` uses both inclusions equally;
            ``pred_subset_gt`` keeps only the Pred ⊄ GT direction;
            ``mixed_subset`` weights the GT ⊄ Pred direction by
            ``reverse_weight``.
        reverse_weight: Weight on the GT-into-Pred subset angle in
            ``mixed_subset`` mode.
    """
    if angle_mode not in MOTION_RELATIONAL_ANGLE_MODES:
        raise ValueError(
            f"unknown motion-relational angle mode: {angle_mode!r}"
        )

    state = pred_native.state
    U_pred, dim_pred = _motion_relational_subspace(
        pred_sketch, state, context, eps, J_phi=J_phi_pred
    )
    if gt_motion_relational is not None:
        U_gt, dim_gt = gt_motion_relational
    else:
        gt_state = gt_native.state if gt_native is not None else state
        U_gt, dim_gt = _motion_relational_subspace(
            gt_sketch, gt_state, context, eps, J_phi=J_phi_gt
        )

    cos2_sum = _principal_cos2_sum(U_pred, U_gt)

    pred_subset_gt_sq = 0.0
    if dim_pred > 0:
        pred_subset_gt_sq = (dim_pred - cos2_sum) / float(dim_pred)

    gt_subset_pred_sq = 0.0
    if dim_gt > 0:
        gt_subset_pred_sq = (dim_gt - cos2_sum) / float(dim_gt)

    d_norm = float(max(dim_pred, dim_gt, 1))
    if angle_mode == "symmetric":
        angle_sq = (dim_pred + dim_gt - 2.0 * cos2_sum) / d_norm
    elif angle_mode == "pred_subset_gt":
        angle_sq = pred_subset_gt_sq
    else:
        angle_sq = pred_subset_gt_sq + reverse_weight * gt_subset_pred_sq

    return SubspaceAngleResult(
        angle=math.sqrt(max(angle_sq, 0.0)),
        dim_pred=int(dim_pred),
        dim_gt=int(dim_gt),
        pred_subset_gt_angle=math.sqrt(max(pred_subset_gt_sq, 0.0)),
        gt_subset_pred_angle=math.sqrt(max(gt_subset_pred_sq, 0.0)),
    )


def _leak_relational_angle(
    gt_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    context: SketchContext,
    eps: float,
    *,
    gt_relation_nullspace: tuple[torch.Tensor, int] | None = None,
) -> SubspaceAngleResult:
    """RMS cosine between Pred motion and GT relation-forbidden directions.

    ``U_P`` spans the motions Pred allows through its dimensional drivers.  The
    row space of GT relation Jacobians spans directions that should be normal to
    any relation-preserving motion.  Their overlap is therefore a scale-free
    measure of relational design-intent leakage:

        leak_relational_angle^2 = ||orth(row(J_R_G))^T U_P||_F^2 / dim(U_P)

    Jacobians are evaluated at Pred's primitive state, whose geometry has been
    overwritten with the original sketch geometry during reconstruction.  Pred
    information is fully carried by ``pred_native``; the Pred sketch object is
    not needed here.

    In the returned ``SubspaceAngleResult``, ``dim_pred`` is the dimension of
    the Pred motion subspace ``U_P`` and ``dim_gt`` is the rank of the GT
    relation row space (``rank(J_R_G)``).  Subset-angle fields are ``None``
    since leak is a one-sided (motion-vs-row-space) comparison.
    """
    U_P_prim, dim_P_prim = _orthonormal_range(pred_native.R, eps)
    if gt_relation_nullspace is not None:
        N_G_prim, dim_N = gt_relation_nullspace
    else:
        gt_rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]
        J_R, n_R = _stack_relation_jacobians_at(gt_rels, pred_native.state, context)
        if J_R is None or n_R == 0:
            N_G_prim = torch.zeros(pred_native.state.num_dof(), 0, dtype=pred_native.R.dtype)
            dim_N = 0
        else:
            N_G_prim, dim_N = _orthonormal_range(J_R.T, eps)
    leak_angle = 0.0
    if dim_P_prim > 0 and dim_N > 0:
        overlap = N_G_prim.T @ U_P_prim
        leak_angle = _leak_angle_from_overlap(overlap, dim_P_prim)
    return SubspaceAngleResult(
        angle=leak_angle,
        dim_pred=int(dim_P_prim),
        dim_gt=int(dim_N),
    )


def compute_h_theta_metrics(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    context: SketchContext,
    eps: float,
    motion_relational_angle_mode: str = "symmetric",
    motion_relational_reverse_weight: float = 0.5,
    *,
    J_phi_pred: torch.Tensor | None = None,
    J_phi_gt: torch.Tensor | None = None,
    gt_motion_full: tuple[torch.Tensor, int] | None = None,
    gt_motion_relational: tuple[torch.Tensor, int] | None = None,
    gt_relation_nullspace: tuple[torch.Tensor, int] | None = None,
) -> HThetaMetrics:
    """Return canonical Htheta plus the RL-oriented relation-intent substitute.

    Public entry point for the Hθ family: this is the single implementation used
    both by the RLOO reward (training) and by the offline evaluator
    (``src/eval/evaluate.py``), so the numbers the evaluator reports are exactly
    the quantities the model optimizes.  It stays in the reward layer because
    ``motion_relational_angle`` uses the reward's symmetric normalization
    (``_motion_relational_angle`` / ``_RESPONSE_ANGLE_NORM``), a training-signal
    choice that coincides with the paper's padded ``theta_motion^rel`` only when
    the pred / GT relational-subspace dimensions match.

    Two related scalars are produced:

    - ``response_full_angle`` is the paper-style diagnostic:
      RMS of full motion-subspace mismatch and relation leak.
    - ``response_relational_angle`` is the reward default:
      RMS of relation-only motion mismatch and relation leak.

    The second scalar deliberately avoids directly rewarding agreement with one
    annotator's dimensional constraint placement.  Dimensional parsimony and FC
    are handled by sibling rewards in the composer.

    All five public angles are kept on the [0, 1] scale:

    - ``motion_full_angle`` and ``leak_relational_angle`` come straight from
      padded principal-angle / RMS-cosine formulas in [0, 1].
    - ``motion_relational_angle`` divides the raw motion-relational angle
      returned by ``_motion_relational_angle`` by the mode-specific maximum so
      that orthogonal subspaces map to 1.  Maxima are: pair-specific symmetric
      maximum for ``symmetric``, ``sqrt(1 + reverse_weight)`` for
      ``mixed_subset``, and 1 for ``pred_subset_gt``.
    - ``response_*_angle`` divides ``sqrt(motion^2 + leak^2)`` by ``sqrt(2)``
      to put the combined RMS angle on the same [0, 1] scale as the components.
    """
    motion_full = _motion_full_angle(
        pred_sketch,
        gt_sketch,
        pred_native,
        gt_native,
        eps,
        J_phi_pred=J_phi_pred,
        J_phi_gt=J_phi_gt,
        gt_motion_full=gt_motion_full,
    )
    leak = _leak_relational_angle(
        gt_sketch,
        pred_native,
        context,
        eps,
        gt_relation_nullspace=gt_relation_nullspace,
    )
    motion_rel = _motion_relational_angle(
        pred_sketch,
        gt_sketch,
        pred_native,
        context,
        eps,
        angle_mode=motion_relational_angle_mode,
        reverse_weight=motion_relational_reverse_weight,
        J_phi_pred=J_phi_pred,
        J_phi_gt=J_phi_gt,
        gt_native=gt_native,
        gt_motion_relational=gt_motion_relational,
    )
    motion_relational_angle = _normalized_motion_relational_angle(
        motion_rel.angle,
        motion_relational_angle_mode,
        motion_rel.dim_pred,
        motion_rel.dim_gt,
        motion_relational_reverse_weight,
    )
    return HThetaMetrics(
        response_full_angle=math.sqrt(
            motion_full.angle ** 2 + leak.angle ** 2
        ) / _RESPONSE_ANGLE_NORM,
        response_relational_angle=math.sqrt(
            motion_relational_angle ** 2 + leak.angle ** 2
        ) / _RESPONSE_ANGLE_NORM,
        motion_full_angle=motion_full.angle,
        motion_relational_angle=motion_relational_angle,
        leak_relational_angle=leak.angle,
        motion_relational_pred_subset_gt_angle=float(
            motion_rel.pred_subset_gt_angle or 0.0
        ),
        motion_relational_gt_subset_pred_angle=float(
            motion_rel.gt_subset_pred_angle or 0.0
        ),
        motion_relational_dim_pred=motion_rel.dim_pred,
        motion_relational_dim_gt=motion_rel.dim_gt,
        motion_full_dim_pred=motion_full.dim_pred,
        motion_full_dim_gt=motion_full.dim_gt,
    )


@dataclass(frozen=True)
class ResponseMetricAsyncHandle:
    """Deferred Htheta-evaluator work that needs solver status at collect time."""

    ctx: RewardContext
    prim_np: np.ndarray
    cons_np: np.ndarray
    length: int


class ResponseMetricEvaluator(BaseReward):
    """Producer for cached Htheta metrics and ``intent_*`` diagnostics.

    This module performs the expensive sketch/native-response/Htheta pipeline
    once, writes per-sample ``HThetaMetrics`` into ``ctx.h_theta_metrics``, and
    returns zero reward scores.  ``MotionAngleReward`` and ``LeakAngleReward``
    are the reward-bearing consumers.
    """

    name = "response_eval"
    is_async = True

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        motion_relational_angle_mode: str = "symmetric",
        motion_relational_reverse_weight: float = 0.5,
        gate: str = "converged",
        num_workers: int = 0,
        weight: float = 1.0,
        enabled: bool = True,
        **unused_legacy_kwargs: Any,
    ):
        super().__init__(weight=weight, enabled=enabled)
        if motion_relational_angle_mode not in MOTION_RELATIONAL_ANGLE_MODES:
            raise ValueError(
                "motion_relational_angle_mode must be one of "
                f"{sorted(MOTION_RELATIONAL_ANGLE_MODES)}, got "
                f"{motion_relational_angle_mode!r}"
            )
        if motion_relational_reverse_weight < 0.0:
            raise ValueError(
                "motion_relational_reverse_weight must be non-negative, got "
                f"{motion_relational_reverse_weight}"
            )
        if gate not in _GATE_MODES:
            raise ValueError(f"gate must be one of {_GATE_MODES}, got {gate!r}")
        if unused_legacy_kwargs:
            logger.debug(
                "ResponseMetricEvaluator ignoring legacy reward-only kwargs: %s",
                sorted(unused_legacy_kwargs),
            )

        self.context = context
        self.eps = eps
        self.motion_relational_angle_mode = motion_relational_angle_mode
        self.motion_relational_reverse_weight = motion_relational_reverse_weight
        self.gate = gate
        self.num_workers = max(0, int(num_workers))
        self.tokenizer = SketchTokenizer(context)
        self._executor: ProcessPoolExecutor | None = None

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_response_metric_worker,
                initargs=(
                    self.context,
                    self.eps,
                    self.motion_relational_angle_mode,
                    self.motion_relational_reverse_weight,
                    self.gate,
                ),
            )
        return self._executor

    def _reset_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is None:
            return
        processes = getattr(executor, "_processes", None) or {}
        for proc in list(processes.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

    def _gate_pass(self, status: str | None) -> bool:
        return _status_passes_gate(status, self.gate)

    def _compute_sample_metrics(
        self,
        ctx: RewardContext,
        i: int,
        length: int,
        prim_i: np.ndarray,
        cons_i_raw: np.ndarray,
        gt_sketch: datalib.Sketch,
        solver_status: str | None,
        stats: ResponseIntentBatchStats,
    ) -> HThetaMetrics | None:
        """Compute Htheta metrics for one sample, or ``None`` when gated/invalid."""
        if not self._gate_pass(solver_status):
            return None

        cons_i, _first_stop_idx = first_stop_and_pad(cons_i_raw, length)
        try:
            pred_sketch, pred_native = get_or_build_pred_native(
                ctx,
                i,
                self.tokenizer,
                cons_i,
                prim_i,
                gt_sketch,
                self.context,
                self.eps,
            )
        except Exception as exc:
            logger.debug(
                "response metric evaluator: skipped sample %d "
                "(build failure: %s)",
                i,
                exc,
            )
            return None

        try:
            gt_native = get_or_compute_gt_native(
                ctx, gt_sketch, self.context, self.eps
            )
            J_phi_pred = get_or_compute_j_phi(
                ctx, pred_sketch, pred_native.R.dtype
            )
            J_phi_gt = get_or_compute_j_phi(
                ctx, gt_sketch, gt_native.R.dtype
            )
            gt_motion_full = get_or_compute_gt_motion_full_subspace(
                ctx, gt_sketch, gt_native, self.eps, J_phi_gt
            )
            gt_motion_relational = get_or_compute_gt_motion_relational_subspace(
                ctx, gt_sketch, gt_native, self.context, self.eps, J_phi_gt
            )
            gt_relation_nullspace = get_or_compute_gt_relation_nullspace(
                ctx, gt_sketch, gt_native, self.context, self.eps
            )
            h_metrics = compute_h_theta_metrics(
                pred_sketch,
                pred_native,
                gt_sketch,
                gt_native,
                self.context,
                self.eps,
                self.motion_relational_angle_mode,
                self.motion_relational_reverse_weight,
                J_phi_pred=J_phi_pred,
                J_phi_gt=J_phi_gt,
                gt_motion_full=gt_motion_full,
                gt_motion_relational=gt_motion_relational,
                gt_relation_nullspace=gt_relation_nullspace,
            )
        except Exception as exc:
            logger.debug(
                "response metric evaluator: skipped sample %d "
                "(angle computation failure: %s)",
                i,
                exc,
            )
            return None

        stats.add_h_theta(h_metrics)
        return h_metrics

    def _score_serial_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str | None],
        length: int,
    ) -> tuple[dict[int, HThetaMetrics], ResponseIntentBatchStats]:
        metric_map: dict[int, HThetaMetrics] = {}
        stats = ResponseIntentBatchStats()

        for i in range(cons_np.shape[0]):
            metrics = self._compute_sample_metrics(
                ctx,
                i,
                length,
                prim_np[i],
                cons_np[i],
                ctx.sketch_original_list[i],
                statuses[i],
                stats,
            )
            if metrics is not None:
                metric_map[i] = metrics

        return metric_map, stats

    def _score_parallel_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str | None],
        length: int,
    ) -> tuple[dict[int, HThetaMetrics], ResponseIntentBatchStats]:
        metric_map: dict[int, HThetaMetrics] = {}
        stats = ResponseIntentBatchStats()

        grouped: dict[int, tuple[datalib.Sketch, list[int]]] = {}
        for i, sketch in enumerate(ctx.sketch_original_list):
            key = id(sketch)
            if key not in grouped:
                grouped[key] = (sketch, [])
            grouped[key][1].append(i)

        if len(grouped) <= 1:
            return self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        payloads = []
        for sketch, indices in grouped.values():
            payloads.append(
                (
                    indices,
                    prim_np[indices].copy(),
                    cons_np[indices].copy(),
                    sketch,
                    [statuses[i] for i in indices],
                    length,
                )
            )

        try:
            executor = self._get_executor()
            futures = [
                executor.submit(_score_response_metric_group, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                part_metrics, part_stats = future.result()
                stats.extend(part_stats)
                for i, metrics in part_metrics:
                    metric_map[i] = metrics
        except Exception as exc:
            logger.warning(
                "parallel response metric evaluator failed; "
                "falling back to serial: %s",
                exc,
            )
            self._reset_executor()
            return self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        return metric_map, stats

    def _compute_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str | None],
        length: int,
    ) -> RewardOutput:
        B = cons_np.shape[0]
        if self.num_workers > 0:
            metric_map, stats = self._score_parallel_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )
        else:
            metric_map, stats = self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        ctx.h_theta_metrics.clear()
        for i in range(B):
            ctx.h_theta_metrics[i] = metric_map.get(i)

        scores = torch.zeros((B, length), dtype=torch.float32)
        return RewardOutput(scores=scores, metrics=stats.to_metrics_dict())

    def submit_async(self, ctx: RewardContext) -> ResponseMetricAsyncHandle:
        """Capture CPU arrays now; actual gated work waits for solver status."""
        return ResponseMetricAsyncHandle(
            ctx=ctx,
            prim_np=ctx.prim_tokens.detach().cpu().numpy(),
            cons_np=ctx.cons_tokens.detach().cpu().numpy(),
            length=int(ctx.cons_tokens.shape[1]),
        )

    def collect_results(self, handle: ResponseMetricAsyncHandle) -> RewardOutput:
        ctx = handle.ctx
        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * handle.cons_np.shape[0]
        return self._compute_from_arrays(
            ctx,
            handle.prim_np,
            handle.cons_np,
            list(statuses),
            handle.length,
        )

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """Synchronous path used by validation and direct unit tests."""
        B, L = ctx.cons_tokens.shape
        assert len(ctx.sketch_original_list) == B

        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * B
        return self._compute_from_arrays(
            ctx,
            ctx.prim_tokens.detach().cpu().numpy(),
            ctx.cons_tokens.detach().cpu().numpy(),
            list(statuses),
            L,
        )


class _CachedAngleReward(BaseReward):
    """Shared implementation for Htheta cache consumers."""

    metric_attr = ""

    def __init__(
        self,
        coeff: float,
        cap: float = 1.0,
        gate: str = "converged",
        shuffle_within_group: bool = False,
        weight: float = 1.0,
        enabled: bool = True,
        context: SketchContext | None = None,
        cap_h_angle: float | None = None,
        **unused_kwargs: Any,
    ):
        del context
        if unused_kwargs:
            logger.debug(
                "%s ignoring producer-only kwargs: %s",
                type(self).__name__,
                sorted(unused_kwargs),
            )
        if cap_h_angle is not None:
            cap = cap_h_angle
        if gate not in _CONSUMER_GATE_MODES:
            raise ValueError(
                f"gate must be one of {_CONSUMER_GATE_MODES}, got {gate!r}"
            )
        super().__init__(weight=weight, enabled=enabled)
        self.coeff = float(coeff)
        self.cap = float(cap)
        self.gate = gate
        self.shuffle_within_group = bool(shuffle_within_group)

    def _gate_pass(self, status: str | None) -> bool:
        return _status_passes_gate(status, self.gate)

    def _gate_weight(self, status: str | None, rate: float | None) -> float:
        """Map solver state to a [0, 1] reward multiplier.

        ``always``/``converged``/``fc`` are 0/1 step gates; ``dof`` is the
        continuous ``dof_reduction_rate`` (0 = no/invalid closure, 1 = FC).
        """
        if self.gate == "dof":
            return 0.0 if rate is None else float(min(max(rate, 0.0), 1.0))
        return 1.0 if _status_passes_gate(status, self.gate) else 0.0

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        B, L = ctx.cons_tokens.shape
        scores = torch.zeros((B, L), dtype=torch.float32)
        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * B
        rates = ctx.per_sample_dof_reduction_rate
        if self.gate == "dof" and rates is None:
            logger.debug(
                "%s gate='dof' but ctx.per_sample_dof_reduction_rate is unset "
                "(SolverReward must run first); contributing zero reward.",
                type(self).__name__,
            )

        cons_np = ctx.cons_tokens.detach().cpu().numpy()
        reward_vals: list[float] = []
        angle_vals: list[float] = []
        cache = ctx.h_theta_metrics
        for i in range(B):
            rate_i = rates[i] if rates is not None else None
            weight = self._gate_weight(statuses[i], rate_i)
            if weight <= 0.0:
                continue
            h_metrics = cache.get(i)
            if h_metrics is None:
                continue
            angle = getattr(h_metrics, self.metric_attr, None)
            if angle is None:
                continue
            angle_f = float(angle)
            sample_reward = weight * _linear_bonus(angle_f, self.cap, self.coeff)
            _cons_i, first_stop_idx = first_stop_and_pad(cons_np[i], L)
            scores[i, first_stop_idx] += sample_reward
            reward_vals.append(sample_reward)
            angle_vals.append(angle_f)

        if self.shuffle_within_group:
            scores = _shuffle_scores_within_group(scores, ctx)

        metrics = {
            f"{self.name}_reward_mean": _mean(reward_vals),
            f"{self.name}_reward_median": _median(reward_vals),
            f"{self.name}_{self.metric_attr}_mean": _mean(angle_vals),
        }
        return RewardOutput(scores=scores, metrics=metrics)


class MotionAngleReward(_CachedAngleReward):
    """Reward consumer for ``motion_relational_angle``."""

    name = "motion_intent"
    metric_attr = "motion_relational_angle"

    def __init__(self, coeff: float, gate: str = "converged", **kwargs: Any):
        super().__init__(coeff=coeff, gate=gate, **kwargs)


class LeakAngleReward(_CachedAngleReward):
    """Reward consumer for ``leak_relational_angle``."""

    name = "leak_intent"
    metric_attr = "leak_relational_angle"

    def __init__(self, coeff: float, gate: str = "fc", **kwargs: Any):
        super().__init__(coeff=coeff, gate=gate, **kwargs)


_PREFIX_DELTA_MODES = ("positive", "signed")


def _iter_constraint_block_prefixes(
    cons_i: np.ndarray,
    first_stop_idx: int,
    prefix_len: int,
    block_size: int,
    max_prefixes: int | None,
):
    """Yield ``(block_idx, block_tail, prefix_cons)`` for each completed block.

    ``cons_i`` must already be truncated by :func:`first_stop_and_pad` so no STOP
    appears before ``first_stop_idx`` and the constraint body is a contiguous run
    of ``block_size``-token blocks starting at ``prefix_len``.  For each completed
    block ``k`` (0-indexed), a prefix sequence is produced that keeps blocks
    ``0..k`` and closes immediately after block ``k`` with a STOP at
    ``block_tail + 1`` (the remainder padded).  ``block_tail`` is the last token
    of block ``k``, i.e. the placement target for that block's prefix reward —
    matching the SolverReward convention of scoring a constraint at its own
    block-end token.
    """
    if block_size <= 0 or first_stop_idx <= prefix_len:
        return
    n_blocks = (first_stop_idx - prefix_len) // block_size
    if max_prefixes is not None:
        n_blocks = min(n_blocks, int(max_prefixes))
    L = cons_i.shape[0]
    for k in range(n_blocks):
        block_tail = prefix_len + (k + 1) * block_size - 1
        stop_pos = block_tail + 1
        if stop_pos >= L:
            break
        prefix = cons_i.copy()
        prefix[stop_pos] = int(ConstraintToken.Stop)
        if stop_pos + 1 < L:
            prefix[stop_pos + 1:] = int(ConstraintToken.Pad)
        yield k, block_tail, prefix


def _prefix_motion_deltas(
    angles: Sequence[float],
    block_tails: Sequence[int],
    coeff: float,
    cap: float,
    delta_mode: str,
    min_delta: float,
) -> tuple[list[tuple[int, float]], list[float]]:
    """Convert a prefix angle trajectory into ``(tail, delta)`` placements.

    ``Phi_t = coeff * (1 - clip(angle_t)/cap)`` and the per-block reward is the
    delta against the previous prefix potential (``Phi_0 = 0`` for the empty
    prefix).  ``positive`` clips to ``max(0, delta - min_delta)`` (non-telescoping,
    rewards each improving block); ``signed`` keeps the sign and zeroes only
    sub-``min_delta`` moves (telescopes to ``Phi_T``).  Returns the non-zero
    placements and the list of strictly-positive deltas (for logging).
    """
    prev_phi = 0.0
    placements: list[tuple[int, float]] = []
    pos_deltas: list[float] = []
    for angle, tail in zip(angles, block_tails):
        phi = _linear_bonus(angle, cap, coeff)
        delta = phi - prev_phi
        prev_phi = phi
        if delta_mode == "positive":
            delta = max(0.0, delta - min_delta)
        elif abs(delta) < min_delta:
            delta = 0.0
        if delta != 0.0:
            placements.append((tail, delta))
            if delta > 0.0:
                pos_deltas.append(delta)
    return placements, pos_deltas


def _prefix_leak_penalties(
    leaks: Sequence[float],
    is_dimensional: Sequence[bool],
    block_tails: Sequence[int],
    coeff: float,
    cap: float,
    min_delta: float,
) -> tuple[list[tuple[int, float]], list[float]]:
    """Convert a prefix leak trajectory into ``(tail, penalty)`` placements.

    Unlike motion, leak must NOT be rewarded as a bonus: ``leak=0`` is trivially
    reached by emitting no dimensions, so a symmetric/positive-delta bonus would
    reward the under-constrained corner.  Instead this is an increase-only
    penalty applied at *dimensional* block tails:

        penalty_t = -coeff * max(0, clip(leak_t) - clip(leak_{t-1}) - min_delta)

    A dimensional constraint that drives a direction GT closes relationally
    raises leak and is penalized at its own block tail; a legitimate dimension
    (orthogonal to GT's relation row space) leaves leak unchanged and is not.
    ``prev`` is the previous prefix's (clipped) leak regardless of block type —
    leak is measured on the whole prefix — but only dimensional blocks are
    charged.  ``prev`` starts at 0 so no-dims prefixes never earn a bonus.
    Returns the non-zero placements and the penalty magnitudes (for logging).
    """
    prev = 0.0
    placements: list[tuple[int, float]] = []
    pen_mags: list[float] = []
    for leak, dim, tail in zip(leaks, is_dimensional, block_tails):
        leak_c = min(max(leak, 0.0), cap) if cap > 0.0 else max(leak, 0.0)
        if dim:
            penalty = -coeff * max(0.0, leak_c - prev - min_delta)
            if penalty != 0.0:
                placements.append((tail, penalty))
                pen_mags.append(-penalty)
        prev = leak_c
    return placements, pen_mags


@dataclass(frozen=True)
class PrefixBlockMetric:
    """Per-prefix response metrics for one completed constraint block.

    Produced by ``PrefixResponseMetricEvaluator`` (one entry per completed block)
    and consumed by ``PrefixMotionAngleReward`` / ``PrefixLeakAngleReward``.

    Fields:
        block_tail: token index of the block's last token (reward placement
            target).
        is_dimensional: whether the block this prefix newly added is a
            dimensional constraint (Distance/DistanceX/Y/Radius/Diameter/Angle).
            Only dimensional blocks receive a prefix leak penalty.
        motion_relational_angle: normalized relation-only motion angle of the
            prefix vs the pseudo target, in [0, 1].
        leak_relational_angle: relation-leak angle of the prefix, in [0, 1].
    """

    block_tail: int
    is_dimensional: bool
    motion_relational_angle: float
    leak_relational_angle: float


@dataclass
class PrefixEvalBatchStats:
    """Aggregation buffer for ``PrefixResponseMetricEvaluator`` logging."""

    num_prefixes_vals: list[int] = field(default_factory=list)
    num_dimensional_vals: list[int] = field(default_factory=list)
    first_motion_vals: list[float] = field(default_factory=list)
    last_motion_vals: list[float] = field(default_factory=list)
    first_leak_vals: list[float] = field(default_factory=list)
    last_leak_vals: list[float] = field(default_factory=list)

    def extend(self, other: "PrefixEvalBatchStats") -> None:
        self.num_prefixes_vals.extend(other.num_prefixes_vals)
        self.num_dimensional_vals.extend(other.num_dimensional_vals)
        self.first_motion_vals.extend(other.first_motion_vals)
        self.last_motion_vals.extend(other.last_motion_vals)
        self.first_leak_vals.extend(other.first_leak_vals)
        self.last_leak_vals.extend(other.last_leak_vals)

    def add_sample(self, blocks: list[PrefixBlockMetric]) -> None:
        if not blocks:
            return
        self.num_prefixes_vals.append(len(blocks))
        self.num_dimensional_vals.append(sum(1 for b in blocks if b.is_dimensional))
        self.first_motion_vals.append(blocks[0].motion_relational_angle)
        self.last_motion_vals.append(blocks[-1].motion_relational_angle)
        self.first_leak_vals.append(blocks[0].leak_relational_angle)
        self.last_leak_vals.append(blocks[-1].leak_relational_angle)

    def to_metrics_dict(self) -> dict[str, float]:
        return {
            "prefix_num_prefixes_mean": _mean(self.num_prefixes_vals),
            "prefix_num_dimensional_blocks_mean": _mean(self.num_dimensional_vals),
            "prefix_first_motion_angle_mean": _mean(self.first_motion_vals),
            "prefix_last_motion_angle_mean": _mean(self.last_motion_vals),
            "prefix_first_leak_mean": _mean(self.first_leak_vals),
            "prefix_last_leak_mean": _mean(self.last_leak_vals),
        }


@dataclass(frozen=True)
class PrefixEvalAsyncHandle:
    """Deferred prefix-eval work that needs solver status at collect time."""

    ctx: RewardContext
    prim_np: np.ndarray
    cons_np: np.ndarray
    length: int


class PrefixResponseMetricEvaluator(BaseReward):
    """Producer for per-prefix response metrics (motion + leak).

    The full-sequence producer (:class:`ResponseMetricEvaluator`) only caches the
    metric of the completed sketch.  This producer re-evaluates the relation-only
    motion angle *and* the relation-leak angle on the growing prefix after each
    completed constraint block, and caches the per-block trajectory in
    ``ctx.prefix_metrics[i]``.  It contributes zero reward; the reward-bearing
    consumers are :class:`PrefixMotionAngleReward` and
    :class:`PrefixLeakAngleReward`.

    Doing both angles in one pass matters because the dominant per-prefix cost is
    rebuilding the prefix sketch and its native response; motion and leak then
    share that work instead of paying it twice.  GT-side subspaces are pure
    functions of ``gt_sketch`` and are reused across prefixes and the K rollouts
    of an RLOO group via the shared ``ctx`` caches; ``J_phi`` for the predicted
    side is identical across prefixes (primitive geometry is overwritten with the
    original) and is computed once per sample.
    """

    name = "prefix_response_eval"
    is_async = True

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        motion_relational_angle_mode: str = "symmetric",
        motion_relational_reverse_weight: float = 0.5,
        gate: str = "converged",
        max_prefixes: int | None = None,
        num_workers: int = 0,
        weight: float = 1.0,
        enabled: bool = True,
        **unused_kwargs: Any,
    ):
        super().__init__(weight=weight, enabled=enabled)
        if gate not in _GATE_MODES:
            raise ValueError(f"gate must be one of {_GATE_MODES}, got {gate!r}")
        if motion_relational_angle_mode not in MOTION_RELATIONAL_ANGLE_MODES:
            raise ValueError(
                "motion_relational_angle_mode must be one of "
                f"{sorted(MOTION_RELATIONAL_ANGLE_MODES)}, got "
                f"{motion_relational_angle_mode!r}"
            )
        if motion_relational_reverse_weight < 0.0:
            raise ValueError(
                "motion_relational_reverse_weight must be non-negative, got "
                f"{motion_relational_reverse_weight}"
            )
        if max_prefixes is not None and max_prefixes <= 0:
            raise ValueError(
                f"max_prefixes must be positive or None, got {max_prefixes}"
            )
        if unused_kwargs:
            logger.debug(
                "PrefixResponseMetricEvaluator ignoring unexpected kwargs: %s",
                sorted(unused_kwargs),
            )

        self.context = context
        self.eps = eps
        self.mode = motion_relational_angle_mode
        self.reverse_weight = float(motion_relational_reverse_weight)
        self.gate = gate
        self.max_prefixes = None if max_prefixes is None else int(max_prefixes)
        self.num_workers = max(0, int(num_workers))
        self._block_size = int(context.align_constraint_token_length_to)
        self._prefix_len = 1
        self._dim_token_ids = set(int(t) for t in DIMENSIONAL_CONSTRAINT_TOKEN_IDS)
        self.tokenizer = SketchTokenizer(context)
        self._executor: ProcessPoolExecutor | None = None

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_prefix_eval_worker,
                initargs=(
                    self.context,
                    self.eps,
                    self.mode,
                    self.reverse_weight,
                    self.gate,
                    self.max_prefixes,
                ),
            )
        return self._executor

    def _reset_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is None:
            return
        processes = getattr(executor, "_processes", None) or {}
        for proc in list(processes.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

    def _gate_pass(self, status: str | None) -> bool:
        return _status_passes_gate(status, self.gate)

    def _block_is_dimensional(self, cons_i: np.ndarray, block_idx: int) -> bool:
        head = self._prefix_len + block_idx * self._block_size
        if head >= cons_i.shape[0]:
            return False
        return int(cons_i[head]) in self._dim_token_ids

    def _score_sample(
        self,
        ctx: RewardContext,
        i: int,
        length: int,
        prim_i: np.ndarray,
        cons_i_raw: np.ndarray,
        gt_sketch: datalib.Sketch,
        solver_status: str | None,
        stats: PrefixEvalBatchStats,
    ) -> list[PrefixBlockMetric] | None:
        """Return the per-block prefix metric list for one sample, or ``None``."""
        if not self._gate_pass(solver_status):
            return None

        cons_i, first_stop_idx = first_stop_and_pad(cons_i_raw, length)
        prefixes = list(
            _iter_constraint_block_prefixes(
                cons_i,
                first_stop_idx,
                self._prefix_len,
                self._block_size,
                self.max_prefixes,
            )
        )
        if not prefixes:
            return None

        try:
            gt_native = get_or_compute_gt_native(
                ctx, gt_sketch, self.context, self.eps
            )
            J_phi_gt = get_or_compute_j_phi(ctx, gt_sketch, gt_native.R.dtype)
            gt_motion_relational = get_or_compute_gt_motion_relational_subspace(
                ctx, gt_sketch, gt_native, self.context, self.eps, J_phi_gt
            )
            gt_relation_nullspace = get_or_compute_gt_relation_nullspace(
                ctx, gt_sketch, gt_native, self.context, self.eps
            )
        except Exception as exc:
            logger.debug(
                "prefix eval: skipped sample %d (gt failure: %s)", i, exc
            )
            return None

        J_phi_pred: torch.Tensor | None = None
        blocks: list[PrefixBlockMetric] = []
        for block_idx, block_tail, prefix_cons in prefixes:
            try:
                prefix_sketch = build_pred_sketch(
                    self.tokenizer, prefix_cons, prim_i, gt_sketch
                )
                prefix_native = compute_native_response(
                    prefix_sketch, self.context, eps=self.eps
                )
                if J_phi_pred is None:
                    # Geometry is overwritten with the original sketch, so J_phi
                    # is identical for every prefix of this sample; compute once.
                    J_phi_pred = pairwise_distance_observable(
                        prefix_sketch, dtype=prefix_native.R.dtype
                    ).to(dtype=prefix_native.R.dtype)
                motion_rel = _motion_relational_angle(
                    prefix_sketch,
                    gt_sketch,
                    prefix_native,
                    self.context,
                    self.eps,
                    angle_mode=self.mode,
                    reverse_weight=self.reverse_weight,
                    J_phi_pred=J_phi_pred,
                    gt_motion_relational=gt_motion_relational,
                )
                leak = _leak_relational_angle(
                    gt_sketch,
                    prefix_native,
                    self.context,
                    self.eps,
                    gt_relation_nullspace=gt_relation_nullspace,
                )
            except Exception as exc:
                logger.debug(
                    "prefix eval: skipped prefix of sample %d (%s)", i, exc
                )
                continue
            motion_angle = _normalized_motion_relational_angle(
                motion_rel.angle,
                self.mode,
                motion_rel.dim_pred,
                motion_rel.dim_gt,
                self.reverse_weight,
            )
            blocks.append(
                PrefixBlockMetric(
                    block_tail=int(block_tail),
                    is_dimensional=self._block_is_dimensional(cons_i, block_idx),
                    motion_relational_angle=motion_angle,
                    leak_relational_angle=leak.angle,
                )
            )

        if not blocks:
            return None
        stats.add_sample(blocks)
        return blocks

    def _score_serial_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str | None],
        length: int,
    ) -> tuple[dict[int, list[PrefixBlockMetric]], PrefixEvalBatchStats]:
        metric_map: dict[int, list[PrefixBlockMetric]] = {}
        stats = PrefixEvalBatchStats()
        for i in range(cons_np.shape[0]):
            blocks = self._score_sample(
                ctx,
                i,
                length,
                prim_np[i],
                cons_np[i],
                ctx.sketch_original_list[i],
                statuses[i],
                stats,
            )
            if blocks is not None:
                metric_map[i] = blocks
        return metric_map, stats

    def _score_parallel_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str | None],
        length: int,
    ) -> tuple[dict[int, list[PrefixBlockMetric]], PrefixEvalBatchStats]:
        metric_map: dict[int, list[PrefixBlockMetric]] = {}
        stats = PrefixEvalBatchStats()

        grouped: dict[int, tuple[datalib.Sketch, list[int]]] = {}
        for i, sketch in enumerate(ctx.sketch_original_list):
            key = id(sketch)
            if key not in grouped:
                grouped[key] = (sketch, [])
            grouped[key][1].append(i)

        if len(grouped) <= 1:
            return self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        payloads = []
        for sketch, indices in grouped.values():
            payloads.append(
                (
                    indices,
                    prim_np[indices].copy(),
                    cons_np[indices].copy(),
                    sketch,
                    [statuses[i] for i in indices],
                    length,
                )
            )

        try:
            executor = self._get_executor()
            futures = [
                executor.submit(_score_prefix_eval_group, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                part_metrics, part_stats = future.result()
                stats.extend(part_stats)
                for i, blocks in part_metrics:
                    metric_map[i] = blocks
        except Exception as exc:
            logger.warning(
                "parallel prefix eval failed; falling back to serial: %s", exc
            )
            self._reset_executor()
            return self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        return metric_map, stats

    def _compute_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str | None],
        length: int,
    ) -> RewardOutput:
        B = cons_np.shape[0]
        if self.num_workers > 0:
            metric_map, stats = self._score_parallel_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )
        else:
            metric_map, stats = self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        ctx.prefix_metrics.clear()
        for i in range(B):
            ctx.prefix_metrics[i] = metric_map.get(i)

        scores = torch.zeros((B, length), dtype=torch.float32)
        return RewardOutput(scores=scores, metrics=stats.to_metrics_dict())

    def submit_async(self, ctx: RewardContext) -> PrefixEvalAsyncHandle:
        return PrefixEvalAsyncHandle(
            ctx=ctx,
            prim_np=ctx.prim_tokens.detach().cpu().numpy(),
            cons_np=ctx.cons_tokens.detach().cpu().numpy(),
            length=int(ctx.cons_tokens.shape[1]),
        )

    def collect_results(self, handle: PrefixEvalAsyncHandle) -> RewardOutput:
        ctx = handle.ctx
        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * handle.cons_np.shape[0]
        return self._compute_from_arrays(
            ctx, handle.prim_np, handle.cons_np, list(statuses), handle.length
        )

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """Synchronous path used by validation and direct unit tests."""
        B, L = ctx.cons_tokens.shape
        assert len(ctx.sketch_original_list) == B
        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * B
        return self._compute_from_arrays(
            ctx,
            ctx.prim_tokens.detach().cpu().numpy(),
            ctx.cons_tokens.detach().cpu().numpy(),
            list(statuses),
            L,
        )


class _PrefixCachedReward(BaseReward):
    """Shared implementation for ``ctx.prefix_metrics`` consumers.

    Mirrors :class:`_CachedAngleReward`: a thin reward that reads the per-prefix
    trajectory produced by :class:`PrefixResponseMetricEvaluator`, applies a
    per-sample solver-status gate weight (``always``/``converged``/``fc`` step
    gates or the continuous ``dof`` ramp), and lets the subclass turn the
    trajectory into ``(block_tail, value)`` placements.
    """

    def __init__(
        self,
        coeff: float,
        cap: float = 0.70,
        gate: str = "converged",
        weight: float = 1.0,
        enabled: bool = True,
        context: SketchContext | None = None,
        **unused_kwargs: Any,
    ):
        del context
        if gate not in _CONSUMER_GATE_MODES:
            raise ValueError(
                f"gate must be one of {_CONSUMER_GATE_MODES}, got {gate!r}"
            )
        if unused_kwargs:
            logger.debug(
                "%s ignoring unexpected kwargs: %s",
                type(self).__name__,
                sorted(unused_kwargs),
            )
        super().__init__(weight=weight, enabled=enabled)
        self.coeff = float(coeff)
        self.cap = float(cap)
        self.gate = gate

    def _gate_weight(self, status: str | None, rate: float | None) -> float:
        if self.gate == "dof":
            return 0.0 if rate is None else float(min(max(rate, 0.0), 1.0))
        return 1.0 if _status_passes_gate(status, self.gate) else 0.0

    def _placements(
        self, blocks: list[PrefixBlockMetric]
    ) -> tuple[list[tuple[int, float]], list[float]]:
        """Return ``(placements, log_values)`` for one sample's trajectory."""
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        B, L = ctx.cons_tokens.shape
        scores = torch.zeros((B, L), dtype=torch.float32)
        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * B
        rates = ctx.per_sample_dof_reduction_rate
        if self.gate == "dof" and rates is None:
            logger.debug(
                "%s gate='dof' but ctx.per_sample_dof_reduction_rate is unset "
                "(SolverReward must run first); contributing zero reward.",
                type(self).__name__,
            )
        cache = ctx.prefix_metrics
        reward_vals: list[float] = []
        num_blocks_vals: list[int] = []
        for i in range(B):
            rate_i = rates[i] if rates is not None else None
            weight = self._gate_weight(statuses[i], rate_i)
            if weight <= 0.0:
                continue
            blocks = cache.get(i)
            if not blocks:
                continue
            placements, log_vals = self._placements(blocks)
            if not placements:
                continue
            for pos, val in placements:
                scores[i, pos] += weight * val
            reward_vals.append(weight * sum(v for _, v in placements))
            num_blocks_vals.append(len(placements))

        metrics = {
            f"{self.name}_reward_mean": _mean(reward_vals),
            f"{self.name}_reward_median": _median(reward_vals),
            f"{self.name}_num_blocks_mean": _mean(num_blocks_vals),
        }
        return RewardOutput(scores=scores, metrics=metrics)


class PrefixMotionAngleReward(_PrefixCachedReward):
    """Prefix-shaped consumer variant of :class:`MotionAngleReward`.

    ``MotionAngleReward`` places one ``motion_relational_angle`` bonus at the
    final STOP.  Under the reverse-cumsum (return-to-go) RLOO advantage in
    ``constraint_ar_module_rloo``, that single scalar contributes the *same*
    constant to every preceding token's return, so the bonus pushes all
    constraint tokens uniformly — no within-sequence credit assignment to the
    constraint that actually moved the predicted relational motion subspace
    toward the pseudo target.

    This reward reads the per-prefix motion trajectory cached by
    :class:`PrefixResponseMetricEvaluator` and places a *delta* potential
    ``Phi_t - Phi_{t-1}`` at each completed block's tail, with
    ``Phi_t = coeff * (1 - clip(angle_t)/cap)``.

    ``delta_mode``:
        - ``signed``: the episode sum telescopes to ``Phi_T`` (the final-STOP
          bonus), a potential-based reshape that changes per-token credit
          (variance) but not the optimum.
        - ``positive`` (default): ``max(0, Phi_t - Phi_{t-1} - min_delta)`` does
          NOT telescope (a worsening block is not penalized), so it changes the
          objective and can reach an operating point distinct from a count-based
          parsimony baseline, at the cost of a relation-count-reward / zig-zag
          hacking risk (guarded by ``min_delta`` and the sibling SolverReward
          redundant/conflict penalties).
    """

    name = "prefix_motion"

    def __init__(
        self,
        coeff: float,
        cap: float = 0.70,
        gate: str = "converged",
        delta_mode: str = "positive",
        min_delta: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(coeff=coeff, cap=cap, gate=gate, **kwargs)
        if delta_mode not in _PREFIX_DELTA_MODES:
            raise ValueError(
                f"delta_mode must be one of {_PREFIX_DELTA_MODES}, got "
                f"{delta_mode!r}"
            )
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        self.delta_mode = delta_mode
        self.min_delta = float(min_delta)

    def _placements(
        self, blocks: list[PrefixBlockMetric]
    ) -> tuple[list[tuple[int, float]], list[float]]:
        angles = [b.motion_relational_angle for b in blocks]
        tails = [b.block_tail for b in blocks]
        placements, _pos = _prefix_motion_deltas(
            angles, tails, self.coeff, self.cap, self.delta_mode, self.min_delta
        )
        return placements, [d for _, d in placements]


class PrefixLeakAngleReward(_PrefixCachedReward):
    """Prefix-shaped consumer variant of :class:`LeakAngleReward`.

    Reads the per-prefix leak trajectory cached by
    :class:`PrefixResponseMetricEvaluator` and places an increase-only penalty at
    each *dimensional* block's tail:

        penalty_t = -coeff * max(0, clip(leak_t) - clip(leak_{t-1}) - min_delta)

    The penalty (rather than a leak bonus) is deliberate: ``leak=0`` is trivially
    reached by emitting no dimensions, so a bonus would reward under-constraint.
    Restricting to dimensional blocks matches the meaning of leak — Pred's
    dimensional drivers leaking into GT's relation-forbidden directions — so a
    dimensional constraint that substitutes for a relation GT closes is charged
    at its own block tail, while a legitimate (orthogonal) dimension is not.

    ``gate`` defaults to ``converged``; because there is no positive reward for
    the no-dims corner, the ``dof`` ramp is optional rather than required (unlike
    the full-sequence :class:`LeakAngleReward`).
    """

    name = "prefix_leak"

    def __init__(
        self,
        coeff: float,
        cap: float = 0.70,
        gate: str = "converged",
        min_delta: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(coeff=coeff, cap=cap, gate=gate, **kwargs)
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        self.min_delta = float(min_delta)

    def _placements(
        self, blocks: list[PrefixBlockMetric]
    ) -> tuple[list[tuple[int, float]], list[float]]:
        leaks = [b.leak_relational_angle for b in blocks]
        is_dim = [b.is_dimensional for b in blocks]
        tails = [b.block_tail for b in blocks]
        placements, pen_mags = _prefix_leak_penalties(
            leaks, is_dim, tails, self.coeff, self.cap, self.min_delta
        )
        return placements, pen_mags


class ResponseIntentReward(BaseReward):
    """Deprecated sequence-level reward from Htheta / relation-intent angles.

    The reward is a clipped linear bonus on one selected Htheta/relation-intent
    angle.  The score is written at the first STOP token of each generated
    constraint sequence.

    Deprecated:
        Prefer ``ResponseMetricEvaluator`` + ``MotionAngleReward`` /
        ``LeakAngleReward`` so motion and leak coefficients/gates are explicit
        and the expensive Htheta computation is shared.

    This module assumes solver status has already been computed by
    ``SolverReward`` and can gate itself on solver status.  The default
    ``gate="fc"`` preserves the original FC-only behavior; ``gate="converged"``
    also scores under-constrained but solver-converged candidates, which is
    useful when starting RL from a pretrain checkpoint that has not learned to
    fully constrain sketches yet.
    """

    name = "intent"

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        coeff_h_angle: float = 2.0,
        cap_h_angle: float = 1.0,
        h_angle_reward_metric: str = "response_relational_angle",
        leak_weight: float = 0.5,
        motion_relational_angle_mode: str = "symmetric",
        motion_relational_reverse_weight: float = 0.5,
        gate: str = "fc",
        leak_gate: str | None = None,
        num_workers: int = 0,
        shuffle_within_group: bool = False,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """Create a relation-intent response reward.

        Args:
            context: Sketch tokenization / solver context.
            eps: Numerical rank threshold for SVD and null-space computations.
            coeff_h_angle: Maximum bonus for the selected Htheta-family metric.
            cap_h_angle: Angle/metric value at which the bonus reaches 0.
            h_angle_reward_metric: Metric name.  Defaults to
                ``response_relational_angle``.
            leak_weight: Mixing ratio for ``response_relational_weighted_angle`` only:
                ``(1 - leak_weight) * motion_relational_angle +
                leak_weight * leak_relational_angle``.  ``0.5`` weighs both
                equally (linear analogue of the RMS default); smaller values
                damp the leak channel and with it the incentive to zero leak
                by emitting no dimensions.  Ignored by every other metric.
            motion_relational_angle_mode: How relation-only motion subspaces
                are compared.
            motion_relational_reverse_weight: Reverse-direction weight for the
                ``mixed_subset`` motion-relational angle.
            gate: Solver-status gate.  ``fc`` scores only fully constrained
                samples, ``converged`` scores FC and under-constrained samples,
                and ``always`` scores every reconstructable sample.
            leak_gate: Separate gate for the leak component of
                ``response_relational_weighted_angle`` (defaults to ``gate``).
                Lets the always-wanted ``motion_relational`` term score under
                ``gate`` while the leak term — whose ``leak=0`` minimum is reached
                trivially by emitting no dimensions — is restricted to e.g. ``fc``,
                preventing amplified intent from collapsing to the no-dims corner.
                Has no effect on the single-scalar (non-blend) metrics.
            num_workers: Number of process workers for per-sketch group
                scoring.  ``0`` keeps the serial in-process path.
            shuffle_within_group: Negative control.  When ``True``, the per-sample
                intent reward scalars are permuted within each RLOO group of
                ``ctx.group_sample_size`` rollouts before being placed back at
                each sample's own STOP token.  This preserves the reward
                distribution and the group mean (so the RLOO baseline is
                unchanged) while destroying the sample↔reward correspondence, so
                any remaining S+R̃ effect is attributable to reward *scale* rather
                than to matching the GT.  The logged ``intent_*`` metrics still
                describe the true (unshuffled) angles.
            weight: Composer-level multiplier from ``BaseReward``.
            enabled: Composer-level on/off switch from ``BaseReward``.
        """
        super().__init__(weight=weight, enabled=enabled)
        warnings.warn(
            "ResponseIntentReward is deprecated; use ResponseMetricEvaluator "
            "with MotionAngleReward and/or LeakAngleReward instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.context = context
        self.eps = eps
        self.coeff_h_angle = coeff_h_angle
        self.cap_h_angle = cap_h_angle
        self.h_angle_reward_metric = _validate_h_theta_metric_name(
            h_angle_reward_metric
        )
        if not 0.0 <= leak_weight <= 1.0:
            raise ValueError(
                f"leak_weight must be in [0, 1], got {leak_weight}"
            )
        self.leak_weight = float(leak_weight)
        if motion_relational_angle_mode not in MOTION_RELATIONAL_ANGLE_MODES:
            raise ValueError(
                "motion_relational_angle_mode must be one of "
                f"{sorted(MOTION_RELATIONAL_ANGLE_MODES)}, got "
                f"{motion_relational_angle_mode!r}"
            )
        if motion_relational_reverse_weight < 0.0:
            raise ValueError(
                "motion_relational_reverse_weight must be non-negative, got "
                f"{motion_relational_reverse_weight}"
            )
        if gate not in _GATE_MODES:
            raise ValueError(f"gate must be one of {_GATE_MODES}, got {gate!r}")
        # leak_gate (defaults to gate) gates ONLY the leak component of the linear
        # response_relational_weighted_angle blend; the motion_relational component
        # always uses ``gate``.  This lets the always-wanted relational-structure
        # signal score on Under samples (gate=converged) while the leak term — whose
        # leak=0 minimizer is trivially reached by emitting NO dimensions, i.e. the
        # under-constrained corner — is restricted to FC samples (leak_gate=fc), so
        # amplifying intent cannot collapse into the no-dims corner.  Ignored by the
        # non-blend metrics (they expose a single coupled scalar).
        if leak_gate is None:
            leak_gate = gate
        if leak_gate not in _GATE_MODES:
            raise ValueError(
                f"leak_gate must be one of {_GATE_MODES}, got {leak_gate!r}"
            )
        self.motion_relational_angle_mode = motion_relational_angle_mode
        self.motion_relational_reverse_weight = motion_relational_reverse_weight
        self.gate = gate
        self.leak_gate = leak_gate
        self.num_workers = max(0, int(num_workers))
        self.shuffle_within_group = bool(shuffle_within_group)
        self.tokenizer = SketchTokenizer(context)
        self._executor: ProcessPoolExecutor | None = None

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_response_intent_worker,
                initargs=(
                    self.context,
                    self.eps,
                    self.coeff_h_angle,
                    self.cap_h_angle,
                    self.h_angle_reward_metric,
                    self.leak_weight,
                    self.motion_relational_angle_mode,
                    self.motion_relational_reverse_weight,
                    self.gate,
                    self.leak_gate,
                ),
            )
        return self._executor

    def _reset_executor(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is None:
            return
        processes = getattr(executor, "_processes", None) or {}
        for proc in list(processes.values()):
            try:
                proc.terminate()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

    def _bonus(self, value: float, cap: float, coeff: float) -> float:
        """Instance wrapper for the common clipped angle-to-bonus transform."""
        return _linear_bonus(value, cap, coeff)

    @staticmethod
    def _status_passes(status: str | None, gate: str) -> bool:
        if gate == "always" or status is None:
            return True
        if gate == "converged":
            return status in ("FC", "Under")
        if gate == "fc":
            return status == "FC"
        return False

    def _gate_pass(self, status: str | None) -> bool:
        return self._status_passes(status, self.gate)

    def _leak_gate_pass(self, status: str | None) -> bool:
        return self._status_passes(status, self.leak_gate)

    def _score_h_theta(
        self,
        pred_sketch: datalib.Sketch,
        pred_native: NativeResponse,
        gt_sketch: datalib.Sketch,
        gt_native: NativeResponse,
        stats: ResponseIntentBatchStats,
        *,
        include_leak: bool = True,
        J_phi_pred: torch.Tensor | None = None,
        J_phi_gt: torch.Tensor | None = None,
        gt_motion_full: tuple[torch.Tensor, int] | None = None,
        gt_motion_relational: tuple[torch.Tensor, int] | None = None,
        gt_relation_nullspace: tuple[torch.Tensor, int] | None = None,
    ) -> float:
        """Compute the selected Htheta-family angle and convert it to reward.

        ``include_leak`` is honoured only by ``response_relational_weighted_angle``:
        when ``False`` (the sample passed ``gate`` but not ``leak_gate``), the leak
        term is dropped and the reward is the pure ``motion_relational_angle`` bonus
        at full weight, so the always-wanted relational signal is preserved while the
        leak term is restricted to ``leak_gate`` samples.
        """
        h_metrics = compute_h_theta_metrics(
            pred_sketch,
            pred_native,
            gt_sketch,
            gt_native,
            self.context,
            self.eps,
            self.motion_relational_angle_mode,
            self.motion_relational_reverse_weight,
            J_phi_pred=J_phi_pred,
            J_phi_gt=J_phi_gt,
            gt_motion_full=gt_motion_full,
            gt_motion_relational=gt_motion_relational,
            gt_relation_nullspace=gt_relation_nullspace,
        )
        if self.h_angle_reward_metric == "response_relational_weighted_angle":
            # Linear blend; computed here (not in HThetaMetrics) because the
            # mixing ratio is reward config, not a property of the sketch pair.
            # When the leak term is gated out for this sample (include_leak=False),
            # fall back to pure motion_relational at full weight.
            if include_leak:
                selected_angle = (
                    (1.0 - self.leak_weight) * h_metrics.motion_relational_angle
                    + self.leak_weight * h_metrics.leak_relational_angle
                )
            else:
                selected_angle = h_metrics.motion_relational_angle
        else:
            selected_angle = h_metrics.value(self.h_angle_reward_metric)
        stats.add_h_theta(h_metrics)
        return self._bonus(
            selected_angle,
            self.cap_h_angle,
            self.coeff_h_angle,
        )

    def _score_sample(
        self,
        ctx: RewardContext,
        i: int,
        length: int,
        prim_i: np.ndarray,
        cons_i_raw: np.ndarray,
        gt_sketch: datalib.Sketch,
        solver_status: str,
        stats: ResponseIntentBatchStats,
    ) -> tuple[int, float] | None:
        """Score one generated sample.

        Pred sketch and GT/Pred native responses are routed through
        ``ctx.pred_sketches`` and ``ctx.native_cache`` so that other rewards
        in the same composer (notably ProbeResponseSVDReward) can share them.

        Returns:
            ``(first_stop_idx, reward)`` when sketch reconstruction succeeded
            and the sample passed the configured solver-status gate.  ``None``
            when the sample should not receive an intent-reward score.
        """
        if not self._gate_pass(solver_status):
            return None

        cons_i, first_stop_idx = first_stop_and_pad(cons_i_raw, length)
        try:
            # Reconstruct the generated constraints on top of the original
            # primitive geometry, then build the first-order response map R_C.
            pred_sketch, pred_native = get_or_build_pred_native(
                ctx,
                i,
                self.tokenizer,
                cons_i,
                prim_i,
                gt_sketch,
                self.context,
                self.eps,
            )
        except Exception as exc:
            logger.debug(
                "intent reward: skipped sample %d (build failure: %s)",
                i,
                exc,
            )
            return None

        try:
            gt_native = get_or_compute_gt_native(
                ctx, gt_sketch, self.context, self.eps
            )
            # Cache J_phi by (id(sketch), dtype): pred is reused across the
            # motion_full / motion_relational helpers within this sample, and
            # the gt entry survives across the K rollouts of an RLOO group
            # since the same gt_sketch object is shared.
            J_phi_pred = get_or_compute_j_phi(
                ctx, pred_sketch, pred_native.R.dtype
            )
            J_phi_gt = get_or_compute_j_phi(
                ctx, gt_sketch, gt_native.R.dtype
            )
            # gt-side subspaces are pure functions of gt_sketch and are
            # reused across the K rollouts of an RLOO group (and across the
            # motion_full / motion_relational / leak helpers within one
            # sample).  See get_or_compute_gt_motion_*_subspace docstrings.
            gt_motion_full = get_or_compute_gt_motion_full_subspace(
                ctx, gt_sketch, gt_native, self.eps, J_phi_gt
            )
            gt_motion_relational = get_or_compute_gt_motion_relational_subspace(
                ctx, gt_sketch, gt_native, self.context, self.eps, J_phi_gt
            )
            gt_relation_nullspace = get_or_compute_gt_relation_nullspace(
                ctx, gt_sketch, gt_native, self.context, self.eps
            )
            sample_reward = self._score_h_theta(
                pred_sketch,
                pred_native,
                gt_sketch,
                gt_native,
                stats,
                include_leak=self._leak_gate_pass(solver_status),
                J_phi_pred=J_phi_pred,
                J_phi_gt=J_phi_gt,
                gt_motion_full=gt_motion_full,
                gt_motion_relational=gt_motion_relational,
                gt_relation_nullspace=gt_relation_nullspace,
            )
        except Exception as exc:
            # Reward computation runs inside the training loop.  A malformed
            # rare case should be visible in metrics but should not abort
            # the whole rollout batch.
            logger.debug(
                "intent reward: skipped sample %d (angle computation failure: %s)",
                i,
                exc,
            )
            return None

        stats.reward_vals.append(sample_reward)
        return first_stop_idx, sample_reward

    def _score_serial_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str],
        length: int,
    ) -> tuple[torch.Tensor, ResponseIntentBatchStats]:
        B = cons_np.shape[0]
        scores = torch.zeros((B, length), dtype=torch.float32)
        stats = ResponseIntentBatchStats()

        for i in range(B):
            result = self._score_sample(
                ctx,
                i,
                length,
                prim_np[i],
                cons_np[i],
                ctx.sketch_original_list[i],
                statuses[i],
                stats,
            )
            if result is None:
                continue
            first_stop_idx, sample_reward = result
            scores[i, first_stop_idx] += sample_reward

        return scores, stats

    def _score_parallel_from_arrays(
        self,
        ctx: RewardContext,
        prim_np: np.ndarray,
        cons_np: np.ndarray,
        statuses: list[str],
        length: int,
    ) -> tuple[torch.Tensor, ResponseIntentBatchStats]:
        B = cons_np.shape[0]
        scores = torch.zeros((B, length), dtype=torch.float32)
        stats = ResponseIntentBatchStats()

        grouped: dict[int, tuple[datalib.Sketch, list[int]]] = {}
        for i, sketch in enumerate(ctx.sketch_original_list):
            key = id(sketch)
            if key not in grouped:
                grouped[key] = (sketch, [])
            grouped[key][1].append(i)

        if len(grouped) <= 1:
            return self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        payloads = []
        for sketch, indices in grouped.values():
            payloads.append(
                (
                    indices,
                    prim_np[indices].copy(),
                    cons_np[indices].copy(),
                    sketch,
                    [statuses[i] for i in indices],
                    length,
                )
            )

        try:
            executor = self._get_executor()
            futures = [
                executor.submit(_score_response_intent_group, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                placements, part_stats = future.result()
                stats.extend(part_stats)
                for i, first_stop_idx, sample_reward in placements:
                    scores[i, first_stop_idx] += sample_reward
        except Exception as exc:
            logger.warning(
                "parallel intent reward failed; falling back to serial: %s",
                exc,
            )
            self._reset_executor()
            return self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, length
            )

        return scores, stats

    def _shuffle_scores_within_group(
        self, scores: torch.Tensor, ctx: RewardContext
    ) -> torch.Tensor:
        """Permute per-sample intent scalars within each RLOO group.

        The sequence-level reward placed by ``_score_*_from_arrays`` is a single
        scalar at each sample's first STOP, so ``scores.sum(dim=1)`` recovers the
        per-sample intent reward (0 for gated-out samples).  We permute those
        scalars within each contiguous block of ``K = ctx.group_sample_size`` and
        place the permuted value back at the *receiving* sample's own first STOP.
        Because a permutation preserves the per-group multiset, the group sum (and
        thus the RLOO leave-one-out baseline mean) is unchanged; only the
        sample↔reward correspondence is broken.
        """
        K = ctx.group_sample_size
        B, L = scores.shape
        if not K or K < 2 or B % K != 0:
            return scores

        scalar = scores.sum(dim=1)                                  # (B,)
        cons = ctx.cons_tokens.detach().to("cpu")
        eq_stop = (cons == int(ConstraintToken.Stop)).int()
        any_stop = eq_stop.any(dim=-1)
        first_stop = torch.where(
            any_stop,
            eq_stop.argmax(dim=-1),
            torch.full((B,), L - 1, dtype=torch.long),
        )

        perm = torch.arange(B)
        for g in range(B // K):
            base = g * K
            perm[base : base + K] = base + torch.randperm(K)

        new_scores = torch.zeros_like(scores)
        new_scores[torch.arange(B), first_stop] = scalar[perm]
        return new_scores

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """Compute sequence-level scores and scalar logging metrics for a batch."""
        B, L = ctx.cons_tokens.shape
        assert len(ctx.sketch_original_list) == B

        prim_np = ctx.prim_tokens.detach().cpu().numpy()
        cons_np = ctx.cons_tokens.detach().cpu().numpy()

        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * B

        if self.num_workers > 0:
            scores, stats = self._score_parallel_from_arrays(
                ctx, prim_np, cons_np, statuses, L
            )
        else:
            scores, stats = self._score_serial_from_arrays(
                ctx, prim_np, cons_np, statuses, L
            )

        # Negative control: scramble the sample↔reward correspondence within each
        # RLOO group (scale-matched, baseline-preserving).  Metrics stay true.
        if self.shuffle_within_group:
            scores = self._shuffle_scores_within_group(scores, ctx)

        return RewardOutput(scores=scores, metrics=stats.to_metrics_dict())


_RESPONSE_METRIC_WORKER: ResponseMetricEvaluator | None = None


def _init_response_metric_worker(
    context: SketchContext,
    eps: float,
    motion_relational_angle_mode: str,
    motion_relational_reverse_weight: float,
    gate: str,
) -> None:
    """Initialise one process-pool worker for response metric evaluation."""
    global _RESPONSE_METRIC_WORKER
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _RESPONSE_METRIC_WORKER = ResponseMetricEvaluator(
        context=context,
        eps=eps,
        motion_relational_angle_mode=motion_relational_angle_mode,
        motion_relational_reverse_weight=motion_relational_reverse_weight,
        gate=gate,
        num_workers=0,
    )


def _score_response_metric_group(
    payload: tuple[
        list[int],
        np.ndarray,
        np.ndarray,
        datalib.Sketch,
        list[str | None],
        int,
    ],
) -> tuple[list[tuple[int, HThetaMetrics]], ResponseIntentBatchStats]:
    """Evaluate one GT-sketch group in a worker process."""
    if _RESPONSE_METRIC_WORKER is None:
        raise RuntimeError("response metric worker was not initialised")

    indices, prim_np, cons_np, gt_sketch, statuses, length = payload
    local_b = len(indices)
    ctx = RewardContext(
        prim_tokens=torch.from_numpy(prim_np),
        cons_tokens=torch.from_numpy(cons_np),
        sketch_original_list=[gt_sketch] * local_b,
    )
    stats = ResponseIntentBatchStats()
    metric_items: list[tuple[int, HThetaMetrics]] = []

    for local_i, global_i in enumerate(indices):
        metrics = _RESPONSE_METRIC_WORKER._compute_sample_metrics(
            ctx,
            local_i,
            length,
            prim_np[local_i],
            cons_np[local_i],
            gt_sketch,
            statuses[local_i],
            stats,
        )
        if metrics is not None:
            metric_items.append((global_i, metrics))

    return metric_items, stats


_PREFIX_EVAL_WORKER: PrefixResponseMetricEvaluator | None = None


def _init_prefix_eval_worker(
    context: SketchContext,
    eps: float,
    motion_relational_angle_mode: str,
    motion_relational_reverse_weight: float,
    gate: str,
    max_prefixes: int | None,
) -> None:
    """Initialise one process-pool worker for prefix-eval metric computation."""
    global _PREFIX_EVAL_WORKER
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _PREFIX_EVAL_WORKER = PrefixResponseMetricEvaluator(
        context=context,
        eps=eps,
        motion_relational_angle_mode=motion_relational_angle_mode,
        motion_relational_reverse_weight=motion_relational_reverse_weight,
        gate=gate,
        max_prefixes=max_prefixes,
        num_workers=0,
    )


def _score_prefix_eval_group(
    payload: tuple[
        list[int],
        np.ndarray,
        np.ndarray,
        datalib.Sketch,
        list[str | None],
        int,
    ],
) -> tuple[list[tuple[int, list[PrefixBlockMetric]]], PrefixEvalBatchStats]:
    """Evaluate one GT-sketch group of prefix metrics in a worker."""
    if _PREFIX_EVAL_WORKER is None:
        raise RuntimeError("prefix eval worker was not initialised")

    indices, prim_np, cons_np, gt_sketch, statuses, length = payload
    local_b = len(indices)
    ctx = RewardContext(
        prim_tokens=torch.from_numpy(prim_np),
        cons_tokens=torch.from_numpy(cons_np),
        sketch_original_list=[gt_sketch] * local_b,
    )
    stats = PrefixEvalBatchStats()
    metric_items: list[tuple[int, list[PrefixBlockMetric]]] = []
    for local_i, global_i in enumerate(indices):
        blocks = _PREFIX_EVAL_WORKER._score_sample(
            ctx,
            local_i,
            length,
            prim_np[local_i],
            cons_np[local_i],
            gt_sketch,
            statuses[local_i],
            stats,
        )
        if blocks:
            metric_items.append((global_i, blocks))
    return metric_items, stats


_RESPONSE_INTENT_WORKER: ResponseIntentReward | None = None


def _init_response_intent_worker(
    context: SketchContext,
    eps: float,
    coeff_h_angle: float,
    cap_h_angle: float,
    h_angle_reward_metric: str,
    leak_weight: float,
    motion_relational_angle_mode: str,
    motion_relational_reverse_weight: float,
    gate: str,
    leak_gate: str,
) -> None:
    """Initialise one process-pool worker for response-intent scoring."""
    global _RESPONSE_INTENT_WORKER
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    _RESPONSE_INTENT_WORKER = ResponseIntentReward(
        context=context,
        eps=eps,
        coeff_h_angle=coeff_h_angle,
        cap_h_angle=cap_h_angle,
        h_angle_reward_metric=h_angle_reward_metric,
        leak_weight=leak_weight,
        motion_relational_angle_mode=motion_relational_angle_mode,
        motion_relational_reverse_weight=motion_relational_reverse_weight,
        gate=gate,
        leak_gate=leak_gate,
        num_workers=0,
    )


def _score_response_intent_group(
    payload: tuple[
        list[int],
        np.ndarray,
        np.ndarray,
        datalib.Sketch,
        list[str],
        int,
    ],
) -> tuple[list[tuple[int, int, float]], ResponseIntentBatchStats]:
    """Score one RLOO group in a worker process.

    Grouping the K rollouts for the same GT sketch preserves the GT-side
    caches inside the local RewardContext while still parallelising across
    independent sketches.
    """
    if _RESPONSE_INTENT_WORKER is None:
        raise RuntimeError("response-intent worker was not initialised")

    indices, prim_np, cons_np, gt_sketch, statuses, length = payload
    local_b = len(indices)
    ctx = RewardContext(
        prim_tokens=torch.from_numpy(prim_np),
        cons_tokens=torch.from_numpy(cons_np),
        sketch_original_list=[gt_sketch] * local_b,
    )
    stats = ResponseIntentBatchStats()
    placements: list[tuple[int, int, float]] = []

    for local_i, global_i in enumerate(indices):
        result = _RESPONSE_INTENT_WORKER._score_sample(
            ctx,
            local_i,
            length,
            prim_np[local_i],
            cons_np[local_i],
            gt_sketch,
            statuses[local_i],
            stats,
        )
        if result is None:
            continue
        first_stop_idx, sample_reward = result
        placements.append((global_i, first_stop_idx, sample_reward))

    return placements, stats
