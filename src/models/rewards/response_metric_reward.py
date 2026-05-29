"""ResponseIntentReward — reward-oriented Htheta / relation-intent signal.

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

import numpy as np
import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _is_relational,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.models.rewards.utils.geometry import (
    motion_full_subspace as _motion_full_subspace,
    motion_relational_subspace as _motion_relational_subspace,
    orthonormal_range as _orthonormal_range,
    principal_cos2_sum as _principal_cos2_sum,
    stack_relation_jacobians_at as _stack_relation_jacobians_at,
)
from src.models.rewards.utils.sketch_builder import (
    first_stop_and_pad,
    get_or_build_pred_native,
    get_or_compute_gt_motion_full_subspace,
    get_or_compute_gt_motion_relational_subspace,
    get_or_compute_gt_native,
    get_or_compute_gt_relation_nullspace,
    get_or_compute_j_phi,
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
}


MOTION_RELATIONAL_ANGLE_MODES = {
    "symmetric",
    "pred_subset_gt",
    "mixed_subset",
}


_GATE_MODES = ("always", "converged", "fc")


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
    d_norm = float(max(dim_P_full, dim_G_full, 1))
    angle_sq = min(max((d_norm - cos2_sum) / d_norm, 0.0), 1.0)
    return SubspaceAngleResult(
        angle=math.sqrt(angle_sq),
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
    gt_motion_relational: tuple[torch.Tensor, int] | None = None,
) -> SubspaceAngleResult:
    """Compare relation-only motion subspaces in observable space.

    The reward-oriented sibling of ``_motion_full_angle``: dimensional drivers
    are deliberately ignored and only the primitive motions allowed by the
    sketch's relational constraints are compared.  Pred and GT subspaces are
    both evaluated at ``pred_native.state`` so the principal-angle SVD lives
    in a common coordinate system.

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
        U_gt, dim_gt = _motion_relational_subspace(
            gt_sketch, state, context, eps, J_phi=J_phi_gt
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
    leak_sq = 0.0
    if dim_P_prim > 0 and dim_N > 0:
        overlap = N_G_prim.T @ U_P_prim
        leak_sq = float((overlap ** 2).sum()) / float(dim_P_prim)
    leak_sq = min(max(leak_sq, 0.0), 1.0)
    return SubspaceAngleResult(
        angle=math.sqrt(leak_sq),
        dim_pred=int(dim_P_prim),
        dim_gt=int(dim_N),
    )


def _compute_h_theta_metrics(
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


class ResponseIntentReward(BaseReward):
    """Sequence-level reward from Htheta / relation-intent angles.

    The reward is a clipped linear bonus on one selected Htheta/relation-intent
    angle.  The score is written at the first STOP token of each generated
    constraint sequence.

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
        motion_relational_angle_mode: str = "symmetric",
        motion_relational_reverse_weight: float = 0.5,
        gate: str = "fc",
        num_workers: int = 0,
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
            motion_relational_angle_mode: How relation-only motion subspaces
                are compared.
            motion_relational_reverse_weight: Reverse-direction weight for the
                ``mixed_subset`` motion-relational angle.
            gate: Solver-status gate.  ``fc`` scores only fully constrained
                samples, ``converged`` scores FC and under-constrained samples,
                and ``always`` scores every reconstructable sample.
            num_workers: Number of process workers for per-sketch group
                scoring.  ``0`` keeps the serial in-process path.
            weight: Composer-level multiplier from ``BaseReward``.
            enabled: Composer-level on/off switch from ``BaseReward``.
        """
        super().__init__(weight=weight, enabled=enabled)
        self.context = context
        self.eps = eps
        self.coeff_h_angle = coeff_h_angle
        self.cap_h_angle = cap_h_angle
        self.h_angle_reward_metric = _validate_h_theta_metric_name(
            h_angle_reward_metric
        )
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
                initializer=_init_response_intent_worker,
                initargs=(
                    self.context,
                    self.eps,
                    self.coeff_h_angle,
                    self.cap_h_angle,
                    self.h_angle_reward_metric,
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

    def _bonus(self, value: float, cap: float, coeff: float) -> float:
        """Instance wrapper for the common clipped angle-to-bonus transform."""
        return _linear_bonus(value, cap, coeff)

    def _gate_pass(self, status: str | None) -> bool:
        if self.gate == "always" or status is None:
            return True
        if self.gate == "converged":
            return status in ("FC", "Under")
        if self.gate == "fc":
            return status == "FC"
        return False

    def _score_h_theta(
        self,
        pred_sketch: datalib.Sketch,
        pred_native: NativeResponse,
        gt_sketch: datalib.Sketch,
        gt_native: NativeResponse,
        stats: ResponseIntentBatchStats,
        *,
        J_phi_pred: torch.Tensor | None = None,
        J_phi_gt: torch.Tensor | None = None,
        gt_motion_full: tuple[torch.Tensor, int] | None = None,
        gt_motion_relational: tuple[torch.Tensor, int] | None = None,
        gt_relation_nullspace: tuple[torch.Tensor, int] | None = None,
    ) -> float:
        """Compute the selected Htheta-family angle and convert it to reward."""
        h_metrics = _compute_h_theta_metrics(
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

        return RewardOutput(scores=scores, metrics=stats.to_metrics_dict())


_RESPONSE_INTENT_WORKER: ResponseIntentReward | None = None


def _init_response_intent_worker(
    context: SketchContext,
    eps: float,
    coeff_h_angle: float,
    cap_h_angle: float,
    h_angle_reward_metric: str,
    motion_relational_angle_mode: str,
    motion_relational_reverse_weight: float,
    gate: str,
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
        motion_relational_angle_mode=motion_relational_angle_mode,
        motion_relational_reverse_weight=motion_relational_reverse_weight,
        gate=gate,
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
