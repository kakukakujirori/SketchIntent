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
``motion_full_angle``         -> observable response-subspace angle
``motion_relational_angle``   -> mode-normalized relation-structure angle
``leak_relational_angle``     -> relation-leak angle

The most important distinction is evaluation metric vs training reward.
``response_full_angle`` is the canonical Htheta scalar from the v4 analysis:
observable motion-subspace mismatch plus angular relation leak.  The default
training signal, ``response_relational_angle``, replaces the dimensional
motion-subspace term with relation-only structure angle.  That avoids
rewarding a model for matching one annotator's particular dimensional-constraint
placement while still encouraging relational design intent.

The probe-conditioned response-SVD family lives in
``probe_response_svd_reward.py``.  It shares the same first-order response map
but is a supplement diagnostic, not part of the default intent reward.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
import math
import statistics

import numpy as np
import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _is_dimensional,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.models.rewards.utils.geometry import (
    observable_subspace as _observable_subspace,
    orthonormal_range as _orthonormal_range,
    stack_relation_jacobians_at as _stack_relation_jacobians_at,
)
from src.models.rewards.utils.relation_subspace import (
    RELATION_SUBSPACE_DISTANCE_MODES,
    compute_relation_subspace_distance,
)
from src.models.rewards.utils.sketch_builder import (
    get_or_build_pred_native,
    get_or_compute_gt_native,
)
from src.sketch_handlers.constants import ConstraintToken
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


def _relation_motion_angle_scale(
    mode: str,
    dim_pred: int,
    dim_gt: int,
    reverse_weight: float,
) -> float:
    """Return the mode-specific maximum of the raw relation-structure metric.

    ``compute_relation_subspace_distance`` returns a raw comparison score whose
    ceiling depends on the comparison mode.  ``motion_relational_angle`` divides
    by that ceiling so the relation-motion component is on a [0, 1] angle-like
    scale before it is combined with leak.

    For ``symmetric`` this is the maximum for the current dimension pair.  It
    equals ``sqrt(2)`` when ``dim_pred == dim_gt`` but is smaller for unequal
    dimensions, so orthogonal unequal-dimensional subspaces still map to 1.
    """
    if reverse_weight < 0.0:
        raise ValueError(
            "reverse_weight must be non-negative for relation-motion scaling, "
            f"got {reverse_weight}"
        )
    if mode == "symmetric":
        d_norm = float(max(dim_pred, dim_gt, 1))
        return math.sqrt(max((dim_pred + dim_gt) / d_norm, 1e-12))
    if mode == "mixed_subset":
        return math.sqrt(max(1.0 + reverse_weight, 1e-12))
    return 1.0


def _normalized_relation_motion_angle(
    raw_value: float,
    mode: str,
    dim_pred: int,
    dim_gt: int,
    reverse_weight: float,
) -> float:
    """Normalize raw ``d_relstruct`` into the public motion-relational angle."""
    scale = _relation_motion_angle_scale(mode, dim_pred, dim_gt, reverse_weight)
    return min(max(raw_value / scale, 0.0), 1.0)


def _first_stop_and_pad(cons_tokens_np: np.ndarray, length: int) -> tuple[np.ndarray, int]:
    """Return a constraint sequence truncated after the first STOP token.

    The reward is placed at the first STOP position because this is a
    sequence-level signal.  Tokens after STOP are ignored by padding them before
    sketch reconstruction; otherwise stale sampled tokens could create invalid
    constraints that the model did not mean to emit.

    Corner case: if no STOP token appears (the model ran to the maximum
    generation length without emitting STOP), ``first_stop_idx`` falls back
    to ``length - 1`` and no padding is applied.  The reward is still placed
    at the last position so the sample receives a usable training signal.
    """
    cons_i = cons_tokens_np.copy()
    stop_pos = np.where(cons_i == int(ConstraintToken.Stop))[0]
    first_stop_idx = int(stop_pos[0]) if len(stop_pos) > 0 else (length - 1)
    if 0 <= first_stop_idx < length - 1:
        cons_i[first_stop_idx + 1:] = int(ConstraintToken.Pad)
    return cons_i, first_stop_idx


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
        motion_full_angle: Observable response-subspace mismatch in [0, 1].
        motion_relational_angle: Relation-only null-space mismatch, normalized
            by the mode-specific raw maximum so that orthogonal subspaces map
            to 1 even for unequal dimensions.
        leak_relational_angle: Fraction of Pred's allowed motion that enters
            GT relation-forbidden directions, in [0, 1].
    """

    response_full_angle: float
    response_relational_angle: float
    motion_full_angle: float
    motion_relational_angle: float
    leak_relational_angle: float
    relation_structure_pred_subset_gt: float
    relation_structure_gt_subset_pred: float
    relation_structure_dim_pred: int
    relation_structure_dim_gt: int
    observable_dim_pred: int
    observable_dim_gt: int

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
            "relation_structure_pred_subset_gt": (
                self.relation_structure_pred_subset_gt
            ),
            "relation_structure_gt_subset_pred": (
                self.relation_structure_gt_subset_pred
            ),
            "relation_structure_dim_pred": self.relation_structure_dim_pred,
            "relation_structure_dim_gt": self.relation_structure_dim_gt,
            "observable_dim_pred": self.observable_dim_pred,
            "observable_dim_gt": self.observable_dim_gt,
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
    selected_angle_vals: list[float] = field(default_factory=list)
    response_full_angle_vals: list[float] = field(default_factory=list)
    response_relational_angle_vals: list[float] = field(default_factory=list)
    motion_full_angle_vals: list[float] = field(default_factory=list)
    motion_relational_angle_vals: list[float] = field(default_factory=list)
    leak_relational_angle_vals: list[float] = field(default_factory=list)
    relation_structure_pred_subset_gt_vals: list[float] = field(default_factory=list)
    relation_structure_gt_subset_pred_vals: list[float] = field(default_factory=list)
    relation_structure_dim_pred_vals: list[int] = field(default_factory=list)
    relation_structure_dim_gt_vals: list[int] = field(default_factory=list)
    observable_dim_pred_vals: list[int] = field(default_factory=list)
    observable_dim_gt_vals: list[int] = field(default_factory=list)
    skipped_build: int = 0
    skipped_not_fc: int = 0
    skipped_angle: int = 0

    def add_h_theta(self, metrics: HThetaMetrics, selected_angle: float) -> None:
        """Record one successful Htheta/relation-intent computation."""
        self.response_full_angle_vals.append(metrics.response_full_angle)
        self.response_relational_angle_vals.append(metrics.response_relational_angle)
        self.motion_full_angle_vals.append(metrics.motion_full_angle)
        self.motion_relational_angle_vals.append(metrics.motion_relational_angle)
        self.leak_relational_angle_vals.append(metrics.leak_relational_angle)
        self.selected_angle_vals.append(selected_angle)
        self.relation_structure_pred_subset_gt_vals.append(
            metrics.relation_structure_pred_subset_gt
        )
        self.relation_structure_gt_subset_pred_vals.append(
            metrics.relation_structure_gt_subset_pred
        )
        self.relation_structure_dim_pred_vals.append(
            metrics.relation_structure_dim_pred
        )
        self.relation_structure_dim_gt_vals.append(metrics.relation_structure_dim_gt)
        self.observable_dim_pred_vals.append(metrics.observable_dim_pred)
        self.observable_dim_gt_vals.append(metrics.observable_dim_gt)

    def to_metrics_dict(self) -> dict[str, float | int]:
        """Return flat scalar metrics under ``intent_*`` keys.

        The composer flattens reward metrics without further prefixing, so
        each key already encodes its owning reward (``intent_``) and the
        metric name without redundant repetition.
        """
        return {
            "intent_reward_mean": _mean(self.reward_vals),
            "intent_reward_median": _median(self.reward_vals),
            "intent_selected_angle_mean": _mean(self.selected_angle_vals),
            "intent_selected_angle_median": _median(self.selected_angle_vals),
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
            "intent_relation_structure_pred_subset_gt_mean": _mean(
                self.relation_structure_pred_subset_gt_vals
            ),
            "intent_relation_structure_gt_subset_pred_mean": _mean(
                self.relation_structure_gt_subset_pred_vals
            ),
            "intent_relation_structure_dim_pred_mean": _mean(
                self.relation_structure_dim_pred_vals
            ),
            "intent_relation_structure_dim_gt_mean": _mean(
                self.relation_structure_dim_gt_vals
            ),
            "intent_observable_dim_pred_mean": _mean(
                self.observable_dim_pred_vals
            ),
            "intent_observable_dim_gt_mean": _mean(self.observable_dim_gt_vals),
            "intent_valid_n": len(self.reward_vals),
            "intent_skipped_build": self.skipped_build,
            "intent_skipped_not_fc": self.skipped_not_fc,
            "intent_skipped_angle": self.skipped_angle,
        }


def _observable_motion_subspace_angle(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    eps: float,
) -> tuple[float, float, int, int]:
    """Padded principal-angle metric between observable response subspaces.

    ``R_C`` maps native dimensional-parameter changes to primitive motion.  Its
    raw primitive coordinates mix positions, radii, angles, etc., so canonical
    Htheta compares the span after projection by ``J_phi`` into the common
    pairwise-distance observable coordinate space:

        W_C = range(J_phi R_C)

    Missing dimensions are treated as 90-degree principal angles by normalizing
    with ``max(dim_P, dim_G)``.  This keeps dimension-dropout failures visible
    while putting equal-dimension angular mismatch on a [0, 1] scale.
    """
    U_W_P, dim_P_obs = _observable_subspace(pred_sketch, pred_native, eps)
    U_W_G, dim_G_obs = _observable_subspace(gt_sketch, gt_native, eps)

    # Both sides have no observable motion (fully constrained with no
    # dimensional drivers) — their zero subspaces trivially agree.
    if dim_P_obs == 0 and dim_G_obs == 0:
        return 0.0, 0.0, 0, 0

    d_norm = float(max(dim_P_obs, dim_G_obs, 1))
    if dim_P_obs > 0 and dim_G_obs > 0:
        align = U_W_G.T @ U_W_P
        sigmas = torch.linalg.svdvals(align).clamp(0.0, 1.0)
        cos2_sum = float((sigmas ** 2).sum())
    else:
        cos2_sum = 0.0
    angle_sq = (d_norm - cos2_sum) / d_norm
    angle_sq = min(max(angle_sq, 0.0), 1.0)
    return math.sqrt(angle_sq), angle_sq, int(dim_P_obs), int(dim_G_obs)


def _leak_relational_angle(
    pred_native: NativeResponse,
    gt_rels: list,
    context,
    eps: float,
) -> tuple[float, float]:
    """RMS cosine between Pred motion and GT relation-forbidden directions.

    ``U_P`` spans the motions Pred allows through its dimensional drivers.  The
    row space of GT relation Jacobians spans directions that should be normal to
    any relation-preserving motion.  Their overlap is therefore a scale-free
    measure of relational design-intent leakage:

        leak_relational_angle^2 = ||orth(row(J_R_G))^T U_P||_F^2 / dim(U_P)

    Jacobians are evaluated at Pred's primitive state, whose geometry has been
    overwritten with the original sketch geometry during reconstruction.
    """
    U_P_prim, dim_P_prim = _orthonormal_range(pred_native.R, eps)
    J_R, n_R = _stack_relation_jacobians_at(gt_rels, pred_native.state, context)
    if J_R is not None and dim_P_prim > 0 and n_R > 0:
        N_G_prim, dim_N = _orthonormal_range(J_R.T, eps)
        if dim_N > 0:
            overlap = N_G_prim.T @ U_P_prim
            leak_relational_angle_sq = float((overlap ** 2).sum()) / float(
                dim_P_prim
            )
        else:
            leak_relational_angle_sq = 0.0
    else:
        leak_relational_angle_sq = 0.0
    leak_relational_angle_sq = min(max(leak_relational_angle_sq, 0.0), 1.0)
    return math.sqrt(leak_relational_angle_sq), leak_relational_angle_sq


def _compute_h_theta_metrics(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    context,
    eps: float,
    relstruct_distance_mode: str = "symmetric",
    relstruct_reverse_weight: float = 0.5,
) -> HThetaMetrics:
    """Return canonical Htheta plus the RL-oriented relation-intent substitute.

    Two related scalars are produced:

    - ``response_full_angle`` is the paper-style diagnostic:
      RMS of observable response-subspace mismatch and relation leak.
    - ``response_relational_angle`` is the reward default:
      RMS of relation-only structure mismatch and relation leak.

    The second scalar deliberately avoids directly rewarding agreement with one
    annotator's dimensional constraint placement.  Dimensional parsimony and FC
    are handled by sibling rewards in the composer.

    All five public angles are kept on the [0, 1] scale:

    - ``motion_full_angle`` and ``leak_relational_angle`` come straight from
      padded principal-angle / RMS-cosine formulas in [0, 1].
    - ``motion_relational_angle`` divides the raw ``d_relstruct`` returned by
      ``compute_relation_subspace_distance`` by the mode-specific maximum so
      that orthogonal subspaces map to 1.  Maxima are: pair-specific symmetric
      maximum for ``symmetric``, ``sqrt(1 + reverse_weight)`` for
      ``mixed_subset``, and 1 for ``pred_subset_gt``.
    - ``response_*_angle`` divides ``sqrt(motion^2 + leak^2)`` by ``sqrt(2)``
      to put the combined RMS angle on the same [0, 1] scale as the components.
    """
    gt_rels = [
        c for c in gt_sketch.constraints.values()
        if not _is_dimensional(c)
    ]

    motion_full_angle, motion_full_angle_sq, dim_P_obs, dim_G_obs = (
        _observable_motion_subspace_angle(
            pred_sketch, pred_native, gt_sketch, gt_native, eps
        )
    )
    leak_relational_angle, leak_relational_angle_sq = _leak_relational_angle(
        pred_native, gt_rels, context, eps
    )
    relstruct = compute_relation_subspace_distance(
        pred_sketch,
        gt_sketch,
        pred_native,
        context,
        eps,
        relstruct_distance_mode,
        relstruct_reverse_weight,
    )
    relation_structure_raw = float(relstruct["d_relstruct"])
    relation_structure_dim_pred = int(relstruct["relstruct_dim_pred"])
    relation_structure_dim_gt = int(relstruct["relstruct_dim_gt"])
    motion_relational_angle = _normalized_relation_motion_angle(
        relation_structure_raw,
        relstruct_distance_mode,
        relation_structure_dim_pred,
        relation_structure_dim_gt,
        relstruct_reverse_weight,
    )
    return HThetaMetrics(
        response_full_angle=math.sqrt(
            max(motion_full_angle_sq + leak_relational_angle_sq, 0.0)
        ) / _RESPONSE_ANGLE_NORM,
        response_relational_angle=math.sqrt(
            max(motion_relational_angle ** 2 + leak_relational_angle_sq, 0.0)
        ) / _RESPONSE_ANGLE_NORM,
        motion_full_angle=motion_full_angle,
        motion_relational_angle=motion_relational_angle,
        leak_relational_angle=leak_relational_angle,
        relation_structure_pred_subset_gt=float(
            relstruct["d_relstruct_pred_subset_gt"]
        ),
        relation_structure_gt_subset_pred=float(
            relstruct["d_relstruct_gt_subset_pred"]
        ),
        relation_structure_dim_pred=relation_structure_dim_pred,
        relation_structure_dim_gt=relation_structure_dim_gt,
        observable_dim_pred=dim_P_obs,
        observable_dim_gt=dim_G_obs,
    )


class ResponseIntentReward(BaseReward):
    """Sequence-level reward from Htheta / relation-intent angles.

    The reward is a clipped linear bonus on one selected Htheta/relation-intent
    angle.  The score is written at the first STOP token of each generated
    constraint sequence.

    This module assumes solver status has already been computed by
    ``SolverReward`` and gates itself to ``status == "FC"``.  That keeps the
    intent reward from competing with the lower-level process of making a sketch
    solvable in the first place.
    """

    name = "intent"

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        coeff_h_angle: float = 2.0,
        cap_h_angle: float = 1.0,
        h_angle_reward_metric: str = "response_relational_angle",
        relstruct_distance_mode: str = "symmetric",
        relstruct_reverse_weight: float = 0.5,
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
            relstruct_distance_mode: How relation-only subspaces are compared.
            relstruct_reverse_weight: Reverse-direction weight for
                ``mixed_subset`` relation-structure comparison.
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
        if relstruct_distance_mode not in RELATION_SUBSPACE_DISTANCE_MODES:
            raise ValueError(
                "relstruct_distance_mode must be one of "
                f"{sorted(RELATION_SUBSPACE_DISTANCE_MODES)}, got "
                f"{relstruct_distance_mode!r}"
            )
        if relstruct_reverse_weight < 0.0:
            raise ValueError(
                "relstruct_reverse_weight must be non-negative, got "
                f"{relstruct_reverse_weight}"
            )
        self.relstruct_distance_mode = relstruct_distance_mode
        self.relstruct_reverse_weight = relstruct_reverse_weight
        self.tokenizer = SketchTokenizer(context)

    def _bonus(self, value: float, cap: float, coeff: float) -> float:
        """Instance wrapper for the common clipped angle-to-bonus transform."""
        return _linear_bonus(value, cap, coeff)

    def _score_h_theta(
        self,
        pred_sketch: datalib.Sketch,
        pred_native: NativeResponse,
        gt_sketch: datalib.Sketch,
        gt_native: NativeResponse,
        stats: ResponseIntentBatchStats,
    ) -> float:
        """Compute the selected Htheta-family angle and convert it to reward."""
        h_metrics = _compute_h_theta_metrics(
            pred_sketch,
            pred_native,
            gt_sketch,
            gt_native,
            self.context,
            self.eps,
            self.relstruct_distance_mode,
            self.relstruct_reverse_weight,
        )
        selected_angle = h_metrics.value(self.h_angle_reward_metric)
        stats.add_h_theta(h_metrics, selected_angle)
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
            and the sample passed the FC gate.  ``None`` when the sample should
            not receive an intent-reward score.
        """
        if solver_status != "FC":
            stats.skipped_not_fc += 1
            return None

        cons_i, first_stop_idx = _first_stop_and_pad(cons_i_raw, length)
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
            stats.skipped_build += 1
            return None

        try:
            gt_native = get_or_compute_gt_native(
                ctx, gt_sketch, self.context, self.eps
            )
            sample_reward = self._score_h_theta(
                pred_sketch,
                pred_native,
                gt_sketch,
                gt_native,
                stats,
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
            stats.skipped_angle += 1
            return None

        stats.reward_vals.append(sample_reward)
        return first_stop_idx, sample_reward

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        """Compute sequence-level scores and scalar logging metrics for a batch."""
        B, L = ctx.cons_tokens.shape
        assert len(ctx.sketch_original_list) == B
        scores = torch.zeros((B, L), dtype=torch.float32)

        prim_np = ctx.prim_tokens.detach().cpu().numpy()
        cons_np = ctx.cons_tokens.detach().cpu().numpy()

        statuses = ctx.per_sample_solver_status
        if statuses is None:
            statuses = ["FC"] * B

        stats = ResponseIntentBatchStats()

        for i in range(B):
            result = self._score_sample(
                ctx,
                i,
                L,
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

        return RewardOutput(scores=scores, metrics=stats.to_metrics_dict())
