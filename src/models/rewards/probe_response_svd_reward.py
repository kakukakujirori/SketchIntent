"""ProbeResponseSVDReward — probe-conditioned response-SVD reward.

This module owns the probe-conditioned response-SVD family that was originally
housed next to the Htheta / relation-intent reward.  The two metric families
share the same first-order response map ``R_C = dx / dp_C``, but they ask
different questions:

- Htheta / relation-intent compares the allowed-motion structure of a sketch.
- Case-P first conditions that motion on a shared edit-probe stack ``J_Q`` and
  then compares the observable response operator
  ``T_C = J_phi R_C pinv(J_Q R_C)``.

That makes this reward a useful supplement for response shrinkage / top
response-direction diagnostics, but not part of the default relation-intent
reward.
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
    constraint_jacobian,
    pairwise_distance_observable,
)
from src.models.rewards.base import BaseReward, RewardContext, RewardOutput
from src.models.rewards.utils.probes import (
    build_intrinsic_probes as _build_intrinsic_probes,
)
from src.models.rewards.utils.sketch_builder import (
    get_or_build_pred_native,
    get_or_compute_gt_native,
    get_or_compute_j_phi,
)
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


logger = logging.getLogger(__name__)


PROBE_RESPONSE_SVD_METRICS = {
    "probe_response_subspace",
    "probe_response_top1",
    "probe_response_spectrum_rel",
}


def _validate_probe_response_svd_metric_name(name: str) -> str:
    """Validate and return a config-facing probe-response SVD metric name."""
    if name not in PROBE_RESPONSE_SVD_METRICS:
        valid = sorted(PROBE_RESPONSE_SVD_METRICS)
        raise ValueError(f"metric must be one of {valid}, got {name!r}")
    return name


def _mean(xs: Sequence[float | int]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _median(xs: Sequence[float | int]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _linear_bonus(value: float, cap: float, coeff: float) -> float:
    """Convert a low-is-good metric value into a clipped reward bonus."""
    if cap <= 0.0:
        return 0.0
    clipped = min(max(value, 0.0), cap)
    return coeff * (1.0 - clipped / cap)


def _first_stop_and_pad(cons_tokens_np: np.ndarray, length: int) -> tuple[np.ndarray, int]:
    """Return a constraint sequence truncated after the first STOP token.

    Tokens beyond the first STOP are replaced with Pad so they cannot leak
    spurious constraints into sketch reconstruction.  If the model runs to
    the max generation length without emitting a STOP, ``first_stop_idx``
    falls back to ``length - 1`` and no padding is applied — the reward is
    still placed at the last position.

    A duplicate copy of this helper lives in
    ``src/models/rewards/response_metric_reward.py``; the two should stay in
    sync.  Kept duplicated rather than moved to ``utils/`` because no other
    file currently needs it.
    """
    cons_i = cons_tokens_np.copy()
    stop_pos = np.where(cons_i == int(ConstraintToken.Stop))[0]
    first_stop_idx = int(stop_pos[0]) if len(stop_pos) > 0 else (length - 1)
    if 0 <= first_stop_idx < length - 1:
        cons_i[first_stop_idx + 1:] = int(ConstraintToken.Pad)
    return cons_i, first_stop_idx


@dataclass(frozen=True)
class ProbeResponseSVDMetrics:
    """Case-P probe-conditioned response SVD values for one Pred/GT pair.

    Case-P compares ``T_C = J_phi R_C pinv(J_Q R_C)`` for Pred and GT.
    ``probe_response_subspace`` and ``probe_response_top1`` compare left
    singular-vector directions.  ``probe_response_spectrum_rel`` compares the
    singular-value magnitudes, catching response shrinkage that subspace-only
    metrics can miss.
    """

    probe_response_spectrum_rel: float
    probe_response_subspace: float
    probe_response_top1: float
    sigma_pred_norm: float
    sigma_gt_norm: float
    effective_rank_pred: int
    effective_rank_gt: int
    probe_count: int

    def value(self, metric_name: str) -> float:
        """Return a probe-response SVD metric selected by config-facing name."""
        metric_name = _validate_probe_response_svd_metric_name(metric_name)
        return float(self.to_dict()[metric_name])

    def to_dict(self) -> dict[str, float | int]:
        """Expose the public probe-response SVD metric names."""
        return {
            "probe_response_spectrum_rel": self.probe_response_spectrum_rel,
            "probe_response_subspace": self.probe_response_subspace,
            "probe_response_top1": self.probe_response_top1,
            "sigma_pred_norm": self.sigma_pred_norm,
            "sigma_gt_norm": self.sigma_gt_norm,
            "effective_rank_pred": self.effective_rank_pred,
            "effective_rank_gt": self.effective_rank_gt,
            "probe_count": self.probe_count,
        }


@dataclass
class ProbeResponseSVDBatchStats:
    """Mutable aggregation buffer for Case-P batch-level reward logging."""

    reward_vals: list[float] = field(default_factory=list)
    selected_vals: list[float] = field(default_factory=list)
    spectrum_rel_vals: list[float] = field(default_factory=list)
    subspace_vals: list[float] = field(default_factory=list)
    top1_vals: list[float] = field(default_factory=list)
    sigma_pred_norm_vals: list[float] = field(default_factory=list)
    sigma_gt_norm_vals: list[float] = field(default_factory=list)
    effective_rank_pred_vals: list[int] = field(default_factory=list)
    effective_rank_gt_vals: list[int] = field(default_factory=list)
    probe_count_vals: list[int] = field(default_factory=list)
    skipped_build: int = 0
    skipped_gate: int = 0
    skipped_metric: int = 0

    def add_probe_response_svd(
        self,
        metrics: ProbeResponseSVDMetrics,
        selected_value: float,
    ) -> None:
        """Record one successful Case-P computation."""
        self.selected_vals.append(selected_value)
        self.spectrum_rel_vals.append(metrics.probe_response_spectrum_rel)
        self.subspace_vals.append(metrics.probe_response_subspace)
        self.top1_vals.append(metrics.probe_response_top1)
        self.sigma_pred_norm_vals.append(metrics.sigma_pred_norm)
        self.sigma_gt_norm_vals.append(metrics.sigma_gt_norm)
        self.effective_rank_pred_vals.append(metrics.effective_rank_pred)
        self.effective_rank_gt_vals.append(metrics.effective_rank_gt)
        self.probe_count_vals.append(metrics.probe_count)

    def to_metrics_dict(self) -> dict[str, float | int]:
        """Return flat scalar metrics under ``probe_response_svd_*`` keys."""
        return {
            "probe_response_svd_reward_mean": _mean(self.reward_vals),
            "probe_response_svd_reward_median": _median(self.reward_vals),
            "probe_response_svd_selected_mean": _mean(self.selected_vals),
            "probe_response_svd_selected_median": _median(self.selected_vals),
            "probe_response_svd_spectrum_rel_mean": _mean(self.spectrum_rel_vals),
            "probe_response_svd_subspace_mean": _mean(self.subspace_vals),
            "probe_response_svd_top1_mean": _mean(self.top1_vals),
            "probe_response_svd_sigma_pred_norm_mean": _mean(
                self.sigma_pred_norm_vals
            ),
            "probe_response_svd_sigma_gt_norm_mean": _mean(
                self.sigma_gt_norm_vals
            ),
            "probe_response_svd_effective_rank_pred_mean": _mean(
                self.effective_rank_pred_vals
            ),
            "probe_response_svd_effective_rank_gt_mean": _mean(
                self.effective_rank_gt_vals
            ),
            "probe_response_svd_probe_count_mean": _mean(self.probe_count_vals),
            "probe_response_svd_valid_n": len(self.reward_vals),
            "probe_response_svd_skipped_build": self.skipped_build,
            "probe_response_svd_skipped_gate": self.skipped_gate,
            "probe_response_svd_skipped_metric": self.skipped_metric,
        }


def _compute_probe_response_svd_metrics(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    edit_probes: list,
    context,
    tau_p: float,
    *,
    J_phi_pred: torch.Tensor | None = None,
    J_phi_gt: torch.Tensor | None = None,
) -> ProbeResponseSVDMetrics | None:
    """Return probe-response SVD metrics for a single Pred/GT pair.

    Returning ``None`` means the pair is not comparable for this diagnostic:
    empty observable space, no native dimensional response columns, or no probe
    whose Jacobian can be evaluated on both Pred and GT.
    """
    if J_phi_pred is None:
        J_phi_pred = pairwise_distance_observable(
            pred_sketch,
            dtype=pred_native.R.dtype,
        ).to(dtype=pred_native.R.dtype)
    if J_phi_gt is None:
        J_phi_gt = pairwise_distance_observable(
            gt_sketch,
            dtype=gt_native.R.dtype,
        ).to(dtype=gt_native.R.dtype)
    J_phi_P = J_phi_pred
    J_phi_G = J_phi_gt
    if J_phi_P.shape[0] != J_phi_G.shape[0] or J_phi_P.shape[0] == 0:
        return None
    if pred_native.R.shape[1] == 0 or gt_native.R.shape[1] == 0:
        return None

    # Keep only probes whose Jacobian exists on both sketches.  We use the first
    # row to preserve the scalar-probe convention from the v4 experiments.
    J_Q_P_rows: list[torch.Tensor] = []
    J_Q_G_rows: list[torch.Tensor] = []
    for probe in edit_probes:
        Jq_P = constraint_jacobian(probe, pred_native.state, context)
        Jq_G = constraint_jacobian(probe, gt_native.state, context)
        if Jq_P is None or Jq_G is None:
            continue
        J_Q_P_rows.append(Jq_P[:1])
        J_Q_G_rows.append(Jq_G[:1])
    if not J_Q_P_rows:
        return None

    J_Q_P = torch.cat(J_Q_P_rows, dim=0)
    J_Q_G = torch.cat(J_Q_G_rows, dim=0)
    M_P = J_Q_P @ pred_native.R
    M_G = J_Q_G @ gt_native.R

    # S_C is the minimum-native-parameter response that realizes the shared
    # probe stack; T_C observes that response in pairwise-distance space.
    S_P = pred_native.R @ torch.linalg.pinv(M_P)
    S_G = gt_native.R @ torch.linalg.pinv(M_G)
    T_P = J_phi_P @ S_P
    T_G = J_phi_G @ S_G

    U_P, sig_P, _ = torch.linalg.svd(T_P, full_matrices=False)
    U_G, sig_G, _ = torch.linalg.svd(T_G, full_matrices=False)
    if sig_P.numel() == 0 or sig_G.numel() == 0:
        return None

    sigma_P_norm = float(torch.linalg.norm(sig_P))
    sigma_G_norm = float(torch.linalg.norm(sig_G))
    spectrum_rel = float(torch.linalg.norm(sig_P - sig_G)) / (sigma_G_norm + 1e-12)

    # Tiny singular directions are unstable arbitrary basis choices, so
    # subspace comparison keeps only directions above a relative threshold.
    sig_max = max(float(sig_P[0]), float(sig_G[0]), 1e-30)
    thresh = tau_p * sig_max
    k_eff_P = int((sig_P >= thresh).sum().item())
    k_eff_G = int((sig_G >= thresh).sum().item())
    k_norm = max(k_eff_P, k_eff_G, 1)
    if k_eff_P > 0 and k_eff_G > 0:
        M_align = U_G[:, :k_eff_G].T @ U_P[:, :k_eff_P]
        svals = torch.linalg.svdvals(M_align).clamp(0.0, 1.0)
        cos2_sum = float((svals ** 2).sum())
        subspace_sq = (k_eff_P + k_eff_G - 2.0 * cos2_sum) / float(k_norm)
        subspace = math.sqrt(max(subspace_sq, 0.0))
    elif k_eff_P == 0 and k_eff_G == 0:
        subspace = 0.0
    else:
        subspace = math.sqrt((k_eff_P + k_eff_G) / float(k_norm))

    cos1 = abs(float(U_P[:, 0] @ U_G[:, 0]))
    top1 = math.sqrt(max(2.0 - 2.0 * cos1, 0.0))

    return ProbeResponseSVDMetrics(
        probe_response_spectrum_rel=spectrum_rel,
        probe_response_subspace=subspace,
        probe_response_top1=top1,
        sigma_pred_norm=sigma_P_norm,
        sigma_gt_norm=sigma_G_norm,
        effective_rank_pred=k_eff_P,
        effective_rank_gt=k_eff_G,
        probe_count=int(J_Q_P.shape[0]),
    )


class ProbeResponseSVDReward(BaseReward):
    """Sequence-level reward from one selected probe-response SVD metric.

    This reward is disabled in the default RLOO config.  It can be added as a
    supplement when we want to explicitly reward probe-conditioned observable
    response similarity, especially response-subspace or response-shrinkage
    behavior.
    """

    name = "probe_response_svd"

    def __init__(
        self,
        context: SketchContext,
        coeff: float = 1.0,
        cap: float = 1.0,
        metric: str = "probe_response_subspace",
        probe_source: str = "gt_dim",
        tau_p: float = 0.01,
        eps: float = 1e-9,
        gate: str = "fc",
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """Create a Case-P probe-response SVD reward."""
        super().__init__(weight=weight, enabled=enabled)
        if probe_source not in ("gt_dim", "intrinsic"):
            raise ValueError(
                "probe_source must be 'gt_dim' or 'intrinsic', "
                f"got {probe_source!r}"
            )
        if tau_p <= 0.0:
            raise ValueError(f"tau_p must be positive, got {tau_p}")
        if gate not in ("always", "converged", "fc"):
            raise ValueError(f"gate must be one of always/converged/fc, got {gate!r}")
        self.context = context
        self.coeff = float(coeff)
        self.cap = float(cap)
        self.metric = _validate_probe_response_svd_metric_name(metric)
        self.probe_source = probe_source
        self.tau_p = float(tau_p)
        self.eps = float(eps)
        self.gate = gate
        self.tokenizer = SketchTokenizer(context)

    def _gate_pass(self, status: str | None) -> bool:
        if self.gate == "always" or status is None:
            return True
        if self.gate == "converged":
            return status in ("FC", "Under")
        return status == "FC"

    def _get_gt_probes(
        self,
        gt_sketch: datalib.Sketch,
        cache: dict[int, list],
    ) -> list:
        """Cached GT-probe list keyed by ``id(gt_sketch)`` (replicated in RLOO)."""
        gt_key = id(gt_sketch)
        if gt_key in cache:
            return cache[gt_key]
        if self.probe_source == "gt_dim":
            probes = [
                c for c in gt_sketch.constraints.values()
                if _is_dimensional(c)
            ]
        else:
            probes = _build_intrinsic_probes(gt_sketch)
        cache[gt_key] = probes
        return probes

    def _score_sample(
        self,
        ctx: RewardContext,
        i: int,
        length: int,
        prim_i: np.ndarray,
        cons_i_raw: np.ndarray,
        gt_sketch: datalib.Sketch,
        solver_status: str | None,
        gt_probe_cache: dict[int, list],
        stats: ProbeResponseSVDBatchStats,
    ) -> tuple[int, float] | None:
        if not self._gate_pass(solver_status):
            stats.skipped_gate += 1
            return None

        cons_i, first_stop_idx = _first_stop_and_pad(cons_i_raw, length)
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
                "probe_response_svd: skipped sample %d (build failure: %s)",
                i,
                exc,
            )
            stats.skipped_build += 1
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
            probe_response_svd = _compute_probe_response_svd_metrics(
                pred_sketch,
                pred_native,
                gt_sketch,
                gt_native,
                self._get_gt_probes(gt_sketch, gt_probe_cache),
                self.context,
                self.tau_p,
                J_phi_pred=J_phi_pred,
                J_phi_gt=J_phi_gt,
            )
        except Exception as exc:
            logger.debug(
                "probe_response_svd: skipped sample %d (metric failure: %s)",
                i,
                exc,
            )
            stats.skipped_metric += 1
            return None

        if probe_response_svd is None:
            stats.skipped_metric += 1
            return None

        selected_value = probe_response_svd.value(self.metric)
        stats.add_probe_response_svd(probe_response_svd, selected_value)
        reward = _linear_bonus(selected_value, self.cap, self.coeff)
        stats.reward_vals.append(reward)
        return first_stop_idx, reward

    @torch.no_grad()
    def compute(self, ctx: RewardContext) -> RewardOutput:
        cons_tokens = ctx.cons_tokens
        sketches = ctx.sketch_original_list
        statuses = ctx.per_sample_solver_status
        B, L = cons_tokens.shape
        scores = torch.zeros((B, L), dtype=torch.float32)

        prim_np = ctx.prim_tokens.detach().cpu().numpy()
        cons_np = cons_tokens.detach().cpu().numpy()
        if statuses is None:
            statuses = ["FC"] * B

        stats = ProbeResponseSVDBatchStats()
        gt_probe_cache: dict[int, list] = {}

        for i in range(B):
            result = self._score_sample(
                ctx,
                i,
                L,
                prim_np[i],
                cons_np[i],
                sketches[i],
                statuses[i],
                gt_probe_cache,
                stats,
            )
            if result is None:
                continue
            first_stop_idx, reward = result
            scores[i, first_stop_idx] += reward

        return RewardOutput(scores=scores, metrics=stats.to_metrics_dict())
