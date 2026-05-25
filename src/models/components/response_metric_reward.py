"""ResponseMetricReward — reward-oriented wrappers for v4 response metrics.

This module owns metric-as-reward signals such as canonical Hθ
(``d_H_angle``), the reward-oriented Hθ surrogate
``d_H_relstruct_angle = sqrt(d_relstruct^2 + leak_angle^2)``, and the Case-P
``d_SVD`` family.  Dimensional-count parsimony remains in ``RelStructReward``.
"""
from __future__ import annotations

import math
import statistics

import numpy as np
import torch

import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    _is_dimensional,
    compute_native_response,
    constraint_jacobian,
    pairwise_distance_observable,
)
from src.models.components.dfeas_reward import _build_intrinsic_probes
from src.models.components.relation_subspace_metric import (
    RELATION_SUBSPACE_DISTANCE_MODES,
    compute_relation_subspace_distance,
)
from src.models.components.solver_feedback import SolverFeedback
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import complete_constraint_values


def _orthonormal_range(R: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
    if R.numel() == 0 or R.shape[1] == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    U, S, _Vh = torch.linalg.svd(R, full_matrices=False)
    keep = int((S >= eps).sum().item())
    if keep == 0:
        return torch.zeros(R.shape[0], 0, dtype=R.dtype, device=R.device), 0
    return U[:, :keep].contiguous(), keep


def _stack_relation_jacobians_at(rels, state, context) -> tuple[torch.Tensor | None, int]:
    rows: list[torch.Tensor] = []
    for rel in rels:
        j = constraint_jacobian(rel, state, context)
        if j is None or j.shape[0] == 0:
            continue
        rows.append(j)
    if not rows:
        return None, 0
    Jr = torch.cat(rows, dim=0)
    return Jr, int(Jr.shape[0])


def _observable_subspace(
    sketch: datalib.Sketch,
    native: NativeResponse,
    eps: float,
) -> tuple[torch.Tensor, int]:
    """Orthonormal basis of range(J_phi R) in pairwise-distance space."""
    J_phi = pairwise_distance_observable(
        sketch,
        dtype=native.R.dtype,
    ).to(dtype=native.R.dtype)
    if J_phi.shape[0] == 0 or native.R.shape[1] == 0:
        return torch.zeros(J_phi.shape[0], 0, dtype=native.R.dtype), 0
    return _orthonormal_range(J_phi @ native.R, eps)


H_ANGLE_REWARD_METRICS = {
    "d_H_angle",
    "d_H_relstruct_angle",
    "d_sub_angle",
    "d_relstruct",
    "leak_angle",
}


def compute_h_angle_metrics(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    gt_rels: list,
    context,
    eps: float,
    relstruct_distance_mode: str = "symmetric",
    relstruct_reverse_weight: float = 0.5,
) -> dict[str, float | int]:
    """Return canonical Hθ plus the reward-oriented relation-only substitute."""
    U_W_P, dim_P_obs = _observable_subspace(pred_sketch, pred_native, eps)
    U_W_G, dim_G_obs = _observable_subspace(gt_sketch, gt_native, eps)

    d_norm = float(max(dim_P_obs, dim_G_obs, 1))
    if dim_P_obs > 0 and dim_G_obs > 0:
        align = U_W_G.T @ U_W_P
        sigmas = torch.linalg.svdvals(align).clamp(0.0, 1.0)
        cos2_sum = float((sigmas ** 2).sum())
    else:
        cos2_sum = 0.0
    d_sub_angle_sq = (d_norm - cos2_sum) / d_norm
    d_sub_angle_sq = min(max(d_sub_angle_sq, 0.0), 1.0)
    d_sub_angle = math.sqrt(d_sub_angle_sq)

    U_P_prim, dim_P_prim = _orthonormal_range(pred_native.R, eps)
    J_R, n_R = _stack_relation_jacobians_at(gt_rels, pred_native.state, context)
    if J_R is not None and dim_P_prim > 0 and n_R > 0:
        N_G_prim, dim_N = _orthonormal_range(J_R.T, eps)
        if dim_N > 0:
            overlap = N_G_prim.T @ U_P_prim
            leak_angle_sq = float((overlap ** 2).sum()) / float(dim_P_prim)
        else:
            leak_angle_sq = 0.0
    else:
        leak_angle_sq = 0.0
    leak_angle_sq = min(max(leak_angle_sq, 0.0), 1.0)
    leak_angle = math.sqrt(leak_angle_sq)

    relstruct = compute_relation_subspace_distance(
        pred_sketch,
        gt_sketch,
        pred_native,
        context,
        eps,
        relstruct_distance_mode,
        relstruct_reverse_weight,
    )
    d_relstruct = float(relstruct["d_relstruct"])

    d_h_angle = math.sqrt(max(d_sub_angle_sq + leak_angle_sq, 0.0))
    d_h_relstruct_angle = math.sqrt(max(d_relstruct ** 2 + leak_angle_sq, 0.0))
    return {
        "d_H_angle": d_h_angle,
        "d_H_relstruct_angle": d_h_relstruct_angle,
        "d_sub_angle": d_sub_angle,
        "d_relstruct": d_relstruct,
        "leak_angle": leak_angle,
        "d_relstruct_pred_subset_gt": float(relstruct["d_relstruct_pred_subset_gt"]),
        "d_relstruct_gt_subset_pred": float(relstruct["d_relstruct_gt_subset_pred"]),
        "relstruct_dim_pred": int(relstruct["relstruct_dim_pred"]),
        "relstruct_dim_gt": int(relstruct["relstruct_dim_gt"]),
        "h_obs_dim_pred": int(dim_P_obs),
        "h_obs_dim_gt": int(dim_G_obs),
    }


def compute_d_h_angle(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    gt_rels: list,
    context,
    eps: float,
) -> tuple[float, float, float, int, int]:
    """Backward-compatible canonical Hθ tuple."""
    metrics = compute_h_angle_metrics(
        pred_sketch,
        pred_native,
        gt_sketch,
        gt_native,
        gt_rels,
        context,
        eps,
    )
    return (
        float(metrics["d_H_angle"]),
        float(metrics["d_sub_angle"]),
        float(metrics["leak_angle"]),
        int(metrics["h_obs_dim_pred"]),
        int(metrics["h_obs_dim_gt"]),
    )


def compute_d_svd(
    pred_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    gt_native: NativeResponse,
    edit_probes: list,
    context,
    tau_p: float,
) -> dict[str, float | int] | None:
    """Return Case-P d_SVD metrics for a single Pred/GT pair."""
    J_phi_P = pairwise_distance_observable(
        pred_sketch,
        dtype=pred_native.R.dtype,
    ).to(dtype=pred_native.R.dtype)
    J_phi_G = pairwise_distance_observable(
        gt_sketch,
        dtype=gt_native.R.dtype,
    ).to(dtype=gt_native.R.dtype)
    if J_phi_P.shape[0] != J_phi_G.shape[0] or J_phi_P.shape[0] == 0:
        return None
    if pred_native.R.shape[1] == 0 or gt_native.R.shape[1] == 0:
        return None

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
    d_sigma_rel = float(torch.linalg.norm(sig_P - sig_G)) / (sigma_G_norm + 1e-12)

    sig_max = max(float(sig_P[0]), float(sig_G[0]), 1e-30)
    thresh = tau_p * sig_max
    k_eff_P = int((sig_P >= thresh).sum().item())
    k_eff_G = int((sig_G >= thresh).sum().item())
    k_norm = max(k_eff_P, k_eff_G, 1)
    if k_eff_P > 0 and k_eff_G > 0:
        M_align = U_G[:, :k_eff_G].T @ U_P[:, :k_eff_P]
        svals = torch.linalg.svdvals(M_align).clamp(0.0, 1.0)
        cos2_sum = float((svals ** 2).sum())
        d_sub_trunc_sq = (k_eff_P + k_eff_G - 2.0 * cos2_sum) / float(k_norm)
        d_sub_trunc = math.sqrt(max(d_sub_trunc_sq, 0.0))
    elif k_eff_P == 0 and k_eff_G == 0:
        d_sub_trunc = 0.0
    else:
        d_sub_trunc = math.sqrt((k_eff_P + k_eff_G) / float(k_norm))

    cos1 = abs(float(U_P[:, 0] @ U_G[:, 0]))
    d_top1 = math.sqrt(max(2.0 - 2.0 * cos1, 0.0))

    return {
        "d_SVD_sigma_rel": d_sigma_rel,
        "d_SVD_sub_trunc": d_sub_trunc,
        "d_SVD_top1": d_top1,
        "sigma_P_norm": sigma_P_norm,
        "sigma_G_norm": sigma_G_norm,
        "k_eff_P": k_eff_P,
        "k_eff_G": k_eff_G,
        "k_Q_used": int(J_Q_P.shape[0]),
    }


class ResponseMetricReward(torch.nn.Module):
    """Direct reward for Hθ components and optional d_SVD metrics."""

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        coeff_h_angle: float = 2.0,
        cap_h_angle: float = 1.0,
        h_angle_reward_metric: str = "d_H_relstruct_angle",
        relstruct_distance_mode: str = "symmetric",
        relstruct_reverse_weight: float = 0.5,
        coeff_d_svd: float = 1.0,
        cap_d_svd: float = 1.0,
        d_svd_metric: str = "d_SVD_sub_trunc",
        d_svd_probe_source: str = "gt_dim",
        tau_p: float = 0.01,
        use_h_angle: bool = True,
        use_d_svd: bool = False,
    ):
        super().__init__()
        self.context = context
        self.eps = eps
        self.coeff_h_angle = coeff_h_angle
        self.cap_h_angle = cap_h_angle
        if h_angle_reward_metric not in H_ANGLE_REWARD_METRICS:
            raise ValueError(
                "h_angle_reward_metric must be one of "
                f"{sorted(H_ANGLE_REWARD_METRICS)}, got "
                f"{h_angle_reward_metric!r}"
            )
        if relstruct_distance_mode not in RELATION_SUBSPACE_DISTANCE_MODES:
            raise ValueError(
                "relstruct_distance_mode must be one of "
                f"{sorted(RELATION_SUBSPACE_DISTANCE_MODES)}, got "
                f"{relstruct_distance_mode!r}"
            )
        self.h_angle_reward_metric = h_angle_reward_metric
        self.relstruct_distance_mode = relstruct_distance_mode
        self.relstruct_reverse_weight = relstruct_reverse_weight
        self.coeff_d_svd = coeff_d_svd
        self.cap_d_svd = cap_d_svd
        if d_svd_metric not in (
            "d_SVD_sub_trunc",
            "d_SVD_top1",
            "d_SVD_sigma_rel",
        ):
            raise ValueError(
                "d_svd_metric must be one of "
                "{'d_SVD_sub_trunc', 'd_SVD_top1', 'd_SVD_sigma_rel'}, "
                f"got {d_svd_metric!r}"
            )
        if d_svd_probe_source not in ("gt_dim", "intrinsic"):
            raise ValueError(
                "d_svd_probe_source must be 'gt_dim' or 'intrinsic', "
                f"got {d_svd_probe_source!r}"
            )
        if tau_p <= 0.0:
            raise ValueError(f"tau_p must be positive, got {tau_p}")
        self.d_svd_metric = d_svd_metric
        self.d_svd_probe_source = d_svd_probe_source
        self.tau_p = tau_p
        self.use_h_angle = use_h_angle
        self.use_d_svd = use_d_svd
        self.tokenizer = SketchTokenizer(context)

    def _bonus(self, value: float, cap: float, coeff: float) -> float:
        if cap <= 0.0:
            return 0.0
        clipped = min(max(value, 0.0), cap)
        return coeff * (1.0 - clipped / cap)

    def _build_pred_sketch(
        self,
        cons_tokens_np: np.ndarray,
        prim_tokens_np: np.ndarray,
        sketch_original: datalib.Sketch,
    ) -> datalib.Sketch:
        pred_sketch, _, _ = self.tokenizer.detokenize(
            {"p_val": prim_tokens_np, "c_val": cons_tokens_np},
            allow_invalid_constraints=True,
            reference_sketch=sketch_original,
        )
        if sketch_original is not None:
            pred_sketch = SolverFeedback.overwrite_primitives_with_original(
                pred_sketch, sketch_original
            )
        pred_sketch, _ = complete_constraint_values(
            pred_sketch, skip_invalid_constraints=True
        )
        return pred_sketch

    @torch.no_grad()
    def forward(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch],
        per_sample_status: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        B, L = cons_tokens.shape
        assert len(sketch_original_list) == B
        scores = torch.zeros((B, L), dtype=torch.float32)

        prim_np = prim_tokens.detach().cpu().numpy()
        cons_np = cons_tokens.detach().cpu().numpy()

        if per_sample_status is None:
            per_sample_status = ["FC"] * B

        reward_vals: list[float] = []
        h_reward_distance_vals: list[float] = []
        h_angle_vals: list[float] = []
        h_relstruct_angle_vals: list[float] = []
        h_sub_angle_vals: list[float] = []
        h_relstruct_vals: list[float] = []
        h_leak_angle_vals: list[float] = []
        h_relstruct_pred_subset_gt_vals: list[float] = []
        h_relstruct_gt_subset_pred_vals: list[float] = []
        h_relstruct_dim_pred_vals: list[int] = []
        h_relstruct_dim_gt_vals: list[int] = []
        h_obs_dim_pred_vals: list[int] = []
        h_obs_dim_gt_vals: list[int] = []
        svd_selected_vals: list[float] = []
        svd_sigma_rel_vals: list[float] = []
        svd_sub_trunc_vals: list[float] = []
        svd_top1_vals: list[float] = []
        svd_sigma_pred_norm_vals: list[float] = []
        svd_sigma_gt_norm_vals: list[float] = []
        svd_k_used_vals: list[int] = []

        skipped_build = 0
        skipped_not_fc = 0
        skipped_h_angle = 0
        skipped_d_svd = 0
        gt_native_cache: dict[int, NativeResponse] = {}
        gt_probe_cache: dict[int, list] = {}

        for i in range(B):
            if per_sample_status[i] != "FC":
                skipped_not_fc += 1
                continue

            cons_i = cons_np[i].copy()
            stop_pos = np.where(cons_i == int(ConstraintToken.Stop))[0]
            first_stop_idx = int(stop_pos[0]) if len(stop_pos) > 0 else (L - 1)
            if 0 <= first_stop_idx < L - 1:
                cons_i[first_stop_idx + 1:] = int(ConstraintToken.Pad)

            gt_sketch = sketch_original_list[i]
            try:
                pred_sketch = self._build_pred_sketch(cons_i, prim_np[i], gt_sketch)
                pred_native = compute_native_response(
                    pred_sketch, self.context, eps=self.eps
                )
            except Exception:
                skipped_build += 1
                continue

            gt_rels = [
                c for c in gt_sketch.constraints.values()
                if not _is_dimensional(c)
            ]
            gt_key = id(gt_sketch)
            sample_reward = 0.0

            if self.use_h_angle:
                try:
                    if gt_key not in gt_native_cache:
                        gt_native_cache[gt_key] = compute_native_response(
                            gt_sketch, self.context, eps=self.eps
                        )
                    h_metrics = compute_h_angle_metrics(
                        pred_sketch,
                        pred_native,
                        gt_sketch,
                        gt_native_cache[gt_key],
                        gt_rels,
                        self.context,
                        self.eps,
                        self.relstruct_distance_mode,
                        self.relstruct_reverse_weight,
                    )
                    dha = float(h_metrics["d_H_angle"])
                    dhra = float(h_metrics["d_H_relstruct_angle"])
                    dsa = float(h_metrics["d_sub_angle"])
                    drs = float(h_metrics["d_relstruct"])
                    la = float(h_metrics["leak_angle"])
                    h_reward_distance = float(
                        h_metrics[self.h_angle_reward_metric]
                    )
                    h_angle_vals.append(dha)
                    h_relstruct_angle_vals.append(dhra)
                    h_sub_angle_vals.append(dsa)
                    h_relstruct_vals.append(drs)
                    h_leak_angle_vals.append(la)
                    h_reward_distance_vals.append(h_reward_distance)
                    h_relstruct_pred_subset_gt_vals.append(
                        float(h_metrics["d_relstruct_pred_subset_gt"])
                    )
                    h_relstruct_gt_subset_pred_vals.append(
                        float(h_metrics["d_relstruct_gt_subset_pred"])
                    )
                    h_relstruct_dim_pred_vals.append(
                        int(h_metrics["relstruct_dim_pred"])
                    )
                    h_relstruct_dim_gt_vals.append(
                        int(h_metrics["relstruct_dim_gt"])
                    )
                    h_obs_dim_pred_vals.append(int(h_metrics["h_obs_dim_pred"]))
                    h_obs_dim_gt_vals.append(int(h_metrics["h_obs_dim_gt"]))
                    sample_reward += self._bonus(
                        h_reward_distance, self.cap_h_angle, self.coeff_h_angle
                    )
                except Exception:
                    skipped_h_angle += 1

            if self.use_d_svd:
                try:
                    if gt_key not in gt_native_cache:
                        gt_native_cache[gt_key] = compute_native_response(
                            gt_sketch, self.context, eps=self.eps
                        )
                    if gt_key not in gt_probe_cache:
                        if self.d_svd_probe_source == "gt_dim":
                            gt_probe_cache[gt_key] = [
                                c for c in gt_sketch.constraints.values()
                                if _is_dimensional(c)
                            ]
                        else:
                            gt_probe_cache[gt_key] = _build_intrinsic_probes(gt_sketch)
                    dsvd = compute_d_svd(
                        pred_sketch,
                        pred_native,
                        gt_sketch,
                        gt_native_cache[gt_key],
                        gt_probe_cache[gt_key],
                        self.context,
                        self.tau_p,
                    )
                    if dsvd is None:
                        skipped_d_svd += 1
                    else:
                        selected = float(dsvd[self.d_svd_metric])
                        svd_selected_vals.append(selected)
                        svd_sigma_rel_vals.append(float(dsvd["d_SVD_sigma_rel"]))
                        svd_sub_trunc_vals.append(float(dsvd["d_SVD_sub_trunc"]))
                        svd_top1_vals.append(float(dsvd["d_SVD_top1"]))
                        svd_sigma_pred_norm_vals.append(float(dsvd["sigma_P_norm"]))
                        svd_sigma_gt_norm_vals.append(float(dsvd["sigma_G_norm"]))
                        svd_k_used_vals.append(int(dsvd["k_Q_used"]))
                        sample_reward += self._bonus(
                            selected, self.cap_d_svd, self.coeff_d_svd
                        )
                except Exception:
                    skipped_d_svd += 1

            scores[i, first_stop_idx] += sample_reward
            reward_vals.append(sample_reward)

        def _mean(xs: list) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _median(xs: list) -> float:
            return statistics.median(xs) if xs else 0.0

        metrics: dict[str, float | int] = {
            "response_metric_reward_mean": _mean(reward_vals),
            "response_metric_reward_median": _median(reward_vals),
            "response_metric_h_reward_distance_mean": _mean(h_reward_distance_vals),
            "response_metric_h_reward_distance_median": _median(h_reward_distance_vals),
            "response_metric_d_h_angle_mean": _mean(h_angle_vals),
            "response_metric_d_h_angle_median": _median(h_angle_vals),
            "response_metric_d_h_relstruct_angle_mean": _mean(h_relstruct_angle_vals),
            "response_metric_d_h_relstruct_angle_median": _median(h_relstruct_angle_vals),
            "response_metric_d_sub_angle_mean": _mean(h_sub_angle_vals),
            "response_metric_d_relstruct_mean": _mean(h_relstruct_vals),
            "response_metric_d_relstruct_median": _median(h_relstruct_vals),
            "response_metric_d_relstruct_pred_subset_gt_mean": _mean(h_relstruct_pred_subset_gt_vals),
            "response_metric_d_relstruct_gt_subset_pred_mean": _mean(h_relstruct_gt_subset_pred_vals),
            "response_metric_relstruct_dim_pred_mean": _mean(h_relstruct_dim_pred_vals),
            "response_metric_relstruct_dim_gt_mean": _mean(h_relstruct_dim_gt_vals),
            "response_metric_leak_angle_mean": _mean(h_leak_angle_vals),
            "response_metric_h_obs_dim_pred_mean": _mean(h_obs_dim_pred_vals),
            "response_metric_h_obs_dim_gt_mean": _mean(h_obs_dim_gt_vals),
            "response_metric_d_svd_selected_mean": _mean(svd_selected_vals),
            "response_metric_d_svd_selected_median": _median(svd_selected_vals),
            "response_metric_d_svd_sigma_rel_mean": _mean(svd_sigma_rel_vals),
            "response_metric_d_svd_sub_trunc_mean": _mean(svd_sub_trunc_vals),
            "response_metric_d_svd_top1_mean": _mean(svd_top1_vals),
            "response_metric_d_svd_sigma_pred_norm_mean": _mean(svd_sigma_pred_norm_vals),
            "response_metric_d_svd_sigma_gt_norm_mean": _mean(svd_sigma_gt_norm_vals),
            "response_metric_d_svd_k_used_mean": _mean(svd_k_used_vals),
            "response_metric_valid_n": len(reward_vals),
            "response_metric_skipped_build": skipped_build,
            "response_metric_skipped_not_fc": skipped_not_fc,
            "response_metric_skipped_h_angle": skipped_h_angle,
            "response_metric_skipped_d_svd": skipped_d_svd,
        }
        return scores, metrics
