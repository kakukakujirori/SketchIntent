"""RelStructReward — Plan A″ RL reward for design-intent-aware constraint inference.

This is the RL reward counterpart to the d_H evaluation metric (v4 paper).
Rather than using d_H directly (which carries annotator-variance issues from
d_sub), this reward decomposes the intent-awareness signal into three
independently gateable components:

  **leak_prime**    ||J_{R_G} U_P||_F / sqrt(n_R · dim V_P)
      Measures how much Pred's response subspace "leaks through" GT's
      relational constraints.  Bonus applied at  FC gate (status == "FC").

  **d_relstruct**   subspace distance between null(J_{R_P,rel}) and
                    null(J_{R_G,rel}) in pairwise-distance observable space.
      The default symmetric mode measures relational structure equivalence in
      both directions (unlike leak_prime which is one-sided).  The optional
      directed / mixed modes can soften the penalty when pseudo-GT relations
      are treated as high-precision but incomplete.  Bonus at Converged gate
      (status in {"FC","Under"}).

  **gain_intrinsic** mean per-probe attainability for intrinsic size probes
      (Line→Length, Arc/Circle→Radius).  Lower = better attainability.
      Bonus at  Converged ∧ ¬Redundant gate (status == "FC").

  **dim_penalty**   -η · f(#dimensional_constraints_in_Pred)
      Explicit parsimony penalty to counteract dim inflation.
      Applied at  Converged gate.

Each component can be enabled/disabled individually so the reward can be
built up incrementally (smoke-test leak_prime first, then add d_relstruct,
then gain_intrinsic + dim_penalty) following the staged rollout in
notebooks/plan_dH_rl_reward.md §10.3.

Reward formula (A″):
    r = α  · max(0, 1 − leak_prime    / cap_leak)  · 1[FC]
      + α' · max(0, 1 − d_relstruct   / cap_rs)    · 1[Converged]
      + β  · max(0, 1 − gain_intrinsic / cap_gain) · 1[FC]
      − η  · dim_penalty_value                      · 1[Converged]

The reward is placed at the STOP token position to match SolverReward convention.
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
    _is_relational,
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
    pairwise_distance_observable,
)
from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_solver import compute_null_space
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import complete_constraint_values
from src.models.components.dfeas_reward import _build_intrinsic_probes
from src.models.components.solver_feedback import SolverFeedback

# Re-export helper from metrics_v4 (relative import to avoid circular deps)
try:
    from notebooks.v4_exp.metrics_v4 import (
        _orthonormal_range,
        _stack_relation_jacobians_at,
    )
except ImportError:
    # Inline fallback so the module can be imported outside the notebooks tree
    def _orthonormal_range(R: torch.Tensor, eps: float) -> tuple[torch.Tensor, int]:
        if R.numel() == 0 or R.shape[1] == 0:
            return torch.zeros(R.shape[0], 0, dtype=R.dtype), 0
        U, S, _Vh = torch.linalg.svd(R, full_matrices=False)
        keep = int((S >= eps).sum().item())
        if keep == 0:
            return torch.zeros(R.shape[0], 0, dtype=R.dtype), 0
        return U[:, :keep].contiguous(), keep

    def _stack_relation_jacobians_at(rels, state, context):
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


# ---------------------------------------------------------------------------
# Per-sample scalar computations
# ---------------------------------------------------------------------------

def _compute_leak_prime(
    pred_native: NativeResponse,
    gt_rels: list,
    context,
    eps: float,
) -> float:
    """||J_{R_G} U_P||_F / sqrt(n_R * dim_P_prim).

    J_{R_G} is evaluated at Pred's state (which shares GT primitive values).
    """
    U_P, dim_P = _orthonormal_range(pred_native.R, eps)
    if dim_P == 0:
        return 0.0
    J_R, n_R = _stack_relation_jacobians_at(gt_rels, pred_native.state, context)
    if J_R is None or n_R == 0:
        return 0.0
    leak_op = J_R @ U_P               # (n_R, dim_P)
    leak_sq = float((leak_op ** 2).sum()) / float(n_R * dim_P)
    return math.sqrt(max(leak_sq, 0.0))


def _compute_d_relstruct(
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    pred_native: NativeResponse,
    context,
    eps: float,
    distance_mode: str = "symmetric",
    reverse_weight: float = 0.5,
) -> tuple[float, float, float, int, int]:
    """Subspace distance between null(J_{R_P}) and null(J_{R_G}) in observable space.

    Both Jacobians are evaluated at Pred's state, which carries GT primitive
    coordinates (via overwrite_primitives_with_original), so the null spaces
    live in the same ambient tangent space.

    Let C = ||U_G^T U_P||_F^2, where U_* spans range(J_phi null(J_R_*)).
    The supported distances are:

      symmetric:
        (dim_P + dim_G - 2C) / max(dim_P, dim_G)

      pred_subset_gt:
        (dim_P - C) / dim_P
        This is zero when Pred's relation-only observable null space is
        contained in GT's relation-only observable null space.

      mixed_subset:
        d(P⊂G)^2 + reverse_weight * d(G⊂P)^2

    Returns (selected_distance, pred_subset_gt, gt_subset_pred, dim_pred, dim_gt).
    """
    state = pred_native.state
    n_x = state.num_dof()
    dtype = state.q.dtype

    pred_rels = [c for c in pred_sketch.constraints.values() if _is_relational(c)]
    gt_rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]

    J_R_pred, _ = _stack_relation_jacobians_at(pred_rels, state, context)
    J_R_gt, _ = _stack_relation_jacobians_at(gt_rels, state, context)

    if J_R_pred is None:
        J_R_pred = torch.zeros(0, n_x, dtype=dtype)
    if J_R_gt is None:
        J_R_gt = torch.zeros(0, n_x, dtype=dtype)

    # Null spaces: (n_x, dim_null_*); empty relations → entire R^{n_x}
    N_pred = compute_null_space(J_R_pred, eps=eps)
    N_gt = compute_null_space(J_R_gt, eps=eps)

    # Observable Jacobian: pairwise distances, shape (n_obs, n_x)
    J_phi = pairwise_distance_observable(pred_sketch).to(dtype=dtype)
    if J_phi.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0, 0

    W_pred = J_phi @ N_pred   # (n_obs, dim_null_pred)
    W_gt = J_phi @ N_gt       # (n_obs, dim_null_gt)

    U_pred, dim_pred = _orthonormal_range(W_pred, eps)
    U_gt, dim_gt = _orthonormal_range(W_gt, eps)

    d_norm = float(max(dim_pred, dim_gt, 1))

    if dim_pred == 0 or dim_gt == 0:
        cos2_sum = 0.0
    else:
        M = U_gt.T @ U_pred          # (dim_gt, dim_pred)
        sigmas = torch.linalg.svdvals(M).clamp(0.0, 1.0)
        cos2_sum = float((sigmas ** 2).sum())

    pred_subset_gt_sq = 0.0
    if dim_pred > 0:
        pred_subset_gt_sq = (dim_pred - cos2_sum) / float(dim_pred)

    gt_subset_pred_sq = 0.0
    if dim_gt > 0:
        gt_subset_pred_sq = (dim_gt - cos2_sum) / float(dim_gt)

    if distance_mode == "symmetric":
        d_rs_sq = (dim_pred + dim_gt - 2.0 * cos2_sum) / d_norm
    elif distance_mode == "pred_subset_gt":
        d_rs_sq = pred_subset_gt_sq
    elif distance_mode == "mixed_subset":
        d_rs_sq = pred_subset_gt_sq + reverse_weight * gt_subset_pred_sq
    else:
        raise ValueError(f"unknown relstruct_distance_mode: {distance_mode!r}")

    return (
        math.sqrt(max(d_rs_sq, 0.0)),
        math.sqrt(max(pred_subset_gt_sq, 0.0)),
        math.sqrt(max(gt_subset_pred_sq, 0.0)),
        dim_pred,
        dim_gt,
    )


def _compute_gain_intrinsic(
    pred_native: NativeResponse,
    gt_sketch: datalib.Sketch,
    context,
    eps: float,
) -> tuple[float, int]:
    """Mean per-probe gain for intrinsic size probes (Line→Length, Arc/Circle→Radius).

    Lower = better: gain → 0 means each entity's intrinsic size is fully
    attainable via pred's dimensional constraints.

    Returns (gain_intrinsic, n_probes_used).
    """
    probes = _build_intrinsic_probes(gt_sketch)
    if not probes:
        return 0.0, 0

    gain_sum = 0.0
    count = 0
    for probe in probes:
        try:
            S_pr = edit_response_for_probe(pred_native, probe, context, eps=eps)
            if S_pr is None:
                continue
            Jq = constraint_jacobian(probe, pred_native.state, context)
            if Jq is None:
                continue
            k_q = max(1, Jq.shape[0])
            eye = torch.eye(Jq.shape[0], dtype=Jq.dtype, device=Jq.device)
            gain_residual = Jq @ S_pr - eye
            gain_ms = float(torch.linalg.norm(gain_residual) ** 2) / k_q
            gain_sum += math.sqrt(max(gain_ms, 0.0))
            count += 1
        except Exception:
            continue

    if count == 0:
        return 0.0, 0
    return gain_sum / count, count


# ---------------------------------------------------------------------------
# Reward class
# ---------------------------------------------------------------------------

class RelStructReward(torch.nn.Module):
    """Plan A″ reward: leak_prime + d_relstruct + gain_intrinsic + dim_penalty.

    Args:
        context: tokeniser / sketch handler context.
        eps: SVD rank cutoff (absolute threshold; matches v4 default 1e-9).

        coeff_leak: multiplier α for leak_prime bonus. Default 5.0 follows
            roadmap.md L295-303 for the d_feas leak bonus.
        cap_leak: clip denominator for leak_prime bonus. v4_results §3 shows
            solver_rl=0.073, rel_drop50=0.195, so 0.3 is a reasonable cap.

        coeff_relstruct: multiplier α' for d_relstruct bonus. Default 3.0
            (conservative relative to leak because d_relstruct is novel and
            depends on pseudo-relation quality).
        cap_relstruct: clip denominator for d_relstruct bonus. 0.6 is a
            conservative initial value; tune via offline rescore.
        relstruct_distance_mode: "symmetric" keeps the original two-sided
            principal-angle distance. "pred_subset_gt" rewards Pred's
            relation-only null space being contained in GT's, which is useful
            when pseudo-GT is high-precision but incomplete. "mixed_subset"
            adds a weaker reverse containment term.
        relstruct_reverse_weight: reverse containment weight used only for
            "mixed_subset".

        coeff_gain: multiplier β for gain_intrinsic bonus.
        cap_gain: clip denominator for gain_intrinsic bonus.

        coeff_dim_penalty: multiplier η for the dimensional parsimony penalty.
            Applied only to Converged samples (FC / Under) to avoid penalising
            broken solver states.
        dim_penalty_mode: normalization for the dimensional parsimony term:
            "count" = #dims, "ratio" = #dims/#constraints, and
            "per_primitive" = #dims/#primitives.

        use_leak: enable leak_prime bonus (FC gate).
        use_relstruct: enable d_relstruct bonus (Converged gate).
        use_gain: enable gain_intrinsic bonus (FC gate).
        use_dim_penalty: enable dim count penalty (Converged gate).
            Off by default to avoid doubling with SolverReward's
            penalty_num_dimensional_constraints; set SolverReward's param to 0
            if you enable this.
    """

    def __init__(
        self,
        context: SketchContext,
        eps: float = 1e-9,
        # leak_prime
        coeff_leak: float = 5.0,
        cap_leak: float = 0.3,
        # d_relstruct
        coeff_relstruct: float = 3.0,
        cap_relstruct: float = 0.6,
        relstruct_distance_mode: str = "symmetric",
        relstruct_reverse_weight: float = 0.5,
        # gain_intrinsic
        coeff_gain: float = 3.0,
        cap_gain: float = 0.5,
        # dim penalty
        coeff_dim_penalty: float = 0.05,
        dim_penalty_mode: str = "count",
        # toggles
        use_leak: bool = True,
        use_relstruct: bool = True,
        use_gain: bool = False,
        use_dim_penalty: bool = False,
    ):
        super().__init__()
        self.context = context
        self.eps = eps
        self.coeff_leak = coeff_leak
        self.cap_leak = cap_leak
        self.coeff_relstruct = coeff_relstruct
        self.cap_relstruct = cap_relstruct
        if relstruct_distance_mode not in (
            "symmetric",
            "pred_subset_gt",
            "mixed_subset",
        ):
            raise ValueError(
                "relstruct_distance_mode must be one of "
                "{'symmetric', 'pred_subset_gt', 'mixed_subset'}, "
                f"got {relstruct_distance_mode!r}"
            )
        if relstruct_reverse_weight < 0.0:
            raise ValueError(
                f"relstruct_reverse_weight must be non-negative, got {relstruct_reverse_weight}"
            )
        self.relstruct_distance_mode = relstruct_distance_mode
        self.relstruct_reverse_weight = relstruct_reverse_weight
        self.coeff_gain = coeff_gain
        self.cap_gain = cap_gain
        self.coeff_dim_penalty = coeff_dim_penalty
        if dim_penalty_mode not in ("count", "ratio", "per_primitive"):
            raise ValueError(
                "dim_penalty_mode must be one of {'count', 'ratio', 'per_primitive'}, "
                f"got {dim_penalty_mode!r}"
            )
        self.dim_penalty_mode = dim_penalty_mode
        self.use_leak = use_leak
        self.use_relstruct = use_relstruct
        self.use_gain = use_gain
        self.use_dim_penalty = use_dim_penalty
        self.tokenizer = SketchTokenizer(context)

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

    def _bonus(self, value: float, cap: float, coeff: float) -> float:
        """Bonus = coeff * max(0, 1 - value/cap), capped at coeff."""
        if cap <= 0.0:
            return 0.0
        clipped = min(max(value, 0.0), cap)
        return coeff * (1.0 - clipped / cap)

    def _dim_penalty_value(
        self,
        n_dims: int,
        n_total_constraints: int,
        n_primitives: int,
    ) -> float:
        """Return the normalized dimensional parsimony quantity to penalize."""
        if self.dim_penalty_mode == "count":
            return float(n_dims)
        if self.dim_penalty_mode == "ratio":
            return float(n_dims) / float(max(1, n_total_constraints))
        if self.dim_penalty_mode == "per_primitive":
            return float(n_dims) / float(max(1, n_primitives))
        raise RuntimeError(f"unreachable dim_penalty_mode: {self.dim_penalty_mode!r}")

    @torch.no_grad()
    def forward(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch],
        per_sample_status: list[str] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Compute A″ reward scores.

        Args:
            prim_tokens: (B, L_prim)
            cons_tokens: (B, L_cons)
            sketch_original_list: GT Sketch objects, length B.
            per_sample_status: per-sample solver status strings (from
                SolverReward.collect_results).  Elements are one of
                {"FC","Under","Distorted","Conflicting","Redundant",
                 "Malformed","Timeout"}.  When None, all samples are
                treated as Converged (useful for debugging).

        Returns:
            scores: (B, L_cons) float32 — reward placed at STOP token.
            metrics: dict of scalar diagnostics for W&B logging.
        """
        B, L = cons_tokens.shape
        assert len(sketch_original_list) == B
        scores = torch.zeros((B, L), dtype=torch.float32)

        prim_np = prim_tokens.detach().cpu().numpy()
        cons_np = cons_tokens.detach().cpu().numpy()

        if per_sample_status is None:
            per_sample_status = ["FC"] * B

        # Per-component accumulators
        leak_vals: list[float] = []
        rs_vals: list[float] = []
        rs_pred_subset_gt_vals: list[float] = []
        rs_gt_subset_pred_vals: list[float] = []
        rs_dim_pred_vals: list[int] = []
        rs_dim_gt_vals: list[int] = []
        gain_vals: list[float] = []
        dim_counts: list[int] = []
        total_counts: list[int] = []
        dim_ratio_vals: list[float] = []
        dim_per_primitive_vals: list[float] = []
        dim_penalty_values: list[float] = []
        reward_vals: list[float] = []

        skipped_build = 0
        skipped_not_converged = 0

        for i in range(B):
            status = per_sample_status[i]
            is_fc = (status == "FC")
            is_converged = status in ("FC", "Under")

            # Any component needs at least Converged
            if not is_converged:
                skipped_not_converged += 1
                continue

            cons_i = cons_np[i].copy()
            stop_pos = np.where(cons_i == int(ConstraintToken.Stop))[0]
            first_stop_idx = int(stop_pos[0]) if len(stop_pos) > 0 else (L - 1)
            if 0 <= first_stop_idx < L - 1:
                cons_i[first_stop_idx + 1:] = int(ConstraintToken.Pad)

            gt_sketch = sketch_original_list[i]
            try:
                pred_sketch = self._build_pred_sketch(cons_i, prim_np[i], gt_sketch)
            except Exception:
                skipped_build += 1
                continue

            try:
                pred_native = compute_native_response(pred_sketch, self.context, eps=self.eps)
            except Exception:
                skipped_build += 1
                continue

            gt_rels = [c for c in gt_sketch.constraints.values() if _is_relational(c)]

            sample_reward = 0.0

            # ── leak_prime bonus (FC gate) ────────────────────────────────
            if self.use_leak and is_fc:
                lp = _compute_leak_prime(pred_native, gt_rels, self.context, self.eps)
                leak_vals.append(lp)
                sample_reward += self._bonus(lp, self.cap_leak, self.coeff_leak)

            # ── d_relstruct bonus (Converged gate) ────────────────────────
            if self.use_relstruct:
                try:
                    (
                        drs,
                        drs_pred_subset_gt,
                        drs_gt_subset_pred,
                        drs_dim_pred,
                        drs_dim_gt,
                    ) = _compute_d_relstruct(
                        pred_sketch,
                        gt_sketch,
                        pred_native,
                        self.context,
                        self.eps,
                        distance_mode=self.relstruct_distance_mode,
                        reverse_weight=self.relstruct_reverse_weight,
                    )
                except Exception:
                    drs = self.cap_relstruct  # fallback to worst case
                    drs_pred_subset_gt = self.cap_relstruct
                    drs_gt_subset_pred = self.cap_relstruct
                    drs_dim_pred = 0
                    drs_dim_gt = 0
                rs_vals.append(drs)
                rs_pred_subset_gt_vals.append(drs_pred_subset_gt)
                rs_gt_subset_pred_vals.append(drs_gt_subset_pred)
                rs_dim_pred_vals.append(drs_dim_pred)
                rs_dim_gt_vals.append(drs_dim_gt)
                sample_reward += self._bonus(drs, self.cap_relstruct, self.coeff_relstruct)

            # ── gain_intrinsic bonus (FC gate) ────────────────────────────
            if self.use_gain and is_fc:
                gi, _ = _compute_gain_intrinsic(
                    pred_native, gt_sketch, self.context, self.eps
                )
                gain_vals.append(gi)
                sample_reward += self._bonus(gi, self.cap_gain, self.coeff_gain)

            # ── dim count penalty (Converged gate) ────────────────────────
            n_dims = sum(
                1 for c in pred_sketch.constraints.values() if _is_dimensional(c)
            )
            n_total_constraints = len(pred_sketch.constraints)
            n_primitives = len(pred_sketch.entities)
            dim_penalty_value = self._dim_penalty_value(
                n_dims,
                n_total_constraints,
                n_primitives,
            )
            dim_counts.append(n_dims)
            total_counts.append(n_total_constraints)
            dim_ratio_vals.append(float(n_dims) / float(max(1, n_total_constraints)))
            dim_per_primitive_vals.append(float(n_dims) / float(max(1, n_primitives)))
            dim_penalty_values.append(dim_penalty_value)
            if self.use_dim_penalty:
                sample_reward -= self.coeff_dim_penalty * dim_penalty_value

            scores[i, first_stop_idx] += sample_reward
            reward_vals.append(sample_reward)

        def _mean(xs: list) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _median(xs: list) -> float:
            return statistics.median(xs) if xs else 0.0

        valid_n = len(reward_vals)
        metrics: dict[str, float | int] = {
            "relstruct_reward_mean":          _mean(reward_vals),
            "relstruct_reward_median":        _median(reward_vals),
            "relstruct_leak_mean":            _mean(leak_vals),
            "relstruct_leak_median":          _median(leak_vals),
            "relstruct_d_relstruct_mean":     _mean(rs_vals),
            "relstruct_d_relstruct_median":   _median(rs_vals),
            "relstruct_d_pred_subset_gt_mean": _mean(rs_pred_subset_gt_vals),
            "relstruct_d_gt_subset_pred_mean": _mean(rs_gt_subset_pred_vals),
            "relstruct_obs_dim_pred_mean":     _mean(rs_dim_pred_vals),
            "relstruct_obs_dim_gt_mean":       _mean(rs_dim_gt_vals),
            "relstruct_gain_intrinsic_mean":  _mean(gain_vals),
            "relstruct_gain_intrinsic_median":_median(gain_vals),
            "relstruct_dim_count_mean":       _mean(dim_counts),
            "relstruct_total_count_mean":     _mean(total_counts),
            "relstruct_dim_ratio_mean":       _mean(dim_ratio_vals),
            "relstruct_dim_per_primitive_mean":_mean(dim_per_primitive_vals),
            "relstruct_dim_penalty_value_mean":_mean(dim_penalty_values),
            "relstruct_valid_n":              valid_n,
            "relstruct_skipped_build":        skipped_build,
            "relstruct_skipped_not_converged":skipped_not_converged,
        }
        return scores, metrics
