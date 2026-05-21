"""
edit_response_distance.py — Common Edit Response Distance (Proposed Metric 1).

設計方針
--------
GT と Pred の constraint set を直接比較する代わりに、
**同じ編集要求 q を与えたときの形状応答 S_C^(q) = ∂x/∂q を比較**する。

各 constraint set C は native dimensional parameters p_C を持つ。
線形化すると J_x δx + J_p δp_C = 0 → δx = R_C δp_C, R_C = -pinv(J_x) J_p.
共通編集 q (= h(x)) を実現するための native dim 変化は
δp_C* = pinv(J_q R_C) δq, よって
        S_C^(q) = R_C pinv(J_q R_C).
観測量 φ(x) を介したい場合は M_C^(q) = J_φ S_C^(q) を用いる。

距離:
    d_resp(P, G) = Σ_q w_q || M_P^(q) - M_G^(q) ||_F

実装上の前提
------------
- edit probe q は `datalib.Constraint` (典型的には Distance/Length/...) として渡す。
  そのハンドラの residual = h(x) - target なので ∂residual/∂x = J_h = J_q.
- relational vs dimensional の判別は constraint_type 名で行う
  (notebooks/canonical_sketch.py と同じセット)。
- base_eids を渡すと、応答比較を当該 entity 群の primitive 座標に絞る。
  (gauge / construction geometry を排除したいときに使う。)

参考: src/metrics/null_space_distance.py — Jacobian 計算と SVD ユーティリティを共有。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import sketchgraphs.data as datalib
from sketchgraphs.data import LocalReferenceParameter

from src.sketch_handlers.sketch_solver import (
    SketchState,
    _StateWithQ,
    get_entity_indices,
)


DIMENSIONAL_CONSTRAINTS = {
    "Distance",
    "DistanceX",
    "DistanceY",
    "Length",
    "Diameter",
    "Radius",
    "Angle",
    "Offset",
    "Centerline_Dimension",
    "Minor_Diameter",
    "Major_Diameter",
    "Rho",
    "Block",
    "InternalAlignment",
}


# ---------------------------------------------------------------------------
# Probe signature & collection
# ---------------------------------------------------------------------------

def _ref_strings(constraint: datalib.Constraint) -> tuple[str, ...]:
    return tuple(
        str(p.value)
        for p in constraint.parameters
        if isinstance(p, LocalReferenceParameter)
    )


def _direction(constraint: datalib.Constraint) -> str:
    for p in constraint.parameters:
        if getattr(p, "parameterId", None) == "direction":
            return str(getattr(p, "value", ""))
    return ""


def constraint_signature(constraint: datalib.Constraint) -> tuple:
    """
    Identity for deduplicating probes across sketches.

    Two constraints are the same probe iff (type, sorted refs, direction) match.
    Target value is ignored — for edit probes, only the geometric observable
    h(x) matters; for relation probes, only the equation r(x)=0.
    """
    return (
        constraint.constraint_type.name,
        tuple(sorted(_ref_strings(constraint))),
        _direction(constraint),
    )


def _is_dimensional(constraint: datalib.Constraint) -> bool:
    return constraint.constraint_type.name in DIMENSIONAL_CONSTRAINTS


def _is_relational(constraint: datalib.Constraint) -> bool:
    return constraint.constraint_type.name not in DIMENSIONAL_CONSTRAINTS


def collect_probes(
    sketches: Sequence[datalib.Sketch],
    kind: str,
) -> tuple[list[datalib.Constraint], list[float]]:
    """
    Gather union of probes across sketches with empirical weights.

    Parameters
    ----------
    sketches : annotator sketches sharing a common entity ID layout (e.g.
               post `remap_sketch_to_canonical`).
    kind     : "dim" or "rel".

    Returns
    -------
    probes  : list of representative `datalib.Constraint` (one per signature)
    weights : list of float in [0, 1]; weight = (#sketches containing probe) / N
    """
    if kind not in ("dim", "rel"):
        raise ValueError(f"kind must be 'dim' or 'rel', got {kind!r}")

    selector = _is_dimensional if kind == "dim" else _is_relational
    counts: dict[tuple, int] = {}
    representative: dict[tuple, datalib.Constraint] = {}

    for sketch in sketches:
        seen_in_sketch: set[tuple] = set()
        for c in sketch.constraints.values():
            if not selector(c):
                continue
            sig = constraint_signature(c)
            if sig in seen_in_sketch:
                continue
            seen_in_sketch.add(sig)
            counts[sig] = counts.get(sig, 0) + 1
            representative.setdefault(sig, c)

    n = max(len(sketches), 1)
    sigs = sorted(counts.keys())
    probes = [representative[s] for s in sigs]
    weights = [counts[s] / n for s in sigs]
    return probes, weights


# ---------------------------------------------------------------------------
# Per-constraint Jacobian
# ---------------------------------------------------------------------------

def _residual_fn(constraint: datalib.Constraint, handler, state: SketchState):
    def fn(q: torch.Tensor) -> torch.Tensor:
        s = _StateWithQ(state, q)
        try:
            r = handler.residual(constraint, s)
            return r.reshape(-1)
        except Exception:
            return torch.zeros(1, dtype=q.dtype, device=q.device)
    return fn


def constraint_jacobian(
    constraint: datalib.Constraint,
    state: SketchState,
    context,
) -> torch.Tensor | None:
    """
    Compute (k, n_x) Jacobian of `constraint`'s residual at state.q (q0).

    Returns None when the handler is missing or the constraint references entities
    that are not present in `state` (e.g. probe entity not in this sketch).
    """
    handler = context.get_constraint_handler(constraint.constraint_type.name)
    if handler is None:
        return None

    # Bail out early if any non-external ref is missing from state.
    for ref in _ref_strings(constraint):
        eid = ref.split(".", 1)[0]
        if eid.startswith("-"):
            continue  # external axis is fine
        if eid not in state.slices:
            return None

    fn = _residual_fn(constraint, handler, state)
    try:
        jac = torch.func.jacfwd(fn)(state.q.detach())
    except Exception:
        return None
    if jac.ndim == 1:
        jac = jac.unsqueeze(0)
    return jac


# ---------------------------------------------------------------------------
# Native dimension response  R_C = -pinv(J_x) J_p
# ---------------------------------------------------------------------------

@dataclass
class NativeResponse:
    state: SketchState
    J_x: torch.Tensor          # (m, n_x) full constraint jacobian
    J_p: torch.Tensor          # (m, n_p) -1 on rows of dim constraints
    R: torch.Tensor            # (n_x, n_p) primitive response to native dims
    dim_constraint_ids: list[str]


def compute_native_response(
    sketch: datalib.Sketch,
    context,
    constraint_ids: list[str] | None = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-9,
    rcond: float = 1e-6,
) -> NativeResponse:
    """
    Build the response map of primitive parameters to a constraint set's own
    dimensional values. R = -pinv(J_x) J_p.

    Multi-row dimensional residuals would need per-row p-columns; we currently
    assume each dimensional constraint contributes a single residual row that
    couples to a single dimensional parameter (which holds for the standard
    set: Distance, Length, Diameter, Radius, Angle, ...). Multi-row residuals
    from such constraints are accepted but their extra rows are treated as
    relational (no p-column), which is an approximation.

    Singular value threshold for pinv(J_x):
        eps_eff = max(eps, rcond * sigma_max(J_x))

    The combined absolute+relative threshold prevents two failure modes:
    - eps alone (absolute): misses near-zero-but-nonzero singular values that
      arise when constraint equations are nearly linearly dependent (e.g. a
      sketch with nearly-redundant distance constraints).  Those tiny singular
      values invert to huge factors and make R blow up.
    - rcond alone (relative): a purely relative cutoff would drop weak but real
      constraint directions in very small or poorly-scaled sketches.

    The default rcond=1e-6 (condition number cap of 1e6) cleanly handles rigid
    body modes and near-degenerate constraint Jacobians seen in practice without
    affecting well-conditioned sketches (verified: main-experiment sketches have
    min real SV ≈ 1.7e-2, well above rcond * sigma_max ≈ 2e-5).
    """
    state = SketchState(sketch, dtype=dtype)
    n_x = state.num_dof()

    if constraint_ids is None:
        constraint_ids = list(sketch.constraints.keys())

    jac_rows: list[torch.Tensor] = []
    Jp_col_for_row: list[int] = []
    dim_ids: list[str] = []

    for cid in constraint_ids:
        constraint = sketch.constraints.get(cid)
        if constraint is None:
            continue
        jac = constraint_jacobian(constraint, state, context)
        if jac is None or jac.shape[0] == 0:
            continue

        if _is_dimensional(constraint):
            col_idx = len(dim_ids)
            dim_ids.append(cid)
            jac_rows.append(jac[0:1])
            Jp_col_for_row.append(col_idx)
            for i in range(1, jac.shape[0]):
                jac_rows.append(jac[i:i + 1])
                Jp_col_for_row.append(-1)
        else:
            for i in range(jac.shape[0]):
                jac_rows.append(jac[i:i + 1])
                Jp_col_for_row.append(-1)

    if not jac_rows:
        J_x = torch.zeros(0, n_x, dtype=dtype)
        J_p = torch.zeros(0, 0, dtype=dtype)
        R = torch.zeros(n_x, 0, dtype=dtype)
        return NativeResponse(state, J_x, J_p, R, [])

    J_x = torch.cat(jac_rows, dim=0)
    n_p = len(dim_ids)
    J_p = torch.zeros(J_x.shape[0], n_p, dtype=dtype)
    for row_idx, col in enumerate(Jp_col_for_row):
        if col >= 0:
            J_p[row_idx, col] = -1.0

    if n_p == 0:
        R = torch.zeros(n_x, 0, dtype=dtype)
    else:
        U, S, Vh = torch.linalg.svd(J_x, full_matrices=False)
        eps_eff = max(eps, rcond * float(S[0])) if S.numel() > 0 else eps
        keep = S >= eps_eff
        if not bool(keep.any()):
            R = torch.zeros(n_x, n_p, dtype=dtype)
        else:
            S_inv = torch.diag(1.0 / S[keep])
            J_x_pinv = Vh.T[:, keep] @ S_inv @ U[:, keep].T   # (n_x, m)
            R = -J_x_pinv @ J_p

    return NativeResponse(state, J_x, J_p, R, dim_ids)


# ---------------------------------------------------------------------------
# Edit response  S_C^(q) = R_C pinv(J_q R_C)
# ---------------------------------------------------------------------------

def _abs_pinv(A: torch.Tensor, eps: float, rcond: float = 0.0) -> torch.Tensor:
    """Pseudo-inverse with combined absolute+relative singular value cutoff.

    eps_eff = max(eps, rcond * sigma_max).  Set rcond=0 to use absolute eps only.
    """
    if A.numel() == 0:
        return torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype, device=A.device)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    eps_eff = max(eps, rcond * float(S[0])) if S.numel() > 0 and rcond > 0.0 else eps
    keep = S >= eps_eff
    if not bool(keep.any()):
        return torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype, device=A.device)
    S_inv = torch.diag(1.0 / S[keep])
    return Vh.T[:, keep] @ S_inv @ U[:, keep].T


def edit_response_for_probe(
    native: NativeResponse,
    probe: datalib.Constraint,
    context,
    eps: float = 1e-9,
    rcond: float = 1e-6,
) -> torch.Tensor | None:
    """
    Compute S_C^(q) = R_C pinv(J_q R_C) for a single edit probe q.

    Shape: (n_x, k_q) where k_q is the residual size of the probe (typically 1).
    Returns None if the probe Jacobian could not be computed for this sketch.

    The combined absolute+relative cutoff (eps, rcond) is applied to
    pinv(J_q R_C), matching the convention in compute_native_response.
    """
    Jq = constraint_jacobian(probe, native.state, context)
    if Jq is None:
        return None

    n_x = native.R.shape[0]
    k_q = Jq.shape[0]
    if native.R.shape[1] == 0:
        # No native dimensions → cannot realize any edit; return zero response.
        return torch.zeros(n_x, k_q, dtype=Jq.dtype, device=Jq.device)

    A = Jq @ native.R                                    # (k_q, n_p)
    # eps_A is normalized by ||J_q|| * ||R||_op so that near-zero SVs of A
    # are thresholded relative to the physical scale of the probe, not relative
    # to sigma_max(A) itself (which would be circular when A is already tiny).
    if rcond > 0.0:
        Jq_norm = float(torch.linalg.norm(Jq))
        R_op = float(torch.linalg.norm(native.R, ord=2))
        eps_A = max(eps, rcond * Jq_norm * R_op)
    else:
        eps_A = eps
    A_pinv = _abs_pinv(A, eps=eps_A, rcond=0.0)         # (n_p, k_q)
    return native.R @ A_pinv                             # (n_x, k_q)


# ---------------------------------------------------------------------------
# Common Edit Response Distance
# ---------------------------------------------------------------------------

@dataclass
class EditResponseResult:
    distance: float
    per_probe_distances: list[float | None]
    per_probe_weights: list[float]
    n_dim_gt: int
    n_dim_pred: int
    skipped_probes: int


def compute_edit_response_distance(
    gt_sketch: datalib.Sketch,
    pred_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    context,
    probe_weights: Sequence[float] | None = None,
    base_eids: Iterable[str] | None = None,
    observable_jacobian: torch.Tensor | None = None,
    observable_jacobian_gt: torch.Tensor | None = None,
    observable_jacobian_pred: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-8,
) -> EditResponseResult:
    """
    Compute Common Edit Response Distance (Proposed Metric 1).

    d_resp(P, G) = Σ_q w_q || M_P^(q) - M_G^(q) ||_F

    Default observable: identity on `base_eids` (or the full primitive vector
    if `base_eids` is None). To compare rigid-motion-invariant quantities,
    pass `observable_jacobian = J_φ` of shape (m_φ, n_x).

    When GT and Pred have extra construction entities or otherwise differ in
    full state layout, pass per-sketch projections via
    `observable_jacobian_gt` and `observable_jacobian_pred`. The projected
    observables must have the same row semantics/shape.

    If no projection is provided, the two sketches must share the same full
    primitive layout. If `base_eids` is provided, coordinate projections are
    built separately for each sketch and only those entities are compared.
    """
    gt_native = compute_native_response(gt_sketch, context, dtype=dtype, eps=eps)
    pred_native = compute_native_response(pred_sketch, context, dtype=dtype, eps=eps)

    same_layout = (
        gt_native.state.num_dof() == pred_native.state.num_dof()
        and gt_native.state.entity_ids == pred_native.state.entity_ids
    )

    J_gt = observable_jacobian_gt
    J_pred = observable_jacobian_pred
    if observable_jacobian is not None:
        if J_gt is not None or J_pred is not None:
            raise ValueError(
                "Pass either observable_jacobian or per-sketch "
                "observable_jacobian_gt/pred, not both."
            )
        J_gt = observable_jacobian
        J_pred = observable_jacobian

    if J_gt is None and J_pred is None and base_eids is not None:
        base_eids_list = list(base_eids)
        idx_gt = get_entity_indices(gt_native.state, base_eids_list)
        idx_pred = get_entity_indices(pred_native.state, base_eids_list)
        if len(idx_gt) != len(idx_pred):
            raise ValueError(
                "base_eids projection has different sizes on GT and Pred "
                f"({len(idx_gt)} vs {len(idx_pred)})."
            )
        J_gt = torch.zeros(len(idx_gt), gt_native.state.num_dof(), dtype=dtype)
        for row, j in enumerate(idx_gt):
            J_gt[row, j] = 1.0
        J_pred = torch.zeros(len(idx_pred), pred_native.state.num_dof(), dtype=dtype)
        for row, j in enumerate(idx_pred):
            J_pred[row, j] = 1.0

    if (J_gt is None) != (J_pred is None):
        raise ValueError(
            "observable_jacobian_gt and observable_jacobian_pred must be "
            "provided together."
        )
    if J_gt is not None and J_pred is not None:
        if J_gt.shape[0] != J_pred.shape[0]:
            raise ValueError(
                "GT and Pred observable projections have different output "
                f"sizes ({J_gt.shape[0]} vs {J_pred.shape[0]})."
            )
        if J_gt.shape[1] != gt_native.state.num_dof():
            raise ValueError(
                "observable_jacobian_gt has incompatible input size "
                f"({J_gt.shape[1]} vs {gt_native.state.num_dof()})."
            )
        if J_pred.shape[1] != pred_native.state.num_dof():
            raise ValueError(
                "observable_jacobian_pred has incompatible input size "
                f"({J_pred.shape[1]} vs {pred_native.state.num_dof()})."
            )
    elif not same_layout:
        raise ValueError(
            f"GT and Pred have incompatible primitive layouts "
            f"({gt_native.state.num_dof()} vs {pred_native.state.num_dof()} DoF). "
            "Pass base_eids or per-sketch observable_jacobian_gt/pred to "
            "compare a common observable subset."
        )

    if probe_weights is None:
        probe_weights = [1.0] * len(edit_probes)
    if len(probe_weights) != len(edit_probes):
        raise ValueError("len(probe_weights) must match len(edit_probes)")

    per_probe_distances: list[float | None] = []
    skipped = 0
    total = 0.0

    for q, w in zip(edit_probes, probe_weights):
        S_gt = edit_response_for_probe(gt_native, q, context, eps=eps)
        S_pred = edit_response_for_probe(pred_native, q, context, eps=eps)
        if S_gt is None or S_pred is None:
            per_probe_distances.append(None)
            skipped += 1
            continue

        if J_gt is not None and J_pred is not None:
            M_gt = J_gt @ S_gt
            M_pred = J_pred @ S_pred
        else:
            M_gt = S_gt
            M_pred = S_pred

        d = float(torch.linalg.norm(M_pred - M_gt))
        per_probe_distances.append(d)
        total += w * d

    return EditResponseResult(
        distance=total,
        per_probe_distances=per_probe_distances,
        per_probe_weights=list(probe_weights),
        n_dim_gt=len(gt_native.dim_constraint_ids),
        n_dim_pred=len(pred_native.dim_constraint_ids),
        skipped_probes=skipped,
    )


# ---------------------------------------------------------------------------
# Convenience: build an observable Jacobian from pairwise distances of subnodes
# ---------------------------------------------------------------------------

def pairwise_distance_observable(
    sketch: datalib.Sketch,
    base_eids: Iterable[str] | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build J_φ for φ(x) = pairwise distances between all primitive subnodes
    (Line.start/end, Circle.center, Arc.start/end/center, Point).

    Pairwise distances are rigid-motion-invariant, so this projects out the
    gauge ambiguity that plain primitive-coordinate comparison would mix in.

    Returns
    -------
    J_phi : (m_phi, n_x)
    """
    state = SketchState(sketch, dtype=dtype)
    base_set = set(base_eids) if base_eids is not None else None

    refs: list[str] = []
    for eid, ent in sketch.entities.items():
        if base_set is not None and eid not in base_set:
            continue
        if isinstance(ent, datalib.Line):
            refs.append(f"{eid}.start")
            refs.append(f"{eid}.end")
        elif isinstance(ent, datalib.Circle):
            refs.append(f"{eid}.center")
        elif isinstance(ent, datalib.Arc):
            refs.append(f"{eid}.start")
            refs.append(f"{eid}.end")
            refs.append(f"{eid}.center")
        elif isinstance(ent, datalib.Point):
            refs.append(eid)

    pairs = [(refs[i], refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))]
    if not pairs:
        return torch.zeros(0, state.num_dof(), dtype=dtype)

    def fn(q: torch.Tensor) -> torch.Tensor:
        s = _StateWithQ(state, q)
        ds: list[torch.Tensor] = []
        for a, b in pairs:
            pa = s.get_point(a)
            pb = s.get_point(b)
            ds.append(torch.linalg.norm(pa - pb))
        return torch.stack(ds)

    J = torch.func.jacfwd(fn)(state.q.detach())
    if J.ndim == 1:
        J = J.unsqueeze(0)
    return J
