"""
null_space_distance.py — Design intent evaluation metric via row-space alignment.

Design rationale (Intent Recall):
  Measure the "agreement" between the row space R_GT of the Jacobian of the GT
  primary constraints and the row space R_Pred of the Jacobian of the Pred
  constraints.

  Intent Recall via principal angles θ_i:
    1. Q_GT  : orthonormal basis of R_GT   (n × d_GT)
    2. Q_Pred: orthonormal basis of R_Pred (n × d_Pred)
    3. M = Q_GT^T @ Q_Pred  →  SVD  →  σ_i = cos(θ_i)
    4. Score = Σ σ_i / dim(R_GT)  ∈ [0, 1]

  Why not a Grassmann distance between the null spaces:
    When Pred has no constraints, its null space = the whole space, so the GT
    null space is trivially contained and the distance collapses to 0.

  Difference from the Grassmann distance (sqrt(Σ arccos(σ_i)²)):
    - Intent Recall needs no arccos and is always normalized to [0, 1]
    - a dimension gap (d_Pred < d_GT) is naturally reflected as σ=0 for the deficit
    - samples with different dim remain directly comparable side by side

Pipeline:
  1. Compute the row space R_gt from the Jacobian of GT_primary
  2. For each Pred constraint, test whether its Jacobian row lies in R_gt (primary classification)
  3. Compute the row space R_pred from the Jacobian of the Pred primary constraints
  4. Return Grassmann distance(R_gt, R_pred)

Usage::

    from src.sketch_handlers.context import SketchContext
    from src.metrics.null_space_distance import compute_null_space_distance

    context = SketchContext()
    result = compute_null_space_distance(gt_primary_sketch, pred_sketch, context)
    print(result["grassmann_dist"])
"""

from __future__ import annotations

import torch
import sketchgraphs.data as datalib

from src.sketch_handlers.sketch_solver import (
    SketchState,
    compute_jacobian,
    compute_jacobian_hessians,
    compute_null_space,
    compute_row_space,
    get_entity_indices,
)


# ---------------------------------------------------------------------------
# Intent Recall (between row spaces)
# ---------------------------------------------------------------------------

def intent_recall(R_gt: torch.Tensor, R_pred: torch.Tensor) -> float:
    """
    Intent Recall: measure how much of the GT row-space directions the Pred row
    space covers.

    Computation:
      M = Q_GT^T @ Q_Pred  →  SVD  →  σ_i = cos(θ_i)  (cosines of principal angles)
      Score = Σ σ_i / dim(R_GT)  ∈ [0, 1]

    Interpretation:
      - Score = 1: Pred fully covers all GT design-intent directions
      - Score = 0: Pred does not overlap the GT design-intent directions at all
      - d_pred < d_gt: missing dimensions are naturally reflected as σ=0 (no arbitrary constant)
      - d_pred = 0 (no constraints): Score = 0

    Args:
      R_gt  : (n, d_gt)   orthonormal basis of the GT primary row space
      R_pred: (n, d_pred) orthonormal basis of the Pred row space

    Returns:
      float in [0, 1]
    """
    d_gt = R_gt.shape[1]

    if d_gt == 0:
        # No GT constraints → recall = 1 by definition
        return 1.0

    if R_pred.shape[1] == 0:
        # No Pred constraints → recall = 0
        return 0.0

    M = R_gt.T @ R_pred  # (d_gt, d_pred)
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    S = S.clamp(0.0, 1.0)  # guard against slight out-of-range values from numerical error
    # Normalize Σ σ_i (= Σ cos(θ_i)) by the GT dimension count.
    # When d_pred < d_gt, S has size d_pred < d_gt, so the deficit does not
    # contribute to the sum (equivalent to σ=0).
    return float(S.sum()) / d_gt


def combine_two_jet_recall(
    linear_recall: float,
    quadratic_recall: float | None,
    gt_row_dim: int,
    gt_quadratic_dim: int,
) -> float:
    """
    Combine first- and second-order recall by counting independent GT intent dims.

    The quadratic signature is constructed only from constraints whose first-order
    component vanishes, so gt_quadratic_dim is treated as additional 2-jet intent
    dimension rather than a reweighting knob. If no quadratic intent is present,
    this reduces exactly to the linear recall.
    """
    if gt_row_dim < 0 or gt_quadratic_dim < 0:
        raise ValueError("GT intent dimensions must be non-negative")

    total_dim = gt_row_dim + gt_quadratic_dim
    if total_dim == 0:
        return 1.0

    q_recall = 1.0 if quadratic_recall is None else quadratic_recall
    score = (gt_row_dim * linear_recall + gt_quadratic_dim * q_recall) / total_dim
    return float(score)


# ---------------------------------------------------------------------------
# Quadratic intent signatures
# ---------------------------------------------------------------------------

def symmetric_vectorize(Q: torch.Tensor) -> torch.Tensor:
    """
    Vectorize a symmetric matrix while preserving the Frobenius inner product.

    The upper triangle is returned. Off-diagonal entries are multiplied by
    sqrt(2), so dot(symmetric_vectorize(A), symmetric_vectorize(B)) equals
    trace(A @ B) for symmetric A and B.
    """
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"symmetric_vectorize expects a square matrix, got {tuple(Q.shape)}")

    n = Q.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=Q.dtype, device=Q.device)

    rows, cols = torch.triu_indices(n, n, device=Q.device)
    vec = Q[rows, cols].clone()
    off_diag = rows != cols
    vec[off_diag] *= torch.sqrt(torch.tensor(2.0, dtype=Q.dtype, device=Q.device))
    return vec


def _left_null_space(J: torch.Tensor, eps: float) -> torch.Tensor:
    """Return an orthonormal basis for {lambda | lambda^T J = 0} as columns."""
    m = J.shape[0]
    if m == 0:
        return torch.zeros(0, 0, dtype=J.dtype, device=J.device)

    U, S, _ = torch.linalg.svd(J, full_matrices=True)
    rank = int((S >= eps).sum().item())
    if rank >= m:
        return torch.zeros(m, 0, dtype=J.dtype, device=J.device)
    return U[:, rank:]


def _restrict_derivatives(
    J: torch.Tensor,
    H: torch.Tensor,
    variable_indices: list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variable_indices is None:
        return J, H

    idx = torch.tensor(variable_indices, dtype=torch.long, device=J.device)
    return J[:, idx], H[:, idx, :][:, :, idx]


def _matrix_pinv(A: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute a numerical pseudo-inverse using the same absolute eps convention."""
    if A.numel() == 0:
        return torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype, device=A.device)

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = S >= eps
    if not bool(keep.any()):
        return torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype, device=A.device)

    V = Vh.T[:, keep]
    U_keep = U[:, keep]
    S_inv = torch.diag(1.0 / S[keep])
    return V @ S_inv @ U_keep.T


def _effective_base_derivatives(
    J: torch.Tensor,
    H: torch.Tensor,
    base_indices: list[int],
    eps_linear: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eliminate non-base variables to obtain derivatives on base coordinates.

    For variables split as q = (b, h), first-order feasibility of a base velocity
    u means there exists hidden velocity w with J_b u + J_h w = 0. Equivalently,
    L_h^T J_b u = 0, where columns of L_h span the left null space of J_h.

    The quadratic term is evaluated along the minimum-norm hidden lift
    w = -J_h^+ J_b u, giving an effective Hessian
    L_h^T H[(u, w), (u, w)]. This is the second-order analogue of the projected
    full-null-space row-space computation used for the linear metric.
    """
    n_total = J.shape[1]
    n_base = len(base_indices)
    if n_base == 0:
        return (
            torch.zeros(0, 0, dtype=J.dtype, device=J.device),
            torch.zeros(0, 0, 0, dtype=J.dtype, device=J.device),
        )

    base_set = set(base_indices)
    hidden_indices = [i for i in range(n_total) if i not in base_set]
    idx_base = torch.tensor(base_indices, dtype=torch.long, device=J.device)
    J_base = J[:, idx_base]

    if not hidden_indices:
        H_base = H[:, idx_base, :][:, :, idx_base]
        return J_base, H_base

    idx_hidden = torch.tensor(hidden_indices, dtype=torch.long, device=J.device)
    J_hidden = J[:, idx_hidden]
    left_hidden = _left_null_space(J_hidden, eps=eps_linear)
    n_effective = left_hidden.shape[1]

    if n_effective == 0:
        return (
            torch.zeros(0, n_base, dtype=J.dtype, device=J.device),
            torch.zeros(0, n_base, n_base, dtype=J.dtype, device=J.device),
        )

    J_effective = left_hidden.T @ J_base
    hidden_lift = -_matrix_pinv(J_hidden, eps_linear) @ J_base

    lift = torch.zeros(n_total, n_base, dtype=J.dtype, device=J.device)
    lift[idx_base, :] = torch.eye(n_base, dtype=J.dtype, device=J.device)
    lift[idx_hidden, :] = hidden_lift

    H_left = torch.einsum("mc,mij->cij", left_hidden, H)
    H_effective = torch.einsum("ia,cij,jb->cab", lift, H_left, lift)
    H_effective = 0.5 * (H_effective + H_effective.transpose(1, 2))
    return J_effective, H_effective


def compute_quadratic_signature(
    J: torch.Tensor,
    H: torch.Tensor,
    variable_indices: list[int] | None = None,
    domain_basis: torch.Tensor | None = None,
    eps_linear: float = 1e-8,
    eps_quad: float = 1e-10,
) -> torch.Tensor:
    """
    Compute the second-order intent signature as a row-space basis in Sym^2(T*).

    Only second-order terms whose first-order component vanishes are included.
    Concretely, for each left-null vector lambda with lambda^T J = 0, we form
    Q_lambda = sum_i lambda_i H_i, restrict it to the first-order tangent domain
    T, vectorize the resulting symmetric form, and take the row space spanned by
    those vectors.

    Recall semantics
    ----------------
    When this function is called from compute_null_space_distance with
    include_quadratic=True, the caller passes domain_basis = ker(J_GT) to BOTH
    the GT and Pred invocations. This makes the comparison "Recall in
    Sym^2(T_GT*)": how much of GT's quadratic intent (measured on GT's tangent
    domain) does Pred reproduce. Pred-only quadratic structure that lives on
    directions GT already pins linearly is invisible in this metric — that is
    intentional, mirroring the linear Intent Recall convention.

    Numerical note
    --------------
    eps_quad is an absolute threshold on |vec(Q_λ|_T)|. For sketches with large
    coordinate magnitudes, the second-order entries scale as O(coord^2), so
    floating-point noise can cross 1e-10 and produce spurious quadratic rank.
    Either pre-normalize coordinates to O(1) or pass a larger eps_quad.

    Parameters
    ----------
    J : (m, n)
        Constraint Jacobian.
    H : (m, n, n)
        Hessian for each residual row.
    variable_indices : optional list[int]
        Restrict derivatives to these coordinates before computing signatures.
    domain_basis : optional (n_restricted, k)
        Basis for the domain T on which quadratic forms are compared. If omitted,
        T = ker(J_restricted).
    eps_linear : float
        SVD threshold for first-order rank / null-space decisions.
    eps_quad : float
        SVD threshold for the quadratic signature row space (absolute).

    Returns
    -------
    Q : (k * (k + 1) / 2, r)
        Orthonormal basis of the quadratic signature row space.
    """
    J_restricted, H_restricted = _restrict_derivatives(J, H, variable_indices)
    n = J_restricted.shape[1]

    if domain_basis is None:
        T = compute_null_space(J_restricted, eps=eps_linear)
    else:
        T = domain_basis.to(dtype=J.dtype, device=J.device)

    if T.shape[0] != n:
        raise ValueError(
            f"domain_basis has incompatible shape {tuple(T.shape)} for {n} variables"
        )

    k = T.shape[1]
    feature_dim = k * (k + 1) // 2
    if k == 0 or J_restricted.shape[0] == 0:
        return torch.zeros(feature_dim, 0, dtype=J.dtype, device=J.device)

    left_null = _left_null_space(J_restricted, eps=eps_linear)
    if left_null.shape[1] == 0:
        return torch.zeros(feature_dim, 0, dtype=J.dtype, device=J.device)

    rows: list[torch.Tensor] = []
    for i in range(left_null.shape[1]):
        lam = left_null[:, i]
        Q = torch.einsum("m,mij->ij", lam, H_restricted)
        Q = 0.5 * (Q + Q.T)
        Q_on_T = T.T @ Q @ T
        vec = symmetric_vectorize(Q_on_T)
        if float(vec.norm()) >= eps_quad:
            rows.append(vec)

    if not rows:
        return torch.zeros(feature_dim, 0, dtype=J.dtype, device=J.device)

    Q_rows = torch.stack(rows, dim=0)
    return compute_row_space(Q_rows, eps=eps_quad)


# ---------------------------------------------------------------------------
# Constraint preservability classification
# ---------------------------------------------------------------------------

def classify_pred_constraints(
    pred_sketch: datalib.Sketch,
    gt_row_space: torch.Tensor,
    context,
    thresh: float = 0.95,
    dtype: torch.dtype = torch.float64,
) -> tuple[list[str], list[str]]:
    """
    Classify each Pred constraint as 'primary' or 'auxiliary' w.r.t. GT row space.

    A constraint is **primary** if at least one of its Jacobian rows
    is strongly aligned with the GT row space R_gt:

        ‖R_gt @ R_gt.T @ j‖ / ‖j‖ >= thresh

    In other words, a constraint is primary when its gradient direction is
    sufficiently aligned with the direction space that GT constrains; it is
    auxiliary when it does not overlap (i.e. lies within GT's null space).

    Parameters
    ----------
    pred_sketch : Sketch whose constraints are to be classified
    gt_row_space : (n, r) orthonormal basis of GT primary row space
    context : SketchContext
    thresh : threshold for projection ratio
    dtype : floating type

    Returns
    -------
    primary_ids, auxiliary_ids : lists of constraint IDs
    """
    state = SketchState(pred_sketch, dtype=dtype)
    R = gt_row_space  # (n, r)

    primary: list[str] = []
    auxiliary: list[str] = []

    for cid, constraint in pred_sketch.constraints.items():
        handler_name = constraint.constraint_type.name
        handler = context.get_constraint_handler(handler_name)
        if handler is None:
            auxiliary.append(cid)
            continue

        def _res_fn(q: torch.Tensor) -> torch.Tensor:
            from src.sketch_handlers.sketch_solver import _StateWithQ
            s = _StateWithQ(state, q)
            try:
                return handler.residual(constraint, s)
            except Exception:
                return torch.zeros(1, dtype=q.dtype)

        try:
            jac = torch.func.jacfwd(_res_fn)(state.q.detach())  # (k, n)
            if jac.ndim == 1:
                jac = jac.unsqueeze(0)

            # primary: gradient overlaps the GT row space
            is_primary = False
            for row in jac:
                j_norm = float(row.norm())
                if j_norm < 1e-15:
                    continue
                # projection: R @ R.T @ j (project onto the GT row space)
                proj = R @ (R.T @ row)
                proj_ratio = float(proj.norm()) / j_norm
                if proj_ratio >= thresh:
                    is_primary = True
                    break

            (primary if is_primary else auxiliary).append(cid)

        except Exception:
            auxiliary.append(cid)

    return primary, auxiliary


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def compute_null_space_distance(
    gt_primary_sketch: datalib.Sketch,
    pred_sketch: datalib.Sketch,
    context,
    eps_preserve: float = 0.95,
    eps_rank: float = 1e-8,
    eps_quad: float = 1e-10,
    dtype: torch.dtype = torch.float64,
    skip_pred_primary_extraction: bool = False,
    base_eids: list[str] | None = None,
    include_quadratic: bool = False,
    return_sigmas: bool = False,
) -> dict:
    """
    Compute the row-space Grassmann distance between GT and Pred design intent.

    Parameters
    ----------
    gt_primary_sketch           : Sketch containing only *primary* constraints (the `_mid` files)
    pred_sketch                 : Model prediction sketch (same primitives, no primary/auxiliary distinction)
    context                     : SketchContext providing constraint handlers
    eps_preserve                : threshold for Jacobian-projection-based preservability test
    eps_rank                    : singular value threshold for rank / row-space computation
    eps_quad                    : singular value threshold for quadratic signature rank
    dtype                       : floating type for all tensors
    skip_pred_primary_extraction: if True, skip classify_pred_constraints and treat pred_sketch as
                                  pred_primary_sketch (i.e. all its constraints are already primary).
                                  In this case pred_primary_ids = list(pred_sketch.constraints.keys())
                                  and pred_auxiliary_ids = [].
    base_eids                   : if provided, restricts the null space / row space to the parameters
                                  belonging only to these entity IDs (e.g., ignoring construction geometry).
                                  Quadratic signatures eliminate non-base coordinates as hidden variables
                                  instead of dropping their Hessian columns.
    include_quadratic           : if True, also compare second-order intent signatures in Sym^2(T_gt*).


    Returns
    -------
    dict with keys:
        "intent_recall"         : float  — Intent Recall in [0, 1] (higher = better)
        "linear_intent_recall"  : float  — same as intent_recall, for explicit reporting
        "quadratic_intent_recall": float | None — second-order recall, if requested
        "two_jet_intent_recall" : float  — dimension-weighted linear + quadratic recall
        "gt_row_dim"            : int    — dimension of GT row space (= rank of J_GT)
        "pred_row_dim"          : int    — dimension of Pred primary row space
        "gt_quadratic_dim"      : int    — dimension of GT second-order signature
        "pred_quadratic_dim"    : int    — dimension of Pred second-order signature
        "gt_null_dim"           : int    — GT null space dimension (= DoF remaining)
        "pred_null_dim"         : int    — Pred null space dimension
        "pred_primary_ids"      : list[str]
        "pred_auxiliary_ids"    : list[str]
        "pred_primary_count"    : int
        "pred_auxiliary_count"  : int
    """
    # --- Step 1: GT row space (= directions GT primary constraints restrict) ---
    gt_constraint_ids = list(gt_primary_sketch.constraints.keys())
    if include_quadratic:
        J_gt, H_gt, state_gt = compute_jacobian_hessians(
            gt_primary_sketch, gt_constraint_ids, context, dtype=dtype
        )
    else:
        J_gt, state_gt = compute_jacobian(gt_primary_sketch, gt_constraint_ids, context, dtype=dtype)
        H_gt = None
    
    if base_eids is not None:
        gt_base_indices = get_entity_indices(state_gt, base_eids)
        R_gt = compute_row_space(J_gt, eps=eps_rank, base_indices=gt_base_indices)
        gt_null_dim = len(gt_base_indices) - R_gt.shape[1]
    else:
        R_gt = compute_row_space(J_gt, eps=eps_rank)
        N_gt = compute_null_space(J_gt, eps=eps_rank)
        gt_null_dim = N_gt.shape[1]

    gt_row_dim = R_gt.shape[1]

    # --- Step 2: Classify Pred constraints w.r.t. GT row space ---
    if skip_pred_primary_extraction:
        # pred_sketch is already primary-only; skip classification
        pred_primary_ids = list(pred_sketch.constraints.keys())
        pred_auxiliary_ids = []
    else:
        if base_eids is not None:
            raise NotImplementedError("classify_pred_constraints with base_eids is not implemented yet.")
        pred_primary_ids, pred_auxiliary_ids = classify_pred_constraints(
            pred_sketch, R_gt, context, thresh=eps_preserve, dtype=dtype
        )

    # --- Step 3: Pred row space (primary constraints only) ---
    if include_quadratic:
        J_pred, H_pred, state_pred = compute_jacobian_hessians(
            pred_sketch, pred_primary_ids, context, dtype=dtype
        )
    else:
        J_pred, state_pred = compute_jacobian(pred_sketch, pred_primary_ids, context, dtype=dtype)
        H_pred = None
    
    if base_eids is not None:
        pred_base_indices = get_entity_indices(state_pred, base_eids)
        R_pred = compute_row_space(J_pred, eps=eps_rank, base_indices=pred_base_indices)
        pred_null_dim = len(pred_base_indices) - R_pred.shape[1]
    else:
        R_pred = compute_row_space(J_pred, eps=eps_rank)
        N_pred = compute_null_space(J_pred, eps=eps_rank)
        pred_null_dim = N_pred.shape[1]

    pred_row_dim = R_pred.shape[1]

    # --- Step 4: Intent Recall ---
    recall = intent_recall(R_gt, R_pred)

    linear_sigmas: list[float] | None = None
    if return_sigmas and R_gt.shape[1] > 0 and R_pred.shape[1] > 0:
        M = R_gt.T @ R_pred
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
        linear_sigmas = S.clamp(0.0, 1.0).tolist()

    # --- Step 5: Optional second-order Intent Recall ---
    quadratic_recall: float | None = None
    gt_quadratic_dim = 0
    pred_quadratic_dim = 0

    if include_quadratic:
        assert H_gt is not None and H_pred is not None

        if base_eids is not None:
            J_gt_effective, H_gt_effective = _effective_base_derivatives(
                J_gt,
                H_gt,
                gt_base_indices,
                eps_linear=eps_rank,
            )
            J_pred_effective, H_pred_effective = _effective_base_derivatives(
                J_pred,
                H_pred,
                pred_base_indices,
                eps_linear=eps_rank,
            )
            domain_basis = compute_null_space(J_gt_effective, eps=eps_rank)
            Q_gt = compute_quadratic_signature(
                J_gt_effective,
                H_gt_effective,
                domain_basis=domain_basis,
                eps_linear=eps_rank,
                eps_quad=eps_quad,
            )
            Q_pred = compute_quadratic_signature(
                J_pred_effective,
                H_pred_effective,
                domain_basis=domain_basis,
                eps_linear=eps_rank,
                eps_quad=eps_quad,
            )
        else:
            domain_basis = compute_null_space(J_gt, eps=eps_rank)
            Q_gt = compute_quadratic_signature(
                J_gt,
                H_gt,
                domain_basis=domain_basis,
                eps_linear=eps_rank,
                eps_quad=eps_quad,
            )
            Q_pred = compute_quadratic_signature(
                J_pred,
                H_pred,
                domain_basis=domain_basis,
                eps_linear=eps_rank,
                eps_quad=eps_quad,
            )

        gt_quadratic_dim = Q_gt.shape[1]
        pred_quadratic_dim = Q_pred.shape[1]
        quadratic_recall = intent_recall(Q_gt, Q_pred)

    two_jet_recall = combine_two_jet_recall(
        linear_recall=recall,
        quadratic_recall=quadratic_recall,
        gt_row_dim=gt_row_dim,
        gt_quadratic_dim=gt_quadratic_dim,
    )

    return {
        "intent_recall": recall,
        "linear_intent_recall": recall,
        "quadratic_intent_recall": quadratic_recall,
        "two_jet_intent_recall": two_jet_recall,
        "gt_row_dim": gt_row_dim,
        "pred_row_dim": pred_row_dim,
        "gt_quadratic_dim": gt_quadratic_dim,
        "pred_quadratic_dim": pred_quadratic_dim,
        "gt_null_dim": gt_null_dim,
        "pred_null_dim": pred_null_dim,
        "pred_primary_ids": pred_primary_ids,
        "pred_auxiliary_ids": pred_auxiliary_ids,
        "pred_primary_count": len(pred_primary_ids),
        "pred_auxiliary_count": len(pred_auxiliary_ids),
        "linear_sigmas": linear_sigmas,
    }
