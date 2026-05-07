"""
null_space_distance.py — Design intent evaluation metric via row-space alignment.

設計方針 (Intent Recall):
  GT primary制約の Jacobian の行空間 R_GT と、
  Pred制約の Jacobian の行空間 R_Pred の間の「整合度」を測る。

  主角 θ_i を用いた Intent Recall:
    1. Q_GT  : R_GT の正規直交基底  (n × d_GT)
    2. Q_Pred: R_Pred の正規直交基底 (n × d_Pred)
    3. M = Q_GT^T @ Q_Pred  →  SVD  →  σ_i = cos(θ_i)
    4. Score = Σ σ_i / dim(R_GT)  ∈ [0, 1]

  Null space同士のGrassmann距離を使わない理由:
    Pred=空制約のとき null space=全空間 → GT null spaceが自明に含まれ距離=0になる。

  Grassmann距離 (sqrt(Σ arccos(σ_i)²)) との違い:
    - Intent Recall は arccos 不要・常に [0,1] に正規化される
    - 次元差 (d_Pred < d_GT) は不足分が自然に σ=0 として反映される
    - サンプル間で dim が異なっても横並び比較が可能

Pipeline:
  1. GT_primary の Jacobian から row space R_gt を計算
  2. Pred の各制約について Jacobian row が R_gt に含まれるか判定 (primary分類)
  3. Pred primary制約の Jacobian から row space R_pred を計算
  4. Grassmann distance(R_gt, R_pred) を返す

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
    Intent Recall: GT行空間の方向をPredの行空間がどれだけカバーするかを測る。

    計算:
      M = Q_GT^T @ Q_Pred  →  SVD  →  σ_i = cos(θ_i)  (主角のコサイン)
      Score = Σ σ_i / dim(R_GT)  ∈ [0, 1]

    解釈:
      - Score = 1: PredがGTの全設計意図方向を完全にカバー
      - Score = 0: PredがGTの設計意図方向と全く重ならない
      - d_pred < d_gt: 不足次元は σ=0 として自然に反映（恣意的な定数なし）
      - d_pred = 0 (制約なし): Score = 0

    Args:
      R_gt  : (n, d_gt)   GT primary行空間の正規直交基底
      R_pred: (n, d_pred) Pred行空間の正規直交基底

    Returns:
      float in [0, 1]
    """
    d_gt = R_gt.shape[1]

    if d_gt == 0:
        # GTに制約なし → 定義上 recall = 1
        return 1.0

    if R_pred.shape[1] == 0:
        # Pred に制約なし → recall = 0
        return 0.0

    M = R_gt.T @ R_pred  # (d_gt, d_pred)
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    S = S.clamp(0.0, 1.0)  # 数値誤差で僅かに範囲外になる場合を防ぐ
    # Σ σ_i (= Σ cos(θ_i)) を GT 次元数で正規化
    # d_pred < d_gt のとき S のサイズは d_pred < d_gt であり、
    # 不足分は和に寄与しない (σ=0 と等価)
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

    つまり制約の勾配方向が GT が束縛する方向空間と十分に揃う場合 = primary.
    重ならない（GT の null space 内にある）場合 = auxiliary.

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

            # primary: 勾配が GT row space と重なる
            is_primary = False
            for row in jac:
                j_norm = float(row.norm())
                if j_norm < 1e-15:
                    continue
                # 射影: R @ R.T @ j (GT row space への投影)
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
            J_gt_domain = J_gt[:, gt_base_indices]
            domain_basis = compute_null_space(J_gt_domain, eps=eps_rank)
            Q_gt = compute_quadratic_signature(
                J_gt,
                H_gt,
                variable_indices=gt_base_indices,
                domain_basis=domain_basis,
                eps_linear=eps_rank,
                eps_quad=eps_quad,
            )
            Q_pred = compute_quadratic_signature(
                J_pred,
                H_pred,
                variable_indices=pred_base_indices,
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
