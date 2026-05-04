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


# ---------------------------------------------------------------------------
# Constraint preservability classification
# ---------------------------------------------------------------------------

def classify_pred_constraints(
    pred_sketch: datalib.Sketch,
    gt_row_space: torch.Tensor,
    context,
    eps: float = 1e-3,
    dtype: torch.dtype = torch.float64,
) -> tuple[list[str], list[str]]:
    """
    Classify each Pred constraint as 'primary' or 'auxiliary' w.r.t. GT row space.

    A constraint is **primary** if at least one of its Jacobian rows
    has a non-negligible projection onto the GT row space R_gt:

        ‖R_gt @ R_gt.T @ j‖ / ‖j‖ >= eps

    つまり制約の勾配方向が GT が束縛する方向空間と重なる場合 = primary.
    重ならない（GT の null space 内にある）場合 = auxiliary.

    Parameters
    ----------
    pred_sketch : Sketch whose constraints are to be classified
    gt_row_space : (n, r) orthonormal basis of GT primary row space
    context : SketchContext
    eps : threshold for projection ratio
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
                if proj_ratio >= eps:
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
    eps_preserve: float = 1e-3,
    eps_rank: float = 1e-8,
    dtype: torch.dtype = torch.float64,
    skip_pred_primary_extraction: bool = False,
    base_eids: list[str] | None = None,
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
    dtype                       : floating type for all tensors
    skip_pred_primary_extraction: if True, skip classify_pred_constraints and treat pred_sketch as
                                  pred_primary_sketch (i.e. all its constraints are already primary).
                                  In this case pred_primary_ids = list(pred_sketch.constraints.keys())
                                  and pred_auxiliary_ids = [].
    base_eids                   : if provided, restricts the null space / row space to the parameters
                                  belonging only to these entity IDs (e.g., ignoring construction geometry).


    Returns
    -------
    dict with keys:
        "intent_recall"         : float  — Intent Recall in [0, 1] (higher = better)
        "gt_row_dim"            : int    — dimension of GT row space (= rank of J_GT)
        "pred_row_dim"          : int    — dimension of Pred primary row space
        "gt_null_dim"           : int    — GT null space dimension (= DoF remaining)
        "pred_null_dim"         : int    — Pred null space dimension
        "pred_primary_ids"      : list[str]
        "pred_auxiliary_ids"    : list[str]
        "pred_primary_count"    : int
        "pred_auxiliary_count"  : int
    """
    # --- Step 1: GT row space (= directions GT primary constraints restrict) ---
    gt_constraint_ids = list(gt_primary_sketch.constraints.keys())
    J_gt, state_gt = compute_jacobian(gt_primary_sketch, gt_constraint_ids, context, dtype=dtype)
    
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
            pred_sketch, R_gt, context, eps=eps_preserve, dtype=dtype
        )

    # --- Step 3: Pred row space (primary constraints only) ---
    J_pred, state_pred = compute_jacobian(pred_sketch, pred_primary_ids, context, dtype=dtype)
    
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

    return {
        "intent_recall": recall,
        "gt_row_dim": gt_row_dim,
        "pred_row_dim": pred_row_dim,
        "gt_null_dim": gt_null_dim,
        "pred_null_dim": pred_null_dim,
        "pred_primary_ids": pred_primary_ids,
        "pred_auxiliary_ids": pred_auxiliary_ids,
        "pred_primary_count": len(pred_primary_ids),
        "pred_auxiliary_count": len(pred_auxiliary_ids),
    }
