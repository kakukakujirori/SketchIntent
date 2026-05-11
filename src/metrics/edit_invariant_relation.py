"""
edit_invariant_relation.py — Edit-Invariant Relation Sensitivity (Proposed Metric 2).

設計方針
--------
edit probe q を与えたときに relation r が壊れる度合い:
    s_C(r, q) = || J_r S_C^(q) ||
を比較する。J_r はリレーションの residual の primitive 座標に対する Jacobian、
S_C^(q) は edit response (Metric 1 を参照)。

距離:
    d_inv(P, G) = Σ_q Σ_r w_q w_r | s_P(r, q) - s_G(r, q) |

Pred が GT と同じ relation を明示的に持っている必要はない —
edit q の下でその relation が同じ程度保持されていれば良い、という解釈。

`d_inv_finite` (FreeCAD finite version)
---------------------------------------
`d_inv` は元形状まわりの局所 sensitivity (一次) のみを見るので、
FreeCAD solver の roundtrip や solver branch change のような有限効果は
拾えない (notebooks/summary_histcad_complete_report.md 参照)。

`compute_edit_invariant_relation_finite_distance` は同じ probe / relation 集合の上で
有限 edit を pred と gt 両方に FreeCAD で適用し、edit 後の relation residual を
比較する:

    d_inv_finite(P, G) = Σ_q Σ_r w_q w_r | ||r(x_P*(q*))|| - ||r(x_G*(q*))|| |

ここで x_P*(q*) は pred sketch を target q* で FreeCAD solve した結果、
x_G*(q*) は gt sketch を同じ target q* で FreeCAD solve した結果。
HistCAD cPCSR の binary preservation を連続値化した版とほぼ等価。

参考: src/metrics/edit_response_distance.py — ネイティブ次元応答 R_C と
S_C^(q) の計算を共有。src/metrics/histcad_2d.py — finite edit + FreeCAD solve
ヘルパを共有。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import sketchgraphs.data as datalib

from src.metrics.edit_response_distance import (
    NativeResponse,
    compute_native_response,
    constraint_jacobian,
    edit_response_for_probe,
)


@dataclass
class RelationSensitivityResult:
    distance: float
    s_pred: dict[tuple[int, int], float | None]   # (q_idx, r_idx) -> sensitivity
    s_gt: dict[tuple[int, int], float | None]
    n_dim_gt: int
    n_dim_pred: int
    skipped_pairs: int


def relation_sensitivity_table(
    native: NativeResponse,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    eps: float = 1e-8,
) -> dict[tuple[int, int], float | None]:
    """
    Compute s_C(r, q) = || J_r S_C^(q) || for every (q, r) pair in this sketch.

    Returns a dict keyed by (q_idx, r_idx). A value of None means the probe
    Jacobian could not be computed (e.g. ref entity missing in sketch).
    """
    # Per-probe edit responses S_C^(q): cache once per q.
    S_per_q: list[torch.Tensor | None] = [
        edit_response_for_probe(native, q, context, eps=eps) for q in edit_probes
    ]
    # Per-relation Jacobians J_r: cache once per r.
    Jr_per_r: list[torch.Tensor | None] = [
        constraint_jacobian(r, native.state, context) for r in relation_probes
    ]

    out: dict[tuple[int, int], float | None] = {}
    for qi, S in enumerate(S_per_q):
        for ri, Jr in enumerate(Jr_per_r):
            if S is None or Jr is None:
                out[(qi, ri)] = None
                continue
            out[(qi, ri)] = float(torch.linalg.norm(Jr @ S))
    return out


def compute_edit_invariant_relation_distance(
    gt_sketch: datalib.Sketch,
    pred_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    edit_weights: Sequence[float] | None = None,
    relation_weights: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-8,
) -> RelationSensitivityResult:
    """
    Compute Edit-Invariant Relation Sensitivity Distance (Proposed Metric 2).

    d_inv(P, G) = Σ_q Σ_r w_q w_r | s_P(r,q) - s_G(r,q) |

    The sketches must share a common primitive layout (e.g. via
    `remap_sketch_to_canonical`).
    """
    gt_native = compute_native_response(gt_sketch, context, dtype=dtype, eps=eps)
    pred_native = compute_native_response(pred_sketch, context, dtype=dtype, eps=eps)

    if edit_weights is None:
        edit_weights = [1.0] * len(edit_probes)
    if relation_weights is None:
        relation_weights = [1.0] * len(relation_probes)
    if len(edit_weights) != len(edit_probes):
        raise ValueError("len(edit_weights) must match len(edit_probes)")
    if len(relation_weights) != len(relation_probes):
        raise ValueError("len(relation_weights) must match len(relation_probes)")

    s_gt = relation_sensitivity_table(
        gt_native, edit_probes, relation_probes, context, eps=eps
    )
    s_pred = relation_sensitivity_table(
        pred_native, edit_probes, relation_probes, context, eps=eps
    )

    total = 0.0
    skipped = 0
    for qi, w_q in enumerate(edit_weights):
        for ri, w_r in enumerate(relation_weights):
            sg = s_gt[(qi, ri)]
            sp = s_pred[(qi, ri)]
            if sg is None or sp is None:
                skipped += 1
                continue
            total += w_q * w_r * abs(sp - sg)

    return RelationSensitivityResult(
        distance=total,
        s_pred=s_pred,
        s_gt=s_gt,
        n_dim_gt=len(gt_native.dim_constraint_ids),
        n_dim_pred=len(pred_native.dim_constraint_ids),
        skipped_pairs=skipped,
    )


# ---------------------------------------------------------------------------
# d_inv_finite: post-FreeCAD-solve relation residual distance
# ---------------------------------------------------------------------------

@dataclass
class RelationFiniteResult:
    distance: float                              # Σ wq wr |||r(xP*)|| - ||r(xG*)|||
    res_pred: dict[tuple[int, int, int], float]  # (qi, sign_idx, ri) -> ||r(xP*)||
    res_gt:   dict[tuple[int, int, int], float]
    n_relations: int
    n_probe_signs: int
    n_skipped: int                               # (q, sign) pairs where solve failed
    pred_solve_failures: int
    gt_solve_failures: int


def _residual_norms(
    state,
    relation_probes: Sequence[datalib.Constraint],
    context,
) -> list[float | None]:
    """Per-relation residual norm at the given state. None if eval impossible."""
    out: list[float | None] = []
    for r in relation_probes:
        handler = context.get_constraint_handler(r.constraint_type.name)
        if handler is None:
            out.append(None)
            continue
        try:
            res = handler.residual(r, state).reshape(-1)
            out.append(float(torch.linalg.norm(res)))
        except Exception:
            out.append(None)
    return out


def compute_edit_invariant_relation_finite_distance(
    gt_sketch: datalib.Sketch,
    pred_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    relation_probes: Sequence[datalib.Constraint],
    context,
    converter,
    config,                                      # HistCADConfig (delta_*, signs)
    edit_weights: Sequence[float] | None = None,
    relation_weights: Sequence[float] | None = None,
    dtype: torch.dtype = torch.float64,
) -> RelationFiniteResult:
    """Finite-edit relation-residual distance between pred and gt.

    For each (probe q, sign s):
      - read q0 on pred (pred-anchored convention), set target = q0 + delta(s)
      - apply edit to pred and gt sketches independently
      - FreeCAD-solve both
      - if both solves succeed, evaluate ||r|| of every relation on each
      - accumulate w_q * w_r * | ||r(x_P*)|| - ||r(x_G*)|| |

    Solve failures cause that (q, sign) pair to be skipped (counted in n_skipped).
    Relation eval failures count as 0 contribution for that (q, sign, r) cell.
    Both sketches must share canonical entity IDs.
    """
    from src.metrics.histcad_2d import (
        _apply_edit, _bbox_diag, _compute_delta, _evaluate_observable,
        _solve_via_freecad,
    )
    from src.sketch_handlers.sketch_solver import SketchState

    if edit_weights is None:
        edit_weights = [1.0] * len(edit_probes)
    if relation_weights is None:
        relation_weights = [1.0] * len(relation_probes)
    if len(edit_weights) != len(edit_probes):
        raise ValueError("len(edit_weights) must match len(edit_probes)")
    if len(relation_weights) != len(relation_probes):
        raise ValueError("len(relation_weights) must match len(relation_probes)")

    pred_state0 = SketchState(pred_sketch, dtype=dtype)
    bbox = _bbox_diag(pred_sketch, dtype=dtype)

    res_pred: dict[tuple[int, int, int], float] = {}
    res_gt:   dict[tuple[int, int, int], float] = {}
    total = 0.0
    n_skipped_signs = 0
    pred_fail = 0
    gt_fail = 0
    n_probe_signs = 0

    for qi, q in enumerate(edit_probes):
        q0 = _evaluate_observable(q, pred_state0, context)
        if q0 is None:
            for si, _s in enumerate(config.delta_signs):
                n_skipped_signs += 1
                n_probe_signs += 1
            continue

        for si, sign in enumerate(config.delta_signs):
            n_probe_signs += 1
            delta = _compute_delta(config, sign, q0, bbox)
            target = q0 + delta

            edited_pred = _apply_edit(pred_sketch, q, target)
            edited_gt   = _apply_edit(gt_sketch,   q, target)

            new_pred, sp = _solve_via_freecad(edited_pred, converter)
            new_gt,   sg = _solve_via_freecad(edited_gt,   converter)

            if new_pred is None:
                pred_fail += 1
            if new_gt is None:
                gt_fail += 1
            if new_pred is None or new_gt is None:
                n_skipped_signs += 1
                continue

            state_p = SketchState(new_pred, dtype=dtype)
            state_g = SketchState(new_gt,   dtype=dtype)
            rp = _residual_norms(state_p, relation_probes, context)
            rg = _residual_norms(state_g, relation_probes, context)

            for ri, w_r in enumerate(relation_weights):
                vp, vg = rp[ri], rg[ri]
                if vp is None or vg is None:
                    continue
                res_pred[(qi, si, ri)] = vp
                res_gt[(qi, si, ri)]   = vg
                total += float(edit_weights[qi]) * float(w_r) * abs(vp - vg)

    return RelationFiniteResult(
        distance=total,
        res_pred=res_pred,
        res_gt=res_gt,
        n_relations=len(relation_probes),
        n_probe_signs=n_probe_signs,
        n_skipped=n_skipped_signs,
        pred_solve_failures=pred_fail,
        gt_solve_failures=gt_fail,
    )
