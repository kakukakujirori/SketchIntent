"""Constraint-token-level F1 between a predicted sketch and one or more GT sketches.

The token-level signature is `(type, sorted_refs, direction)` (reusing
`constraint_signature` from `edit_response_distance`), which is parameterization
invariant: same probe regardless of constraint ordering or auxiliary values.

For multi-reference evaluation, take the max F1 across references.

Returned dict keys (per (pred, gt) pair):
- n_pred, n_gt, n_match: signature counts
- precision, recall, f1
- ... plus 'dim_' / 'rel_' prefixed variants restricted to dimensional or
  relational signatures.
"""
from __future__ import annotations

from typing import Iterable

import sketchgraphs.data as datalib
from src.metrics.edit_response_distance import (
    _is_dimensional,
    _is_relational,
    constraint_signature,
)


def _signatures(sketch: datalib.Sketch, selector=None) -> set[tuple]:
    sigs: set[tuple] = set()
    for c in sketch.constraints.values():
        if selector is not None and not selector(c):
            continue
        sigs.add(constraint_signature(c))
    return sigs


def _prf(pred_sigs: set[tuple], gt_sigs: set[tuple]) -> dict[str, float | int]:
    n_pred = len(pred_sigs)
    n_gt = len(gt_sigs)
    n_match = len(pred_sigs & gt_sigs)
    precision = (n_match / n_pred) if n_pred > 0 else 0.0
    recall = (n_match / n_gt) if n_gt > 0 else 0.0
    if precision + recall <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_match": n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def pair_f1(pred: datalib.Sketch, gt: datalib.Sketch) -> dict[str, float | int]:
    """Compute combined / dimensional / relational F1 for one (pred, gt) pair."""
    out: dict[str, float | int] = {}
    combined = _prf(_signatures(pred), _signatures(gt))
    for k, v in combined.items():
        out[k] = v
    dim_part = _prf(_signatures(pred, _is_dimensional),
                    _signatures(gt, _is_dimensional))
    for k, v in dim_part.items():
        out[f"dim_{k}"] = v
    rel_part = _prf(_signatures(pred, _is_relational),
                    _signatures(gt, _is_relational))
    for k, v in rel_part.items():
        out[f"rel_{k}"] = v
    return out


def multi_ref_f1(
    pred: datalib.Sketch,
    refs: Iterable[datalib.Sketch],
) -> dict[str, float | int]:
    """Max F1 over references (overall + dim + rel).

    For each F1 family we pick the reference that maximises its F1; we then
    surface the precision / recall / counts from that same reference.
    Returns a row dict ready for CSV output. Empty `refs` yields zeros.
    """
    ref_list = list(refs)
    if not ref_list:
        return {
            "n_pred": 0, "n_gt": 0, "n_match": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "dim_n_pred": 0, "dim_n_gt": 0, "dim_n_match": 0,
            "dim_precision": 0.0, "dim_recall": 0.0, "dim_f1": 0.0,
            "rel_n_pred": 0, "rel_n_gt": 0, "rel_n_match": 0,
            "rel_precision": 0.0, "rel_recall": 0.0, "rel_f1": 0.0,
        }
    per_ref = [pair_f1(pred, g) for g in ref_list]

    def _pick(prefix: str) -> dict[str, float | int]:
        f1_key = f"{prefix}f1" if prefix else "f1"
        best_idx = max(range(len(per_ref)), key=lambda i: per_ref[i][f1_key])
        row = per_ref[best_idx]
        keys = ["n_pred", "n_gt", "n_match", "precision", "recall", "f1"]
        return {f"{prefix}{k}": row[f"{prefix}{k}"] for k in keys}

    out: dict[str, float | int] = {}
    out.update(_pick(""))
    out.update(_pick("dim_"))
    out.update(_pick("rel_"))
    return out
