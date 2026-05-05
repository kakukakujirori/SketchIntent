from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import sketchgraphs.data as datalib

from src.metrics.null_space_distance import compute_null_space_distance
from src.sketch_handlers.context import SketchContext


def _line(eid: str, start: tuple[float, float], end: tuple[float, float]) -> datalib.Line:
    return datalib.Line.from_info({
        "id": eid,
        "startPoint": np.array(start, dtype=float),
        "endPoint": np.array(end, dtype=float),
        "isConstruction": False,
    })


def _constraint(context: SketchContext, name: str, refs: tuple[str, ...]):
    handler = context.get_constraint_handler(name)
    assert handler is not None
    return handler.create_constraint(name, refs)


def _square_sketch(context: SketchContext, include_equal: bool) -> datalib.Sketch:
    entities = OrderedDict([
        ("0", _line("0", (0.0, 0.0), (1.0, 0.0))),
        ("1", _line("1", (1.0, 0.0), (1.0, 1.0))),
        ("2", _line("2", (1.0, 1.0), (0.0, 1.0))),
        ("3", _line("3", (0.0, 1.0), (0.0, 0.0))),
    ])
    constraints = OrderedDict([
        ("c0", _constraint(context, "Coincident", ("0.end", "1.start"))),
        ("c1", _constraint(context, "Coincident", ("1.end", "2.start"))),
        ("c2", _constraint(context, "Coincident", ("2.end", "3.start"))),
        ("c3", _constraint(context, "Coincident", ("3.end", "0.start"))),
        ("c4", _constraint(context, "Vertical", ("1",))),
        ("c5", _constraint(context, "Vertical", ("3",))),
        ("c6", _constraint(context, "Horizontal", ("2",))),
    ])
    if include_equal:
        constraints["c7"] = _constraint(context, "Equal", ("0", "2"))
    return datalib.Sketch(entities=entities, constraints=constraints)


def test_quadratic_recall_detects_equal_length_missing_from_linear_row_space():
    context = SketchContext()
    gt = _square_sketch(context, include_equal=True)
    pred_without_equal = _square_sketch(context, include_equal=False)

    result = compute_null_space_distance(
        gt_primary_sketch=gt,
        pred_sketch=pred_without_equal,
        context=context,
        skip_pred_primary_extraction=True,
        include_quadratic=True,
    )

    assert result["linear_intent_recall"] == pytest.approx(1.0, abs=1e-8)
    assert result["gt_quadratic_dim"] >= 1
    assert result["pred_quadratic_dim"] == 0
    assert result["quadratic_intent_recall"] == pytest.approx(0.0, abs=1e-8)


def test_quadratic_recall_matches_when_equal_length_is_present():
    context = SketchContext()
    gt = _square_sketch(context, include_equal=True)
    pred_with_equal = _square_sketch(context, include_equal=True)

    result = compute_null_space_distance(
        gt_primary_sketch=gt,
        pred_sketch=pred_with_equal,
        context=context,
        skip_pred_primary_extraction=True,
        include_quadratic=True,
    )

    assert result["linear_intent_recall"] == pytest.approx(1.0, abs=1e-8)
    assert result["gt_quadratic_dim"] >= 1
    assert result["pred_quadratic_dim"] == result["gt_quadratic_dim"]
    assert result["quadratic_intent_recall"] == pytest.approx(1.0, abs=1e-8)


def test_linear_constraints_have_empty_quadratic_signature():
    context = SketchContext()
    sketch = _square_sketch(context, include_equal=False)

    result = compute_null_space_distance(
        gt_primary_sketch=sketch,
        pred_sketch=sketch,
        context=context,
        skip_pred_primary_extraction=True,
        include_quadratic=True,
    )

    assert result["linear_intent_recall"] == pytest.approx(1.0, abs=1e-8)
    assert result["gt_quadratic_dim"] == 0
    assert result["pred_quadratic_dim"] == 0
    assert result["quadratic_intent_recall"] == pytest.approx(1.0, abs=1e-8)
