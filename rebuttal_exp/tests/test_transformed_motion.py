"""The B-injected pipeline must reproduce the paper metrics when B is the identity."""
from __future__ import annotations

from collections import OrderedDict

import pytest
import sketchgraphs.data as datalib
import torch

from rebuttal_exp.primitive_charts import C1, C2, build_global_chart, identity_chart
from rebuttal_exp.parameterization_stress import response_rank
from rebuttal_exp.tests.fixtures import arc, circle, constraint, line
from rebuttal_exp.transformed_motion import (
    TransformedMotion,
    cutoff_pinv,
    transformed_pair_metrics,
)
from src.eval.motion_metrics import SketchMotion, pair_paper_metrics
from src.metrics.edit_response_distance import compute_native_response
from src.metrics.motion_subspace import orthonormal_range
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_solver import SketchState

EPS = 1e-9
SCALE = 1.0
ANGLE_METRICS = ("theta_full", "theta_rel", "theta_leak")


@pytest.fixture(scope="module")
def context() -> SketchContext:
    return SketchContext()


def _constrained_sketch(context: SketchContext, with_equal: bool) -> datalib.Sketch:
    entities = OrderedDict([
        ("0", line("0", 0.0, 0.0, 2.0, 0.0)),
        ("1", line("1", 2.0, 0.0, 2.0, 2.0)),
        ("2", arc("2", 3.5, 1.0, 0.8, start_param=0.2, end_param=2.1)),
        ("3", circle("3", -2.0, 1.0, 0.6)),
    ])
    constraints = OrderedDict([
        ("c0", constraint(context, "Coincident", ("0.end", "1.start"))),
        ("c1", constraint(context, "Horizontal", ("0",))),
        ("c2", constraint(context, "Radius", ("3",), value=0.6)),
        ("c3", constraint(context, "Distance", ("0.start", "1.end"), value=2.8284271247461903)),
    ])
    if with_equal:
        constraints["c4"] = constraint(context, "Equal", ("2", "3"))
    return datalib.Sketch(entities=entities, constraints=constraints)


def _motions(sketch, context, scenario, base_eids=None):
    state = SketchState(sketch)
    chart = (identity_chart(state) if scenario == "C0"
             else build_global_chart(sketch, state, scenario, SCALE))
    return TransformedMotion(sketch, context, base_eids, chart, eps=EPS)


def test_identity_chart_reproduces_pair_paper_metrics(context):
    pred = _constrained_sketch(context, with_equal=False)
    gt = _constrained_sketch(context, with_equal=True)

    reference = pair_paper_metrics(
        SketchMotion(pred, context, None, eps=EPS), SketchMotion(gt, context, None, eps=EPS)
    )
    transformed = transformed_pair_metrics(
        _motions(pred, context, "C0"), _motions(gt, context, "C0")
    )

    assert reference is not None and transformed is not None
    for key, value in reference.items():
        assert transformed[key] == pytest.approx(value, rel=1e-8, abs=1e-10), key


def test_identity_chart_preserves_subspace_ranks(context):
    sketch = _constrained_sketch(context, with_equal=True)
    native = SketchMotion(sketch, context, None, eps=EPS)
    transformed = _motions(sketch, context, "C0")

    assert (transformed.k_full, transformed.k_rel) == (native.k_full, native.k_rel)
    assert transformed.k_response == orthonormal_range(native.native.R, EPS)[1]
    assert transformed.n_dim_cols == native.n_dim_cols
    assert transformed.m_phi == native.m_phi
    assert transformed.n_y == native.native.J_x.shape[1]


def test_experiment_records_the_response_rank_used_by_theta_leak(context):
    sketch = _constrained_sketch(context, with_equal=True)
    native = SketchMotion(sketch, context, None, eps=EPS)
    transformed = _motions(sketch, context, C1)

    assert response_rank(native) == orthonormal_range(native.native.R, EPS)[1]
    assert response_rank(transformed) == orthonormal_range(transformed.R_t, EPS)[1]


def test_alternative_charts_change_the_variable_count_as_expected(context):
    sketch = _constrained_sketch(context, with_equal=True)
    n_x = SketchState(sketch).num_dof()

    assert _motions(sketch, context, C1).n_y == n_x - 4, "two lines lose two variables each"
    assert _motions(sketch, context, C2).n_y == n_x + 1, "one arc gains one variable"


def test_response_is_recomputed_not_pushed_through_b(context):
    """R~ must solve the transformed system, which R pushed through B does not."""
    sketch = _constrained_sketch(context, with_equal=True)
    native = compute_native_response(sketch, context, eps=EPS)
    motion = _motions(sketch, context, C1)
    B = motion.chart.B

    residual = torch.linalg.norm(native.J_x @ B @ motion.R_t + native.J_p)
    assert float(residual) < 1e-6
    assert motion.R_t.shape == (motion.n_y, native.J_p.shape[1])


def test_relational_subspace_is_invariant_because_it_avoids_the_pseudoinverse(context):
    """theta_rel compares range(J_phi ker(J_R)), which B leaves unchanged."""
    pred = _constrained_sketch(context, with_equal=False)
    gt = _constrained_sketch(context, with_equal=True)
    native = transformed_pair_metrics(
        _motions(pred, context, "C0"), _motions(gt, context, "C0")
    )
    transformed = transformed_pair_metrics(
        _motions(pred, context, C1), _motions(gt, context, C1)
    )
    assert transformed["theta_rel"] == pytest.approx(native["theta_rel"], abs=1e-6)


def test_metrics_are_none_when_a_side_was_skipped(context):
    sketch = _constrained_sketch(context, with_equal=True)
    usable = _motions(sketch, context, "C0")

    degenerate = line("0", 0.0, 0.0, 1.0, 0.0)
    degenerate.endParam = degenerate.startParam
    broken = datalib.Sketch(
        entities=OrderedDict([("0", degenerate)]), constraints=OrderedDict()
    )
    skipped = _motions(broken, context, C1)

    assert not skipped.usable
    assert skipped.skip_reason == "degenerate_line_span"
    assert transformed_pair_metrics(skipped, usable) is None
    assert transformed_pair_metrics(usable, skipped) is None


def test_gauge_residuals_are_reported_for_every_jacobian(context):
    motion = _motions(_constrained_sketch(context, with_equal=True), context, C1)
    assert set(motion.gauge_residuals) == {
        "gauge_residual_J_x", "gauge_residual_J_R", "gauge_residual_J_phi"
    }
    assert all(v >= 0.0 for v in motion.gauge_residuals.values())


def test_cutoff_pinv_matches_the_native_response_convention(context):
    sketch = _constrained_sketch(context, with_equal=True)
    native = compute_native_response(sketch, context, eps=EPS)
    recomputed = -cutoff_pinv(native.J_x, eps=EPS) @ native.J_p
    assert torch.allclose(recomputed, native.R, atol=1e-12)


def test_chart_built_for_another_sketch_is_rejected(context):
    small = _constrained_sketch(context, with_equal=True)
    big = _constrained_sketch(context, with_equal=True)
    big.entities["4"] = circle("4", 5.0, 5.0, 1.0)

    chart = identity_chart(SketchState(small))
    with pytest.raises(AssertionError):
        TransformedMotion(big, context, None, chart, eps=EPS)
