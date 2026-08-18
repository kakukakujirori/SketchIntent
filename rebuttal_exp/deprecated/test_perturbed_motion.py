"""The cached evaluator must reproduce pair_paper_metrics, and only R may drive it."""
from __future__ import annotations

from collections import OrderedDict

import pytest
import sketchgraphs.data as datalib
import torch

from rebuttal_exp.perturbed_motion import (
    PairEvaluator,
    motion_with_response,
    pred_response_state,
)
from rebuttal_exp.response_perturbation import (
    MAIN_ARM,
    OBSERVABLE_NULL_ARM,
    build_response_geometry,
    perturbed_response,
)
from rebuttal_exp.tests.fixtures import arc, circle, constraint, line
from src.eval.motion_metrics import SketchMotion, pair_paper_metrics
from src.metrics.edit_response_distance import pairwise_distance_observable
from src.sketch_handlers.context import SketchContext

EPS = 1e-9
RCOND = 1e-6
ETAS = (0.01, 0.10, 0.50, 1.00)
SHARED_METRICS = ("theta_full", "theta_rel", "theta_leak", "placement_sim",
                  "k_full_pred", "k_full_gt", "k_rel_pred", "k_rel_gt",
                  "n_dim_pred", "n_dim_gt")


@pytest.fixture(scope="module")
def context() -> SketchContext:
    return SketchContext()


def _sketch(context: SketchContext, with_equal: bool) -> datalib.Sketch:
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


@pytest.fixture(scope="module")
def pair(context):
    pred_sketch = _sketch(context, with_equal=False)
    gt_sketch = _sketch(context, with_equal=True)
    pred = SketchMotion(pred_sketch, context, None, eps=EPS)
    gt = SketchMotion(gt_sketch, context, None, eps=EPS)
    j_phi = pairwise_distance_observable(pred_sketch, base_eids=None).to(
        dtype=pred.native.R.dtype
    )
    geometry = build_response_geometry(pred.native, j_phi, eps=EPS, rcond=RCOND)
    return pred, gt, j_phi, geometry


def test_evaluator_reproduces_pair_paper_metrics_at_the_baseline(pair):
    pred, gt, j_phi, _geometry = pair
    reference = pair_paper_metrics(pred, gt)

    fast = PairEvaluator(pred, gt).metrics(pred_response_state(pred.native.R, j_phi, EPS))

    assert reference is not None and fast is not None
    for key in SHARED_METRICS:
        assert fast[key] == pytest.approx(reference[key], rel=1e-8, abs=1e-10), key


def test_substituted_response_agrees_with_the_paper_path(pair):
    """The fast path stays exact once R is no longer the minimum-norm solution."""
    pred, gt, j_phi, geometry = pair
    R_eta, _diag = perturbed_response(geometry, MAIN_ARM, 0.5, seed=17, eps=EPS, rcond=RCOND)

    slow = pair_paper_metrics(motion_with_response(pred, R_eta, j_phi), gt)
    fast = PairEvaluator(pred, gt).metrics(pred_response_state(R_eta, j_phi, EPS))

    for key in SHARED_METRICS:
        assert fast[key] == pytest.approx(slow[key], rel=1e-8, abs=1e-10), key


def test_motion_with_response_rebuilds_only_the_response_dependent_fields(pair):
    pred, _gt, j_phi, geometry = pair
    R_eta, _diag = perturbed_response(geometry, MAIN_ARM, 0.5, seed=2, eps=EPS, rcond=RCOND)

    swapped = motion_with_response(pred, R_eta, j_phi)

    assert swapped.sketch is pred.sketch
    assert swapped.context is pred.context
    assert swapped.U_rel is pred.U_rel and swapped.k_rel == pred.k_rel
    assert swapped.native.state is pred.native.state
    assert torch.equal(swapped.native.R, R_eta)
    assert torch.equal(pred.native.R, geometry.R0), "the base motion must not be mutated"
    assert not torch.equal(swapped.U_full, pred.U_full)


@pytest.mark.parametrize("eta", ETAS)
def test_relational_angle_is_untouched_by_the_response_choice(pair, eta):
    pred, gt, j_phi, geometry = pair
    evaluator = PairEvaluator(pred, gt)
    baseline = evaluator.metrics(pred_response_state(pred.native.R, j_phi, EPS))

    for arm in (MAIN_ARM, OBSERVABLE_NULL_ARM):
        R_eta, _diag = perturbed_response(geometry, arm, eta, seed=4, eps=EPS, rcond=RCOND)
        moved = pair_paper_metrics(motion_with_response(pred, R_eta, j_phi), gt)
        assert moved["theta_rel"] == baseline["theta_rel"]


@pytest.mark.parametrize("eta", ETAS)
def test_observable_null_arm_moves_only_the_raw_space_metric(pair, eta):
    """theta_full and S_place live in observable space, which this arm cannot reach."""
    pred, gt, j_phi, geometry = pair
    evaluator = PairEvaluator(pred, gt)
    baseline = evaluator.metrics(pred_response_state(pred.native.R, j_phi, EPS))

    R_eta, _diag = perturbed_response(geometry, OBSERVABLE_NULL_ARM, eta, seed=9)
    moved = evaluator.metrics(pred_response_state(R_eta, j_phi, EPS))

    assert moved["theta_full"] == pytest.approx(baseline["theta_full"], abs=1e-9)
    assert moved["placement_sim"] == pytest.approx(baseline["placement_sim"], abs=1e-9)


def test_main_arm_moves_the_observable_metrics(pair):
    pred, gt, j_phi, geometry = pair
    evaluator = PairEvaluator(pred, gt)
    baseline = evaluator.metrics(pred_response_state(pred.native.R, j_phi, EPS))

    R_eta, _diag = perturbed_response(geometry, MAIN_ARM, 1.0, seed=21, eps=EPS, rcond=RCOND)
    moved = evaluator.metrics(pred_response_state(R_eta, j_phi, EPS))

    assert abs(moved["theta_full"] - baseline["theta_full"]) > 1e-6


def test_incomparable_observable_layouts_produce_no_metrics(context):
    pred = SketchMotion(_sketch(context, with_equal=True), context, None, eps=EPS)
    smaller = datalib.Sketch(
        entities=OrderedDict([("0", circle("0", 0.0, 0.0, 1.0))]), constraints=OrderedDict()
    )
    gt = SketchMotion(smaller, context, None, eps=EPS)
    j_phi = pairwise_distance_observable(pred.sketch, base_eids=None).to(
        dtype=pred.native.R.dtype
    )

    evaluator = PairEvaluator(pred, gt)
    assert evaluator.metrics(pred_response_state(pred.native.R, j_phi, EPS)) is None
