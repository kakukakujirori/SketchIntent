"""A perturbed response must stay an exact solution of the linearised system."""
from __future__ import annotations

from collections import OrderedDict

import pytest
import sketchgraphs.data as datalib
import torch

from rebuttal_exp.response_perturbation import (
    INCONSISTENT_LINEAR_SYSTEM,
    MAIN_ARM,
    NOT_UNDERCONSTRAINED,
    NO_OBSERVABLE_NULL_DIRECTION,
    NO_NONZERO_OBSERVABLE_RESPONSE,
    NO_NONZERO_RAW_RESPONSE,
    OBSERVABLE_NULL_ARM,
    build_response_geometry,
    cutoff_null_space,
    perturbed_response,
    relation_visibility,
    stacked_observable_null,
)
from rebuttal_exp.tests.fixtures import arc, circle, constraint, line
from src.metrics.edit_response_distance import (
    _is_relational,
    compute_native_response,
    pairwise_distance_observable,
)
from src.metrics.motion_subspace import stack_relation_jacobians_at
from src.sketch_handlers.context import SketchContext

EPS = 1e-9
RCOND = 1e-6
ETA = 0.25


@pytest.fixture(scope="module")
def context() -> SketchContext:
    return SketchContext()


def _sketch(context: SketchContext) -> datalib.Sketch:
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
    return datalib.Sketch(entities=entities, constraints=constraints)


def _geometry(context: SketchContext):
    sketch = _sketch(context)
    native = compute_native_response(sketch, context, eps=EPS)
    j_phi = pairwise_distance_observable(sketch, base_eids=None).to(dtype=native.R.dtype)
    return sketch, native, j_phi, build_response_geometry(native, j_phi, eps=EPS, rcond=RCOND)


# ---------------------------------------------------------------------------
# Null-space construction
# ---------------------------------------------------------------------------

def test_cutoff_null_space_keeps_the_full_null_space_of_a_wide_matrix():
    """The economy SVD returns only m right singular vectors and would truncate it."""
    J = torch.zeros(2, 6, dtype=torch.float64)
    J[0, 0] = 1.0
    J[1, 1] = 1.0

    N = cutoff_null_space(J, EPS, RCOND)

    assert N.shape == (6, 4)
    assert torch.allclose(J @ N, torch.zeros(2, 4, dtype=torch.float64), atol=1e-12)
    _U, _S, Vh_economy = torch.linalg.svd(J, full_matrices=False)
    assert Vh_economy.shape[0] == 2, "economy SVD cannot express this null space"


def test_stacked_null_space_annihilates_both_jacobians(context):
    _sk, native, j_phi, geometry = _geometry(context)
    G = geometry.G

    assert torch.allclose(native.J_x @ G, torch.zeros(native.J_x.shape[0], G.shape[1],
                                                      dtype=G.dtype), atol=1e-9)
    assert torch.allclose(j_phi @ G, torch.zeros(j_phi.shape[0], G.shape[1], dtype=G.dtype),
                          atol=1e-9)
    assert geometry.observable_null_dim <= geometry.raw_null_dim


def test_stacked_null_space_is_not_a_numerical_intersection(context):
    """Built from one SVD of [J_x; J_phi], so no tolerance is spent twice."""
    _sk, native, j_phi, _geo = _geometry(context)
    G = stacked_observable_null(native.J_x, j_phi, EPS, RCOND)
    stacked = torch.cat([native.J_x, j_phi], dim=0)
    assert torch.linalg.matrix_rank(stacked, rtol=1e-8) + G.shape[1] == native.J_x.shape[1]


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", [MAIN_ARM, OBSERVABLE_NULL_ARM])
def test_perturbed_response_still_solves_the_linearised_system(context, arm):
    _sk, native, _j_phi, geometry = _geometry(context)
    R_eta, diagnostics = perturbed_response(
        geometry, arm, ETA, seed=7, eps=EPS, rcond=RCOND
    )

    baseline = torch.linalg.norm(native.J_x @ native.R + native.J_p)
    perturbed = torch.linalg.norm(native.J_x @ R_eta + native.J_p)
    assert float(perturbed) == pytest.approx(float(baseline), abs=1e-9)
    assert diagnostics["constraint_residual"] < 1e-9
    assert diagnostics["nullspace_residual"] < 1e-9
    assert not torch.allclose(R_eta, native.R)


def test_main_arm_moves_each_column_by_eta_times_its_observable_norm(context):
    _sk, _native, j_phi, geometry = _geometry(context)
    R_eta, _diag = perturbed_response(geometry, MAIN_ARM, ETA, seed=3, eps=EPS, rcond=RCOND)

    increment = j_phi @ (R_eta - geometry.R0)
    for column in range(int(geometry.R0.shape[1])):
        expected = ETA * float(geometry.observable_column_norms[column])
        assert float(torch.linalg.norm(increment[:, column])) == pytest.approx(expected, rel=1e-8)


def test_observable_null_arm_leaves_the_observable_response_untouched(context):
    _sk, _native, j_phi, geometry = _geometry(context)
    R_eta, _diag = perturbed_response(geometry, OBSERVABLE_NULL_ARM, ETA, seed=3)

    assert torch.allclose(j_phi @ R_eta, j_phi @ geometry.R0, atol=1e-9)
    increment = R_eta - geometry.R0
    for column in range(int(geometry.R0.shape[1])):
        expected = ETA * float(geometry.raw_column_norms[column])
        assert float(torch.linalg.norm(increment[:, column])) == pytest.approx(expected, rel=1e-8)


def test_eta_zero_returns_the_minimum_norm_response_itself(context):
    _sk, _native, _j_phi, geometry = _geometry(context)
    for arm in (MAIN_ARM, OBSERVABLE_NULL_ARM):
        R_eta, diagnostics = perturbed_response(geometry, arm, 0.0, seed=1)
        assert R_eta is geometry.R0
        assert diagnostics["raw_perturbation_ratio"] == 0.0
        assert diagnostics["constraint_residual"] == pytest.approx(
            geometry.consistency_residual
        )
        assert diagnostics["nullspace_residual"] == 0.0


def test_samples_are_reproducible_and_seed_dependent(context):
    _sk, _native, _j_phi, geometry = _geometry(context)
    first, _ = perturbed_response(geometry, MAIN_ARM, ETA, seed=11, eps=EPS, rcond=RCOND)
    again, _ = perturbed_response(geometry, MAIN_ARM, ETA, seed=11, eps=EPS, rcond=RCOND)
    other, _ = perturbed_response(geometry, MAIN_ARM, ETA, seed=12, eps=EPS, rcond=RCOND)

    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fully_constrained_response_is_flagged_not_perturbed(context):
    sketch = datalib.Sketch(
        entities=OrderedDict([("0", circle("0", 0.0, 0.0, 1.0))]),
        constraints=OrderedDict(),
    )
    native = compute_native_response(sketch, context, eps=EPS)
    j_phi = pairwise_distance_observable(sketch, base_eids=None).to(dtype=native.R.dtype)
    fake = type(native)(native.state, torch.eye(3, dtype=torch.float64),
                        torch.zeros(3, 0, dtype=torch.float64),
                        torch.zeros(3, 0, dtype=torch.float64), [])

    geometry = build_response_geometry(fake, j_phi, eps=EPS, rcond=RCOND)
    assert geometry.raw_null_dim == 0
    assert geometry.skip_reason == NOT_UNDERCONSTRAINED
    assert geometry.arm_skip_reason(MAIN_ARM) == NOT_UNDERCONSTRAINED


def test_inconsistent_linear_system_is_a_separate_category(context):
    _sk, native, j_phi, _geo = _geometry(context)
    broken = type(native)(native.state, native.J_x, native.J_p,
                          torch.zeros_like(native.R), native.dim_constraint_ids)

    geometry = build_response_geometry(broken, j_phi, eps=EPS, rcond=RCOND)
    assert geometry.consistency_residual > 1e-6
    assert geometry.skip_reason == INCONSISTENT_LINEAR_SYSTEM


def test_absent_observable_null_directions_skip_only_that_arm(context):
    _sk, native, j_phi, geometry = _geometry(context)
    dense = type(native)(native.state, native.J_x, native.J_p, native.R,
                         native.dim_constraint_ids)
    full_observable = torch.eye(native.J_x.shape[1], dtype=j_phi.dtype)

    without_null = build_response_geometry(dense, full_observable, eps=EPS, rcond=RCOND)
    assert without_null.observable_null_dim == 0
    assert without_null.arm_skip_reason(OBSERVABLE_NULL_ARM) == NO_OBSERVABLE_NULL_DIRECTION
    assert without_null.arm_skip_reason(MAIN_ARM) == ""
    assert geometry.observable_null_dim > 0


def test_zero_response_columns_receive_no_noise(context):
    _sk, native, j_phi, geometry = _geometry(context)
    zeroed = geometry.R0.clone()
    zeroed[:, 0] = 0.0
    silent = build_response_geometry(
        type(native)(native.state, native.J_x, native.J_p, zeroed, native.dim_constraint_ids),
        j_phi, eps=EPS, rcond=RCOND,
    )

    assert silent.n_zero_observable_columns >= 1
    R_eta, _diag = perturbed_response(silent, MAIN_ARM, ETA, seed=5, eps=EPS, rcond=RCOND)
    assert torch.allclose(R_eta[:, 0], zeroed[:, 0], atol=1e-15)


def test_an_arm_skips_a_response_when_every_driver_column_is_zero(context):
    _sk, native, j_phi, _unused_geometry = _geometry(context)
    zeroed = torch.zeros_like(native.R)
    silent = build_response_geometry(
        type(native)(native.state, native.J_x, native.J_p, zeroed,
                     native.dim_constraint_ids),
        j_phi, eps=EPS, rcond=RCOND, consistency_tol=float("inf"),
    )

    assert silent.arm_skip_reason(MAIN_ARM) == NO_NONZERO_OBSERVABLE_RESPONSE
    assert silent.arm_skip_reason(OBSERVABLE_NULL_ARM) == NO_NONZERO_RAW_RESPONSE


# ---------------------------------------------------------------------------
# Stratification key for the auxiliary arm
# ---------------------------------------------------------------------------

def test_relation_visibility_separates_pure_gauge_from_relation_visible(context):
    sketch, native, j_phi, geometry = _geometry(context)
    rels = [c for c in sketch.constraints.values() if _is_relational(c)]
    J_R, _n = stack_relation_jacobians_at(rels, native.state, context)

    seen = relation_visibility(J_R, geometry.G, eps=EPS, rcond=RCOND)
    assert seen["relation_null_overlap_rank"] <= geometry.observable_null_dim
    assert seen["relation_null_overlap_norm"] >= 0.0

    blind = relation_visibility(None, geometry.G, eps=EPS, rcond=RCOND)
    assert blind == dict(relation_null_overlap_norm=0.0, relation_null_overlap_rank=0)
