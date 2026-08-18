"""A perturbed response must stay an exact solution of the linearised system, and eta must
mean the same observable-space size for every driver column."""
from __future__ import annotations

from collections import OrderedDict

import pytest
import sketchgraphs.data as datalib
import torch

from rebuttal_exp.response_perturbation import (
    INCONSISTENT_LINEAR_SYSTEM,
    NOT_UNDERCONSTRAINED,
    NO_NONZERO_OBSERVABLE_RESPONSE,
    build_response_geometry,
    cutoff_null_space,
    perturbed_response,
)
from rebuttal_exp.tests.fixtures import arc, circle, constraint, line
from src.metrics.edit_response_distance import (
    compute_native_response,
    pairwise_distance_observable,
)
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


def _geometry_of(context: SketchContext, sketch: datalib.Sketch):
    native = compute_native_response(sketch, context, eps=EPS)
    j_phi = pairwise_distance_observable(sketch, base_eids=None).to(dtype=native.R.dtype)
    return sketch, native, j_phi, build_response_geometry(native, j_phi, eps=EPS, rcond=RCOND)


def _geometry(context: SketchContext):
    return _geometry_of(context, _sketch(context))


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


def test_observable_gain_max_is_the_supremum_of_the_guarded_quantity(context):
    """No unit direction in ker(J_x) can beat sigma_max(J_phi N), which the guard compares to."""
    _sk, _native, j_phi, geometry = _geometry(context)
    generator = torch.Generator().manual_seed(0)
    coefficients = torch.randn(geometry.N.shape[1], 64, generator=generator, dtype=torch.float64)
    directions = geometry.N @ (coefficients / torch.linalg.norm(coefficients, dim=0))

    gains = torch.linalg.norm(j_phi @ directions, dim=0)

    assert float(gains.max()) <= geometry.observable_gain_max * (1 + 1e-9)


# ---------------------------------------------------------------------------
# Admissibility and the eta guarantee
# ---------------------------------------------------------------------------

def test_perturbed_response_still_solves_the_linearised_system(context):
    _sk, native, _j_phi, geometry = _geometry(context)
    R_eta, diagnostics = perturbed_response(geometry, ETA, seed=7)

    baseline = torch.linalg.norm(native.J_x @ native.R + native.J_p)
    perturbed = torch.linalg.norm(native.J_x @ R_eta + native.J_p)
    assert float(perturbed) == pytest.approx(float(baseline), abs=1e-9)
    assert diagnostics["constraint_residual"] < 1e-9
    assert diagnostics["nullspace_residual"] < 1e-9
    assert not torch.allclose(R_eta, native.R)


def test_each_column_moves_by_eta_times_its_observable_norm(context):
    """The whole point of the scheme: eta is an observable-space size, not a raw one."""
    _sk, _native, j_phi, geometry = _geometry(context)
    R_eta, diagnostics = perturbed_response(geometry, ETA, seed=3)

    assert diagnostics["n_columns_guarded"] == 0
    increment = j_phi @ (R_eta - geometry.R0)
    for column in range(int(geometry.R0.shape[1])):
        expected = ETA * float(geometry.observable_column_norms[column])
        assert float(torch.linalg.norm(increment[:, column])) == pytest.approx(expected, rel=1e-8)


def test_eta_zero_returns_the_minimum_norm_response_itself(context):
    _sk, _native, _j_phi, geometry = _geometry(context)
    R_eta, diagnostics = perturbed_response(geometry, 0.0, seed=1)

    assert R_eta is geometry.R0
    assert diagnostics["raw_perturbation_ratio"] == 0.0
    assert diagnostics["constraint_residual"] == pytest.approx(geometry.consistency_residual)
    assert diagnostics["nullspace_residual"] == 0.0
    assert diagnostics["n_columns_guarded"] == 0


def test_samples_are_reproducible_and_seed_dependent(context):
    _sk, _native, _j_phi, geometry = _geometry(context)
    first, _ = perturbed_response(geometry, ETA, seed=11)
    again, _ = perturbed_response(geometry, ETA, seed=11)
    other, _ = perturbed_response(geometry, ETA, seed=12)

    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


# ---------------------------------------------------------------------------
# The draw guard
# ---------------------------------------------------------------------------

def test_a_draw_below_the_gain_floor_leaves_its_column_untouched(context):
    """Below the floor the quotient eta ||M_j|| / ||J_phi n|| is roundoff, not a direction."""
    _sk, _native, _j_phi, geometry = _geometry(context)
    R_eta, diagnostics = perturbed_response(geometry, ETA, seed=3, gain_floor=1e9)

    assert torch.equal(R_eta, geometry.R0)
    assert diagnostics["n_columns_guarded"] == int(geometry.R0.shape[1])
    assert diagnostics["raw_perturbation_ratio"] == 0.0
    assert diagnostics["min_observable_gain"] < 1e9


def test_the_floor_governs_the_response_verdict_and_the_draw_alike(context):
    """One threshold, two scales: raising it must move both tests together."""
    sketch = _sketch(context)
    native = compute_native_response(sketch, context, eps=EPS)
    j_phi = pairwise_distance_observable(sketch, base_eids=None).to(dtype=native.R.dtype)

    strict = build_response_geometry(native, j_phi, eps=EPS, rcond=RCOND, gain_floor=1e9)
    R_eta, diagnostics = perturbed_response(strict, ETA, seed=3)

    assert strict.exactly_invariant, "no direction can pass a floor above the gain supremum"
    assert torch.equal(R_eta, strict.R0)
    assert diagnostics["n_columns_guarded"] == int(strict.R0.shape[1])


def test_an_observable_invisible_null_space_is_exactly_invariant(context):
    """ker(J_x) = rigid motion + the Line's parameterisation gauge, both invisible to phi.

    phi is a complete rigid invariant (pairwise landmark distances plus radii) and the Line's
    (pnt, dir, startParam, endParam) carries two redundancies that leave both endpoints
    fixed, so J_phi N vanishes and J_phi R cannot depend on the response choice.
    """
    sketch = datalib.Sketch(
        entities=OrderedDict([
            ("0", line("0", 0.0, 0.0, 2.0, 0.0)),
            ("1", circle("1", 0.0, 0.0, 0.5)),
        ]),
        constraints=OrderedDict([
            ("c0", constraint(context, "Coincident", ("0.start", "1.center"))),
            ("c1", constraint(context, "Radius", ("1",), value=0.5)),
            ("c2", constraint(context, "Distance", ("0.start", "0.end"), value=2.0)),
        ]),
    )
    _sk, _native, j_phi, geometry = _geometry_of(context, sketch)

    assert geometry.raw_null_dim > 0
    assert geometry.observable_free_rank == 0
    assert geometry.exactly_invariant
    R_eta, diagnostics = perturbed_response(geometry, 1.0, seed=4)
    assert torch.equal(R_eta, geometry.R0)
    assert diagnostics["n_columns_guarded"] == int(geometry.R0.shape[1])
    assert torch.allclose(j_phi @ R_eta, j_phi @ geometry.R0)


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


def test_inconsistent_linear_system_is_a_separate_category(context):
    _sk, native, j_phi, _geo = _geometry(context)
    broken = type(native)(native.state, native.J_x, native.J_p,
                          torch.zeros_like(native.R), native.dim_constraint_ids)

    geometry = build_response_geometry(broken, j_phi, eps=EPS, rcond=RCOND)
    assert geometry.consistency_residual > 1e-6
    assert geometry.skip_reason == INCONSISTENT_LINEAR_SYSTEM


def test_zero_response_columns_receive_no_noise(context):
    _sk, native, j_phi, geometry = _geometry(context)
    zeroed = geometry.R0.clone()
    zeroed[:, 0] = 0.0
    silent = build_response_geometry(
        type(native)(native.state, native.J_x, native.J_p, zeroed, native.dim_constraint_ids),
        j_phi, eps=EPS, rcond=RCOND, consistency_tol=float("inf"),
    )

    assert silent.n_zero_observable_columns >= 1
    R_eta, diagnostics = perturbed_response(silent, ETA, seed=5)
    assert torch.allclose(R_eta[:, 0], zeroed[:, 0], atol=1e-15)
    assert diagnostics["n_columns_guarded"] == 0, "a zero column is not a guarded draw"


def test_a_response_with_no_observable_motion_at_all_is_skipped(context):
    _sk, native, j_phi, _unused = _geometry(context)
    zeroed = torch.zeros_like(native.R)
    silent = build_response_geometry(
        type(native)(native.state, native.J_x, native.J_p, zeroed,
                     native.dim_constraint_ids),
        j_phi, eps=EPS, rcond=RCOND, consistency_tol=float("inf"),
    )

    assert silent.skip_reason == NO_NONZERO_OBSERVABLE_RESPONSE
