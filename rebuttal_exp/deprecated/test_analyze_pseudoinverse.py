"""Ordering stability must survive ties, and the eta = 0 gate must not be fooled by them."""
from __future__ import annotations

import pandas as pd
import pytest

from rebuttal_exp.analyze_pseudoinverse import (
    ORDERING_TOL,
    flip_rate,
    identity_gate_failed,
    identity_report,
    ordering_violations,
    paired_samples,
    variant_ordering,
)

VARIANTS = ["heldout", "ours", "base"]


def _medians(theta_full: dict, placement: dict | None = None) -> dict:
    return dict(
        theta_full=theta_full,
        theta_rel=theta_full,
        theta_leak=theta_full,
        placement_sim=placement if placement is not None else theta_full,
    )


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_sub_tolerance_reordering_is_a_tie_not_a_violation():
    """theta_leak medians are exactly zero for many variants; noise must not score there."""
    native = _medians({"heldout": 0.0, "ours": 0.0, "base": 0.0})
    jittered = _medians({"heldout": 1e-8, "ours": -2e-8, "base": 3e-9})

    inverted, comparable = ordering_violations(native, jittered, "theta_leak", VARIANTS)
    assert (inverted, comparable) == (0, 0)


def test_a_genuine_reversal_is_counted():
    native = _medians({"heldout": 0.10, "ours": 0.20, "base": 0.60})
    swapped = _medians({"heldout": 0.30, "ours": 0.20, "base": 0.60})

    inverted, comparable = ordering_violations(native, swapped, "theta_full", VARIANTS)
    assert comparable == 3
    assert inverted == 1


def test_only_pairs_the_baseline_resolves_are_comparable():
    native = _medians({"heldout": 0.10, "ours": 0.10 + ORDERING_TOL / 2, "base": 0.60})
    moved = _medians({"heldout": 0.20, "ours": 0.10, "base": 0.60})

    inverted, comparable = ordering_violations(native, moved, "theta_full", VARIANTS)
    assert comparable == 2, "the two near-equal variants are not a resolvable pair"
    assert inverted == 0


def test_variants_missing_from_one_side_are_skipped():
    native = _medians({"heldout": 0.10, "ours": 0.50, "base": 0.60})
    partial = _medians({"heldout": 0.10, "base": 0.60})

    _inverted, comparable = ordering_violations(native, partial, "theta_full", VARIANTS)
    assert comparable == 1


def test_similarity_metrics_are_ordered_the_other_way():
    medians = _medians({}, placement={"heldout": 0.9, "ours": 0.5, "base": 0.1})
    assert variant_ordering(medians, "placement_sim") == ["heldout", "ours", "base"]

    angles = _medians({"heldout": 0.1, "ours": 0.5, "base": 0.9})
    assert variant_ordering(angles, "theta_full") == ["heldout", "ours", "base"]


def test_tied_variants_are_ordered_deterministically_by_name():
    tied = _medians({"heldout": 0.0, "ours": 1e-9, "base": -1e-9})
    assert variant_ordering(tied, "theta_full") == ["base", "heldout", "ours"]


def test_flip_rate_ignores_ties_on_either_side():
    native = {("r0", "a"): {"theta_full": 0.10}, ("r0", "b"): {"theta_full": 0.10}}
    other = {("r0", "a"): {"theta_full": 0.90}, ("r0", "b"): {"theta_full": 0.10}}
    rate, comparisons = flip_rate(native, other, "theta_full")
    assert comparisons == 0 and rate != rate  # NaN: nothing was comparable

    native = {("r0", "a"): {"theta_full": 0.10}, ("r0", "b"): {"theta_full": 0.50}}
    other = {("r0", "a"): {"theta_full": 0.60}, ("r0", "b"): {"theta_full": 0.20}}
    rate, comparisons = flip_rate(native, other, "theta_full")
    assert (rate, comparisons) == (1.0, 1)


# ---------------------------------------------------------------------------
# The eta = 0 gate
# ---------------------------------------------------------------------------

RANKS = dict(k_full_pred=[2, 1], k_rel_pred=[3, 3], k_response_pred=[2, 2], n_dim_pred=[4, 4])


def _frames(perturbed: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = dict(rid=["r0", "r1"], variant=["ours", "ours"], gt=["A", "B"])
    native = pd.DataFrame(dict(
        **keys, sketch_id=["s", "s"], pred_label=["ours", "ours"], status=["ok", "ok"],
        theta_full=[0.4, 0.0], theta_rel=[0.2, 0.0],
        theta_leak=[0.1, 0.0], placement_sim=[0.8, 1.0], **RANKS,
    ))
    samples = pd.DataFrame(dict(
        **keys, sketch_id=["s", "s"], arm=["main", "main"], eta=[0.0, 0.0], sample=[0, 0],
        **RANKS, **perturbed,
    ))
    return native, samples


def test_gate_passes_when_the_sampling_path_reproduces_the_paper_path():
    native, samples = _frames(dict(
        theta_full=[0.4, 0.0], theta_rel=[0.2, 0.0],
        theta_leak=[0.1, 0.0], placement_sim=[0.8, 1.0],
    ))
    gate = identity_report(native, samples)
    assert not identity_gate_failed(gate)
    assert int(gate["n_mismatch_resolved"].sum()) == 0


def test_gate_tolerates_the_padded_angle_noise_floor_only_near_zero():
    """sqrt(1 - cos2/max) lifts machine eps to ~1e-8 exactly where the angle is ~0."""
    native, samples = _frames(dict(
        theta_full=[0.4, 3e-8], theta_rel=[0.2, 0.0],
        theta_leak=[0.1, 0.0], placement_sim=[0.8, 1.0],
    ))
    assert not identity_gate_failed(identity_report(native, samples))

    native, samples = _frames(dict(
        theta_full=[0.4 + 3e-8, 0.0], theta_rel=[0.2, 0.0],
        theta_leak=[0.1, 0.0], placement_sim=[0.8, 1.0],
    ))
    gate = identity_report(native, samples)
    assert identity_gate_failed(gate), "a resolved value must match to machine precision"


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def test_each_sample_carries_its_own_records_baseline():
    native, samples = _frames(dict(
        theta_full=[0.9, 0.5], theta_rel=[0.2, 0.0],
        theta_leak=[0.1, 0.0], placement_sim=[0.3, 0.4],
    ))
    paired = paired_samples(native, samples)

    assert list(paired["theta_full_0"]) == [0.4, 0.0]
    assert list(paired["theta_full"]) == [0.9, 0.5]
    assert list(paired["pred_label"]) == ["ours", "ours"]


def test_records_without_a_baseline_are_dropped():
    native, samples = _frames(dict(
        theta_full=[0.9, 0.5], theta_rel=[0.2, 0.0],
        theta_leak=[0.1, 0.0], placement_sim=[0.3, 0.4],
    ))
    orphan = samples.copy()
    orphan["rid"] = ["unknown", "r1"]

    assert len(paired_samples(native, orphan)) == 1


if __name__ == "__main__":
    pytest.main([__file__])
