"""Tie handling is the whole experiment, so it is tested rather than trusted.

The synthetic samples cover K = 2 and K = 3, the S_place direction flip, ambiguous rank sums
that must stay unresolved, and the degenerate all-tied case that is coherent and ambiguous at
the same time. The last test is a regression gate: independent(all) on the real motion_eval.csv
must reproduce paper Table 1, otherwise the loader disagrees with Sec. 5.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rebuttal_exp.shared_reference import (
    ANGLES,
    METRICS,
    PLACE,
    PRIMARY_ATOL,
    TABLE1_REFERENCE,
    Sample,
    analyse_sample,
    average_ranks,
    best_reference_set,
    check_table1_reproduction,
    envelope,
    exp4_table,
    joint_selection,
    load_samples,
    rank_sums,
    rank_table,
    regret,
)

MOTION_EVAL = (
    Path(__file__).resolve().parents[2] / "fusion360_annotation_" / "eda_code" / "out" / "motion_eval.csv"
)


def make_sample(values: dict[str, dict[str, float]], variant: str = "heldout") -> Sample:
    refs = tuple(sorted(next(iter(values.values()))))
    return Sample(
        rid=f"0:s.FCStd:{'+'.join(refs)}__q=Q",
        batch="0",
        filename="s.FCStd",
        variant=variant,
        refs=refs,
        values=values,
    )


def test_best_set_absorbs_ties_inside_the_tolerance_only():
    scores = {"A": 0.0, "B": 1e-8, "C": 1e-3}
    assert best_reference_set(scores, "theta_full", PRIMARY_ATOL) == frozenset({"A", "B"})
    assert best_reference_set(scores, "theta_full", 0.0) == frozenset({"A"})
    assert best_reference_set(scores, "theta_full", 1e-2) == frozenset({"A", "B", "C"})


def test_place_direction_is_inverted():
    scores = {"A": 0.2, "B": 0.9}
    assert best_reference_set(scores, PLACE, PRIMARY_ATOL) == frozenset({"B"})
    assert best_reference_set(scores, "theta_rel", PRIMARY_ATOL) == frozenset({"A"})


def test_average_ranks_share_positions_between_tied_references():
    ranks = average_ranks({"A": 0.0, "B": 1e-9, "C": 0.5}, "theta_full", PRIMARY_ATOL)
    assert ranks == {"A": 1.5, "B": 1.5, "C": 3.0}
    strict = average_ranks({"A": 0.0, "B": 1e-9, "C": 0.5}, "theta_full", 0.0)
    assert strict == {"A": 1.0, "B": 2.0, "C": 3.0}
    place = average_ranks({"A": 0.1, "B": 0.9}, PLACE, PRIMARY_ATOL)
    assert place == {"A": 2.0, "B": 1.0}


def test_coherent_sample_has_zero_regret_and_a_unique_joint_reference():
    sample = make_sample(
        {
            "theta_full": {"A": 0.1, "B": 0.4},
            "theta_rel": {"A": 0.0, "B": 0.3},
            "theta_leak": {"A": 0.2, "B": 0.5},
            PLACE: {"A": 0.9, "B": 0.4},
        }
    )
    record = analyse_sample(sample, PRIMARY_ATOL)
    assert record["coherent"] is True
    assert record["k_joint"] == "A"
    assert record["ambiguous"] is False
    assert all(v == 0.0 for v in record["regret"].values())
    assert all(not flag for flag in record["incompatible"].values())


def test_two_two_split_leaves_the_rank_sum_ambiguous_and_is_not_broken():
    sample = make_sample(
        {
            "theta_full": {"A": 0.1, "B": 0.4},
            "theta_rel": {"A": 0.1, "B": 0.4},
            "theta_leak": {"A": 0.5, "B": 0.2},
            PLACE: {"A": 0.3, "B": 0.8},
        }
    )
    record = analyse_sample(sample, PRIMARY_ATOL)
    assert record["ambiguous"] is True
    assert record["k_joint"] is None
    assert record["shared"] is None and record["regret"] is None
    assert record["rank_optimal"] == ("A", "B")
    assert record["coherent"] is False
    assert record["incompatible"][("theta_rel", PLACE)] is True
    # the envelope must bracket both admissible resolutions, per metric
    assert record["envelope"]["theta_full"] == (0.4, 0.1)
    assert record["envelope"][PLACE] == (0.3, 0.8)


def test_all_metrics_tied_is_coherent_and_ambiguous():
    sample = make_sample({m: {"A": 0.25, "B": 0.25 + 1e-9} for m in METRICS})
    record = analyse_sample(sample, PRIMARY_ATOL)
    assert record["coherent"] is True
    assert record["ambiguous"] is True
    assert set(record["rank_optimal"]) == {"A", "B"}


def test_three_references_rank_and_regret():
    sample = make_sample(
        {
            "theta_full": {"A": 0.1, "B": 0.2, "C": 0.6},
            "theta_rel": {"A": 0.3, "B": 0.1, "C": 0.7},
            "theta_leak": {"A": 0.2, "B": 0.1, "C": 0.9},
            PLACE: {"A": 0.5, "B": 0.7, "C": 0.1},
        }
    )
    ranks = rank_table(sample, PRIMARY_ATOL)
    sums = rank_sums(ranks, sample.refs)
    assert sums == {"A": 7.0, "B": 5.0, "C": 12.0}
    k_joint, winners = joint_selection(sums)
    assert (k_joint, winners) == ("B", ("B",))
    record = analyse_sample(sample, PRIMARY_ATOL)
    assert record["regret"]["theta_full"] == pytest.approx(0.1)
    assert record["regret"][PLACE] == pytest.approx(0.0)
    assert record["regret"]["theta_rel"] == pytest.approx(0.0)


def test_regret_is_never_negative_and_matches_the_direction():
    sample = make_sample(
        {
            "theta_full": {"A": 0.1, "B": 0.4},
            "theta_rel": {"A": 0.1, "B": 0.4},
            "theta_leak": {"A": 0.1, "B": 0.4},
            PLACE: {"A": 0.2, "B": 0.9},
        }
    )
    for metric in ANGLES:
        assert regret(sample, metric, "B") == pytest.approx(0.3)
        assert regret(sample, metric, "A") == 0.0
    assert regret(sample, PLACE, "A") == pytest.approx(0.7)
    assert regret(sample, PLACE, "B") == 0.0


def test_envelope_brackets_every_admissible_choice():
    sample = make_sample(
        {
            "theta_full": {"A": 0.1, "B": 0.4, "C": 0.9},
            "theta_rel": {"A": 0.1, "B": 0.4, "C": 0.9},
            "theta_leak": {"A": 0.1, "B": 0.4, "C": 0.9},
            PLACE: {"A": 0.1, "B": 0.4, "C": 0.9},
        }
    )
    worst, best = envelope(sample, "theta_full", ("A", "B"))
    assert (worst, best) == (0.4, 0.1)
    worst, best = envelope(sample, PLACE, ("A", "B"))
    assert (worst, best) == (0.1, 0.4)


def test_loader_accepts_the_legacy_asgn_sim_column(tmp_path: Path):
    csv_path = tmp_path / "motion_eval.csv"
    csv_path.write_text(
        "batch,filename,rid,variant,gt,theta_full,theta_rel,theta_leak,asgn_sim\n"
        "0,s.FCStd,r1,heldout,A,0.1,0.2,0.3,0.9\n"
        "0,s.FCStd,r1,heldout,B,0.4,0.5,0.6,0.2\n"
        "0,s.FCStd,r2,heldout,A,0.1,0.2,0.3,\n"  # incomplete row, dropped
    )
    samples, counts, column = load_samples(csv_path)
    assert column == "asgn_sim"
    assert counts["rows_with_incomplete_metrics"] == 1
    assert counts["groups_dropped_fewer_than_two_refs"] == 1
    assert len(samples) == 1
    assert samples[0].values[PLACE] == {"A": 0.9, "B": 0.2}


@pytest.mark.skipif(not MOTION_EVAL.exists(), reason="motion_eval.csv not available")
def test_independent_aggregation_reproduces_paper_table1():
    samples, counts, _column = load_samples(MOTION_EVAL)
    assert counts["groups_dropped_fewer_than_two_refs"] == 0
    rows = exp4_table([analyse_sample(s, PRIMARY_ATOL) for s in samples], PRIMARY_ATOL)
    assert check_table1_reproduction(rows) == []
    lookup = {(r["variant"], r["metric"]): r for r in rows}
    for variant, expected in TABLE1_REFERENCE.items():
        for metric, value in zip(METRICS, expected[:4]):
            assert lookup[(variant, metric)]["independent_all"] == pytest.approx(value, abs=5e-4)
