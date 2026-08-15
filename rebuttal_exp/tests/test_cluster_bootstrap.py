"""The bootstrap must resample whole sketches and must be reproducible from its seed."""
from __future__ import annotations

import numpy as np

from rebuttal_exp.common import (
    cluster_index_groups,
    percentile_ci,
    sketch_cluster_bootstrap,
)

CLUSTERS = ["s0", "s0", "s0", "s1", "s1", "s2"]


def test_groups_keep_every_row_of_a_sketch_together():
    groups = cluster_index_groups(CLUSTERS)
    assert [sorted(g.tolist()) for g in groups] == [[0, 1, 2], [3, 4], [5]]


def test_resampled_rows_arrive_as_whole_sketches():
    seen = []
    sketch_cluster_bootstrap(CLUSTERS, lambda rows: seen.append(rows.tolist()) or 0.0,
                             n_replicates=25, seed=7)
    memberships = [{0, 1, 2}, {3, 4}, {5}]
    assert len(seen) == 25
    for rows in seen:
        counts = {row: rows.count(row) for row in set(rows)}
        multiplicities = []
        for members in memberships:
            drawn = {counts.get(row, 0) for row in members}
            assert len(drawn) == 1, "a sketch was split across the resample"
            multiplicities.append(drawn.pop())
        assert sum(multiplicities) == len(memberships), "one draw per original sketch"


def test_same_seed_reproduces_the_interval():
    statistic = lambda rows: float(np.mean(rows))  # noqa: E731
    first = sketch_cluster_bootstrap(CLUSTERS, statistic, n_replicates=200, seed=3)
    second = sketch_cluster_bootstrap(CLUSTERS, statistic, n_replicates=200, seed=3)
    other = sketch_cluster_bootstrap(CLUSTERS, statistic, n_replicates=200, seed=4)

    assert np.array_equal(first, second)
    assert percentile_ci(first) == percentile_ci(second)
    assert not np.array_equal(first, other)


def test_undefined_replicates_are_reported_not_silently_dropped():
    replicates = sketch_cluster_bootstrap(
        CLUSTERS, lambda rows: float("nan") if len(set(rows.tolist())) < 6 else 1.0,
        n_replicates=100, seed=1,
    )
    _low, _high, n_valid = percentile_ci(replicates)
    assert 0 < n_valid < 100
    assert np.isnan(replicates).sum() == 100 - n_valid


def test_statistic_exceptions_become_invalid_replicates():
    def explode(rows):
        raise ValueError("undefined here")

    replicates = sketch_cluster_bootstrap(CLUSTERS, explode, n_replicates=10, seed=0)
    assert np.isnan(replicates).all()
    assert percentile_ci(replicates)[2] == 0
