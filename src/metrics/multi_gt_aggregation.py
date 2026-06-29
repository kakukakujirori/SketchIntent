"""
multi_gt_aggregation.py — multi-reference scoring for design-intent metrics.

Provides post-processing common to the setting where each sketch has 3 human
GTs:

  - nearest-GT scoring:    d(P, {G_j}) = min_j d(P, G_j)
  - human-human variation: median_{a<b} d(G_a, G_b)
  - normalized score:      d(P, {G_j}) / (variation + eps)

`d(., .)` may be any symmetric or asymmetric distance, independent of this
module.

Data layout helpers
-------------------
In the future we plan to consolidate annotations into a single folder with a
naming scheme like `xxxxxx_gt[0,2,3].FCStd`, so a loader for that layout is
provided. The current 3-directory layout (used in
notebooks/exp1_human_baseline.py) is also handled by `discover_per_annotator_dir`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def nearest_gt(distances: Sequence[float]) -> float:
    """min_j d(P, G_j). Raises if no distances supplied."""
    if not distances:
        raise ValueError("nearest_gt requires at least one distance")
    return min(distances)


def human_human_variation(distances: Sequence[float]) -> float:
    """median over all GT pairs. Returns 0.0 when fewer than 2 GTs."""
    if len(distances) < 1:
        return 0.0
    return float(np.median(distances))


def normalized_distance(
    pred_to_gts: Sequence[float],
    human_pairwise: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """
    \\hat d(P) = min_j d(P, G_j) / (median_{a<b} d(G_a, G_b) + eps)

    Interpretation:
      < 1   Pred is closer to some GT than typical human-human disagreement
      ~ 1   Pred sits at the human variation level
      > 1   Pred falls outside human agreement
    """
    return nearest_gt(pred_to_gts) / (human_human_variation(human_pairwise) + eps)


def pairwise_distances(
    items: Sequence,
    distance_fn: Callable[[object, object], float],
) -> list[float]:
    """All d(items[a], items[b]) for a < b. Convenience for human variation."""
    return [distance_fn(items[a], items[b]) for a, b in combinations(range(len(items)), 2)]


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GtBundle:
    """A sketch and its multiple human GTs at known absolute paths."""
    name: str                                  # e.g. "137627"
    canonical_path: str | None                 # path to the canonical .py, if any
    gt_paths: dict[str, str]                   # annotator -> .FCStd path


_GT_SUFFIX_RE = re.compile(r"^(?P<base>.+)_gt(?P<idx>-?\d+)\.FCStd$", re.IGNORECASE)


def discover_per_annotator_dirs(
    annotator_dirs: Mapping[str, str],
) -> list[GtBundle]:
    """
    Current layout (one directory per annotator):

        Eko/
            137627.FCStd
            137627.py
        Sagar/
            137627.FCStd
        Ryotaro/
            137627.FCStd

    Returns one GtBundle per filename present in *every* annotator dir.
    """
    file_sets = {
        ann: {
            f for f in os.listdir(d)
            if f.endswith(".FCStd")
            and not f.endswith("_mid.FCStd")
            and not f.startswith("_")
        }
        for ann, d in annotator_dirs.items()
    }
    common = sorted(set.intersection(*file_sets.values())) if file_sets else []

    bundles: list[GtBundle] = []
    for filename in common:
        base = os.path.splitext(filename)[0]
        canonical = None
        for d in annotator_dirs.values():
            cand = os.path.join(d, base + ".py")
            if os.path.exists(cand):
                canonical = cand
                break
        bundles.append(
            GtBundle(
                name=base,
                canonical_path=canonical,
                gt_paths={ann: os.path.join(annotator_dirs[ann], filename)
                          for ann in annotator_dirs},
            )
        )
    return bundles


def discover_combined_gt_dir(
    directory: str,
    annotator_index: Mapping[str, int] = {"Eko": 0, "Sagar": 2, "Ryotaro": 3},
) -> list[GtBundle]:
    """
    Planned layout (one directory holding all GT variants):

        directory/
            137627_gt0.FCStd     (Eko    = index 0)
            137627_gt2.FCStd     (Sagar  = index 2)
            137627_gt3.FCStd     (Ryotaro= index 3)
            137627.py            (canonical, optional)

    `annotator_index` maps annotator name -> the integer suffix used in the
    filename. Default mirrors the user's [0, 2, 3] convention.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)

    by_base: dict[str, dict[int, str]] = {}
    canonical_for: dict[str, str] = {}

    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if not os.path.isfile(path):
            continue
        m = _GT_SUFFIX_RE.match(entry)
        if m:
            base = m.group("base")
            idx = int(m.group("idx"))
            by_base.setdefault(base, {})[idx] = path
            continue
        if entry.endswith(".py"):
            canonical_for[os.path.splitext(entry)[0]] = path

    bundles: list[GtBundle] = []
    index_to_annotator = {v: k for k, v in annotator_index.items()}
    for base, idx_to_path in sorted(by_base.items()):
        gt_paths = {
            index_to_annotator[idx]: path
            for idx, path in idx_to_path.items()
            if idx in index_to_annotator
        }
        if not gt_paths:
            continue
        bundles.append(
            GtBundle(
                name=base,
                canonical_path=canonical_for.get(base),
                gt_paths=gt_paths,
            )
        )
    return bundles
