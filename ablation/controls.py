"""Synthetic-control variants shared by evaluation scripts."""
from __future__ import annotations


def relation_dropout_variant(sketch, *, drop_rate: float, seed: int):
    """Drop a deterministic fraction of relational constraints."""
    import copy
    import random

    from src.metrics.edit_response_distance import _is_relational, constraint_signature

    rel_ids = sorted(
        [cid for cid, c in sketch.constraints.items() if _is_relational(c)],
        key=lambda cid: (constraint_signature(sketch.constraints[cid]), cid),
    )
    n = len(rel_ids)
    nd = max(0, min(n, int(round(n * drop_rate / 100.0))))
    if nd == 0:
        drop: set = set()
    elif nd == n:
        drop = set(rel_ids)
    else:
        drop = set(random.Random(seed).sample(rel_ids, nd))
    out = copy.deepcopy(sketch)
    out.constraints = {k: v for k, v in out.constraints.items() if k not in drop}
    return out, n, n - nd, nd

