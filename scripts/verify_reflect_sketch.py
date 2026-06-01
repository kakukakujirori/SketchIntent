"""Validate sketch_utils.reflect_sketch on real data (research plan §16.9.1).

Reflection must be a distance-preserving involution: every pairwise distance
between geometric landmarks (points, line endpoints, arc/circle centers, arc
start/mid/end) is preserved, and applying it twice is the identity.  This is the
correctness guarantee that lets a constraint set generated on a reflected view
be scored on the original sketch (flip-augmentation).

Note the circle "rightmost perimeter" point is intentionally NOT used as a
landmark: it is frame-dependent (maps to the leftmost point under reflection) and
so is not reflection-equivariant — only the center is.

Run:
    conda run -n freecad python scripts/verify_reflect_sketch.py
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import torch
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data._entity import EntityType
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.flip_augment import (
    reflect_primitive_value_tokens, _X_COORD_ATTR_IDS, _Y_COORD_ATTR_IDS,
)
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import normalize_sketch, reflect_sketch

DATA = "data_processed/sg_filtered_unique_valid.npy"
MAX_SKETCHES = 600


def landmarks(sketch) -> np.ndarray:
    pts = []
    for e in sketch.entities.values():
        t = e.type
        if t == EntityType.Point:
            pts.append((e.x, e.y))
        elif t == EntityType.Line:
            pts += [tuple(map(float, e.start_point)), tuple(map(float, e.end_point))]
        elif t == EntityType.Circle:
            pts.append((e.xCenter, e.yCenter))  # center only (equivariant)
        elif t == EntityType.Arc:
            pts += [tuple(map(float, p)) for p in (e.start_point, e.mid_point, e.end_point, e.center_point)]
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def pdist_sorted(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.zeros(0)
    return np.sort(np.array([math.dist(a, b) for a, b in itertools.combinations(pts, 2)]))


def float_fields(sketch) -> np.ndarray:
    return np.array([getattr(e, f) for e in sketch.entities.values()
                     for f in getattr(e, "float_ids", [])], dtype=float)


def main() -> None:
    seqs = flat_array.load_dictionary_flat(DATA)["sequences"]
    n_check = n_arc = fails = 0
    max_dist_err = max_idemp_err = 0.0
    for i in range(min(4000, len(seqs))):
        try:
            sk = datalib.sketch_from_sequence(seqs[i])
        except Exception:
            continue
        base = pdist_sorted(landmarks(sk))
        if base.size == 0:
            continue
        has_arc = any(e.type == EntityType.Arc for e in sk.entities.values())
        for axis in ("x", "y"):
            sk1 = copy.deepcopy(sk)
            reflect_sketch(sk1, axis)
            pd1 = pdist_sorted(landmarks(sk1))
            derr = float(np.max(np.abs(pd1 - base))) if pd1.shape == base.shape and pd1.size else float("inf")
            sk2 = copy.deepcopy(sk1)
            reflect_sketch(sk2, axis)
            f0, f2 = float_fields(sk), float_fields(sk2)
            ierr = float(np.max(np.abs(f0 - f2))) if f0.shape == f2.shape and f0.size else float("inf")
            max_dist_err = max(max_dist_err, derr)
            max_idemp_err = max(max_idemp_err, ierr)
            fails += int(derr > 1e-6 or ierr > 1e-6)
        n_check += 1
        n_arc += int(has_arc)
        if n_check >= MAX_SKETCHES:
            break

    print(f"checked {n_check} sketches ({n_arc} with arcs)")
    print(f"max pairwise-distance error under reflection: {max_dist_err:.3e}")
    print(f"max double-reflect identity error:            {max_idemp_err:.3e}")
    print("isometry/involution:", "PASS" if fails == 0 else f"FAIL ({fails} sketches)")
    token_level_consistency()


def token_level_consistency() -> None:
    """The token-space reflect_primitive_value_tokens (used in the RLOO training
    loop) must agree with tokenizing the geometrically-reflected sketch."""
    import numpy as np
    tok = SketchTokenizer(SketchContext())
    num_bins = SketchContext().num_quantization_bins
    seqs = flat_array.load_dictionary_flat(DATA)["sequences"]
    # Non-arc sketches: token-level and object-level reflection must be token-identical.
    # Arc sketches: object-level flips `clockwise`, swapping start/end LABELS on
    # re-tokenize, so tokens differ even though the geometry is the same reflected
    # arc (arcs are point-encoded; negating each point's X bin is exact).  So the
    # equality assertion is restricted to non-arc sketches.
    # token-flip negates the bin CENTER exactly (dequant(flip(q)) = -dequant(q));
    # object-level re-quantizes the negated continuous value with floor.  These
    # can disagree by AT MOST 1 bin (floor asymmetry, e.g. v=0 -> bin 32 vs
    # flip -> 31) = 1/num_bins ~ 0.016 of the coordinate range, far below the
    # training primitive_noise std 0.15.  So the correctness criterion is
    # "max bin discrepancy <= 1 on the non-arc coordinate tokens", not exact equality.
    n_noarc = n_arc = max_bindiff = 0
    for i in range(min(8000, len(seqs))):
        try:
            sk = datalib.sketch_from_sequence(seqs[i])
        except Exception:
            continue
        has_arc = any(e.type == EntityType.Arc for e in sk.entities.values())
        if has_arc:
            n_arc += 1
            continue
        normalize_sketch(sk)
        pt, _ = tok.tokenize_primitives(sk)
        p_val = torch.from_numpy(pt["p_val"]); p_attr = torch.from_numpy(pt["p_attr"])
        coord = np.isin(pt["p_attr"], _X_COORD_ATTR_IDS + _Y_COORD_ATTR_IDS)
        for axis in ("x", "y"):
            tok_ref = reflect_primitive_value_tokens(p_val, p_attr, axis, num_bins).numpy()
            skr = copy.deepcopy(sk); reflect_sketch(skr, axis)
            obj_ref = tok.tokenize_primitives(skr)[0]["p_val"]
            max_bindiff = max(max_bindiff, int(np.abs(tok_ref[coord] - obj_ref[coord]).max(initial=0)))
        n_noarc += 1
        if n_noarc >= 400:
            break
    print(f"token-level vs object-level (non-arc): checked {n_noarc}, "
          f"max coord bin discrepancy = {max_bindiff} bin(s) ->",
          "PASS (<=1 = sub-quantization, < noise)" if max_bindiff <= 1 else "FAIL")
    print(f"  (skipped {n_arc} arc sketches: start/end-label difference, geometrically equivalent)")


if __name__ == "__main__":
    main()
