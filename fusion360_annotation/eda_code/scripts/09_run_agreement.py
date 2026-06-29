from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import Counter
from itertools import combinations
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from fusion360_annotation.eda_code.bundle_loader import build_bundle
from src.eval.manifest import discover_manifest
from src.eval.progress import progress
from src.metrics.edit_response_distance import _is_dimensional, _ref_strings, constraint_signature  # noqa: E402
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
POSITIONAL = {"Distance", "DistanceX", "DistanceY", "Angle", "Length"}


def ent_of(ref: str) -> str:
    return ref.split(".", 1)[0]


def dim_sigs(sketch, shared_eids):
    out = set()
    for c in sketch.constraints.values():
        if not _is_dimensional(c) or c.constraint_type.name not in POSITIONAL:
            continue
        refs = _ref_strings(c)
        if refs and all(ent_of(r) in shared_eids for r in refs):
            out.add((c.constraint_type.name, tuple(sorted(refs))))
    return out


def all_sigs(sketch):
    return [constraint_signature(c) for c in sketch.constraints.values()]


def f1(a: list, b: list) -> float:
    ca, cb = Counter(a), Counter(b)
    tp = sum((ca & cb).values())
    if not a or not b:
        return 0.0
    p, r = tp / len(a), tp / len(b)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "out" / "agreement.csv"))
    args = ap.parse_args()

    manifest, _stats = discover_manifest()
    items = [
        (batch, base, files)
        for batch in sorted(manifest, key=int)
        for base, files in manifest[batch].items()
    ]
    if args.limit is not None:
        items = items[: args.limit]
    ctx = SketchContext()
    converter = SketchFreecadConverter(ctx, verbose=False)
    rows = []
    for batch, base, files in progress(items, desc="agreement", total=len(items)):
        try:
            bundle = build_bundle(batch, base, files, converter)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {batch}/{base}: {str(exc)[:80]}")
            continue
        sketches = {n: bundle.sketches[n] for n in sorted(files)}
        shared = set.intersection(*(set(s.entities) for s in sketches.values()))
        sig_sets = {n: dim_sigs(s, shared) for n, s in sketches.items()}
        tok = {n: all_sigs(s) for n, s in sketches.items()}
        for a, b in combinations(sorted(sketches), 2):
            union = sig_sets[a] | sig_sets[b]
            jac = (len(sig_sets[a] & sig_sets[b]) / len(union)) if union else None
            rows.append(
                dict(
                    batch=batch,
                    filename=base,
                    a=a,
                    b=b,
                    jaccard=jac,
                    token_f1=f1(tok[a], tok[b]),
                    n_dim_a=len(sig_sets[a]),
                    n_dim_b=len(sig_sets[b]),
                )
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "filename", "a", "b", "jaccard", "token_f1", "n_dim_a", "n_dim_b"])
        writer.writeheader()
        writer.writerows(rows)
    jacc = [r["jaccard"] for r in rows if r["jaccard"] is not None]
    f1s = [r["token_f1"] for r in rows]
    print(f"pairs={len(rows)}  jaccard mean={st.fmean(jacc):.3f} median={st.median(jacc):.3f}  token-F1 mean={st.fmean(f1s):.3f} median={st.median(f1s):.3f}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
