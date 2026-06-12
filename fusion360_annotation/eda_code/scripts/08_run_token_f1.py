from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from fusion360_annotation.eda_code.bundle_loader import build_bundle
from fusion360_annotation.eda_code.config import BASE_POLICY
from fusion360_annotation.eda_code.controls import relation_dropout_variant
from fusion360_annotation.eda_code.file_bundle import (
    base_policy_pass,
    build_base_eids,
    dropout_variant,
    stable_seed,
)
from fusion360_annotation.eda_code.manifest import discover_manifest
from fusion360_annotation.eda_code.progress import progress
from fusion360_annotation.eda_code.runtime_warnings import (
    validate_model_outputs,
    warn_missing_model_roots,
)
from src.metrics.constraint_token_f1 import pair_f1  # noqa: E402
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", default=None)
    ap.add_argument("--dim-drop-rates", default="25,50,75,100")
    ap.add_argument("--rel-drop-rates", default="50,100")
    ap.add_argument("--geometry-tol", type=float, default=5e-1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "out" / "token_f1.csv"))
    args = ap.parse_args()

    dim_rates = [float(x) for x in args.dim_drop_rates.split(",") if x]
    rel_rates = [float(x) for x in args.rel_drop_rates.split(",") if x]
    manifest, _stats = discover_manifest()
    batches = args.batches or sorted(manifest, key=int)
    items = [
        (batch, base, files)
        for batch in batches
        for base, files in manifest.get(batch, {}).items()
    ]
    if args.limit is not None:
        items = items[: args.limit]
    warn_missing_model_roots(script="08_run_token_f1.py", batches=batches)
    validate_model_outputs(items, script="08_run_token_f1.py")
    ctx = SketchContext()
    converter = SketchFreecadConverter(ctx, verbose=False)

    rows = []
    for batch, base, files in progress(items, desc="token f1", total=len(items)):
        try:
            bundle = build_bundle(batch, base, files, converter, geometry_tol=args.geometry_tol)
        except Exception as exc:  # noqa: BLE001
            ann = ", ".join(f"{k}:{v}" for k, v in sorted(files.items()))
            print(f"  [skip] batch={batch} base={base} annotations=[{ann}]: {str(exc)[:100]}")
            continue
        build_base_eids(bundle.canonical_sketch, BASE_POLICY)
        names = tuple(sorted(files))
        model_ok = {
            label: base_policy_pass(bundle.audits[label], BASE_POLICY)[0]
            for label in bundle.loaded_model_labels()
        }

        for refs in combinations(names, 2):
            heldout = next(n for n in names if n not in refs)
            rid = f"{batch}:{base}:{'+'.join(refs)}__q={heldout}"
            ok = all(base_policy_pass(bundle.audits[n], BASE_POLICY)[0] for n in names if n in bundle.audits)
            if not ok:
                continue

            variants = [("heldout", bundle.sketches[heldout])]
            for rate in dim_rates:
                sd = stable_seed(batch, base, rid, "dim", rate)
                pred, *_ = dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=sd)
                variants.append((f"dim_drop{rate:g}", pred))
            for rate in rel_rates:
                sd = stable_seed(batch, base, rid, "rel", rate)
                pred, *_ = relation_dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=sd)
                variants.append((f"rel_drop{rate:g}", pred))
            for label in bundle.loaded_model_labels():
                if model_ok.get(label):
                    variants.append((label, bundle.sketches[label]))

            for g in refs:
                score = pair_f1(bundle.sketches[g], bundle.sketches[g])
                rows.append(dict(batch=batch, filename=base, rid=rid, variant="source_self", gt=g, token_f1=score["f1"], dim_f1=score["dim_f1"], rel_f1=score["rel_f1"]))
            for vname, vsk in variants:
                for g in refs:
                    score = pair_f1(vsk, bundle.sketches[g])
                    rows.append(dict(batch=batch, filename=base, rid=rid, variant=vname, gt=g, token_f1=score["f1"], dim_f1=score["dim_f1"], rel_f1=score["rel_f1"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "filename", "rid", "variant", "gt", "token_f1", "dim_f1", "rel_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
