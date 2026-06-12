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
    dropout_variant,
    histcad_score,
    source_probes,
    stable_seed,
    validate_probes_on_source,
)
from fusion360_annotation.eda_code.manifest import discover_manifest
from fusion360_annotation.eda_code.progress import progress
from fusion360_annotation.eda_code.runtime_warnings import (
    validate_model_outputs,
    warn_missing_model_roots,
)
from src.metrics.histcad_2d import HistCADConfig  # noqa: E402
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
    ap.add_argument("--newton-max-iter", type=int, default=80)
    ap.add_argument("--exclude-pp-coincident", action="store_true")
    ap.add_argument("--out", default=str(HERE / "out" / "oes_eval.csv"))
    args = ap.parse_args()

    dim_rates = [float(x) for x in args.dim_drop_rates.split(",") if x]
    rel_rates = [float(x) for x in args.rel_drop_rates.split(",") if x]
    cfg = HistCADConfig(
        delta_basis="bbox_diag",
        delta_scale=0.01,
        delta_absolute=1.0,
        delta_signs=(-1.0, 1.0),
        newton_max_iter=args.newton_max_iter,
        newton_tol=1e-6,
        er_target_tol=1e-3,
        relation_tol_default=1e-3,
        relation_source="gt",
    )
    manifest, _stats = discover_manifest()
    batches = args.batches or sorted(manifest, key=int)
    items = [
        (batch, base, files)
        for batch in batches
        for base, files in manifest.get(batch, {}).items()
    ]
    if args.limit is not None:
        items = items[: args.limit]
    warn_missing_model_roots(script="07_run_oes_eval.py", batches=batches)
    validate_model_outputs(items, script="07_run_oes_eval.py")
    ctx = SketchContext()
    converter = SketchFreecadConverter(ctx, verbose=False)

    rows = []
    for batch, base, files in progress(items, desc="oes eval", total=len(items)):
        try:
            bundle = build_bundle(batch, base, files, converter, geometry_tol=args.geometry_tol)
        except Exception as exc:  # noqa: BLE001
            ann = ", ".join(f"{k}:{v}" for k, v in sorted(files.items()))
            print(f"  [skip] batch={batch} base={base} annotations=[{ann}]: {str(exc)[:100]}")
            continue
        names = tuple(sorted(files))
        if not all(base_policy_pass(bundle.audits[n], BASE_POLICY)[0] for n in names if n in bundle.audits):
            print(f"  [gate-skip] {batch}/{base}")
            continue

        src_info = {}
        for g in names:
            probes, _weights, rels, _ = source_probes(
                bundle.sketches[g],
                exclude_point_point_coincident=args.exclude_pp_coincident,
            )
            mask = validate_probes_on_source(bundle.sketches[g], probes, rels, ctx, converter, cfg)
            src_info[g] = {
                "probes": [p for p, m in zip(probes, mask) if m],
                "rels": rels,
                "n_before": len(probes),
                "n_after": int(sum(mask)) if mask else 0,
            }

        model_ok = {
            label: base_policy_pass(bundle.audits[label], BASE_POLICY)[0]
            for label in bundle.loaded_model_labels()
        }
        model_cache = {}

        def score(vname, pred_sk, g, rid):
            info = src_info[g]
            if not info["probes"]:
                return None
            er, cpcsr, oes, _np, _nr, n_mf, n_if, err = histcad_score(
                pred_sketch=pred_sk,
                gt_sketch=bundle.sketches[g],
                edit_probes=info["probes"],
                rels=info["rels"],
                context=ctx,
                converter=converter,
                config=cfg,
            )
            return dict(
                batch=batch,
                filename=base,
                rid=rid,
                variant=vname,
                gt=g,
                er=er,
                cpcsr=cpcsr,
                oes=oes,
                n_probes=info["n_after"],
                n_probes_raw=info["n_before"],
                er_modify_fail=n_mf,
                er_insert_fail=n_if,
                error=err,
            )

        for refs in combinations(names, 2):
            heldout = next(n for n in names if n not in refs)
            rid = f"{batch}:{base}:{'+'.join(refs)}__q={heldout}"

            for g in refs:
                row = score("source_self", bundle.sketches[g], g, rid)
                if row:
                    rows.append(row)

            variants = [("heldout", bundle.sketches[heldout])]
            for rate in dim_rates:
                sd = stable_seed(batch, base, rid, "dim", rate)
                pred, *_ = dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=sd)
                variants.append((f"dim_drop{rate:g}", pred))
            for rate in rel_rates:
                sd = stable_seed(batch, base, rid, "rel", rate)
                pred, *_ = relation_dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=sd)
                variants.append((f"rel_drop{rate:g}", pred))

            for vname, vsk in variants:
                for g in refs:
                    row = score(vname, vsk, g, rid)
                    if row:
                        rows.append(row)
            for label in bundle.loaded_model_labels():
                if not model_ok.get(label):
                    continue
                for g in refs:
                    key = (label, g)
                    if key not in model_cache:
                        model_cache[key] = score(label, bundle.sketches[label], g, rid)
                    row = model_cache[key]
                    if row:
                        rows.append({**row, "rid": rid})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["batch", "filename", "rid", "variant", "gt", "er", "cpcsr", "oes", "n_probes", "n_probes_raw", "er_modify_fail", "er_insert_fail", "error"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
