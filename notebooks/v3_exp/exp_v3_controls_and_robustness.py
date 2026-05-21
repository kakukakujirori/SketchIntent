"""
exp_v3_controls_and_robustness.py — extra checks for summary_histcad_response_metric_v3.

This script evaluates positive/negative controls for d_feas under the same
source-conditioned multi-reference setting used in exp_constrained_response_metric:

  - source_self: each source GT evaluated against itself
  - heldout: intent-equivalent human heldout
  - dim_drop{r}: heldout with r% dimensional constraints removed
  - rel_drop{r}: heldout with r% relational constraints removed
  - Vitruvion: model output, when available

It also sweeps the absolute SVD pseudo-inverse cutoff eps in one pass.  This is
much cheaper than re-running the full data loading and source validation for
each eps value separately.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
from itertools import combinations
from statistics import median

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.dirname(__file__)
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from exp_constrained_response_metric import compute_metrics
from exp_source_conditioned_dropout import (
    DEFAULT_BATCHES,
    HistCADConfig,
    base_policy_pass,
    build_base_eids,
    build_file_bundle,
    dropout_variant,
    finite_or_none,
    get_fcstd_map,
    parse_human_specs,
    parse_percentages,
    parse_signs,
    parse_vitruvion_specs,
    safe_median,
    source_probes,
    stable_seed,
    validate_probes_on_source,
)
from src.metrics.edit_response_distance import constraint_signature, _is_relational
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def parse_eps_values(text: str) -> list[float]:
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--eps-values must contain at least one value")
    return vals


def sorted_rel_ids(sketch) -> list[str]:
    return sorted(
        [cid for cid, c in sketch.constraints.items() if _is_relational(c)],
        key=lambda cid: (constraint_signature(sketch.constraints[cid]), cid),
    )


def relation_dropout_variant(sketch, *, drop_rate: float, seed: int):
    rel_ids = sorted_rel_ids(sketch)
    n = len(rel_ids)
    nd = max(0, min(n, int(round(n * drop_rate / 100.0))))
    if nd == 0:
        drop = set()
    elif nd == n:
        drop = set(rel_ids)
    else:
        drop = set(random.Random(seed).sample(rel_ids, nd))
    out = copy.deepcopy(sketch)
    out.constraints = {k: v for k, v in out.constraints.items() if k not in drop}
    return out, n, n - nd, nd


def write_csv(path: str, rows: list[dict]):
    full = repo_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(full, "w", newline="") as f:
        if not keys:
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {full}")


def evaluate_variant_for_eps(
    *,
    bundle,
    refs,
    rotation_id: str,
    variant_name: str,
    pred_sketch,
    source_info,
    base_eids,
    context,
    eps: float,
    rcond: float = 1e-6,
    source_validated: bool = True,
):
    per_source = []
    detail_rows = []
    for gt_name in refs:
        info = source_info[gt_name]
        if source_validated:
            probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
            weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
        else:
            probes, weights = info["probes"], info["weights"]
        if not probes:
            continue

        res = compute_metrics(
            pred_sketch=pred_sketch,
            gt_sketch=bundle.sketches[gt_name],
            edit_probes=probes,
            edit_weights=weights,
            rels=info["rels"],
            context=context,
            base_eids=base_eids,
            eps=eps,
            rcond=rcond,
        )
        row = {
            "batch": bundle.batch,
            "filename": bundle.filename,
            "rotation_id": rotation_id,
            "refs": "|".join(refs),
            "variant": variant_name,
            "gt": gt_name,
            "eps": eps,
            "source_validated": source_validated,
            "n_edit_probes": len(probes),
            "n_probes_before_validation": info["n_before_validation"],
            "n_probes_after_validation": info["n_after_validation"],
            **res,
            "error": "",
        }
        per_source.append(row)
        detail_rows.append(row)

    summary = {
        "batch": bundle.batch,
        "filename": bundle.filename,
        "rotation_id": rotation_id,
        "refs": "|".join(refs),
        "variant": variant_name,
        "eps": eps,
        "source_validated": source_validated,
        "valid_sources": len(per_source),
        "error": "",
    }
    for key in (
        "d_feas",
        "d_feas_tilde",
        "d_feas_gain",
        "d_feas_leak",
        "d_feas_tilde_leak",
        "d_leak",
        "d_resp",
        "n_dim_pred",
        "n_dim_gt",
        "n_relations",
    ):
        vals = [finite_or_none(r.get(key)) for r in per_source]
        vals = [v for v in vals if v is not None]
        if key.startswith("d_"):
            summary[f"min_{key}"] = min(vals) if vals else None
            summary[f"median_{key}"] = safe_median(vals)
        else:
            summary[f"median_{key}"] = safe_median(vals)
    return summary, detail_rows


def evaluate_rotation(
    *,
    bundle,
    refs,
    heldout,
    dim_drop_rates,
    rel_drop_rates,
    eps_values,
    rcond: float = 1e-6,
    context,
    converter,
    histcad_cfg,
    base_policy,
):
    rotation_id = f"{bundle.batch}:{bundle.filename}:{'+'.join(refs)}__query={heldout}"
    participants = set(refs) | {heldout}
    bad = []
    for name in participants:
        ok, reason = base_policy_pass(bundle.audits[name], base_policy)
        if not ok:
            bad.append(f"{name}:{reason}")
    if bad:
        return [{
            "batch": bundle.batch,
            "filename": bundle.filename,
            "rotation_id": rotation_id,
            "variant": "error",
            "error": ",".join(bad),
        }], []

    base_eids = build_base_eids(bundle.canonical_sketch, base_policy)
    source_info = {}
    for gt_name in refs:
        gt_sketch = bundle.sketches[gt_name]
        probes, weights, rels, _rel_weights = source_probes(gt_sketch)
        mask = validate_probes_on_source(
            gt_sketch, probes, rels, context, converter, histcad_cfg)
        source_info[gt_name] = {
            "probes": probes,
            "weights": weights,
            "rels": rels,
            "valid_mask": mask,
            "n_before_validation": len(probes),
            "n_after_validation": sum(mask) if mask else 0,
        }

    heldout_sketch = bundle.sketches[heldout]
    variants: list[tuple[str, object, dict]] = []
    variants.append(("heldout", heldout_sketch, {}))

    for rate in dim_drop_rates:
        seed = stable_seed(bundle.batch, bundle.filename, rotation_id, "dim", rate)
        pred, n_orig, n_kept, n_dropped = dropout_variant(
            heldout_sketch, drop_rate=rate, seed=seed)
        variants.append((
            f"dim_drop{rate:g}",
            pred,
            {"n_dim_original": n_orig, "n_dim_kept": n_kept, "n_dim_dropped": n_dropped},
        ))

    for rate in rel_drop_rates:
        seed = stable_seed(bundle.batch, bundle.filename, rotation_id, "rel", rate)
        pred, n_orig, n_kept, n_dropped = relation_dropout_variant(
            heldout_sketch, drop_rate=rate, seed=seed)
        variants.append((
            f"rel_drop{rate:g}",
            pred,
            {"n_rel_original": n_orig, "n_rel_kept": n_kept, "n_rel_dropped": n_dropped},
        ))

    for model_label in bundle.loaded_model_labels():
        ok, _reason = base_policy_pass(bundle.audits[model_label], base_policy)
        if ok:
            variants.append((model_label, bundle.sketches[model_label], {}))

    summary_rows = []
    detail_rows = []
    for eps in eps_values:
        # Source-self positive control: evaluate each source GT against itself,
        # then aggregate over the two source choices by min/median.
        per_source_self = []
        source_self_detail = []
        for gt_name in refs:
            row, drows = evaluate_variant_for_eps(
                bundle=bundle,
                refs=(gt_name,),
                rotation_id=rotation_id,
                variant_name="source_self",
                pred_sketch=bundle.sketches[gt_name],
                source_info={gt_name: source_info[gt_name]},
                base_eids=base_eids,
                context=context,
                eps=eps,
                rcond=rcond,
            )
            per_source_self.append(row)
            source_self_detail.extend(drows)
        sself = {
            "batch": bundle.batch,
            "filename": bundle.filename,
            "rotation_id": rotation_id,
            "refs": "|".join(refs),
            "variant": "source_self",
            "eps": eps,
            "source_validated": True,
            "valid_sources": len(per_source_self),
            "error": "",
        }
        for key in (
            "d_feas",
            "d_feas_tilde",
            "d_feas_gain",
            "d_feas_leak",
            "d_feas_tilde_leak",
            "d_leak",
            "d_resp",
        ):
            vals = [finite_or_none(r.get(f"min_{key}")) for r in per_source_self]
            vals = [v for v in vals if v is not None]
            sself[f"min_{key}"] = min(vals) if vals else None
            sself[f"median_{key}"] = safe_median(vals)
        summary_rows.append(sself)
        detail_rows.extend(source_self_detail)

        for name, pred_sketch, meta in variants:
            srow, drows = evaluate_variant_for_eps(
                bundle=bundle,
                refs=refs,
                rotation_id=rotation_id,
                variant_name=name,
                pred_sketch=pred_sketch,
                source_info=source_info,
                base_eids=base_eids,
                context=context,
                eps=eps,
                rcond=rcond,
            )
            srow.update(meta)
            summary_rows.append(srow)
            detail_rows.extend(drows)

    return summary_rows, detail_rows


def print_summary(rows: list[dict]):
    valid = [r for r in rows if not r.get("error")]
    print("\n--- V3 controls and eps robustness ---")
    eps_values = sorted({float(r["eps"]) for r in valid})
    fixed = [
        "source_self",
        "heldout",
        "dim_drop50",
        "dim_drop100",
        "rel_drop50",
        "rel_drop100",
    ]
    extra: list[str] = []
    for r in valid:
        v = r.get("variant")
        if v and v not in fixed and v not in extra:
            extra.append(v)
    variants = fixed + extra
    for eps in eps_values:
        erows = [r for r in valid if float(r["eps"]) == eps]
        print(f"\n  [eps={eps:g}] n={len(erows)}")
        for variant in variants:
            rs = [r for r in erows if r["variant"] == variant]
            if not rs:
                continue
            print(
                f"    {variant:14s} n={len(rs):4d}"
                f"  med_d_feas={safe_median([r.get('min_d_feas') for r in rs])}"
                f"  p75_d_feas={percentile([r.get('min_d_feas') for r in rs], 75)}"
                f"  med_gain={safe_median([r.get('median_d_feas_gain') for r in rs])}"
                f"  med_leak={safe_median([r.get('median_d_feas_leak') for r in rs])}"
            )


def percentile(vals, q):
    xs = sorted(float(v) for v in vals if finite_or_none(v) is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--vitruvion-root",
        default=None,
        help="Legacy single-model root. Prefer --vitruvion-spec.",
    )
    parser.add_argument(
        "--vitruvion-spec",
        action="append",
        default=None,
        help="label:path, repeatable. Defaults to base + solver_rl when neither flag given.",
    )
    parser.add_argument("--batches", nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--human", action="append", default=None)
    parser.add_argument("--ref-size", type=int, default=2)
    parser.add_argument("--dim-drop-rates", default="50,100")
    parser.add_argument("--rel-drop-rates", default="50,100")
    parser.add_argument("--eps-values", default="1e-7,1e-9,1e-11")
    parser.add_argument(
        "--rcond", type=float, default=1e-6,
        help="Relative singular value cutoff for pinv(J_x): eps_eff = max(eps, rcond*sigma_max). "
             "Sweeping eps with rcond=0 isolates the absolute-eps effect.")
    parser.add_argument("--base-policy", default="canonical_solid",
                        choices=("none", "canonical_solid", "canonical_all"))
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument("--canonical-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delta-basis", default="bbox_diag")
    parser.add_argument("--delta-scale", type=float, default=0.01)
    parser.add_argument("--delta-absolute", type=float, default=1.0)
    parser.add_argument("--delta-signs", default="-1,+1")
    parser.add_argument("--newton-max-iter", type=int, default=80)
    parser.add_argument("--newton-tol", type=float, default=1e-6)
    parser.add_argument("--er-target-tol", type=float, default=1e-3)
    parser.add_argument("--relation-tol", type=float, default=1e-3)
    parser.add_argument("--output-csv",
                        default="artifacts/v3_controls_scores.csv")
    parser.add_argument("--details-csv",
                        default="artifacts/v3_controls_details.csv")
    return parser.parse_args()


def main():
    import _fast
    import exp_constrained_response_metric as Mcrm

    _fast.install(Mcrm, wrap_evaluate_rotation=False)
    _fast.install(sys.modules[__name__])

    args = parse_args()
    humans_by_batch = parse_human_specs(args.human)
    dim_drop_rates = parse_percentages(args.dim_drop_rates)
    rel_drop_rates = parse_percentages(args.rel_drop_rates)
    eps_values = parse_eps_values(args.eps_values)
    histcad_cfg = HistCADConfig(
        delta_basis=args.delta_basis,
        delta_scale=args.delta_scale,
        delta_absolute=args.delta_absolute,
        delta_signs=parse_signs(args.delta_signs),
        newton_max_iter=args.newton_max_iter,
        newton_tol=args.newton_tol,
        er_target_tol=args.er_target_tol,
        relation_tol_default=args.relation_tol,
        relation_source="gt",
    )

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    n_files = 0
    model_specs = parse_vitruvion_specs(args.vitruvion_spec, args.vitruvion_root)
    if model_specs:
        print("[Vitruvion specs] " + ", ".join(f"{l}={p}" for l, p in model_specs))

    for batch in args.batches:
        humans = [(n, d) for n, d in humans_by_batch.get(batch, [])
                  if os.path.isdir(d)]
        if len(humans) < args.ref_size + 1:
            print(f"[skip] batch {batch}: need {args.ref_size + 1} humans")
            continue
        file_maps = {n: get_fcstd_map(d) for n, d in humans}
        common = sorted(set.intersection(*(set(m) for m in file_maps.values())))
        print(f"Batch {batch}: common={len(common)}")

        for filename in common:
            if args.limit is not None and n_files >= args.limit:
                break
            n_files += 1
            try:
                bundle = build_file_bundle(
                    batch=batch,
                    filename=filename,
                    humans=humans,
                    file_maps=file_maps,
                    converter=converter,
                    geometry_tol=args.geometry_tol,
                    canonical_dir=args.canonical_dir,
                    model_specs=model_specs,
                )
            except Exception as exc:
                print(f"  [skip] {batch}/{filename}: {exc}")
                continue

            names = tuple(name for name, _ in humans)
            n_rot = 0
            for refs in combinations(names, args.ref_size):
                for heldout in names:
                    if heldout in refs:
                        continue
                    srows, drows = evaluate_rotation(
                        bundle=bundle,
                        refs=tuple(refs),
                        heldout=heldout,
                        dim_drop_rates=dim_drop_rates,
                        rel_drop_rates=rel_drop_rates,
                        eps_values=eps_values,
                        rcond=args.rcond,
                        context=context,
                        converter=converter,
                        histcad_cfg=histcad_cfg,
                        base_policy=args.base_policy,
                    )
                    summary_rows.extend(srows)
                    detail_rows.extend(drows)
                    n_rot += 1
            print(f"  [done] {batch}/{filename}: {n_rot} rotations")

        if args.limit is not None and n_files >= args.limit:
            break

    print_summary(summary_rows)
    write_csv(args.output_csv, summary_rows)
    write_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
