"""exp_v4_controls.py — evaluate v4 candidate metrics on the standard
source-conditioned multi-reference setting.

This mirrors ``notebooks/exp_v3_controls_and_robustness.py`` but additionally
computes the v4 candidate metrics defined in ``metrics_v4.py`` (Cases D, H,
K, M, I, J) for the same rotations / variants.  The v3 ``d_feas`` and its
gain / leak components are emitted alongside via ``compute_metrics`` from
``exp_constrained_response_metric``, so a single CSV can be used for
side-by-side comparison.

Variants per rotation:
  source_self     — each source GT against itself (definitional 0)
  heldout         — the heldout human (intent-equivalent positive control)
  dim_dropR       — heldout with R% dimensional constraints removed
  rel_dropR       — heldout with R% relational constraints removed
  <model>         — every loaded Vitruvion variant (base, solver_rl, ...)

The default 5 humans (Eko, Johnny, Ryotaro, Sagar, Tanaka) across batches
{0, 4, 6, 8, 9, 11} come from ``DEFAULT_HUMANS`` in
``exp_source_conditioned_dropout``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys
from itertools import combinations

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from exp_constrained_response_metric import compute_metrics  # noqa: E402
from exp_source_conditioned_dropout import (  # noqa: E402
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
from src.metrics.edit_response_distance import (  # noqa: E402
    constraint_signature,
    _is_relational,
)
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter

from metrics_v4 import V4Hyperparams, compute_metrics_v4  # noqa: E402


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


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


METRIC_KEYS_V3 = (
    "d_feas",
    "d_feas_gain",
    "d_feas_leak",
)
METRIC_KEYS_V4 = (
    # subspace family
    "d_sub", "align_loss", "rank_loss",
    "d_H", "leak_prime",
    # q-retaining family
    "d_K", "gain_K", "leak_K", "effort_K",
    "d_M", "gain_M", "leak_M", "effort_M",
    "gain_I", "gain_J",
    # case P (d_SVD family)
    "d_SVD_sigma_rel", "d_SVD_sub_trunc", "d_SVD_top1",
    # diagnostics
    "dim_P", "dim_G",
    "n_p_pred", "n_p_gt", "n_R_rows", "n_relations", "k_Q_stack",
    "k_Q_used", "k_eff_P", "k_eff_G", "sigma_P_norm", "sigma_G_norm",
)


def evaluate_variant(
    *,
    bundle,
    refs,
    rotation_id: str,
    variant_name: str,
    pred_sketch,
    source_info,
    base_eids,
    context,
    hp: V4Hyperparams,
):
    per_source = []
    detail_rows = []
    for gt_name in refs:
        info = source_info[gt_name]
        probes = [p for p, m in zip(info["probes"], info["valid_mask"]) if m]
        weights = [w for w, m in zip(info["weights"], info["valid_mask"]) if m]
        if not probes:
            continue

        # v3 metrics (d_feas + components)
        res_v3 = compute_metrics(
            pred_sketch=pred_sketch,
            gt_sketch=bundle.sketches[gt_name],
            edit_probes=probes,
            edit_weights=weights,
            rels=info["rels"],
            context=context,
            base_eids=base_eids,
            eps=hp.eps,
        )
        # v4 metrics
        res_v4 = compute_metrics_v4(
            pred_sketch=pred_sketch,
            gt_sketch=bundle.sketches[gt_name],
            edit_probes=probes,
            edit_weights=weights,
            rels=info["rels"],
            context=context,
            base_eids=base_eids,
            hp=hp,
        )
        row = {
            "batch": bundle.batch,
            "filename": bundle.filename,
            "rotation_id": rotation_id,
            "refs": "|".join(refs),
            "variant": variant_name,
            "gt": gt_name,
            "n_edit_probes": len(probes),
            "n_probes_before_validation": info["n_before_validation"],
            "n_probes_after_validation": info["n_after_validation"],
            **{k: res_v3.get(k) for k in METRIC_KEYS_V3},
            **{k: res_v4.get(k) for k in METRIC_KEYS_V4},
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
        "valid_sources": len(per_source),
        "error": "",
    }
    metric_keys = METRIC_KEYS_V3 + METRIC_KEYS_V4
    for key in metric_keys:
        vals = [finite_or_none(r.get(key)) for r in per_source]
        vals = [v for v in vals if v is not None]
        if not vals:
            summary[f"min_{key}"] = None
            summary[f"median_{key}"] = None
        else:
            summary[f"min_{key}"] = min(vals)
            summary[f"median_{key}"] = safe_median(vals)
    return summary, detail_rows


def evaluate_rotation(
    *,
    bundle,
    refs,
    heldout,
    dim_drop_rates,
    rel_drop_rates,
    context,
    converter,
    histcad_cfg,
    base_policy,
    hp: V4Hyperparams,
    eval_models: bool = True,
    exclude_pp_coincident: bool = False,
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
        probes, weights, rels, _rel_weights = source_probes(
            gt_sketch, exclude_point_point_coincident=exclude_pp_coincident)
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

    if eval_models:
        for model_label in bundle.loaded_model_labels():
            ok, _reason = base_policy_pass(bundle.audits[model_label], base_policy)
            if ok:
                variants.append((model_label, bundle.sketches[model_label], {}))

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    # source_self positive control — each GT against itself.
    per_source_self = []
    self_details = []
    for gt_name in refs:
        srow, drows = evaluate_variant(
            bundle=bundle,
            refs=(gt_name,),
            rotation_id=rotation_id,
            variant_name="source_self",
            pred_sketch=bundle.sketches[gt_name],
            source_info={gt_name: source_info[gt_name]},
            base_eids=base_eids,
            context=context,
            hp=hp,
        )
        per_source_self.append(srow)
        self_details.extend(drows)
    summ_self = {
        "batch": bundle.batch,
        "filename": bundle.filename,
        "rotation_id": rotation_id,
        "refs": "|".join(refs),
        "variant": "source_self",
        "valid_sources": len(per_source_self),
        "error": "",
    }
    for key in METRIC_KEYS_V3 + METRIC_KEYS_V4:
        vals = [finite_or_none(r.get(f"min_{key}")) for r in per_source_self]
        vals = [v for v in vals if v is not None]
        summ_self[f"min_{key}"] = min(vals) if vals else None
        summ_self[f"median_{key}"] = safe_median(vals)
    summary_rows.append(summ_self)
    detail_rows.extend(self_details)

    for name, pred_sketch, meta in variants:
        srow, drows = evaluate_variant(
            bundle=bundle,
            refs=refs,
            rotation_id=rotation_id,
            variant_name=name,
            pred_sketch=pred_sketch,
            source_info=source_info,
            base_eids=base_eids,
            context=context,
            hp=hp,
        )
        srow.update(meta)
        summary_rows.append(srow)
        detail_rows.extend(drows)

    return summary_rows, detail_rows


def print_summary(rows: list[dict]):
    valid = [r for r in rows if not r.get("error")]
    print("\n--- V4 controls summary (median by variant) ---")
    variants_ordered: list[str] = []
    seen = set()
    for r in valid:
        v = r.get("variant")
        if v and v not in seen:
            seen.add(v)
            variants_ordered.append(v)
    # Put canonical controls first.
    canonical = ["source_self", "heldout", "dim_drop50", "dim_drop100",
                 "rel_drop50", "rel_drop100"]
    head = [v for v in canonical if v in seen]
    tail = [v for v in variants_ordered if v not in head]
    order = head + tail

    key_metrics = [
        "d_feas",      # v3 baseline
        "d_sub", "d_H",                # probe-free
        "d_K", "d_M",                  # q-retaining
        "gain_K", "leak_K",
        "gain_M", "leak_M",
        "gain_I", "gain_J",
        "d_SVD_sigma_rel", "d_SVD_sub_trunc", "d_SVD_top1",  # case P
    ]
    for variant in order:
        rs = [r for r in valid if r["variant"] == variant]
        if not rs:
            continue
        line = f"  {variant:14s} n={len(rs):4d}"
        for m in key_metrics:
            v = safe_median([r.get(f"min_{m}") for r in rs])
            if v is None:
                line += f"  {m}=NA"
            else:
                line += f"  {m}={v:.4f}"
        print(line)


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
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--mu-k", type=float, default=1.0,
                        help="Relation-preservation weight for Case K / M")
    parser.add_argument("--lambda-k", type=float, default=1e-3,
                        help="Effort regulariser for Case K / M")
    parser.add_argument("--tau-j", type=float, default=1.0,
                        help="Reach scale for Case J")
    parser.add_argument("--tau-p", type=float, default=0.01,
                        help="Case P SVD truncation threshold (relative to "
                             "max(sigma_P[0], sigma_G[0]))")
    parser.add_argument("--base-policy", default="canonical_solid",
                        choices=("none", "canonical_solid", "canonical_all"))
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument("--canonical-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-batch", type=int, default=None)
    parser.add_argument("--delta-basis", default="bbox_diag")
    parser.add_argument("--delta-scale", type=float, default=0.01)
    parser.add_argument("--delta-absolute", type=float, default=1.0)
    parser.add_argument("--delta-signs", default="-1,+1")
    parser.add_argument("--newton-max-iter", type=int, default=80)
    parser.add_argument("--newton-tol", type=float, default=1e-6)
    parser.add_argument("--er-target-tol", type=float, default=1e-3)
    parser.add_argument("--relation-tol", type=float, default=1e-3)
    parser.add_argument("--no-eval-models", action="store_true")
    parser.add_argument("--no-bbox-normalization", action="store_true")
    parser.add_argument("--bbox-normalize-dst-length", type=float, default=1.0)
    parser.add_argument(
        "--exclude-pp-coincident",
        action="store_true",
        help="Task-5 ablation: filter point-point Coincident constraints out of "
             "the relational set used by source_probes (and thus J_R / leak_prime / "
             "leak_K). Point-line/line-line Coincident are retained.",
    )
    parser.add_argument("--coordinate-scale", type=float, default=1.0)
    parser.add_argument("--output-csv",
                        default="artifacts/v4_exp/v4_controls_scores.csv")
    parser.add_argument("--details-csv",
                        default="artifacts/v4_exp/v4_controls_details.csv")
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
    hp = V4Hyperparams(
        eps=args.eps,
        mu_k=args.mu_k,
        lambda_k=args.lambda_k,
        tau_j=args.tau_j,
        tau_p=args.tau_p,
    )
    print(f"[V4 hyperparams] eps={hp.eps} mu_k={hp.mu_k} lambda_k={hp.lambda_k} tau_j={hp.tau_j} tau_p={hp.tau_p}")

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    n_files = 0
    bbox_dst = None if args.no_bbox_normalization else args.bbox_normalize_dst_length
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

        n_batch_files = 0
        for filename in common:
            if args.limit is not None and n_files >= args.limit:
                break
            if args.limit_per_batch is not None and n_batch_files >= args.limit_per_batch:
                break
            n_files += 1
            n_batch_files += 1
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
                    coordinate_scale=args.coordinate_scale,
                    bbox_normalize_dst_length=bbox_dst,
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
                        context=context,
                        converter=converter,
                        histcad_cfg=histcad_cfg,
                        base_policy=args.base_policy,
                        hp=hp,
                        eval_models=not args.no_eval_models,
                        exclude_pp_coincident=args.exclude_pp_coincident,
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
