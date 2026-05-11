"""
exp_multiref_normalization.py — fixed-probe multi-reference evaluator.

This script recomputes all scores from sketches.  It does not read any prior
metric CSVs.

For each file and each rotation:

    refs    = K human annotations used as the multi-reference GT set
    heldout = one remaining human annotation
    model   = Vitruvion output, when available

it constructs one fixed edit-probe set from the refs, then evaluates every
comparison on that same probe set:

    h_scale     = median_{i != j in refs} d_resp(ref_i, ref_j)
    heldout     = min_j d_resp(heldout, ref_j) / h_scale
    Vitruvion   = min_j d_resp(Vitruvion, ref_j) / h_scale

The default probe set is `ref_union`: the union of dimensional constraints
appearing in the reference humans.  Use `--probe-mode ref_intersection` for
only dimensions shared by all refs, or `--probe-mode ref_consensus` with
`--probe-min-count`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations, permutations
from statistics import median
from typing import Mapping, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.dirname(__file__)
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

import sketchgraphs.data as datalib  # noqa: E402

from src.metrics.edit_response_distance import (  # noqa: E402
    collect_probes,
    compute_edit_response_distance,
    constraint_signature,
    pairwise_distance_observable,
)
from src.metrics.edit_invariant_relation import (  # noqa: E402
    compute_edit_invariant_relation_distance,
    compute_edit_invariant_relation_finite_distance,
)
from src.metrics.histcad_2d import (  # noqa: E402
    HistCADConfig,
    compute_histcad_2d_pair,
    compute_histcad_2d_pair_freecad,
)
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.freecad_utils import (  # noqa: E402
    ensure_freecad_imported,
    load_freecad_commands,
)
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

from canonical_sketch import (  # noqa: E402
    build_entity_remap,
    find_canonical_py,
    load_sketch_from_fcstd,
    load_sketch_from_py,
    metric_base_eids,
    remap_sketch_to_canonical,
)


HUMAN_ROOTS = "/home/sturk/Downloads/fiverr"
DEFAULT_HUMANS: dict[str, list[tuple[str, str]]] = {
    "0": [
        ("Johnny", f"{HUMAN_ROOTS}/Johnny/0"),
        ("Sagar", f"{HUMAN_ROOTS}/Sagar/0"),
        ("Ryotaro", f"{HUMAN_ROOTS}/Ryotaro/0"),
    ],
    "9": [
        ("Eko", f"{HUMAN_ROOTS}/Eko/9"),
        ("Sagar", f"{HUMAN_ROOTS}/Sagar/9"),
        ("Ryotaro", f"{HUMAN_ROOTS}/Ryotaro/9"),
    ],
}


@dataclass
class FileBundle:
    batch: str
    filename: str
    canonical_sketch: datalib.Sketch
    sketches: dict[str, datalib.Sketch]
    audits: dict[str, dict]
    model_status: str
    model_dof: int | None
    model_solver_status: str | None


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def parse_signs(text: str) -> tuple[float, ...]:
    signs = tuple(float(x) for x in text.split(",") if x.strip())
    return signs or (-1.0, 1.0)


def parse_human_specs(specs: Sequence[str] | None) -> dict[str, list[tuple[str, str]]]:
    if not specs:
        return {b: list(v) for b, v in DEFAULT_HUMANS.items()}
    out: dict[str, list[tuple[str, str]]] = {}
    for spec in specs:
        try:
            batch, name, path = spec.split(":", 2)
        except ValueError as exc:
            raise ValueError(
                "--human must be formatted as BATCH:NAME:DIR"
            ) from exc
        out.setdefault(batch, []).append((name, path))
    return out


def get_fcstd_map(directory: str) -> dict[str, str]:
    if not os.path.isdir(directory):
        return {}
    return {
        f: f
        for f in os.listdir(directory)
        if f.endswith(".FCStd")
        and not f.endswith("_mid.FCStd")
        and not f.startswith("_")
    }


def vitruvion_solver_status(path: str) -> tuple[int | None, str | None]:
    FreeCAD, _, _ = ensure_freecad_imported()
    doc = None
    try:
        doc, fc_sketch = load_freecad_commands(path)
        status = fc_sketch.getSolverStatus()
        if isinstance(status, dict):
            return status.get("DoF"), status.get("Status")
        return None, str(status)
    except Exception:
        return None, "load_error"
    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except Exception:
                pass


def base_policy_pass(audit: Mapping, base_policy: str) -> tuple[bool, str]:
    if base_policy == "none":
        return True, ""
    if base_policy == "canonical_solid":
        ok = int(audit["missing_solid"]) == 0
        return ok, "" if ok else "missing canonical solid"
    if base_policy == "canonical_all":
        ok = int(audit["missing_solid"]) == 0 and int(audit["missing_construction"]) == 0
        return ok, "" if ok else "missing canonical geometry"
    raise ValueError(base_policy)


def build_base_eids(canonical_sketch: datalib.Sketch, base_policy: str):
    return None if base_policy == "none" else metric_base_eids(
        base_policy, canonical_sketch
    )


def model_path_for(root: str, batch: str, filename: str) -> str | None:
    path = os.path.join(root, batch, os.path.splitext(filename)[0] + ".py")
    return path if os.path.exists(path) else None


def build_file_bundle(
    *,
    batch: str,
    filename: str,
    humans: Sequence[tuple[str, str]],
    file_maps: Mapping[str, Mapping[str, str]],
    model_path: str | None,
    converter: SketchFreecadConverter,
    geometry_tol: float,
    canonical_dir: str | None,
    keep_all_model: bool,
) -> FileBundle:
    data_dirs = {name: directory for name, directory in humans}
    annotators = tuple(data_dirs.keys())
    canonical_path = find_canonical_py(
        filename, data_dirs, annotators, canonical_dir=canonical_dir
    )
    canonical_sketch = load_sketch_from_py(canonical_path, converter)

    raw: dict[str, datalib.Sketch] = {}
    for name, directory in humans:
        raw[name] = load_sketch_from_fcstd(
            os.path.join(directory, file_maps[name][filename]), converter
        )

    model_status = "missing"
    model_dof: int | None = None
    model_solver_status: str | None = None
    if model_path is not None:
        model_dof, model_solver_status = vitruvion_solver_status(model_path)
        if keep_all_model or (model_dof == 2 and model_solver_status == "Converged"):
            raw["Vitruvion"] = load_sketch_from_py(model_path, converter)
            model_status = "loaded"
        else:
            model_status = "filtered"

    sketches: dict[str, datalib.Sketch] = {}
    audits: dict[str, dict] = {}
    for name, full in raw.items():
        source_to_new, extras, audit = build_entity_remap(
            full, canonical_sketch, geometry_tol
        )
        remapped, dropped = remap_sketch_to_canonical(
            full, canonical_sketch, source_to_new, extras
        )
        audit["dropped_constraints"] = dropped
        sketches[name] = remapped
        audits[name] = audit

    return FileBundle(
        batch=batch,
        filename=filename,
        canonical_sketch=canonical_sketch,
        sketches=sketches,
        audits=audits,
        model_status=model_status,
        model_dof=model_dof,
        model_solver_status=model_solver_status,
    )


def select_ref_probes(
    ref_sketches: Sequence[datalib.Sketch],
    mode: str,
    min_count: int | None,
    weighting: str,
) -> tuple[list[datalib.Constraint], list[float], int]:
    probes, frac_weights = collect_probes(ref_sketches, kind="dim")
    n_refs = len(ref_sketches)
    if mode == "ref_union":
        keep = [True] * len(probes)
    elif mode == "ref_intersection":
        keep = [w >= 1.0 - 1e-12 for w in frac_weights]
    elif mode == "ref_consensus":
        threshold = min_count if min_count is not None else max(1, math.ceil(n_refs / 2))
        keep = [round(w * n_refs) >= threshold for w in frac_weights]
    else:
        raise ValueError(mode)

    out_probes: list[datalib.Constraint] = []
    out_weights: list[float] = []
    for probe, frac, ok in zip(probes, frac_weights, keep):
        if not ok:
            continue
        out_probes.append(probe)
        if weighting == "uniform":
            out_weights.append(1.0)
        elif weighting == "ref_fraction":
            out_weights.append(frac)
        else:
            raise ValueError(weighting)
    return out_probes, out_weights, len(probes)


def relation_union_from_refs(
    ref_sketches: Sequence[datalib.Sketch],
) -> tuple[list[datalib.Constraint], list[float]]:
    return collect_probes(ref_sketches, kind="rel")


def finite_or_none(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def norm_by_scale(value: float | None, scale: float | None, eps: float) -> float | None:
    if value is None or scale is None:
        return None
    if abs(scale) <= eps:
        return 0.0 if abs(value) <= eps else math.inf
    return value / scale


def metric_pair(
    *,
    batch: str,
    filename: str,
    rotation_id: str,
    role: str,
    pred_name: str,
    gt_name: str,
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    edit_weights: Sequence[float],
    relation_probes: Sequence[datalib.Constraint],
    relation_weights: Sequence[float],
    context: SketchContext,
    converter: SketchFreecadConverter,
    histcad_cfg: HistCADConfig,
    solver: str,
    base_eids: Sequence[str] | None,
    compute_finite: bool,
) -> dict:
    row = {
        "batch": batch,
        "filename": filename,
        "rotation_id": rotation_id,
        "role": role,
        "pred": pred_name,
        "gt": gt_name,
        "n_edit_probes": len(edit_probes),
        "n_ref_rel_probes": len(relation_probes),
        "error": "",
    }

    try:
        j_gt = j_pred = None
        if base_eids is not None:
            j_gt = pairwise_distance_observable(gt_sketch, base_eids=base_eids)
            j_pred = pairwise_distance_observable(pred_sketch, base_eids=base_eids)
        r1 = compute_edit_response_distance(
            gt_sketch=gt_sketch,
            pred_sketch=pred_sketch,
            edit_probes=edit_probes,
            context=context,
            probe_weights=edit_weights,
            observable_jacobian_gt=j_gt,
            observable_jacobian_pred=j_pred,
        )
        row.update({
            "d_resp": r1.distance,
            "n_dim_pred": r1.n_dim_pred,
            "n_dim_gt": r1.n_dim_gt,
            "skipped_resp": r1.skipped_probes,
        })
    except Exception as exc:
        row["error"] = (row["error"] + f" d_resp:{exc}").strip()[:300]

    try:
        r2 = compute_edit_invariant_relation_distance(
            gt_sketch=gt_sketch,
            pred_sketch=pred_sketch,
            edit_probes=edit_probes,
            relation_probes=relation_probes,
            context=context,
            edit_weights=edit_weights,
            relation_weights=relation_weights,
        )
        row.update({"d_inv": r2.distance, "skipped_inv": r2.skipped_pairs})
    except Exception as exc:
        row["error"] = (row["error"] + f" d_inv:{exc}").strip()[:300]

    if compute_finite:
        try:
            gt_rels = [
                c for c in gt_sketch.constraints.values()
                if c.constraint_type.name not in {
                    "Distance", "DistanceX", "DistanceY", "Length", "Diameter",
                    "Radius", "Angle", "Offset", "Centerline_Dimension",
                    "Minor_Diameter", "Major_Diameter", "Rho", "Block",
                    "InternalAlignment",
                }
            ]
            r3 = compute_edit_invariant_relation_finite_distance(
                gt_sketch=gt_sketch,
                pred_sketch=pred_sketch,
                edit_probes=edit_probes,
                relation_probes=gt_rels,
                context=context,
                converter=converter,
                config=histcad_cfg,
                edit_weights=edit_weights,
                relation_weights=[1.0] * len(gt_rels),
            )
            row.update({
                "d_inv_finite": r3.distance,
                "d_inv_finite_skipped": r3.n_skipped,
            })
        except Exception as exc:
            row["error"] = (row["error"] + f" d_inv_finite:{exc}").strip()[:300]

    try:
        if solver == "freecad":
            h = compute_histcad_2d_pair_freecad(
                pred_sketch=pred_sketch,
                gt_sketch=gt_sketch,
                edit_probes=edit_probes,
                context=context,
                converter=converter,
                config=histcad_cfg,
            )
        else:
            h = compute_histcad_2d_pair(
                pred_sketch=pred_sketch,
                gt_sketch=gt_sketch,
                edit_probes=edit_probes,
                context=context,
                config=histcad_cfg,
            )
        row.update({
            "histcad_er": h.er_rate,
            "histcad_cpcsr": h.cpcsr_rate,
            "histcad_oes": h.oes_rate,
            "histcad_n_probes": h.n_probes,
            "histcad_n_relations": h.n_relations,
        })
    except Exception as exc:
        row["error"] = (row["error"] + f" histcad:{exc}").strip()[:300]

    return row


def best_by_d_resp(rows: Sequence[dict]) -> dict | None:
    valid = [r for r in rows if finite_or_none(r.get("d_resp")) is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: finite_or_none(r.get("d_resp")) or math.inf)


def max_metric(rows: Sequence[dict], key: str) -> float | None:
    vals = [finite_or_none(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def verdict(model_norm, model_max_oes, heldout_norm, min_oes: float, eps: float) -> str:
    if model_max_oes is None:
        return "no_model_score"
    if model_max_oes < min_oes:
        return "model_not_histcad_high"
    if model_norm is None:
        return "no_model_norm"
    if math.isinf(model_norm):
        return "HIGH_OES_MODEL_NONZERO_REF_SCALE_ZERO"
    if model_norm <= 1.0 + eps:
        return "HIGH_OES_WITHIN_REF_SCALE"
    if heldout_norm is not None and not math.isinf(heldout_norm):
        if model_norm > 1.5 * max(heldout_norm, eps):
            return "HIGH_OES_MODEL_GT_1.5X_HELDOUT"
    return "HIGH_OES_ABOVE_REF_SCALE"


def evaluate_rotation(
    *,
    bundle: FileBundle,
    refs: tuple[str, ...],
    heldout: str,
    context: SketchContext,
    converter: SketchFreecadConverter,
    histcad_cfg: HistCADConfig,
    base_policy: str,
    probe_mode: str,
    probe_min_count: int | None,
    probe_weighting: str,
    solver: str,
    compute_finite: bool,
    min_model_oes: float,
    eps: float,
) -> tuple[dict, list[dict]]:
    rotation_id = f"{bundle.batch}:{bundle.filename}:{'+'.join(refs)}__heldout={heldout}"
    participants = set(refs) | {heldout}
    row = {
        "batch": bundle.batch,
        "filename": bundle.filename,
        "rotation_id": rotation_id,
        "refs": "|".join(refs),
        "heldout": heldout,
        "ref_size": len(refs),
        "probe_mode": probe_mode,
        "probe_weighting": probe_weighting,
        "model_status": bundle.model_status,
        "model_dof": bundle.model_dof,
        "model_solver_status": bundle.model_solver_status,
        "error": "",
    }

    bad = []
    for name in participants:
        ok, reason = base_policy_pass(bundle.audits[name], base_policy)
        if not ok:
            bad.append(f"{name}:{reason}")
    if bad:
        row["error"] = "human_audit:" + ",".join(bad)
        return row, []

    ref_sketches = [bundle.sketches[name] for name in refs]
    edit_probes, edit_weights, n_ref_dim_union = select_ref_probes(
        ref_sketches, probe_mode, probe_min_count, probe_weighting
    )
    rel_probes, rel_weights = relation_union_from_refs(ref_sketches)
    base_eids = build_base_eids(bundle.canonical_sketch, base_policy)

    row.update({
        "n_ref_dim_union": n_ref_dim_union,
        "n_edit_probes": len(edit_probes),
        "n_ref_rel_union": len(rel_probes),
    })

    detail_rows: list[dict] = []

    ref_pair_rows = []
    for pred, gt in permutations(refs, 2):
        pr = metric_pair(
            batch=bundle.batch,
            filename=bundle.filename,
            rotation_id=rotation_id,
            role="ref_ref",
            pred_name=pred,
            gt_name=gt,
            pred_sketch=bundle.sketches[pred],
            gt_sketch=bundle.sketches[gt],
            edit_probes=edit_probes,
            edit_weights=edit_weights,
            relation_probes=rel_probes,
            relation_weights=rel_weights,
            context=context,
            converter=converter,
            histcad_cfg=histcad_cfg,
            solver=solver,
            base_eids=base_eids,
            compute_finite=compute_finite,
        )
        ref_pair_rows.append(pr)
        detail_rows.append(pr)

    h_vals = [finite_or_none(r.get("d_resp")) for r in ref_pair_rows]
    h_vals = [v for v in h_vals if v is not None]
    h_scale = float(median(h_vals)) if h_vals else None
    row["h_scale_ordered_median"] = h_scale
    row["n_ref_ordered_pairs"] = len(h_vals)

    heldout_rows = []
    for gt in refs:
        pr = metric_pair(
            batch=bundle.batch,
            filename=bundle.filename,
            rotation_id=rotation_id,
            role="heldout_ref",
            pred_name=heldout,
            gt_name=gt,
            pred_sketch=bundle.sketches[heldout],
            gt_sketch=bundle.sketches[gt],
            edit_probes=edit_probes,
            edit_weights=edit_weights,
            relation_probes=rel_probes,
            relation_weights=rel_weights,
            context=context,
            converter=converter,
            histcad_cfg=histcad_cfg,
            solver=solver,
            base_eids=base_eids,
            compute_finite=compute_finite,
        )
        heldout_rows.append(pr)
        detail_rows.append(pr)

    hb = best_by_d_resp(heldout_rows)
    heldout_best = finite_or_none(hb.get("d_resp")) if hb else None
    row.update({
        "heldout_best_ref": hb.get("gt") if hb else "",
        "heldout_best_d_resp": heldout_best,
        "heldout_norm": norm_by_scale(heldout_best, h_scale, eps),
        "heldout_max_oes": max_metric(heldout_rows, "histcad_oes"),
    })

    model_rows = []
    if bundle.model_status == "loaded":
        ok, reason = base_policy_pass(bundle.audits["Vitruvion"], base_policy)
        if not ok:
            row["error"] = (row["error"] + f" model_audit:{reason}").strip()
        else:
            for gt in refs:
                pr = metric_pair(
                    batch=bundle.batch,
                    filename=bundle.filename,
                    rotation_id=rotation_id,
                    role="model_ref",
                    pred_name="Vitruvion",
                    gt_name=gt,
                    pred_sketch=bundle.sketches["Vitruvion"],
                    gt_sketch=bundle.sketches[gt],
                    edit_probes=edit_probes,
                    edit_weights=edit_weights,
                    relation_probes=rel_probes,
                    relation_weights=rel_weights,
                    context=context,
                    converter=converter,
                    histcad_cfg=histcad_cfg,
                    solver=solver,
                    base_eids=base_eids,
                    compute_finite=compute_finite,
                )
                model_rows.append(pr)
                detail_rows.append(pr)

    mb = best_by_d_resp(model_rows)
    model_best = finite_or_none(mb.get("d_resp")) if mb else None
    model_norm = norm_by_scale(model_best, h_scale, eps)
    model_max_oes = max_metric(model_rows, "histcad_oes")
    row.update({
        "model_best_ref": mb.get("gt") if mb else "",
        "model_best_d_resp": model_best,
        "model_norm": model_norm,
        "model_best_oes": finite_or_none(mb.get("histcad_oes")) if mb else None,
        "model_max_oes": model_max_oes,
        "model_valid_refs": len([r for r in model_rows if finite_or_none(r.get("d_resp")) is not None]),
        "verdict": verdict(model_norm, model_max_oes, row["heldout_norm"], min_model_oes, eps),
    })
    return row, detail_rows


def write_csv(path: str, rows: Sequence[dict]) -> None:
    full = repo_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
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
    print(f"Saved CSV: {full}")


def print_summary(rows: Sequence[dict]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("verdict") or "no_verdict"
        counts[key] = counts.get(key, 0) + 1
    print("\n--- Fixed-probe multi-reference summary ---")
    print(f"rotation rows: {len(rows)}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--vitruvion-root", default="/home/sturk/Downloads/fiverr/Vitruvion_2026-05-03_17-08-23_step_56400")
    p.add_argument("--batches", nargs="+", default=["0", "9"])
    p.add_argument("--human", action="append", default=None,
                   help="Override/add humans as BATCH:NAME:DIR. If omitted, defaults to batches 0 and 9.")
    p.add_argument("--ref-size", type=int, default=2)
    p.add_argument("--probe-mode", default="ref_union",
                   choices=("ref_union", "ref_intersection", "ref_consensus"))
    p.add_argument("--probe-min-count", type=int, default=None)
    p.add_argument("--probe-weighting", default="ref_fraction",
                   choices=("ref_fraction", "uniform"))
    p.add_argument("--base-policy", default="canonical_solid",
                   choices=("none", "canonical_solid", "canonical_all"))
    p.add_argument("--geometry-tol", type=float, default=5e-1)
    p.add_argument("--canonical-dir", default=None)
    p.add_argument("--keep-all-model", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--solver", default="freecad", choices=("newton", "freecad"))
    p.add_argument("--with-d-inv-finite", action="store_true")
    p.add_argument("--delta-basis", default="bbox_diag",
                   choices=("bbox_diag", "current", "absolute"))
    p.add_argument("--delta-scale", type=float, default=0.01)
    p.add_argument("--delta-absolute", type=float, default=1.0)
    p.add_argument("--delta-signs", default="-1,+1")
    p.add_argument("--newton-max-iter", type=int, default=80)
    p.add_argument("--newton-tol", type=float, default=1e-6)
    p.add_argument("--er-target-tol", type=float, default=1e-3)
    p.add_argument("--relation-tol", type=float, default=1e-3)
    p.add_argument("--relation-source", default="gt",
                   choices=("gt", "pred", "intersection", "union"))
    p.add_argument("--model-min-oes", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--output-csv", default="artifacts/multiref_fixed_probe_scores.csv")
    p.add_argument("--details-csv", default="artifacts/multiref_fixed_probe_pair_details.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    humans_by_batch = parse_human_specs(args.human)
    cfg = HistCADConfig(
        delta_basis=args.delta_basis,
        delta_scale=args.delta_scale,
        delta_absolute=args.delta_absolute,
        delta_signs=parse_signs(args.delta_signs),
        newton_max_iter=args.newton_max_iter,
        newton_tol=args.newton_tol,
        er_target_tol=args.er_target_tol,
        relation_tol_default=args.relation_tol,
        relation_source=args.relation_source,
    )

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    n_files_seen = 0
    for batch in args.batches:
        humans = humans_by_batch.get(batch, [])
        humans = [(name, d) for name, d in humans if os.path.isdir(d)]
        if len(humans) < args.ref_size + 1:
            print(f"[skip] batch {batch}: need at least {args.ref_size + 1} humans, got {len(humans)}")
            continue

        file_maps = {name: get_fcstd_map(directory) for name, directory in humans}
        common = sorted(set.intersection(*(set(m) for m in file_maps.values())))
        print(
            f"Batch {batch}: "
            + " | ".join(f"{name}={len(file_maps[name])}" for name, _ in humans)
            + f" | common={len(common)}"
        )

        for filename in common:
            if args.limit is not None and n_files_seen >= args.limit:
                break
            n_files_seen += 1
            model_path = model_path_for(args.vitruvion_root, batch, filename)
            try:
                bundle = build_file_bundle(
                    batch=batch,
                    filename=filename,
                    humans=humans,
                    file_maps=file_maps,
                    model_path=model_path,
                    converter=converter,
                    geometry_tol=args.geometry_tol,
                    canonical_dir=args.canonical_dir,
                    keep_all_model=args.keep_all_model,
                )
            except Exception as exc:
                print(f"  [skip] {batch}/{filename}: {exc}")
                continue

            human_names = tuple(name for name, _ in humans)
            n_rot = 0
            for refs in combinations(human_names, args.ref_size):
                for heldout in human_names:
                    if heldout in refs:
                        continue
                    row, details = evaluate_rotation(
                        bundle=bundle,
                        refs=tuple(refs),
                        heldout=heldout,
                        context=context,
                        converter=converter,
                        histcad_cfg=cfg,
                        base_policy=args.base_policy,
                        probe_mode=args.probe_mode,
                        probe_min_count=args.probe_min_count,
                        probe_weighting=args.probe_weighting,
                        solver=args.solver,
                        compute_finite=args.with_d_inv_finite,
                        min_model_oes=args.model_min_oes,
                        eps=args.eps,
                    )
                    summary_rows.append(row)
                    detail_rows.extend(details)
                    n_rot += 1
            print(f"  [done] {batch}/{filename}: {n_rot} rotations model={bundle.model_status}")

        if args.limit is not None and n_files_seen >= args.limit:
            break

    print_summary(summary_rows)
    write_csv(args.output_csv, summary_rows)
    write_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
