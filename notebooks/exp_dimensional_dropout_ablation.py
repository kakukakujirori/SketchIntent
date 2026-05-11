"""
exp_dimensional_dropout_ablation.py — GT-only dimensional dropout ablation.

This experiment tests whether HistCAD-style OES can reward sketches with fewer
native dimensional constraints even when the removed dimensions are part of a
human GT annotation.

For each file and each 2-ref/1-heldout rotation:

    refs      = two human annotations used as evaluation GTs
    query     = heldout human annotation
    pred(r)   = query with r% of its dimensional constraints removed

The edit probes are fixed from the reference GT dimensions by default.  Each
dropout variant is evaluated against both refs from scratch:

    HistCAD: solve edits on pred(r), preserve ref relational constraints
    d_resp : compare pred(r)'s first-order edit response with each ref

Lower d_resp is better; higher HistCAD OES is better.  If the hypothesis is
correct, OES should rise as more dimensions are removed, while d_resp should
worsen as more dimensions are removed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from statistics import median
from typing import Mapping, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
NOTEBOOKS_DIR = os.path.dirname(__file__)
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

import sketchgraphs.data as datalib  # noqa: E402
import torch  # noqa: E402

from src.metrics.edit_response_distance import (  # noqa: E402
    DIMENSIONAL_CONSTRAINTS,
    collect_probes,
    compute_native_response,
    compute_edit_response_distance,
    constraint_signature,
    constraint_jacobian,
    edit_response_for_probe,
    pairwise_distance_observable,
)
from src.metrics.histcad_2d import (  # noqa: E402
    HistCADConfig,
    compute_histcad_2d_pair,
    compute_histcad_2d_pair_freecad,
)
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

from canonical_sketch import (  # noqa: E402
    build_entity_remap,
    find_canonical_py,
    load_sketch_from_fcstd,
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


def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def parse_signs(text: str) -> tuple[float, ...]:
    signs = tuple(float(x) for x in text.split(",") if x.strip())
    return signs or (-1.0, 1.0)


def parse_percentages(text: str) -> list[float]:
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("--drop-rates must contain at least one percentage")
    for value in vals:
        if value < 0.0 or value > 100.0:
            raise ValueError(f"drop percentage must be in [0,100], got {value}")
    return vals


def parse_human_specs(specs: Sequence[str] | None) -> dict[str, list[tuple[str, str]]]:
    if not specs:
        return {b: list(v) for b, v in DEFAULT_HUMANS.items()}
    out: dict[str, list[tuple[str, str]]] = {}
    for spec in specs:
        try:
            batch, name, path = spec.split(":", 2)
        except ValueError as exc:
            raise ValueError("--human must be formatted as BATCH:NAME:DIR") from exc
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


def is_dimensional_constraint(constraint: datalib.Constraint) -> bool:
    return constraint.constraint_type.name in DIMENSIONAL_CONSTRAINTS


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


def build_file_bundle(
    *,
    batch: str,
    filename: str,
    humans: Sequence[tuple[str, str]],
    file_maps: Mapping[str, Mapping[str, str]],
    converter: SketchFreecadConverter,
    geometry_tol: float,
    canonical_dir: str | None,
) -> FileBundle:
    data_dirs = {name: directory for name, directory in humans}
    annotators = tuple(data_dirs.keys())
    canonical_path = find_canonical_py(
        filename, data_dirs, annotators, canonical_dir=canonical_dir
    )
    canonical_sketch = load_sketch_from_fcstd(
        os.path.join(data_dirs[annotators[0]], file_maps[annotators[0]][filename]),
        converter,
    )
    # Prefer the canonical script when available; it may contain cleaner entity
    # ordering than any one FCStd annotation.
    try:
        from canonical_sketch import load_sketch_from_py  # local import for parity
        canonical_sketch = load_sketch_from_py(canonical_path, converter)
    except Exception:
        pass

    raw: dict[str, datalib.Sketch] = {}
    for name, directory in humans:
        raw[name] = load_sketch_from_fcstd(
            os.path.join(directory, file_maps[name][filename]), converter
        )

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
    )


def finite_or_none(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def safe_median(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if finite_or_none(v) is not None]
    return float(median(vals)) if vals else None


def norm_by_scale(value: float | None, scale: float | None, eps: float) -> float | None:
    if value is None or scale is None:
        return None
    if abs(scale) <= eps:
        return 0.0 if abs(value) <= eps else math.inf
    return value / scale


def stable_seed(*parts: object) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sorted_dimensional_ids(sketch: datalib.Sketch) -> list[str]:
    ids = [
        cid for cid, c in sketch.constraints.items()
        if is_dimensional_constraint(c)
    ]
    return sorted(ids, key=lambda cid: (constraint_signature(sketch.constraints[cid]), cid))


def dropout_variant(
    sketch: datalib.Sketch,
    *,
    drop_rate: float,
    seed: int,
) -> tuple[datalib.Sketch, int, int, int]:
    dim_ids = sorted_dimensional_ids(sketch)
    n_dim = len(dim_ids)
    n_drop = int(round(n_dim * drop_rate / 100.0))
    n_drop = max(0, min(n_dim, n_drop))

    if n_drop == 0:
        drop_ids: set[str] = set()
    elif n_drop == n_dim:
        drop_ids = set(dim_ids)
    else:
        rng = random.Random(seed)
        drop_ids = set(rng.sample(dim_ids, n_drop))

    out = copy.deepcopy(sketch)
    out.constraints = {
        cid: c
        for cid, c in out.constraints.items()
        if cid not in drop_ids
    }
    return out, n_dim, n_dim - n_drop, n_drop


def query_edit_probes(
    *,
    query_sketch: datalib.Sketch,
    ref_sketches: Sequence[datalib.Sketch],
    all_human_sketches: Sequence[datalib.Sketch],
    source: str,
    weighting: str,
) -> tuple[list[datalib.Constraint], list[float], str]:
    if source == "heldout":
        probes, weights = collect_probes([query_sketch], kind="dim")
    elif source == "ref_union":
        probes, weights = collect_probes(ref_sketches, kind="dim")
    elif source == "ref_intersection":
        probes, weights = collect_probes(ref_sketches, kind="dim")
        kept = [(p, w) for p, w in zip(probes, weights) if w >= 1.0 - 1e-12]
        probes = [p for p, _ in kept]
        weights = [w for _, w in kept]
    elif source == "all_humans_union":
        probes, weights = collect_probes(all_human_sketches, kind="dim")
    else:
        raise ValueError(source)

    if weighting == "uniform":
        weights = [1.0] * len(probes)
    elif weighting != "source_fraction":
        raise ValueError(weighting)
    return probes, weights, source


def response_distance(
    *,
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    edit_weights: Sequence[float],
    context: SketchContext,
    base_eids: Sequence[str] | None,
) -> tuple[float | None, int | None, int | None, int | None, str]:
    try:
        j_gt = j_pred = None
        if base_eids is not None:
            j_gt = pairwise_distance_observable(gt_sketch, base_eids=base_eids)
            j_pred = pairwise_distance_observable(pred_sketch, base_eids=base_eids)
        result = compute_edit_response_distance(
            gt_sketch=gt_sketch,
            pred_sketch=pred_sketch,
            edit_probes=edit_probes,
            context=context,
            probe_weights=edit_weights,
            observable_jacobian_gt=j_gt,
            observable_jacobian_pred=j_pred,
        )
        return (
            result.distance,
            result.n_dim_pred,
            result.n_dim_gt,
            result.skipped_probes,
            "",
        )
    except Exception as exc:
        return None, None, None, None, ("d_resp:" + str(exc))[:300]


def histcad_score(
    *,
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    context: SketchContext,
    converter: SketchFreecadConverter,
    config: HistCADConfig,
    solver: str,
) -> tuple[float | None, float | None, float | None, int | None, int | None, str]:
    try:
        if solver == "freecad":
            result = compute_histcad_2d_pair_freecad(
                pred_sketch=pred_sketch,
                gt_sketch=gt_sketch,
                edit_probes=edit_probes,
                context=context,
                converter=converter,
                config=config,
            )
        else:
            result = compute_histcad_2d_pair(
                pred_sketch=pred_sketch,
                gt_sketch=gt_sketch,
                edit_probes=edit_probes,
                context=context,
                config=config,
            )
        return (
            result.er_rate,
            result.cpcsr_rate,
            result.oes_rate,
            result.n_probes,
            result.n_relations,
            "",
        )
    except Exception as exc:
        return None, None, None, None, None, ("histcad:" + str(exc))[:300]


def edit_attainability_error(
    *,
    sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    edit_weights: Sequence[float],
    context: SketchContext,
    eps: float = 1e-8,
) -> tuple[float | None, int]:
    """Weighted mean ||J_q S_C^(q) - I|| over probes.

    This is the response-framework counterpart of edit reachability.  A sketch
    with no native dimensions has S=0, hence scalar probes contribute ~1 even
    though HistCAD can add a temporary dimension and solve the edit.
    """
    try:
        native = compute_native_response(sketch, context, eps=eps)
    except Exception:
        return None, len(edit_probes)

    total = 0.0
    total_w = 0.0
    skipped = 0
    for probe, weight in zip(edit_probes, edit_weights):
        j_q = constraint_jacobian(probe, native.state, context)
        s_q = edit_response_for_probe(native, probe, context, eps=eps)
        if j_q is None or s_q is None:
            skipped += 1
            continue
        gain = j_q @ s_q
        ident = torch.eye(gain.shape[0], dtype=gain.dtype, device=gain.device)
        total += float(weight) * float(torch.linalg.norm(gain - ident))
        total_w += float(weight)
    if total_w <= 0:
        return None, skipped
    return total / total_w, skipped


def pair_metrics(
    *,
    batch: str,
    filename: str,
    rotation_id: str,
    sample_id: int,
    drop_rate: float,
    pred_name: str,
    gt_name: str,
    pred_sketch: datalib.Sketch,
    gt_sketch: datalib.Sketch,
    edit_probes: Sequence[datalib.Constraint],
    edit_weights: Sequence[float],
    context: SketchContext,
    converter: SketchFreecadConverter,
    histcad_cfg: HistCADConfig,
    solver: str,
    base_eids: Sequence[str] | None,
    eps: float,
) -> dict:
    d_resp, n_dim_pred_native, n_dim_gt, skipped_resp, e1 = response_distance(
        pred_sketch=pred_sketch,
        gt_sketch=gt_sketch,
        edit_probes=edit_probes,
        edit_weights=edit_weights,
        context=context,
        base_eids=base_eids,
    )
    er, cpcsr, oes, n_hist_probes, n_hist_rels, e2 = histcad_score(
        pred_sketch=pred_sketch,
        gt_sketch=gt_sketch,
        edit_probes=edit_probes,
        context=context,
        converter=converter,
        config=histcad_cfg,
        solver=solver,
    )
    pred_attain, pred_attain_skipped = edit_attainability_error(
        sketch=pred_sketch,
        edit_probes=edit_probes,
        edit_weights=edit_weights,
        context=context,
        eps=eps,
    )
    gt_attain, gt_attain_skipped = edit_attainability_error(
        sketch=gt_sketch,
        edit_probes=edit_probes,
        edit_weights=edit_weights,
        context=context,
        eps=eps,
    )
    error = " ".join(e for e in (e1, e2) if e)
    return {
        "batch": batch,
        "filename": filename,
        "rotation_id": rotation_id,
        "sample_id": sample_id,
        "drop_rate": drop_rate,
        "pred": pred_name,
        "gt": gt_name,
        "d_resp": d_resp,
        "n_dim_pred_native": n_dim_pred_native,
        "n_dim_gt": n_dim_gt,
        "skipped_resp": skipped_resp,
        "histcad_er": er,
        "histcad_cpcsr": cpcsr,
        "histcad_oes": oes,
        "histcad_n_probes": n_hist_probes,
        "histcad_n_relations": n_hist_rels,
        "pred_attain_error": pred_attain,
        "pred_attain_skipped": pred_attain_skipped,
        "gt_attain_error": gt_attain,
        "gt_attain_skipped": gt_attain_skipped,
        "error": error,
    }


def summarize_variant(
    *,
    base: dict,
    detail_rows: Sequence[dict],
    h_scale: float | None,
    eps: float,
) -> dict:
    d_vals = [finite_or_none(r.get("d_resp")) for r in detail_rows]
    oes_vals = [finite_or_none(r.get("histcad_oes")) for r in detail_rows]
    er_vals = [finite_or_none(r.get("histcad_er")) for r in detail_rows]
    cpcsr_vals = [finite_or_none(r.get("histcad_cpcsr")) for r in detail_rows]
    pred_attain_vals = [finite_or_none(r.get("pred_attain_error")) for r in detail_rows]
    gt_attain_vals = [finite_or_none(r.get("gt_attain_error")) for r in detail_rows]
    best_d = min([v for v in d_vals if v is not None], default=None)
    max_oes = max([v for v in oes_vals if v is not None], default=None)
    row = dict(base)
    row.update({
        "best_d_resp": best_d,
        "norm_d_resp": norm_by_scale(best_d, h_scale, eps),
        "median_d_resp": safe_median(d_vals),
        "max_histcad_oes": max_oes,
        "median_histcad_oes": safe_median(oes_vals),
        "median_histcad_er": safe_median(er_vals),
        "median_histcad_cpcsr": safe_median(cpcsr_vals),
        "median_pred_attain_error": safe_median(pred_attain_vals),
        "median_gt_attain_error": safe_median(gt_attain_vals),
        "valid_pairs": sum(v is not None for v in d_vals),
        "h_scale_ordered_median": h_scale,
    })
    return row


def evaluate_rotation(
    *,
    bundle: FileBundle,
    refs: tuple[str, ...],
    heldout: str,
    drop_rates: Sequence[float],
    samples_per_rate: int,
    probe_source: str,
    probe_weighting: str,
    context: SketchContext,
    converter: SketchFreecadConverter,
    histcad_cfg: HistCADConfig,
    base_policy: str,
    solver: str,
    eps: float,
) -> tuple[list[dict], list[dict]]:
    rotation_id = f"{bundle.batch}:{bundle.filename}:{'+'.join(refs)}__query={heldout}"
    participants = set(refs) | {heldout}
    bad = []
    for name in participants:
        ok, reason = base_policy_pass(bundle.audits[name], base_policy)
        if not ok:
            bad.append(f"{name}:{reason}")
    if bad:
        return ([{
            "batch": bundle.batch,
            "filename": bundle.filename,
            "rotation_id": rotation_id,
            "refs": "|".join(refs),
            "heldout": heldout,
            "error": "human_audit:" + ",".join(bad),
        }], [])

    ref_sketches = [bundle.sketches[name] for name in refs]
    heldout_sketch = bundle.sketches[heldout]
    all_humans = [bundle.sketches[name] for name in participants]
    edit_probes, edit_weights, actual_probe_source = query_edit_probes(
        query_sketch=heldout_sketch,
        ref_sketches=ref_sketches,
        all_human_sketches=all_humans,
        source=probe_source,
        weighting=probe_weighting,
    )
    base_eids = build_base_eids(bundle.canonical_sketch, base_policy)

    # Human reference scale on the same fixed query probes.
    ref_pair_d: list[float | None] = []
    for pred in refs:
        for gt in refs:
            if pred == gt:
                continue
            d_resp, *_ = response_distance(
                pred_sketch=bundle.sketches[pred],
                gt_sketch=bundle.sketches[gt],
                edit_probes=edit_probes,
                edit_weights=edit_weights,
                context=context,
                base_eids=base_eids,
            )
            ref_pair_d.append(d_resp)
    h_scale = safe_median(ref_pair_d)

    dim_ids = sorted_dimensional_ids(heldout_sketch)
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    for drop_rate in drop_rates:
        sample_count = 1 if drop_rate in (0.0, 100.0) else samples_per_rate
        for sample_id in range(sample_count):
            seed = stable_seed(bundle.batch, bundle.filename, rotation_id, drop_rate, sample_id)
            pred_variant, n_dim_original, n_dim_kept, n_dim_dropped = dropout_variant(
                heldout_sketch,
                drop_rate=drop_rate,
                seed=seed,
            )
            base = {
                "batch": bundle.batch,
                "filename": bundle.filename,
                "rotation_id": rotation_id,
                "refs": "|".join(refs),
                "heldout": heldout,
                "sample_id": sample_id,
                "drop_rate": drop_rate,
                "keep_rate": 100.0 - drop_rate,
                "probe_source": actual_probe_source,
                "probe_weighting": probe_weighting,
                "n_edit_probes": len(edit_probes),
                "n_query_dim_original": len(dim_ids),
                "n_dim_original": n_dim_original,
                "n_dim_kept": n_dim_kept,
                "n_dim_dropped": n_dim_dropped,
                "solver": solver,
                "error": "",
            }
            variant_details: list[dict] = []
            for gt in refs:
                detail = pair_metrics(
                    batch=bundle.batch,
                    filename=bundle.filename,
                    rotation_id=rotation_id,
                    sample_id=sample_id,
                    drop_rate=drop_rate,
                    pred_name=f"{heldout}_drop{drop_rate:g}",
                    gt_name=gt,
                    pred_sketch=pred_variant,
                    gt_sketch=bundle.sketches[gt],
                    edit_probes=edit_probes,
                    edit_weights=edit_weights,
                    context=context,
                    converter=converter,
                    histcad_cfg=histcad_cfg,
                    solver=solver,
                    base_eids=base_eids,
                    eps=eps,
                )
                variant_details.append(detail)
                detail_rows.append(detail)
            summary = summarize_variant(
                base=base,
                detail_rows=variant_details,
                h_scale=h_scale,
                eps=eps,
            )
            if any(r.get("error") for r in variant_details):
                summary["error"] = " | ".join(
                    sorted({r["error"] for r in variant_details if r.get("error")})
                )[:300]
            summary_rows.append(summary)

    return summary_rows, detail_rows


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
    valid = [r for r in rows if not r.get("error")]
    print("\n--- Dimensional dropout ablation summary ---")
    print(f"rows: {len(rows)} | valid: {len(valid)}")
    by_rate: dict[float, list[dict]] = {}
    for row in valid:
        by_rate.setdefault(float(row["drop_rate"]), []).append(row)
    for rate in sorted(by_rate):
        rs = by_rate[rate]
        oes = [finite_or_none(r.get("max_histcad_oes")) for r in rs]
        resp = [finite_or_none(r.get("norm_d_resp")) for r in rs]
        er = [finite_or_none(r.get("median_histcad_er")) for r in rs]
        attain = [finite_or_none(r.get("median_pred_attain_error")) for r in rs]
        print(
            f"  drop={rate:5.1f}% n={len(rs):4d} "
            f"median_OES={safe_median(oes)} "
            f"median_ER={safe_median(er)} "
            f"median_norm_d_resp={safe_median(resp)} "
            f"median_attain_err={safe_median(attain)}"
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--batches", nargs="+", default=["0", "9"])
    p.add_argument("--human", action="append", default=None,
                   help="Override/add humans as BATCH:NAME:DIR. If omitted, defaults to batches 0 and 9.")
    p.add_argument("--ref-size", type=int, default=2)
    p.add_argument("--drop-rates", default="100,75,50,25,0",
                   help="Comma-separated percentages of heldout dimensional constraints to remove.")
    p.add_argument("--samples-per-rate", type=int, default=1,
                   help="Random dropout samples for intermediate rates. 0%% and 100%% are unique.")
    p.add_argument("--probe-source", default="ref_union",
                   choices=("heldout", "ref_union", "ref_intersection", "all_humans_union"))
    p.add_argument("--probe-weighting", default="uniform",
                   choices=("uniform", "source_fraction"))
    p.add_argument("--base-policy", default="canonical_solid",
                   choices=("none", "canonical_solid", "canonical_all"))
    p.add_argument("--geometry-tol", type=float, default=5e-1)
    p.add_argument("--canonical-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--solver", default="freecad", choices=("newton", "freecad"))
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
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--output-csv", default="artifacts/dimensional_dropout_ablation_scores.csv")
    p.add_argument("--details-csv", default="artifacts/dimensional_dropout_ablation_pair_details.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    humans_by_batch = parse_human_specs(args.human)
    drop_rates = parse_percentages(args.drop_rates)
    if args.samples_per_rate < 1:
        raise ValueError("--samples-per-rate must be >= 1")

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
            try:
                bundle = build_file_bundle(
                    batch=batch,
                    filename=filename,
                    humans=humans,
                    file_maps=file_maps,
                    converter=converter,
                    geometry_tol=args.geometry_tol,
                    canonical_dir=args.canonical_dir,
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
                    rows, details = evaluate_rotation(
                        bundle=bundle,
                        refs=tuple(refs),
                        heldout=heldout,
                        drop_rates=drop_rates,
                        samples_per_rate=args.samples_per_rate,
                        probe_source=args.probe_source,
                        probe_weighting=args.probe_weighting,
                        context=context,
                        converter=converter,
                        histcad_cfg=cfg,
                        base_policy=args.base_policy,
                        solver=args.solver,
                        eps=args.eps,
                    )
                    summary_rows.extend(rows)
                    detail_rows.extend(details)
                    n_rot += 1
            print(f"  [done] {batch}/{filename}: {n_rot} rotations")

        if args.limit is not None and n_files_seen >= args.limit:
            break

    print_summary(summary_rows)
    write_csv(args.output_csv, summary_rows)
    write_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
