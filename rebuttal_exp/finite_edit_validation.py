"""Experiment A: do the tangent-space metrics predict real solver behaviour under finite edits?

For every ordered pair of annotations (P, G) of the same sketch, every dimensional driver
they both contain is given the same signed finite increment on both sides, both sketches
are solved by FreeCAD from independently solved baselines, and the exact nonlinear
observable change of each side is compared. The pre-edit paper metrics are computed once
per pair and are never specialised to a driver.

Outputs (see rebuttal_exp/README.md):
    run_config.json  environment.json  pair_metrics.csv  driver_matches.csv
    baseline_solves.csv  trials.csv
"""
from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketchgraphs.data as datalib  # noqa: E402

from rebuttal_exp.common import (  # noqa: E402
    BASE_POLICY,
    build_base_eids,
    build_context,
    discover_items,
    finalize_run_config,
    load_bundle,
    prepare_output_dir,
    select_shard,
    usable_labels,
    write_csv,
    write_environment,
)
from rebuttal_exp.driver_matching import (  # noqa: E402
    DriverMatch,
    dimensional_drivers,
    match_common_drivers,
)
from rebuttal_exp.exact_observable import (  # noqa: E402
    ObservableLayout,
    build_observable_layout,
    evaluate_observable,
    layouts_match,
)
from rebuttal_exp.solved_sketch import constraint_residual_norm, solve_from  # noqa: E402
from src.eval.motion_metrics import SketchMotion, pair_paper_metrics  # noqa: E402
from src.eval.progress import progress  # noqa: E402
from src.metrics.histcad_2d import (  # noqa: E402
    HistCADConfig,
    _bbox_diag,
    _evaluate_observable,
    _set_quantity,
)
from src.sketch_handlers.sketch_solver import SketchState  # noqa: E402

DEFAULT_LENGTH_ALPHAS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
DEFAULT_ANGLE_DEGREES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SIGNS = (-1.0, 1.0)
D_FINITE_EPS = 1e-12
QUANTITY_RTOL = 1e-6
HUMAN_PAIR = "human"
MODEL_HUMAN_PAIR = "model_human"

PAIR_METRIC_COLUMNS = [
    "batch", "sketch_id", "pair_id", "pair_kind", "annotation_p", "annotation_g",
    "theta_full", "theta_rel", "theta_leak", "placement_sim", "placement_distance",
    "asgn_min", "set_sim", "omp_sim",
    "k_full_pred", "k_full_gt", "k_rel_pred", "k_rel_gt", "n_dim_pred", "n_dim_gt",
    "n_common_drivers", "status",
]
DRIVER_MATCH_COLUMNS = [
    "batch", "sketch_id", "pair_id", "pair_kind", "annotation_p", "annotation_g",
    "driver_signature", "driver_type", "magnitude_kind",
    "driver_index_p", "driver_index_g", "q0_p", "q0_g",
    "driver_residual_p", "driver_residual_g",
]
BASELINE_COLUMNS = [
    "batch", "sketch_id", "label", "label_kind", "status", "solve_ok",
    "observable_dim", "residual", "bbox_diag", "elapsed_s", "n_drivers",
]
TRIAL_COLUMNS = [
    "trial_id", "batch", "sketch_id", "pair_id", "pair_kind",
    "annotation_p", "annotation_g",
    "driver_signature", "driver_type", "driver_index_p", "driver_index_g",
    "sign", "magnitude_kind", "magnitude_value", "delta_deg", "delta_internal",
    "q0_p", "q0_g", "driver_residual_p", "driver_residual_g", "target_p", "target_g",
    "baseline_ok_p", "baseline_ok_g", "solve_ok_p", "solve_ok_g",
    "status_p", "status_g", "residual_p", "residual_g",
    "achieved_p", "achieved_g", "target_miss_p", "target_miss_g",
    "response_norm_p", "response_norm_g", "d_finite",
    "theta_full", "theta_rel", "theta_leak", "placement_distance",
    "skip_reason",
]


@dataclass
class Collectors:
    pairs: list = field(default_factory=list)
    driver_matches: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)


@dataclass
class Baseline:
    label: str
    sketch: datalib.Sketch
    state: SketchState
    layout: ObservableLayout
    phi: np.ndarray
    residual: float
    drivers: dict
    status: str
    elapsed_s: float


@dataclass(frozen=True)
class EditOutcome:
    status: str
    solve_ok: bool
    achieved: float | None = None
    target_miss: float | None = None
    residual: float | None = None
    response: np.ndarray | None = None

    @property
    def response_norm(self) -> float | None:
        return float(np.linalg.norm(self.response)) if self.response is not None else None


def prepare_baseline(label: str, sketch, base_eids, converter, context) -> Baseline | str:
    outcome = solve_from(sketch, converter)
    if outcome.sketch is None:
        return outcome.status
    solved = outcome.sketch
    layout = build_observable_layout(solved, base_eids)
    if layout.dimension == 0:
        return "empty_observable"
    state = SketchState(solved)
    drivers, _skips = dimensional_drivers(solved, state, context)
    return Baseline(
        label=label,
        sketch=solved,
        state=state,
        layout=layout,
        phi=evaluate_observable(solved, layout),
        residual=constraint_residual_norm(solved, state, context),
        drivers=drivers,
        status=outcome.status,
        elapsed_s=outcome.elapsed_s,
    )


def apply_finite_edit(
    baseline: Baseline, cid: str, target: float, bbox: float, converter, context, target_tol: float
) -> EditOutcome:
    edited = copy.deepcopy(baseline.sketch)
    edited.constraints[cid] = _set_quantity(edited.constraints[cid], target)
    outcome = solve_from(edited, converter)
    if outcome.sketch is None:
        return EditOutcome(outcome.status, False)

    solved = outcome.sketch
    state = SketchState(solved)
    achieved = _evaluate_observable(solved.constraints[cid], state, context)
    if achieved is None:
        return EditOutcome("observable_eval_failed", False)

    miss = abs(achieved - target) / max(bbox, 1e-12)
    residual = constraint_residual_norm(solved, state, context)
    response = evaluate_observable(solved, baseline.layout) - baseline.phi
    status = outcome.status if miss <= target_tol else f"target_miss:{miss:.3g}"
    return EditOutcome(status, miss <= target_tol, achieved, miss, residual, response)


def finite_discrepancy(response_p: np.ndarray, response_g: np.ndarray) -> float:
    numerator = float(np.linalg.norm(response_p - response_g))
    denominator = (
        float(np.linalg.norm(response_p)) + float(np.linalg.norm(response_g)) + D_FINITE_EPS
    )
    return numerator / denominator


def magnitude_grid(match: DriverMatch, bbox: float, length_alphas, angle_degrees):
    """(magnitude_kind, magnitude_value, delta_deg, |delta| in internal units) per grid point."""
    if match.kind == "angle":
        for degrees in angle_degrees:
            yield "angle", degrees, degrees, math.radians(degrees)
    else:
        for alpha in length_alphas:
            yield "length", alpha, None, alpha * bbox


def directed_pairs(humans: list[str], models: list[str], pair_kinds) -> list[tuple[str, str, str]]:
    pairs = []
    for p, g in itertools.permutations(humans + models, 2):
        p_is_human, g_is_human = p in humans, g in humans
        if p_is_human and g_is_human:
            kind = HUMAN_PAIR
        elif p_is_human or g_is_human:
            kind = MODEL_HUMAN_PAIR
        else:
            continue
        if kind in pair_kinds:
            pairs.append((p, g, kind))
    return pairs


def pair_metric_rows(item, bundle, pairs, base_eids, context, eps) -> dict:
    """Directed pair metrics from the bundle sketches, i.e. the state the paper reports."""
    motions: dict[str, SketchMotion | None] = {}

    def motion(label: str):
        if label not in motions:
            try:
                motions[label] = SketchMotion(bundle.sketches[label], context, base_eids, eps=eps)
            except Exception as exc:  # noqa: BLE001
                motions[label] = None
                print(f"    [motion-fail] {item.key}:{label}: {str(exc)[:80]}")
        return motions[label]

    rows = {}
    for p, g, kind in pairs:
        mp, mg = motion(p), motion(g)
        metrics = pair_paper_metrics(mp, mg) if (mp and mg) else None
        row = dict(
            batch=item.batch, sketch_id=item.sketch_id, pair_id=f"{p}->{g}", pair_kind=kind,
            annotation_p=p, annotation_g=g, n_common_drivers=0,
            status="ok" if metrics else "metrics_unavailable",
        )
        if metrics:
            row.update(metrics)
            row["placement_distance"] = 1.0 - float(metrics["placement_sim"])
        rows[(p, g)] = row
    return rows


def process_sketch(item, converter, context, config, collectors: Collectors) -> None:
    bundle = load_bundle(item, converter, config["geometry_tol"])
    humans, models = usable_labels(bundle, item)
    if not humans:
        collectors.skips["sketch_failed_base_policy"] += 1
        return

    pairs = directed_pairs(humans, models, config["pair_kinds"])
    if not pairs:
        return
    labels = sorted({label for p, g, _ in pairs for label in (p, g)})
    base_eids = build_base_eids(bundle.canonical_sketch, BASE_POLICY)
    bbox = _bbox_diag(bundle.canonical_sketch)

    baselines = _prepare_baselines(item, bundle, labels, humans, base_eids, bbox,
                                   converter, context, collectors)
    rows = pair_metric_rows(item, bundle, pairs, base_eids, context, config["eps"])
    edit_cache: dict[tuple, EditOutcome] = {}

    for p, g, _kind in pairs:
        pair_row = rows[(p, g)]
        collectors.pairs.append(pair_row)
        if p not in baselines or g not in baselines:
            pair_row["status"] = "baseline_solve_failure"
            continue
        side_p, side_g = baselines[p], baselines[g]
        if not layouts_match(side_p.layout, side_g.layout):
            pair_row["status"] = "observable_layout_mismatch"
            collectors.skips["observable_layout_mismatch"] += 1
            continue

        matches, skips = match_common_drivers(
            side_p.drivers, side_g.drivers, quantity_rtol=QUANTITY_RTOL
        )
        collectors.skips.update(skips)
        pair_row["n_common_drivers"] = len(matches)

        for match in matches:
            collectors.driver_matches.append(dict(
                batch=item.batch, sketch_id=item.sketch_id, pair_id=pair_row["pair_id"],
                pair_kind=pair_row["pair_kind"], annotation_p=p, annotation_g=g,
                driver_signature=repr(match.signature), driver_type=match.driver_type,
                magnitude_kind=match.kind,
                driver_index_p=match.slot_p.cid, driver_index_g=match.slot_g.cid,
                q0_p=match.slot_p.q0, q0_g=match.slot_g.q0,
                driver_residual_p=match.slot_p.baseline_residual,
                driver_residual_g=match.slot_g.baseline_residual,
            ))
            run_driver_trials(item, pair_row, match, side_p, side_g, bbox,
                              converter, context, config, edit_cache, collectors)


def _prepare_baselines(item, bundle, labels, humans, base_eids, bbox, converter, context, collectors):
    baselines: dict[str, Baseline] = {}
    for label in labels:
        prepared = prepare_baseline(label, bundle.sketches[label], base_eids, converter, context)
        label_kind = "human" if label in humans else "model"
        if isinstance(prepared, str):
            collectors.baselines.append(dict(
                batch=item.batch, sketch_id=item.sketch_id, label=label, label_kind=label_kind,
                status=prepared, solve_ok=False, bbox_diag=bbox,
            ))
            collectors.skips["baseline_solve_failure"] += 1
            continue
        baselines[label] = prepared
        collectors.baselines.append(dict(
            batch=item.batch, sketch_id=item.sketch_id, label=label, label_kind=label_kind,
            status=prepared.status, solve_ok=True, observable_dim=prepared.layout.dimension,
            residual=prepared.residual, bbox_diag=bbox, elapsed_s=round(prepared.elapsed_s, 4),
            n_drivers=len(prepared.drivers),
        ))
    return baselines


def run_driver_trials(
    item, pair_row, match, side_p, side_g, bbox, converter, context, config, edit_cache, collectors
):
    for sign in SIGNS:
        for kind, value, delta_deg, delta_magnitude in magnitude_grid(
            match, bbox, config["length_alphas"], config["angle_degrees"]
        ):
            delta = sign * delta_magnitude
            target_p = match.slot_p.q0 + delta
            target_g = match.slot_g.q0 + delta
            row = dict(
                trial_id=f"{item.key}|{pair_row['pair_id']}|{match.signature!r}"
                         f"|{sign:+.0f}|{kind}:{value:g}",
                batch=item.batch, sketch_id=item.sketch_id, pair_id=pair_row["pair_id"],
                pair_kind=pair_row["pair_kind"],
                annotation_p=pair_row["annotation_p"], annotation_g=pair_row["annotation_g"],
                driver_signature=repr(match.signature), driver_type=match.driver_type,
                driver_index_p=match.slot_p.cid, driver_index_g=match.slot_g.cid,
                sign=int(sign), magnitude_kind=kind, magnitude_value=value,
                delta_deg=_number(sign * delta_deg if delta_deg is not None else None),
                delta_internal=delta,
                q0_p=match.slot_p.q0, q0_g=match.slot_g.q0,
                driver_residual_p=match.slot_p.baseline_residual,
                driver_residual_g=match.slot_g.baseline_residual,
                target_p=target_p, target_g=target_g,
                baseline_ok_p=True, baseline_ok_g=True,
                theta_full=pair_row.get("theta_full", ""),
                theta_rel=pair_row.get("theta_rel", ""),
                theta_leak=pair_row.get("theta_leak", ""),
                placement_distance=pair_row.get("placement_distance", ""),
                skip_reason="",
            )

            if match.requires_positive_target and min(target_p, target_g) <= 0.0:
                row.update(skip_reason="invalid_target_domain",
                           solve_ok_p=False, solve_ok_g=False,
                           status_p="invalid_target_domain", status_g="invalid_target_domain")
                collectors.trials.append(row)
                continue

            outcome_p = cached_edit(edit_cache, side_p, match.slot_p.cid, target_p, bbox,
                                    converter, context, config)
            outcome_g = cached_edit(edit_cache, side_g, match.slot_g.cid, target_g, bbox,
                                    converter, context, config)
            row.update(
                solve_ok_p=outcome_p.solve_ok, solve_ok_g=outcome_g.solve_ok,
                status_p=outcome_p.status, status_g=outcome_g.status,
                residual_p=_number(outcome_p.residual), residual_g=_number(outcome_g.residual),
                achieved_p=_number(outcome_p.achieved), achieved_g=_number(outcome_g.achieved),
                target_miss_p=_number(outcome_p.target_miss),
                target_miss_g=_number(outcome_g.target_miss),
                response_norm_p=_number(outcome_p.response_norm),
                response_norm_g=_number(outcome_g.response_norm),
            )
            if outcome_p.solve_ok and outcome_g.solve_ok:
                row["d_finite"] = finite_discrepancy(outcome_p.response, outcome_g.response)
            else:
                row["skip_reason"] = "solve_failure"
            collectors.trials.append(row)


def cached_edit(cache, side: Baseline, cid: str, target: float, bbox, converter, context, config):
    key = (side.label, cid, f"{target:.17g}")
    if key not in cache:
        cache[key] = apply_finite_edit(
            side, cid, target, bbox, converter, context, config["target_tol"]
        )
    return cache[key]


def _number(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def build_config(args) -> dict:
    return dict(
        batches=args.batches,
        limit=args.limit,
        geometry_tol=args.geometry_tol,
        eps=args.eps,
        pair_kinds=tuple(sorted(k.strip() for k in args.pair_kinds.split(",") if k.strip())),
        length_alphas=tuple(float(a) for a in args.length_alphas.split(",") if a),
        angle_degrees=tuple(float(d) for d in args.angle_degrees.split(",") if d),
        signs=SIGNS,
        base_policy=BASE_POLICY,
        target_tol=HistCADConfig().er_target_tol,
        quantity_rtol=QUANTITY_RTOL,
        d_finite_eps=D_FINITE_EPS,
        delta_length_basis="canonical_bbox_diag",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--pair-kinds", default=f"{HUMAN_PAIR},{MODEL_HUMAN_PAIR}")
    parser.add_argument("--length-alphas", default=",".join(str(a) for a in DEFAULT_LENGTH_ALPHAS))
    parser.add_argument("--angle-degrees", default=",".join(str(d) for d in DEFAULT_ANGLE_DEGREES))
    parser.add_argument("--shard", default="0/1", help="i/n, split deterministically by sketch key")
    parser.add_argument("--out", default="artifacts/rebuttal/finite_edit")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shard_index, n_shards = (int(part) for part in args.shard.split("/"))
    config = build_config(args)
    out_dir = Path(args.out)
    if n_shards > 1:
        out_dir = out_dir / f"shard_{shard_index:03d}_of_{n_shards:03d}"
    prepare_output_dir(out_dir, dict(config, shard=args.shard), force=args.force)
    # Hash only the modules that generate the finite-edit trials.  In particular,
    # do not include the downstream analyser: changing a plot or aggregation must
    # not make the recorded generation digest disagree with otherwise unchanged
    # trial-generation code.
    generation_scripts = [
        Path(__file__),
        Path(__file__).with_name("common.py"),
        Path(__file__).with_name("driver_matching.py"),
        Path(__file__).with_name("exact_observable.py"),
        Path(__file__).with_name("solved_sketch.py"),
    ]
    write_environment(out_dir, sys.argv, generation_scripts)

    items = select_shard(discover_items(args.batches, args.limit), shard_index, n_shards)
    context, converter = build_context()
    collectors = Collectors()

    for item in progress(items, desc="finite edit", total=len(items)):
        try:
            process_sketch(item, converter, context, config, collectors)
        except Exception as exc:  # noqa: BLE001
            collectors.skips[f"sketch_error:{type(exc).__name__}"] += 1
            print(f"  [skip] {item.key}: {str(exc)[:120]}")

    trial_ids = [row["trial_id"] for row in collectors.trials]
    assert len(trial_ids) == len(set(trial_ids)), "trial ids are not unique"

    write_csv(out_dir / "pair_metrics.csv", collectors.pairs, PAIR_METRIC_COLUMNS)
    write_csv(out_dir / "driver_matches.csv", collectors.driver_matches, DRIVER_MATCH_COLUMNS)
    write_csv(out_dir / "baseline_solves.csv", collectors.baselines, BASELINE_COLUMNS)
    write_csv(out_dir / "trials.csv", collectors.trials, TRIAL_COLUMNS)
    finalize_run_config(
        out_dir,
        n_sketches=len(items),
        n_pairs=len(collectors.pairs),
        n_driver_matches=len(collectors.driver_matches),
        n_trials=len(collectors.trials),
        skips=dict(collectors.skips),
    )
    print(f"sketches={len(items)} pairs={len(collectors.pairs)} "
          f"drivers={len(collectors.driver_matches)} trials={len(collectors.trials)}")
    print(f"skips: {dict(collectors.skips)}")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
