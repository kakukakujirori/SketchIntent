"""Experiment C: are the paper metrics stable under an alternative primitive chart?

Reproduces the row structure of ``ablation/run_motion.py`` (the Sec. 5 table) and
recomputes every (variant, reference) record once per scenario:

    native  the existing pair_paper_metrics path
    C0      this pipeline with B = I, an integration test against native
    C1      Line  6 -> (start point, end point)          4
    C2      Arc   5 -> (start, sweep midpoint, end)      6

Dropout variants are drawn with the same ``stable_seed`` as the ablation driver, so every
scenario sees the identical variant sketches and the comparison is paired.

Outputs (see rebuttal_exp/README.md):
    run_config.json  environment.json  transformed_pair_metrics.parquet
    transform_diagnostics.parquet  skip_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketchgraphs.data as datalib  # noqa: E402

from ablation.controls import relation_dropout_variant  # noqa: E402
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
from rebuttal_exp.primitive_charts import (  # noqa: E402
    C0,
    C1,
    C2,
    NATIVE,
    SCENARIOS,
    GlobalChart,
    build_global_chart,
    identity_chart,
)
from rebuttal_exp.transformed_motion import TransformedMotion, transformed_pair_metrics  # noqa: E402
from src.eval.file_bundle import dropout_variant, stable_seed  # noqa: E402
from src.eval.motion_metrics import SketchMotion, pair_paper_metrics  # noqa: E402
from src.eval.progress import progress  # noqa: E402
from src.metrics.histcad_2d import _bbox_diag  # noqa: E402
from src.metrics.motion_subspace import orthonormal_range  # noqa: E402
from src.sketch_handlers.sketch_solver import SketchState  # noqa: E402

DEFAULT_SCENARIOS = (NATIVE, C1, C2)
DEFAULT_DIM_RATES = (25.0, 50.0, 75.0, 100.0)
DEFAULT_REL_RATES = (50.0, 100.0)
REQUIRED_PRIMITIVE = {C1: datalib.Line, C2: datalib.Arc}

METRIC_COLUMNS = [
    "theta_full", "theta_rel", "theta_leak",
    "placement_sim", "asgn_min", "set_sim", "omp_sim",
    "k_full_pred", "k_full_gt", "k_rel_pred", "k_rel_gt",
    "k_response_pred", "k_response_gt",
    "n_dim_pred", "n_dim_gt", "n_y_pred", "n_y_gt",
]
PAIR_COLUMNS = ["batch", "filename", "sketch_id", "rid", "variant", "gt", "scenario",
                *METRIC_COLUMNS, "status"]
DIAGNOSTIC_COLUMNS = [
    "batch", "sketch_id", "label", "scenario", "eid", "entity_type", "skip_reason",
    "rank_H", "rank_B", "expected_rank", "left_inverse", "identity_error",
    "reconstruction_error", "raw_state_reconstruction_error",
    "angle_reconstruction_error", "point_reconstruction_error",
    "span", "radius", "circumcircle_condition",
    "angle_branch_start", "angle_branch_end", "y_redundancy_dim",
    "gauge_residual_J_x", "gauge_residual_J_R", "gauge_residual_J_phi",
    "n_x", "n_y", "motion_skip_reason",
]


@dataclass
class Collectors:
    pairs: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)


def scenario_applies(scenario: str, sketch: datalib.Sketch) -> bool:
    required = REQUIRED_PRIMITIVE.get(scenario)
    if required is None:
        return True
    return any(isinstance(e, required) for e in sketch.entities.values())


def chart_for(sketch: datalib.Sketch, scenario: str, scale: float):
    state = SketchState(sketch)
    if scenario == C0:
        return identity_chart(state)
    return build_global_chart(sketch, state, scenario, scale)


def response_rank(motion) -> int:
    """Rank of the raw dimensional-response map actually used by theta_leak.

    ``SketchMotion`` stores the native response as ``native.R``.  The transformed
    pipeline caches the corresponding rank of ``R_t`` because that is the response
    whose parameterization dependence Experiment C measures.
    """
    if hasattr(motion, "k_response"):
        return int(motion.k_response)
    return int(orthonormal_range(motion.native.R, motion.eps)[1])


def build_variants(bundle, item, refs, heldout, model_labels, dim_rates, rel_rates):
    """The Sec. 5 variant set, seeded exactly like ablation/run_motion.py."""
    rid = f"{item.batch}:{item.base_fcstd}:{'+'.join(refs)}__q={heldout}"
    variants = [("heldout", heldout, bundle.sketches[heldout])]
    for rate in dim_rates:
        seed = stable_seed(item.batch, item.base_fcstd, rid, "dim", rate)
        pred, *_ = dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=seed)
        variants.append((f"dim_drop{rate:g}", f"{rid}:d{rate:g}", pred))
    for rate in rel_rates:
        seed = stable_seed(item.batch, item.base_fcstd, rid, "rel", rate)
        pred, *_ = relation_dropout_variant(bundle.sketches[heldout], drop_rate=rate, seed=seed)
        variants.append((f"rel_drop{rate:g}", f"{rid}:r{rate:g}", pred))
    for label in model_labels:
        variants.append((label, label, bundle.sketches[label]))
    return rid, variants


def process_sketch(item, converter, context, config, collectors: Collectors) -> None:
    bundle = load_bundle(item, converter, config["geometry_tol"])
    humans, models = usable_labels(bundle, item)
    if not humans:
        collectors.skips["sketch_failed_base_policy"] += 1
        return

    base_eids = build_base_eids(bundle.canonical_sketch, BASE_POLICY)
    scale = max(_bbox_diag(bundle.canonical_sketch), 1.0)
    eps = config["eps"]
    scenarios = [s for s in config["scenarios"]
                 if scenario_applies(s, bundle.canonical_sketch)]
    for skipped in set(config["scenarios"]) - set(scenarios):
        collectors.skips[f"{skipped}:sketch_has_no_target_primitive"] += 1

    charts: dict[tuple[str, str], GlobalChart] = {}
    motions: dict[tuple[str, str], object] = {}
    rejected: dict[tuple[str, str], TransformedMotion] = {}

    def motion(scenario: str, label: str, sketch):
        key = (scenario, label)
        if key in motions:
            return motions[key]
        try:
            if scenario == NATIVE:
                motions[key] = SketchMotion(sketch, context, base_eids, eps=eps)
            else:
                if key not in charts:
                    charts[key] = chart_for(sketch, scenario, scale)
                built = TransformedMotion(sketch, context, base_eids, charts[key], eps=eps)
                motions[key] = built if built.usable else None
                if not built.usable:
                    rejected[key] = built
                    collectors.skips[f"{scenario}:{built.skip_reason}"] += 1
        except Exception as exc:  # noqa: BLE001
            motions[key] = None
            collectors.skips[f"{scenario}:motion_error:{type(exc).__name__}"] += 1
            print(f"    [motion-fail] {item.key}:{scenario}:{label}: {str(exc)[:80]}")
        return motions[key]

    def emit(scenario, rid, variant, gt_label, pred_motion, gt_motion):
        row = dict(batch=item.batch, filename=item.base_fcstd, sketch_id=item.sketch_id,
                   rid=rid, variant=variant, gt=gt_label, scenario=scenario)
        if pred_motion is None or gt_motion is None:
            row["status"] = "motion_unavailable"
        else:
            metrics = (pair_paper_metrics if scenario == NATIVE else transformed_pair_metrics)(
                pred_motion, gt_motion
            )
            if metrics is None:
                row["status"] = "metrics_unavailable"
            else:
                row.update(metrics)
                row.update(
                    k_response_pred=response_rank(pred_motion),
                    k_response_gt=response_rank(gt_motion),
                )
                row["status"] = "ok"
        collectors.pairs.append(row)

    names = item.human_labels
    for refs in combinations(names, 2):
        heldout = next(n for n in names if n not in refs)
        rid, variants = build_variants(bundle, item, refs, heldout, models,
                                       config["dim_rates"], config["rel_rates"])
        for scenario in scenarios:
            for gt_label in refs:
                gt_motion = motion(scenario, gt_label, bundle.sketches[gt_label])
                emit(scenario, rid, "source_self", gt_label, gt_motion, gt_motion)
            for variant, label, sketch in variants:
                pred_motion = motion(scenario, label, sketch)
                for gt_label in refs:
                    emit(scenario, rid, variant, gt_label,
                         pred_motion, motion(scenario, gt_label, bundle.sketches[gt_label]))

    _record_diagnostics(item, bundle, humans, models, scenarios, scale,
                        {**rejected, **{k: v for k, v in motions.items() if v is not None}},
                        charts, collectors)


def _record_diagnostics(item, bundle, humans, models, scenarios, scale, motions,
                        charts, collectors) -> None:
    """Block-level chart diagnostics, once per geometry source rather than per variant."""
    for scenario in scenarios:
        if scenario == NATIVE:
            continue
        for label in list(humans) + list(models):
            sketch = bundle.sketches[label]
            chart = charts.get((scenario, label)) or chart_for(sketch, scenario, scale)
            built = motions.get((scenario, label))
            residuals = getattr(built, "gauge_residuals", {}) or {}
            common = dict(batch=item.batch, sketch_id=item.sketch_id, label=label,
                          scenario=scenario, n_x=chart.n_x, n_y=chart.n_y,
                          motion_skip_reason=getattr(built, "skip_reason", "")
                          if built is not None else "motion_unavailable",
                          **residuals)
            for block in chart.blocks:
                if block.entity_type not in ("Line", "Arc") or not block.diagnostics:
                    continue
                collectors.diagnostics.append(dict(
                    common, eid=block.eid, entity_type=block.entity_type,
                    skip_reason=block.skip_reason, **block.diagnostics,
                ))


def build_config(args) -> dict:
    return dict(
        batches=args.batches,
        limit=args.limit,
        geometry_tol=args.geometry_tol,
        eps=args.eps,
        scenarios=tuple(s.strip() for s in args.scenarios.split(",") if s.strip()),
        dim_rates=tuple(float(r) for r in args.dim_drop_rates.split(",") if r),
        rel_rates=tuple(float(r) for r in args.rel_drop_rates.split(",") if r),
        base_policy=BASE_POLICY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--dim-drop-rates", default=",".join(f"{r:g}" for r in DEFAULT_DIM_RATES))
    parser.add_argument("--rel-drop-rates", default=",".join(f"{r:g}" for r in DEFAULT_REL_RATES))
    parser.add_argument("--shard", default="0/1", help="i/n, split deterministically by sketch key")
    parser.add_argument("--out", default="artifacts/rebuttal/parameterization")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    unknown = set(config["scenarios"]) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"unknown scenarios: {sorted(unknown)}")
    if NATIVE not in config["scenarios"]:
        raise SystemExit("the native scenario is the delta baseline and must be included")

    shard_index, n_shards = (int(part) for part in args.shard.split("/"))
    out_dir = Path(args.out)
    if n_shards > 1:
        out_dir = out_dir / f"shard_{shard_index:03d}_of_{n_shards:03d}"
    prepare_output_dir(out_dir, dict(config, shard=args.shard), force=args.force)
    write_environment(out_dir, sys.argv, Path(__file__).parent.glob("*.py"))

    items = select_shard(discover_items(args.batches, args.limit), shard_index, n_shards)
    context, converter = build_context()
    collectors = Collectors()

    for item in progress(items, desc="parameterization", total=len(items)):
        try:
            process_sketch(item, converter, context, config, collectors)
        except Exception as exc:  # noqa: BLE001
            collectors.skips[f"sketch_error:{type(exc).__name__}"] += 1
            print(f"  [skip] {item.key}: {str(exc)[:120]}")

    pairs = pd.DataFrame(collectors.pairs).reindex(columns=PAIR_COLUMNS)
    pairs.to_parquet(out_dir / "transformed_pair_metrics.parquet", index=False)
    diagnostics = pd.DataFrame(collectors.diagnostics).reindex(columns=DIAGNOSTIC_COLUMNS)
    diagnostics.to_parquet(out_dir / "transform_diagnostics.parquet", index=False)
    write_csv(
        out_dir / "skip_summary.csv",
        [dict(reason=reason, count=count) for reason, count in sorted(collectors.skips.items())],
        ["reason", "count"],
    )
    finalize_run_config(out_dir, n_sketches=len(items), n_pair_rows=len(pairs),
                        n_diagnostic_rows=len(diagnostics), skips=dict(collectors.skips))
    print(f"sketches={len(items)} pair_rows={len(pairs)} diagnostic_rows={len(diagnostics)}")
    print(f"status: {pairs['status'].value_counts().to_dict()}")
    print(f"skips: {dict(collectors.skips)}")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
