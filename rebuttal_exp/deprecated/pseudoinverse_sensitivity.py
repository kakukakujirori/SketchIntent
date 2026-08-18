"""Experiment D: how much do the paper metrics depend on the minimum-norm response?

Reproduces the row structure of ``ablation/run_motion.py`` (the Sec. 5 table) and, for
every Pred side, replaces ``R0 = -pinv(J_x) J_p`` by admissible alternatives
``R0 + N Z``:

    native           the existing pair_paper_metrics path; the delta baseline and the gate
    main             free motion sized in observable space (rebuttal_experiments.md 4.5)
    observable_null  free motion drawn from ker([J_x; J_phi]) and sized in raw space

Only the Pred side is perturbed; GT keeps the current definition, so the variance has one
source. The two arms normalise eta in different spaces and are never pooled.

Outputs (see rebuttal_exp/README.md):
    run_config.json  environment.json  native_pair_metrics.parquet
    perturbation_metrics.parquet  response_metadata.csv  pair_null_visibility.csv
    skip_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from rebuttal_exp.perturbed_motion import PairEvaluator, pred_response_state  # noqa: E402
from rebuttal_exp.response_perturbation import (  # noqa: E402
    ARMS,
    CONSISTENCY_TOL,
    MAIN_ARM,
    OBSERVABLE_NULL_ARM,
    ResponseGeometry,
    build_response_geometry,
    perturbed_response,
    relation_visibility,
)
from src.eval.file_bundle import dropout_variant, stable_seed  # noqa: E402
from src.eval.motion_metrics import SketchMotion, pair_paper_metrics  # noqa: E402
from src.eval.progress import progress  # noqa: E402
from src.metrics.edit_response_distance import pairwise_distance_observable  # noqa: E402
from src.metrics.motion_subspace import orthonormal_range  # noqa: E402

torch.set_num_threads(1)
"""The per-sample matrices are small enough that intra-op threading costs more than it
saves, and a single core keeps this run out of the way of the training job."""

DEFAULT_ETAS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_DIM_RATES = (25.0, 50.0, 75.0, 100.0)
DEFAULT_REL_RATES = (50.0, 100.0)
DEFAULT_SAMPLES = 50
DEFAULT_NULL_SAMPLES = 20
BASELINE_ETA = 0.0

METRIC_COLUMNS = [
    "theta_full", "theta_rel", "theta_leak", "placement_sim",
    "k_full_pred", "k_full_gt", "k_rel_pred", "k_rel_gt", "k_response_pred",
    "n_dim_pred", "n_dim_gt",
]
RECORD_KEYS = ["batch", "sketch_id", "rid", "variant", "gt"]
NATIVE_COLUMNS = [*RECORD_KEYS, "pred_label", *METRIC_COLUMNS, "status"]
SAMPLE_COLUMNS = [*RECORD_KEYS, "arm", "eta", "sample", *METRIC_COLUMNS,
                  "raw_perturbation_ratio", "constraint_residual", "nullspace_residual"]
RESPONSE_COLUMNS = [
    "batch", "sketch_id", "label", "raw_null_dim", "observable_free_rank",
    "observable_null_dim", "consistency_residual", "r0_frobenius",
    "n_zero_observable_columns", "n_zero_raw_columns", "n_drivers",
    "n_x", "m_phi", "skip_reason",
]
VISIBILITY_COLUMNS = [
    "batch", "sketch_id", "pred_label", "gt_label", "observable_null_dim",
    "relation_null_overlap_norm", "relation_null_overlap_rank",
]


SAMPLE_SCHEMA = pa.schema(
    [(name, pa.string()) for name in RECORD_KEYS]
    + [("arm", pa.string()), ("eta", pa.float64()), ("sample", pa.int32())]
    + [(name, pa.float64() if name.startswith(("theta", "placement")) else pa.int32())
       for name in METRIC_COLUMNS]
    + [(name, pa.float64()) for name in
       ("raw_perturbation_ratio", "constraint_residual", "nullspace_residual")]
)
CHUNK_ROWS = 100_000


class SampleWriter:
    """Streams the per-sample rows to parquet; the full run does not fit in memory."""

    def __init__(self, path: Path):
        self.writer = pq.ParquetWriter(path, SAMPLE_SCHEMA, compression="zstd")
        self.buffer: list[dict] = []
        self.n_rows = 0

    def extend(self, rows: list[dict]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        frame = pd.DataFrame(self.buffer).reindex(columns=SAMPLE_SCHEMA.names)
        self.writer.write_table(pa.Table.from_pandas(frame, schema=SAMPLE_SCHEMA,
                                                     preserve_index=False))
        self.n_rows += len(self.buffer)
        self.buffer = []

    def close(self) -> None:
        self.flush()
        self.writer.close()


@dataclass
class Collectors:
    samples: SampleWriter
    native: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    visibility: list = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)


@dataclass
class PredCache:
    motion: SketchMotion
    j_phi: torch.Tensor
    geometry: ResponseGeometry
    fingerprint: tuple


def response_fingerprint(motion: SketchMotion) -> tuple:
    """Detects any in-place mutation of the quantities the sampling loop must not touch."""
    return (
        float(torch.linalg.norm(motion.native.R)) if motion.native.R.numel() else 0.0,
        float(torch.linalg.norm(motion.native.state.q)),
        float(torch.linalg.norm(motion.U_rel)) if motion.U_rel.numel() else 0.0,
        int(motion.k_rel),
    )


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


def plan_records(bundle, item, models, config) -> tuple[list[tuple], dict]:
    """Every (rid, variant, gt) row of the Sec. 5 table, and the sketch behind each label."""
    records: list[tuple] = []
    sketches: dict[str, object] = {}
    for refs in combinations(item.human_labels, 2):
        heldout = next(n for n in item.human_labels if n not in refs)
        rid, variants = build_variants(bundle, item, refs, heldout, models,
                                       config["dim_rates"], config["rel_rates"])
        for gt_label in refs:
            sketches[gt_label] = bundle.sketches[gt_label]
            records.append((rid, "source_self", gt_label, gt_label))
        for variant, label, sketch in variants:
            sketches[label] = sketch
            for gt_label in refs:
                records.append((rid, variant, label, gt_label))
    return records, sketches


def sample_plan(config) -> list[tuple[str, float, int]]:
    """(arm, eta, sample_index) triples; eta = 0 is drawn once and only under the main arm."""
    plan = [(MAIN_ARM, BASELINE_ETA, 0)]
    counts = {MAIN_ARM: config["n_samples"], OBSERVABLE_NULL_ARM: config["n_null_samples"]}
    for arm in config["arms"]:
        for eta in config["etas"]:
            if eta == BASELINE_ETA:
                continue
            plan.extend((arm, eta, index) for index in range(counts[arm]))
    return plan


def process_sketch(item, converter, context, config, collectors: Collectors) -> None:
    bundle = load_bundle(item, converter, config["geometry_tol"])
    humans, models = usable_labels(bundle, item)
    if not humans:
        collectors.skips["sketch_failed_base_policy"] += 1
        return

    base_eids = build_base_eids(bundle.canonical_sketch, BASE_POLICY)
    eps, rcond = config["eps"], config["rcond"]
    records, sketches = plan_records(bundle, item, models, config)

    cache: dict[str, PredCache] = {}
    for label, sketch in sketches.items():
        motion = SketchMotion(sketch, context, base_eids, eps=eps)
        j_phi = pairwise_distance_observable(sketch, base_eids=base_eids).to(
            dtype=motion.native.R.dtype
        )
        geometry = build_response_geometry(
            motion.native, j_phi, eps=eps, rcond=rcond,
            consistency_tol=config["consistency_tol"],
        )
        cache[label] = PredCache(motion, j_phi, geometry, response_fingerprint(motion))
        collectors.responses.append(dict(
            batch=item.batch, sketch_id=item.sketch_id, label=label,
            n_x=int(motion.native.J_x.shape[1]), m_phi=motion.m_phi,
            **geometry.as_record(),
        ))
        if geometry.skip_reason:
            collectors.skips[f"response:{geometry.skip_reason}"] += 1
        else:
            for arm in config["arms"]:
                reason = geometry.arm_skip_reason(arm)
                if reason:
                    collectors.skips[f"response:{arm}:{reason}"] += 1

    evaluators: dict[tuple[str, str], PairEvaluator] = {}
    for _rid, _variant, pred_label, gt_label in records:
        key = (pred_label, gt_label)
        if key in evaluators:
            continue
        evaluators[key] = PairEvaluator(cache[pred_label].motion, cache[gt_label].motion)
        collectors.visibility.append(dict(
            batch=item.batch, sketch_id=item.sketch_id,
            pred_label=pred_label, gt_label=gt_label,
            observable_null_dim=cache[pred_label].geometry.observable_null_dim,
            **relation_visibility(evaluators[key].J_R_at_pred, cache[pred_label].geometry.G,
                                  eps=eps, rcond=rcond),
        ))

    for rid, variant, pred_label, gt_label in records:
        metrics = pair_paper_metrics(cache[pred_label].motion, cache[gt_label].motion)
        row = dict(batch=item.batch, sketch_id=item.sketch_id, rid=rid,
                   variant=variant, gt=gt_label, pred_label=pred_label)
        row.update(
            status="metrics_unavailable" if metrics is None else "ok",
            k_response_pred=int(orthonormal_range(cache[pred_label].motion.native.R, eps)[1]),
        )
        if metrics is not None:
            row.update(metrics)
        collectors.native.append(row)

    by_pred: dict[str, list[tuple]] = {}
    for rid, variant, pred_label, gt_label in records:
        by_pred.setdefault(pred_label, []).append((rid, variant, gt_label))

    rows: list[dict] = []
    for arm, eta, index in sample_plan(config):
        for pred_label, targets in by_pred.items():
            entry = cache[pred_label]
            if entry.geometry.arm_skip_reason(arm):
                continue
            seed = stable_seed(config["seed"], item.sketch_id, pred_label, arm, eta, index)
            R_eta, diagnostics = perturbed_response(
                entry.geometry, arm, eta, seed, eps=eps, rcond=rcond
            )
            if diagnostics["constraint_residual"] > config["consistency_tol"]:
                collectors.skips[f"sample:{arm}:inconsistent_linear_system"] += 1
                continue
            state = pred_response_state(R_eta, entry.j_phi, eps)
            for rid, variant, gt_label in targets:
                metrics = evaluators[(pred_label, gt_label)].metrics(state)
                if metrics is None:
                    continue
                rows.append(dict(
                    batch=item.batch, sketch_id=item.sketch_id, rid=rid,
                    variant=variant, gt=gt_label, arm=arm, eta=eta, sample=index,
                    **metrics, **diagnostics,
                ))
    collectors.samples.extend(rows)

    for label, entry in cache.items():
        if response_fingerprint(entry.motion) != entry.fingerprint:
            raise AssertionError(f"{item.key}:{label}: the sampling loop mutated the base motion")


def build_config(args) -> dict:
    return dict(
        batches=args.batches,
        limit=args.limit,
        geometry_tol=args.geometry_tol,
        eps=args.eps,
        rcond=args.rcond,
        consistency_tol=args.consistency_tol,
        arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()),
        etas=tuple(sorted({float(e) for e in args.etas.split(",") if e})),
        n_samples=args.n_samples,
        n_null_samples=args.null_samples,
        seed=args.seed,
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
    parser.add_argument("--rcond", type=float, default=1e-6)
    parser.add_argument("--consistency-tol", type=float, default=CONSISTENCY_TOL)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--etas", default=",".join(f"{e:g}" for e in DEFAULT_ETAS))
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--null-samples", type=int, default=DEFAULT_NULL_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim-drop-rates", default=",".join(f"{r:g}" for r in DEFAULT_DIM_RATES))
    parser.add_argument("--rel-drop-rates", default=",".join(f"{r:g}" for r in DEFAULT_REL_RATES))
    parser.add_argument("--shard", default="0/1", help="i/n, split deterministically by sketch key")
    parser.add_argument("--out", default="artifacts/rebuttal/pseudoinverse")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    unknown = set(config["arms"]) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    if BASELINE_ETA not in config["etas"]:
        raise SystemExit("eta = 0 is the delta baseline and must be included")

    shard_index, n_shards = (int(part) for part in args.shard.split("/"))
    out_dir = Path(args.out)
    if n_shards > 1:
        out_dir = out_dir / f"shard_{shard_index:03d}_of_{n_shards:03d}"
    prepare_output_dir(out_dir, dict(config, shard=args.shard), force=args.force)
    write_environment(out_dir, sys.argv, Path(__file__).parent.glob("*.py"))

    items = select_shard(discover_items(args.batches, args.limit), shard_index, n_shards)
    context, converter = build_context()
    collectors = Collectors(samples=SampleWriter(out_dir / "perturbation_metrics.parquet"))

    try:
        for item in progress(items, desc="pseudoinverse", total=len(items)):
            try:
                process_sketch(item, converter, context, config, collectors)
            except AssertionError:
                raise
            except Exception as exc:  # noqa: BLE001
                collectors.skips[f"sketch_error:{type(exc).__name__}"] += 1
                print(f"  [skip] {item.key}: {str(exc)[:120]}")
    finally:
        collectors.samples.close()

    native = pd.DataFrame(collectors.native).reindex(columns=NATIVE_COLUMNS)
    native.to_parquet(out_dir / "native_pair_metrics.parquet", index=False)
    n_sample_rows = collectors.samples.n_rows
    write_csv(out_dir / "response_metadata.csv", collectors.responses, RESPONSE_COLUMNS)
    write_csv(out_dir / "pair_null_visibility.csv", collectors.visibility, VISIBILITY_COLUMNS)
    write_csv(
        out_dir / "skip_summary.csv",
        [dict(reason=reason, count=count) for reason, count in sorted(collectors.skips.items())],
        ["reason", "count"],
    )
    finalize_run_config(
        out_dir, n_sketches=len(items), n_native_rows=len(native), n_sample_rows=n_sample_rows,
        n_responses=len(collectors.responses), skips=dict(collectors.skips),
    )
    print(f"sketches={len(items)} native_rows={len(native)} sample_rows={n_sample_rows}")
    print(f"skips: {dict(collectors.skips)}")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
