"""Experiment D: how much do the paper metrics depend on the minimum-norm response?

Reproduces the row structure of ``ablation/run_motion.py`` (the Sec. 5 table) and, for
every Pred side, replaces ``R0 = -pinv(J_x) J_p`` by admissible alternatives ``R0 + N Z``
drawn as described in ``response_perturbation``: one uniform direction per driver column,
sized so that its observable displacement is exactly ``eta`` times that column's baseline
observable response, with draws below the observable-gain floor discarded.

Only the Pred side is perturbed, so the variance has one source; GT keeps the current
definition. Responses whose whole null space is invisible to J_phi are recorded with
``exactly_invariant`` rather than sampled: for them ``J_phi R`` does not depend on the
choice at all.

Outputs (see rebuttal_exp/README.md):
    run_config.json  environment.json  native_pair_metrics.parquet
    perturbation_metrics.parquet  response_metadata.csv  skip_summary.csv
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
    CONSISTENCY_TOL,
    OBSERVABLE_GAIN_FLOOR,
    ResponseGeometry,
    build_response_geometry,
    perturbed_response,
)
from rebuttal_exp.variants import build_variants  # noqa: E402
from src.eval.file_bundle import stable_seed  # noqa: E402
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
BASELINE_ETA = 0.0

HUMAN_KIND = "human"
MODEL_KIND = "model"
DROPOUT_KIND = "dropout_control"

METRIC_COLUMNS = [
    "theta_full", "theta_rel", "theta_leak", "placement_sim",
    "k_full_pred", "k_full_gt", "k_rel_pred", "k_rel_gt", "k_response_pred",
    "n_dim_pred", "n_dim_gt",
]
RECORD_KEYS = ["batch", "sketch_id", "rid", "variant", "gt"]
NATIVE_COLUMNS = [*RECORD_KEYS, "pred_label", *METRIC_COLUMNS, "status"]
SAMPLE_DIAGNOSTICS = [
    "raw_perturbation_ratio", "constraint_residual", "nullspace_residual",
    "min_observable_gain",
]
RESPONSE_COLUMNS = [
    "batch", "sketch_id", "label", "label_kind", "raw_null_dim", "observable_free_rank",
    "observable_gain_max", "exactly_invariant", "consistency_residual", "r0_frobenius",
    "n_zero_observable_columns", "n_drivers", "n_x", "m_phi", "skip_reason",
]


SAMPLE_SCHEMA = pa.schema(
    [(name, pa.string()) for name in RECORD_KEYS]
    + [("eta", pa.float64()), ("sample", pa.int32())]
    + [(name, pa.float64() if name.startswith(("theta", "placement")) else pa.int32())
       for name in METRIC_COLUMNS]
    + [(name, pa.float64()) for name in SAMPLE_DIAGNOSTICS]
    + [("n_columns_guarded", pa.int32())]
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


def plan_records(bundle, item, models, config) -> tuple[list[tuple], dict, dict]:
    """Every (rid, variant, gt) row of the Sec. 5 table, the sketch behind each label, and
    which population that label belongs to.

    ``label_kind`` is recorded here, where it is known from the bundle, so no downstream
    consumer has to recover it by pattern-matching a label name.
    """
    records: list[tuple] = []
    sketches: dict[str, object] = {}
    kinds: dict[str, str] = {}
    for refs in combinations(item.human_labels, 2):
        heldout = next(n for n in item.human_labels if n not in refs)
        rid, variants = build_variants(bundle, item, refs, heldout, models,
                                       config["dim_rates"], config["rel_rates"])
        for gt_label in refs:
            sketches[gt_label] = bundle.sketches[gt_label]
            kinds[gt_label] = HUMAN_KIND
            records.append((rid, "source_self", gt_label, gt_label))
        for variant, label, sketch in variants:
            sketches[label] = sketch
            kinds[label] = (
                HUMAN_KIND if variant == "heldout"
                else DROPOUT_KIND if variant.startswith(("dim_drop", "rel_drop"))
                else MODEL_KIND
            )
            for gt_label in refs:
                records.append((rid, variant, label, gt_label))
    return records, sketches, kinds


def sample_plan(config) -> list[tuple[float, int]]:
    """(eta, sample_index) pairs; eta = 0 is the delta baseline and is drawn once."""
    plan = [(BASELINE_ETA, 0)]
    for eta in config["etas"]:
        if eta == BASELINE_ETA:
            continue
        plan.extend((eta, index) for index in range(config["n_samples"]))
    return plan


def process_sketch(item, converter, context, config, collectors: Collectors) -> None:
    bundle = load_bundle(item, converter, config["geometry_tol"])
    humans, models = usable_labels(bundle, item)
    if not humans:
        collectors.skips["sketch_failed_base_policy"] += 1
        return

    base_eids = build_base_eids(bundle.canonical_sketch, BASE_POLICY)
    eps, rcond = config["eps"], config["rcond"]
    records, sketches, kinds = plan_records(bundle, item, models, config)

    cache: dict[str, PredCache] = {}
    for label, sketch in sketches.items():
        motion = SketchMotion(sketch, context, base_eids, eps=eps)
        j_phi = pairwise_distance_observable(sketch, base_eids=base_eids).to(
            dtype=motion.native.R.dtype
        )
        geometry = build_response_geometry(
            motion.native, j_phi, eps=eps, rcond=rcond,
            consistency_tol=config["consistency_tol"], gain_floor=config["gain_floor"],
        )
        cache[label] = PredCache(motion, j_phi, geometry, response_fingerprint(motion))
        collectors.responses.append(dict(
            batch=item.batch, sketch_id=item.sketch_id, label=label,
            label_kind=kinds[label],
            n_x=int(motion.native.J_x.shape[1]), m_phi=motion.m_phi,
            **geometry.as_record(),
        ))
        if geometry.skip_reason:
            collectors.skips[f"response:{geometry.skip_reason}"] += 1
        elif geometry.exactly_invariant:
            collectors.skips["response:exactly_invariant"] += 1

    evaluators: dict[tuple[str, str], PairEvaluator] = {}
    for _rid, _variant, pred_label, gt_label in records:
        key = (pred_label, gt_label)
        if key not in evaluators:
            evaluators[key] = PairEvaluator(cache[pred_label].motion, cache[gt_label].motion)

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
    for eta, index in sample_plan(config):
        for pred_label, targets in by_pred.items():
            entry = cache[pred_label]
            if entry.geometry.skip_reason or entry.geometry.exactly_invariant:
                continue
            # An exactly-invariant response would contribute a row of zeros to every eta and
            # pull the reported medians toward a movement it cannot have. Its population is
            # reported by invariance_coverage.csv instead.
            seed = stable_seed(config["seed"], item.sketch_id, pred_label, eta, index)
            R_eta, diagnostics = perturbed_response(entry.geometry, eta, seed)
            if diagnostics["constraint_residual"] > config["consistency_tol"]:
                collectors.skips["sample:inconsistent_linear_system"] += 1
                continue
            state = pred_response_state(R_eta, entry.j_phi, eps)
            for rid, variant, gt_label in targets:
                metrics = evaluators[(pred_label, gt_label)].metrics(state)
                if metrics is None:
                    continue
                rows.append(dict(
                    batch=item.batch, sketch_id=item.sketch_id, rid=rid,
                    variant=variant, gt=gt_label, eta=eta, sample=index,
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
        gain_floor=args.gain_floor,
        etas=tuple(sorted({float(e) for e in args.etas.split(",") if e})),
        n_samples=args.n_samples,
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
    parser.add_argument("--gain-floor", type=float, default=OBSERVABLE_GAIN_FLOOR,
                        help="discard a draw whose ||J_phi n|| falls below this")
    parser.add_argument("--etas", default=",".join(f"{e:g}" for e in DEFAULT_ETAS))
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim-drop-rates", default=",".join(f"{r:g}" for r in DEFAULT_DIM_RATES))
    parser.add_argument("--rel-drop-rates", default=",".join(f"{r:g}" for r in DEFAULT_REL_RATES))
    parser.add_argument("--shard", default="0/1", help="i/n, split deterministically by sketch key")
    parser.add_argument("--out", default="artifacts/rebuttal/pseudoinverse")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
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
