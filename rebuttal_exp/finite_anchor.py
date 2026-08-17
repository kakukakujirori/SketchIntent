"""Does the real solver's small finite response look like the minimum-norm prediction?

Experiment D shows how much the metrics *could* move if a different admissible response
were chosen. This asks the complementary empirical question: when FreeCAD actually solves a
small edit from the current geometry, how close is the observable change it produces to
``J_phi R_0[:, q]``, the minimum-norm tangent prediction for that driver.

This is an anchor, not a proof: FreeCAD is not claimed to implement any minimum-norm
convention. It only records which branch an incremental solve from a nearby state picks in
practice. Baseline solving, edit application and the success rule are imported unchanged
from the experiment A runner, so the two experiments cannot drift apart.

Outputs:
    finite_anchor_run_config.json  finite_anchor_environment.json  finite_anchor.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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
from rebuttal_exp.finite_edit_validation import (  # noqa: E402
    SIGNS,
    apply_finite_edit,
    prepare_baseline,
)
from rebuttal_exp.response_perturbation import cutoff_null_space  # noqa: E402
from src.eval.progress import progress  # noqa: E402
from src.metrics.edit_response_distance import (  # noqa: E402
    compute_native_response,
    pairwise_distance_observable,
)
from src.metrics.histcad_2d import HistCADConfig, _bbox_diag  # noqa: E402

DEFAULT_LENGTH_ALPHA = 0.001
DEFAULT_ANGLE_DEGREES = 0.1
ZERO_PREDICTION_TOL = 1e-12

ANCHOR_COLUMNS = [
    "batch", "sketch_id", "label", "label_kind", "driver_signature", "driver_type",
    "driver_cid", "response_column", "magnitude_kind", "sign", "delta_internal", "q0",
    "target", "solve_ok", "status", "target_miss", "cosine", "normalized_error",
    "finite_rate_norm", "tangent_norm", "raw_null_dim", "skip_reason",
]


@dataclass
class Collectors:
    rows: list = field(default_factory=list)
    skips: Counter = field(default_factory=Counter)


def cosine_and_error(finite_rate: np.ndarray, tangent: np.ndarray) -> tuple[float, float]:
    finite_norm = float(np.linalg.norm(finite_rate))
    tangent_norm = float(np.linalg.norm(tangent))
    if finite_norm < ZERO_PREDICTION_TOL or tangent_norm < ZERO_PREDICTION_TOL:
        return float("nan"), float("nan")
    cosine = float(finite_rate @ tangent) / (finite_norm * tangent_norm)
    error = float(np.linalg.norm(finite_rate - tangent)) / tangent_norm
    return cosine, error


def process_sketch(item, converter, context, config, collectors: Collectors) -> None:
    bundle = load_bundle(item, converter, config["geometry_tol"])
    humans, models = usable_labels(bundle, item)
    if not humans:
        collectors.skips["sketch_failed_base_policy"] += 1
        return

    base_eids = build_base_eids(bundle.canonical_sketch, BASE_POLICY)
    bbox = _bbox_diag(bundle.canonical_sketch)
    for label in list(humans) + list(models):
        label_kind = "human" if label in humans else "model"
        baseline = prepare_baseline(label, bundle.sketches[label], base_eids, converter, context)
        if isinstance(baseline, str):
            collectors.skips[f"baseline:{baseline}"] += 1
            continue
        _anchor_annotation(item, baseline, label_kind, base_eids, bbox,
                           converter, context, config, collectors)


def _anchor_annotation(item, baseline, label_kind, base_eids, bbox,
                       converter, context, config, collectors: Collectors) -> None:
    """The tangent prediction is built on the solved baseline, the same state the edit starts from."""
    native = compute_native_response(
        baseline.sketch, context, eps=config["eps"], rcond=config["rcond"]
    )
    j_phi = pairwise_distance_observable(baseline.sketch, base_eids=base_eids).to(
        dtype=native.R.dtype
    )
    if int(j_phi.shape[0]) != baseline.layout.dimension:
        collectors.skips["observable_layout_mismatch"] += 1
        return

    tangent = (j_phi @ native.R).numpy() if native.R.shape[1] else np.zeros((j_phi.shape[0], 0))
    columns = {cid: index for index, cid in enumerate(native.dim_constraint_ids)}
    raw_null_dim = int(cutoff_null_space(
        native.J_x, eps=config["eps"], rcond=config["rcond"]
    ).shape[1])

    for signature, slot in baseline.drivers.items():
        column = columns.get(slot.cid)
        if column is None:
            collectors.skips["driver_not_in_response_columns"] += 1
            continue
        magnitude = (math.radians(config["angle_degrees"]) if slot.kind == "angle"
                     else config["length_alpha"] * bbox)
        for sign in SIGNS:
            delta = sign * magnitude
            target = slot.q0 + delta
            row = dict(
                batch=item.batch, sketch_id=item.sketch_id, label=baseline.label,
                label_kind=label_kind, driver_signature=repr(signature),
                driver_type=signature[0], driver_cid=slot.cid, response_column=column,
                magnitude_kind=slot.kind, sign=int(sign), delta_internal=delta,
                q0=slot.q0, target=target, raw_null_dim=raw_null_dim, skip_reason="",
            )
            if signature[0] in _POSITIVE_TYPES and target <= 0.0:
                row.update(skip_reason="invalid_target_domain", solve_ok=False,
                           status="invalid_target_domain")
                collectors.rows.append(row)
                continue

            outcome = apply_finite_edit(baseline, slot.cid, target, bbox, converter, context,
                                        config["target_tol"])
            row.update(solve_ok=outcome.solve_ok, status=outcome.status,
                       target_miss=outcome.target_miss)
            if outcome.solve_ok and outcome.response is not None:
                cosine, error = cosine_and_error(outcome.response / delta, tangent[:, column])
                row.update(
                    cosine=cosine, normalized_error=error,
                    finite_rate_norm=float(np.linalg.norm(outcome.response / delta)),
                    tangent_norm=float(np.linalg.norm(tangent[:, column])),
                )
            else:
                row["skip_reason"] = "solve_failure"
            collectors.rows.append(row)


_POSITIVE_TYPES = frozenset(
    {"Radius", "Diameter", "Minor_Diameter", "Major_Diameter", "Length", "Distance"}
)


def summarise(rows: list[dict]) -> None:
    usable = [r for r in rows if r.get("cosine") is not None and not _is_nan(r.get("cosine"))]
    if not usable:
        print("no usable anchor trials")
        return
    cosines = np.array([r["cosine"] for r in usable], dtype=float)
    errors = np.array([r["normalized_error"] for r in usable], dtype=float)
    print(f"usable trials: {len(usable)} of {len(rows)}")
    for kind in sorted({r["magnitude_kind"] for r in usable}):
        mask = np.array([r["magnitude_kind"] == kind for r in usable])
        print(f"  {kind}: n={int(mask.sum())} "
              f"cos median={np.median(cosines[mask]):.4f} "
              f"Q10={np.quantile(cosines[mask], 0.10):.4f} "
              f"frac>0.99={float((cosines[mask] > 0.99).mean()):.3f} "
              f"rel err median={np.median(errors[mask]):.4f}")


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def build_config(args) -> dict:
    return dict(
        batches=args.batches,
        limit=args.limit,
        geometry_tol=args.geometry_tol,
        eps=args.eps,
        rcond=args.rcond,
        length_alpha=args.length_alpha,
        angle_degrees=args.angle_degrees,
        target_tol=HistCADConfig().er_target_tol,
        base_policy=BASE_POLICY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--rcond", type=float, default=1e-6)
    parser.add_argument("--length-alpha", type=float, default=DEFAULT_LENGTH_ALPHA)
    parser.add_argument("--angle-degrees", type=float, default=DEFAULT_ANGLE_DEGREES)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--out", default="artifacts/rebuttal/pseudoinverse")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    shard_index, n_shards = (int(part) for part in args.shard.split("/"))
    out_dir = Path(args.out)
    if n_shards > 1:
        out_dir = out_dir / f"shard_{shard_index:03d}_of_{n_shards:03d}"
    prepare_output_dir(out_dir, dict(config, shard=args.shard), force=args.force,
                       filename="finite_anchor_run_config.json")
    write_environment(out_dir, sys.argv, Path(__file__).parent.glob("*.py"),
                      filename="finite_anchor_environment.json")

    items = select_shard(discover_items(args.batches, args.limit), shard_index, n_shards)
    context, converter = build_context()
    collectors = Collectors()

    for item in progress(items, desc="finite-anchor", total=len(items)):
        try:
            process_sketch(item, converter, context, config, collectors)
        except Exception as exc:  # noqa: BLE001
            collectors.skips[f"sketch_error:{type(exc).__name__}"] += 1
            print(f"  [skip] {item.key}: {str(exc)[:120]}")

    write_csv(out_dir / "finite_anchor.csv", collectors.rows, ANCHOR_COLUMNS)
    finalize_run_config(out_dir, n_sketches=len(items), n_rows=len(collectors.rows),
                        skips=dict(collectors.skips), filename="finite_anchor_run_config.json")
    summarise(collectors.rows)
    print(f"skips: {dict(collectors.skips)}")
    print(f"wrote -> {out_dir / 'finite_anchor.csv'}")


if __name__ == "__main__":
    main()
