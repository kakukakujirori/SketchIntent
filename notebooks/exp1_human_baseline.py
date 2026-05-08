import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.metrics.null_space_distance import compute_null_space_distance  # noqa: E402
from src.sketch_handlers.context import SketchContext  # noqa: E402
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter  # noqa: E402

from canonical_sketch import (  # noqa: E402
    CANONICAL_METRICS,
    build_canonical_comparison_sketches,
    metric_base_eids,
    metric_include_quadratic,
    metric_valid,
)

EKO = "/home/sturk/Downloads/fiverr/Eko/9"
SAGAR = "/home/sturk/Downloads/fiverr/Sagar/9"
RYOTARO = "/home/sturk/Downloads/fiverr/Ryotaro/9"
ANNOTATORS = ("Eko", "Sagar", "Ryotaro")


def get_file_map(directory: str):
    return {
        filename: filename
        for filename in os.listdir(directory)
        if filename.endswith(".FCStd")
        and not filename.endswith("_mid.FCStd")
        and not filename.startswith("_")
    }


def print_file_overlap(file_maps):
    common = set.intersection(*(set(files) for files in file_maps.values()))
    print(
        "Input files: "
        + " | ".join(f"{ann}={len(files)}" for ann, files in file_maps.items())
        + f" | common={len(common)}"
    )


def format_score(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "N/A"


def format_int_pair(left: float | None, right: float | None, width: int = 3) -> str:
    return (
        f"{int(round(left)):>{width}d}/{int(round(right)):<{width}d}"
        if left is not None and right is not None
        else f"{'NA':>{width}}/{'NA':<{width}}"
    )


def format_mean_std(mean: float | None, std: float | None) -> str:
    return f"{mean:.4f} (+/-{std:.4f})" if mean is not None else "N/A"


def summarize_metric_rows(rows: list) -> dict | None:
    ok_rows = [r for r in rows if not r.get("error") and r.get("linear") is not None]
    if not ok_rows:
        return None

    def arr(key, source_rows=None):
        source_rows = ok_rows if source_rows is None else source_rows
        values = [r[key] for r in source_rows if r.get(key) is not None]
        return np.array(values, dtype=float) if values else None

    quadratic_rows = [
        r
        for r in ok_rows
        if r.get("quadratic") is not None and (r.get("gt_quad_dim") or 0) > 0
    ]
    linear = arr("linear")
    quadratic = arr("quadratic", quadratic_rows)
    two_jet = arr("two_jet")
    gt_quad = arr("gt_quad_dim", quadratic_rows)
    pred_quad = arr("pred_quad_dim", quadratic_rows)

    return {
        "n": len(ok_rows),
        "linear_mean": float(np.mean(linear)),
        "linear_std": float(np.std(linear)),
        "quadratic_mean": None if quadratic is None else float(np.mean(quadratic)),
        "quadratic_std": None if quadratic is None else float(np.std(quadratic)),
        "quadratic_n": 0 if quadratic is None else len(quadratic),
        "two_jet_mean": None if two_jet is None else float(np.mean(two_jet)),
        "two_jet_std": None if two_jet is None else float(np.std(two_jet)),
        "gt_row_dim": float(np.mean(arr("gt_row_dim"))),
        "pred_row_dim": float(np.mean(arr("pred_row_dim"))),
        "gt_quad_dim": None if gt_quad is None else float(np.mean(gt_quad)),
        "pred_quad_dim": None if pred_quad is None else float(np.mean(pred_quad)),
    }


def print_geometry_audit(audit_rows: list[dict], valid_rows: list[dict], filewise: bool) -> None:
    print("\n--- Geometry Audit ---")
    for annotator in ANNOTATORS:
        rows = [r for r in audit_rows if r["annotator"] == annotator]
        if not rows:
            continue
        print(
            f"{annotator:7s} | files={len(rows):2d} | "
            f"missing solid files={sum(r['missing_solid'] > 0 for r in rows):2d} | "
            f"missing construction files={sum(r['missing_construction'] > 0 for r in rows):2d} | "
            f"extra solid files={sum(r['extra_solid'] > 0 for r in rows):2d} | "
            f"extra construction files={sum(r['extra_construction'] > 0 for r in rows):2d} | "
            f"flag changes={sum(r['construction_flag_changed'] for r in rows):2d} | "
            f"dropped constraints={sum(r['dropped_constraints'] for r in rows):2d}"
        )

    for metric in CANONICAL_METRICS:
        rows = [r for r in valid_rows if r["metric"] == metric]
        print(f"{metric:15s} valid files = {sum(r['valid'] for r in rows)}/{len(rows)}")

    if not filewise:
        return

    print("\n--- Filewise Geometry Audit ---")
    valid_by_file_metric = {(r["filename"], r["metric"]): r for r in valid_rows}
    rows_by_file = defaultdict(list)
    for row in audit_rows:
        rows_by_file[row["filename"]].append(row)

    for filename in sorted(rows_by_file):
        valid_text = " | ".join(
            f"{metric}={'OK' if valid_by_file_metric[(filename, metric)]['valid'] else 'SKIP'}"
            for metric in CANONICAL_METRICS
        )
        print(f"\n{filename} | {valid_text}")
        print(
            f"{'Ann':<8} {'match':>5} {'miss S/C':>9} {'extra S/C':>9} "
            f"{'flag':>5} {'drop':>5} {'align dx/dy':>17} "
            f"{'missing IDs':<20} {'extra IDs':<20}"
        )
        for row in sorted(rows_by_file[filename], key=lambda r: ANNOTATORS.index(r["annotator"])):
            print(
                f"{row['annotator']:<8} {row['matched']:>5d} "
                f"{row['missing_solid']:>3d}/{row['missing_construction']:<3d} "
                f"{row['extra_solid']:>3d}/{row['extra_construction']:<3d} "
                f"{row['construction_flag_changed']:>5d} {row['dropped_constraints']:>5d} "
                f"{row['alignment_dx']:>8.3f}/{row['alignment_dy']:<8.3f} "
                f"{row['missing_ids'][:20]:<20} {row['extra_ids'][:20]:<20}"
            )


def print_filewise_results(rows: list[dict]) -> None:
    if not rows:
        return

    print("\n--- Filewise Results ---")
    ann_order = {ann: i for i, ann in enumerate(ANNOTATORS)}
    rows_by_metric_file = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rows_by_metric_file[row["metric"]][row["filename"]].append(row)

    for metric in CANONICAL_METRICS:
        print(f"\n[{metric}]")
        for filename in sorted(rows_by_metric_file.get(metric, {})):
            file_rows = sorted(
                rows_by_metric_file[metric][filename],
                key=lambda r: (ann_order[r["pred"]], ann_order[r["gt"]]),
            )
            print(f"\n{filename}")
            print(
                f"{'Pred':<8} {'GT':<8} {'Dim(P/G)':>8} "
                f"{'Linear':>10} {'Quadratic':>10} {'2-Jet':>10} "
                f"{'d_lin(G/P)':>12} {'d_quad(G/P)':>13} {'Status':<48}"
            )
            for row in file_rows:
                status = "OK" if not row.get("error") else f"ERR: {row['error']}"
                if len(status) > 48:
                    status = status[:45] + "..."
                quadratic = row.get("quadratic") if (row.get("gt_quad_dim") or 0) > 0 else None
                print(
                    f"{row['pred']:<8} {row['gt']:<8} "
                    f"{format_int_pair(row.get('pred_dimensional_constraints'), row.get('gt_dimensional_constraints'))} "
                    f"{format_score(row.get('linear')):>10} "
                    f"{format_score(quadratic):>10} "
                    f"{format_score(row.get('two_jet')):>10} "
                    f"{format_int_pair(row.get('gt_row_dim'), row.get('pred_row_dim'), width=5)} "
                    f"{format_int_pair(row.get('gt_quad_dim'), row.get('pred_quad_dim'), width=6)} "
                    f"{status:<48}"
                )


def print_aggregate_results(rows: list[dict]):
    print("\n--- Aggregate Results ---")
    rows_by_metric = defaultdict(list)
    for row in rows:
        rows_by_metric[row["metric"]].append(row)

    for metric in CANONICAL_METRICS:
        print(f"\n[{metric}]")
        metric_rows = rows_by_metric.get(metric, [])
        for pred_ann in ANNOTATORS:
            for gt_ann in ANNOTATORS:
                if pred_ann == gt_ann:
                    continue
                pair_rows = [
                    r
                    for r in metric_rows
                    if r["pred"] == pred_ann and r["gt"] == gt_ann
                ]
                summary = summarize_metric_rows(pair_rows)
                if summary is None:
                    continue
                print(
                    f"Pred: {pred_ann:7s} | GT: {gt_ann:7s} | "
                    f"Linear = {format_mean_std(summary['linear_mean'], summary['linear_std'])} | "
                    f"Quadratic = {format_mean_std(summary['quadratic_mean'], summary['quadratic_std'])} "
                    f"[n={summary['quadratic_n']}] | "
                    f"2-Jet = {format_mean_std(summary['two_jet_mean'], summary['two_jet_std'])} | "
                    f"d_lin GT/Pred = {summary['gt_row_dim']:.1f}/{summary['pred_row_dim']:.1f} | "
                    f"d_quad GT/Pred = {format_int_pair(summary['gt_quad_dim'], summary['pred_quad_dim'], width=1)} | "
                    f"n = {summary['n']:2d}"
                )


def write_csv(path, fieldnames, rows):
    if not rows or not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {path}")


def compute_pair_metric(
    bundle,
    metric,
    pred_ann,
    gt_ann,
    base_eids,
    include_quadratic,
    context,
    dump_sigmas,
):
    row = {
        "filename": bundle.filename,
        "metric": metric,
        "pred": pred_ann,
        "gt": gt_ann,
        "include_quadratic": include_quadratic,
        "gt_dimensional_constraints": bundle.dimensional_counts[gt_ann],
        "pred_dimensional_constraints": bundle.dimensional_counts[pred_ann],
        "error": "",
    }

    try:
        res = compute_null_space_distance(
            gt_primary_sketch=bundle.remapped_relational_sketches[gt_ann],
            pred_sketch=bundle.remapped_relational_sketches[pred_ann],
            context=context,
            skip_pred_primary_extraction=True,
            base_eids=base_eids,
            include_quadratic=include_quadratic,
            return_sigmas=dump_sigmas,
        )
        row.update(
            {
                "linear": res["linear_intent_recall"],
                "quadratic": res["quadratic_intent_recall"],
                "two_jet": res["two_jet_intent_recall"],
                "gt_row_dim": res["gt_row_dim"],
                "pred_row_dim": res["pred_row_dim"],
                "gt_quad_dim": res["gt_quadratic_dim"],
                "pred_quad_dim": res["pred_quadratic_dim"],
                "sigmas": ",".join(f"{s:.6f}" for s in res["linear_sigmas"])
                if res.get("linear_sigmas")
                else "",
            }
        )
    except Exception as exc:
        print(f"Error processing {bundle.filename} [{metric}] Pred={pred_ann} GT={gt_ann}: {exc}")
        row.update(
            {
                "linear": None,
                "quadratic": None,
                "two_jet": None,
                "gt_row_dim": None,
                "pred_row_dim": None,
                "gt_quad_dim": None,
                "pred_quad_dim": None,
                "sigmas": "",
                "error": str(exc),
            }
        )

    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Human baseline with canonical geometry metrics.")
    parser.add_argument("--eko", default=EKO)
    parser.add_argument("--sagar", default=SAGAR)
    parser.add_argument("--ryotaro", default=RYOTARO)
    parser.add_argument(
        "--canonical-dir",
        default=None,
        help="Directory containing canonical xxx.py files. Defaults to annotator dirs.",
    )
    parser.add_argument("--geometry-tol", type=float, default=5e-1)
    parser.add_argument(
        "--filewise",
        action="store_true",
        help="Print filewise scores and filewise geometry audit.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(REPO_ROOT, "artifacts", "exp1_per_file.csv"),
        help="Path for per-file pair metrics CSV output.",
    )
    parser.add_argument(
        "--audit-csv",
        default=os.path.join(REPO_ROOT, "artifacts", "exp1_geometry_audit.csv"),
        help="Path for per-file geometry audit CSV output.",
    )
    parser.add_argument("--dump-sigmas", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dirs = {"Eko": args.eko, "Sagar": args.sagar, "Ryotaro": args.ryotaro}
    file_maps = {ann: get_file_map(directory) for ann, directory in data_dirs.items()}
    common_files = sorted(set.intersection(*(set(files) for files in file_maps.values())))
    print_file_overlap(file_maps)

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False)
    metric_rows = []
    audit_rows = []
    valid_rows = []

    for filename in common_files:
        try:
            bundle = build_canonical_comparison_sketches(
                filename=filename,
                data_dirs=data_dirs,
                file_maps=file_maps,
                annotators=ANNOTATORS,
                converter=converter,
                geometry_tol=args.geometry_tol,
                canonical_dir=args.canonical_dir,
            )
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")
            continue

        for annotator, audit in bundle.geometry_audits.items():
            audit_rows.append({"filename": filename, "annotator": annotator, **audit})

        for metric in CANONICAL_METRICS:
            valid, reason = metric_valid(metric, bundle.geometry_audits)
            include_quadratic = metric_include_quadratic(metric, bundle.geometry_audits)
            valid_rows.append(
                {
                    "filename": filename,
                    "metric": metric,
                    "valid": valid,
                    "reason": reason,
                    "include_quadratic": include_quadratic,
                }
            )
            if not valid:
                continue

            base_eids = metric_base_eids(metric, bundle.canonical_sketch)
            for pred_ann in ANNOTATORS:
                for gt_ann in ANNOTATORS:
                    if pred_ann == gt_ann:
                        continue
                    metric_rows.append(
                        compute_pair_metric(
                            bundle=bundle,
                            metric=metric,
                            pred_ann=pred_ann,
                            gt_ann=gt_ann,
                            base_eids=base_eids,
                            include_quadratic=include_quadratic,
                            context=context,
                            dump_sigmas=args.dump_sigmas,
                        )
                    )

    print_geometry_audit(audit_rows, valid_rows, args.filewise)
    if args.filewise:
        print_filewise_results(metric_rows)
    print_aggregate_results(metric_rows)

    metric_fieldnames = [
        "filename",
        "metric",
        "pred",
        "gt",
        "include_quadratic",
        "linear",
        "quadratic",
        "two_jet",
        "gt_row_dim",
        "pred_row_dim",
        "gt_quad_dim",
        "pred_quad_dim",
        "gt_dimensional_constraints",
        "pred_dimensional_constraints",
        "sigmas",
        "error",
    ]
    audit_fieldnames = [
        "filename",
        "annotator",
        "matched",
        "missing_solid",
        "missing_construction",
        "extra_solid",
        "extra_construction",
        "construction_flag_changed",
        "dropped_constraints",
        "alignment_dx",
        "alignment_dy",
        "missing_ids",
        "extra_ids",
        "flag_changed_ids",
    ]
    write_csv(args.output_csv, metric_fieldnames, metric_rows)
    write_csv(args.audit_csv, audit_fieldnames, audit_rows)


if __name__ == "__main__":
    main()
