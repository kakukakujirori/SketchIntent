"""Plot Pareto diagnostics from full-Fiverr evaluate.py JSON outputs.

Default usage:
    conda run -n freecad python paper/scripts/plot_fiverr_eval_pareto.py

Custom inputs:
    conda run -n freecad python paper/scripts/plot_fiverr_eval_pareto.py \
      --input "Ours step59800=artifacts/fiverr_eval/ours_step59800.json" \
      --input "DP c1 step21200=artifacts/fiverr_eval/dp_c1_step21200.json"

Outputs:
    artifacts/fiverr_eval/summary.csv
    artifacts/fiverr_eval/summary.md
    artifacts/fiverr_eval/fiverr_pareto_fc_vs_angles.png
    artifacts/fiverr_eval/fiverr_pareto_fc_vs_independent.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DIMENSIONAL_RATIO_TYPES = (
    "Distance",
    "DistanceX",
    "DistanceY",
    "Angle",
    "Diameter",
    "Radius",
    "Length",
)

DEFAULT_INPUTS = (
    ("S step12700", "artifacts/fiverr_eval/s_noml_step12700.json"),
    ("S+M step12200", "artifacts/fiverr_eval/sm_m4.5_step12200.json"),
    ("Ours M4.5/L1.5 step52200", "artifacts/fiverr_eval/ours_step52200.json"),
    ("Ours M4.5/L1.5 step59800", "artifacts/fiverr_eval/ours_step59800.json"),
    ("DP c1 step21200", "artifacts/fiverr_eval/dp_c1_step21200.json"),
    ("DP c0.3 step22500", "artifacts/fiverr_eval/dp_c03_step22500.json"),
    ("DP c3 step15400", "artifacts/fiverr_eval/dp_c3_step15400.json"),
    ("M_DP step13300", "artifacts/fiverr_eval/m_dp_step13300.json"),
    ("M2.25/L0.75 step36000", "artifacts/fiverr_eval/m225_l075_step36000.json"),
)


SUMMARY_COLUMNS = (
    "label",
    "id",
    "n",
    "fc_rate",
    "under_rate",
    "distorted_rate",
    "solver_error_rate",
    "valid_dof_mean",
    "dof_mean",
    "relational_ratio",
    "dimensional_ratio",
    "distance_family_ratio",
    "angle_family_ratio",
    "raw_distance_ratio",
    "axis_distance_ratio",
    "key_relational_ratio",
    "n_scored",
    "motion_full_angle_mean",
    "motion_full_angle_median",
    "motion_relational_angle_mean",
    "motion_relational_angle_median",
    "leak_dimensional_angle_mean",
    "leak_dimensional_angle_median",
    "placement_sim_mean",
    "placement_sim_median",
)


def parse_input(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--input must be LABEL=PATH")
    return label, path


def metric(stats: dict[str, Any], name: str, field: str) -> float:
    value = stats.get(name, {}).get(field)
    return float(value) if value is not None else float("nan")


def ratio_from_per_type(comp: dict[str, Any], names: tuple[str, ...]) -> float:
    total = float(comp.get("n_constraints") or 0)
    if total <= 0:
        return float("nan")
    per_type = comp.get("per_type") or {}
    return float(sum(per_type.get(name, 0) for name in names) / total)


def comp_ratio(
    comp: dict[str, Any],
    key: str,
    *,
    fallback_names: tuple[str, ...],
) -> float:
    value = comp.get(key)
    if value is not None:
        return float(value)
    return ratio_from_per_type(comp, fallback_names)


def row_from_json(label: str, path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    solver = data["solver"]
    comp = data["composition"]
    ref = data["reference"]
    return {
        "label": label,
        "id": path.stem,
        "n": solver["n"],
        "fc_rate": solver["fc_rate"],
        "under_rate": solver["under_rate"],
        "distorted_rate": solver["distorted_rate"],
        "solver_error_rate": solver["solver_error_rate"],
        "valid_dof_mean": solver["valid_dof_mean"],
        "dof_mean": solver["dof_mean"],
        "relational_ratio": comp["relational_ratio"],
        "dimensional_ratio": comp_ratio(
            comp,
            "dimensional_ratio",
            fallback_names=DIMENSIONAL_RATIO_TYPES,
        ),
        "distance_family_ratio": comp_ratio(
            comp,
            "distance_family_ratio",
            fallback_names=("Distance", "DistanceX", "DistanceY"),
        ),
        "angle_family_ratio": comp_ratio(
            comp,
            "angle_family_ratio",
            fallback_names=("Angle",),
        ),
        "raw_distance_ratio": comp_ratio(
            comp,
            "raw_distance_ratio",
            fallback_names=("Distance",),
        ),
        "axis_distance_ratio": comp_ratio(
            comp,
            "axis_distance_ratio",
            fallback_names=("DistanceX", "DistanceY"),
        ),
        "key_relational_ratio": float(comp["key_relational_ratio"]),
        "n_scored": ref["n_scored"],
        "motion_full_angle_mean": metric(ref, "motion_full_angle", "mean"),
        "motion_full_angle_median": metric(ref, "motion_full_angle", "median"),
        "motion_relational_angle_mean": metric(ref, "motion_relational_angle", "mean"),
        "motion_relational_angle_median": metric(ref, "motion_relational_angle", "median"),
        "leak_dimensional_angle_mean": metric(ref, "leak_dimensional_angle", "mean"),
        "leak_dimensional_angle_median": metric(ref, "leak_dimensional_angle", "median"),
        "placement_sim_mean": metric(ref, "placement_sim", "mean"),
        "placement_sim_median": metric(ref, "placement_sim", "median"),
    }


def write_summary(rows: list[dict[str, Any]], outdir: Path) -> None:
    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = outdir / "summary.md"
    with md_path.open("w") as f:
        f.write(
            "| model | FC | valid DoF | dim-ratio | theta-full | "
            "theta-rel | theta-align | S_place | raw-dist |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['label']} | {row['fc_rate']:.3f} | "
                f"{row['valid_dof_mean']:.3f} | "
                f"{row['dimensional_ratio']:.3f} | "
                f"{row['motion_full_angle_mean']:.3f} | "
                f"{row['motion_relational_angle_mean']:.3f} | "
                f"{row['leak_dimensional_angle_mean']:.3f} | "
                f"{row['placement_sim_mean']:.3f} | "
                f"{row['raw_distance_ratio']:.3f} |\n"
            )


def color_for(label: str) -> str:
    if label.startswith("Ours"):
        return "#d62728"
    if label.startswith("DP"):
        return "#1f77b4"
    if label.startswith("M_DP"):
        return "#9467bd"
    if label.startswith("M2"):
        return "#2ca02c"
    if label.startswith("S+M"):
        return "#ff7f0e"
    if label.startswith("S"):
        return "#8c564b"
    return "#666666"


def short_label(label: str) -> str:
    return label.replace("Ours M4.5/L1.5 ", "Ours ").replace(" step", "\n")


def plot(rows: list[dict[str, Any]], outdir: Path) -> None:
    specs = (
        ("motion_relational_angle_mean", "Motion relational angle"),
        ("leak_dimensional_angle_mean", "Dimensional leak angle"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=180)
    for ax, (key, title) in zip(axes, specs):
        for row in rows:
            ax.scatter(
                row["fc_rate"],
                row[key],
                s=55,
                c=color_for(row["label"]),
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.annotate(
                short_label(row["label"]),
                (row["fc_rate"], row[key]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(title)
        ax.set_xlabel("Fully constrained rate (higher is better)")
        ax.set_ylabel(f"{title} (lower is better)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.12, 0.80)
    fig.tight_layout()
    fig.savefig(outdir / "fiverr_pareto_fc_vs_angles.png")

    independent_specs = (
        ("dimensional_ratio", "Dimensional ratio", "lower is better"),
        ("motion_full_angle_mean", "Full motion angle", "lower is better"),
        ("placement_sim_mean", "S_place / assignment similarity", "higher is better"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=180)
    for ax, (key, title, direction) in zip(axes, independent_specs):
        for row in rows:
            ax.scatter(
                row["fc_rate"],
                row[key],
                s=55,
                c=color_for(row["label"]),
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.annotate(
                short_label(row["label"]),
                (row["fc_rate"], row[key]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(title)
        ax.set_xlabel("Fully constrained rate (higher is better)")
        ax.set_ylabel(f"{title} ({direction})")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.12, 0.80)
    fig.tight_layout()
    fig.savefig(outdir / "fiverr_pareto_fc_vs_independent.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, help="LABEL=PATH JSON input")
    parser.add_argument("--outdir", default="artifacts/fiverr_eval")
    args = parser.parse_args()

    inputs = args.input if args.input else list(DEFAULT_INPUTS)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for label, path_str in inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"[warn] missing input: {path}")
            continue
        rows.append(row_from_json(label, path))
    if not rows:
        raise SystemExit("no input JSON files found")

    write_summary(rows, outdir)
    plot(rows, outdir)
    print(outdir / "summary.csv")
    print(outdir / "summary.md")
    print(outdir / "fiverr_pareto_fc_vs_angles.png")
    print(outdir / "fiverr_pareto_fc_vs_independent.png")


if __name__ == "__main__":
    main()
