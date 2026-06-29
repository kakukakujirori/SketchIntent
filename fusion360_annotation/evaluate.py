#!/usr/bin/env python
"""Quantitative evaluation of a model-output directory against the fiverr benchmark.

    python evaluate.py <Pred dir> [<GT dir>]

``<Pred dir>`` is one ``model_outputs/<run>/`` directory containing batch
subdirectories ``0..11`` with per-sketch ``<sketch_id>.py`` and a ``results.json``
(the layout produced by ``src/test.py``).  ``<GT dir>`` defaults to ``data/SketchIntent-292``
and holds the human annotations ``<sketch_id>_<label>.FCStd`` plus the canonical
input ``<sketch_id>.py``.

Three metric families are reported:

A. Population solver metrics (read from ``results.json``; no GT, no FreeCAD reload):
   DoF, FC / Under / Distorted / SolverError rates, and the raw final-status
   breakdown (Redundant / Conflicting / Malformed / Distorted).  ``valid_dof_mean``
   is the honest health number -- fc_rate hides the close-or-abandon bimodality.

B. Composition / over-dim KPI (parsed from the predicted constraint set; no GT):
   DimensionalRatio (= Distance/DistanceX/DistanceY/Angle), KeyRelational /
   relational ratios, and the raw Distance-vs-DistanceX/Y split.

C. Reference geometry metrics (predicted sketch vs each human GT; FreeCAD + the
   eda_code canonical remap).  motion_full / motion_relational / leak /
   response_full / response_relational angles (identical implementation to the
   training reward, so the numbers line up with the paper) plus asgn_sim (S_place,
   the dimensional-placement axis).

Notes
-----
* ``response_full_angle`` is reported but is a pro-over-dim metric (its GT is the
  fully-dimensioned human), so do NOT use it for checkpoint / ablation selection.
  Discriminate the leak arms on leak + Distance-vs-DistanceX/Y + matched DoF/FC.
* HistCAD OES is a heavy baseline-comparison artifact, not a selection metric, so
  it is omitted here.  Use ``fusion360_annotation/eda_code`` if you need it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constraint-type families (mirror src/sketch_handlers/constants.py, by name so
# we can count straight from the FreeCAD ``Sketcher.Constraint('Type', ...)``
# call in the predicted .py without tokenizing).
# ---------------------------------------------------------------------------
DIMENSIONAL_TYPES = frozenset(
    {"Distance", "DistanceX", "DistanceY", "Length", "Angle", "Diameter", "Radius"}
)
DISTANCE_FAMILY_TYPES = frozenset({"Distance", "DistanceX", "DistanceY"})
# FreeCAD emits ``Symmetric`` for the token named ``Mirror``; accept both spellings.
KEY_RELATIONAL_TYPES = frozenset(
    {"Coincident", "Equal", "Tangent", "Parallel", "Perpendicular", "Mirror",
     "Symmetric", "Concentric"}
)

_CONSTRAINT_RE = re.compile(r"Sketcher\.Constraint\(\s*'(?P<type>\w+)'")


# ===========================================================================
# A. Population solver metrics  (results.json)
# ===========================================================================
def load_results(pred_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for results_path in sorted(pred_dir.glob("*/results.json")):
        with results_path.open() as f:
            rows.extend(json.load(f))
    return rows


def solver_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0}
    status = Counter(r.get("status") for r in results)
    final = Counter(r.get("final_status") for r in results)
    dofs = [r.get("final_dof") for r in results if r.get("final_dof") is not None]
    # valid_dof: DoF over solver-converged outputs (Solved + UnderConstrained),
    # the honest health number that fc_rate hides.
    valid_dofs = [
        r["final_dof"]
        for r in results
        if r.get("final_dof") is not None
        and r.get("status") in {"Solved", "UnderConstrained"}
    ]
    return {
        "n": n,
        "fc_rate": status["Solved"] / n,
        "under_rate": status["UnderConstrained"] / n,
        "distorted_rate": status["Distorted"] / n,
        "solver_error_rate": status["SolverError"] / n,
        "dof_mean": _mean(dofs),
        "valid_dof_mean": _mean(valid_dofs),
        "final_status": dict(final),
    }


# ===========================================================================
# B. Composition / over-dim KPI  (predicted constraint set)
# ===========================================================================
def composition_metrics(pred_dir: Path) -> dict:
    per_type: Counter = Counter()
    n_sketches = 0
    per_sketch: dict[str, dict] = {}
    for py_path in sorted(pred_dir.glob("*/*.py")):
        text = py_path.read_text()
        types = _CONSTRAINT_RE.findall(text)
        if not types:
            continue
        n_sketches += 1
        per_type.update(types)
        # Per-sketch composition (batch-qualified id) for paired statistics.
        sid = f"{py_path.parent.name}/{py_path.stem}"
        c = Counter(types)
        t = len(types)
        ndim = sum(v for k, v in c.items() if k in DIMENSIONAL_TYPES)
        per_sketch[sid] = {
            "n_constraints": t,
            "dimensional_ratio": ndim / t,
            "rel_ratio": (t - ndim) / t,
            "raw_distance_ratio": c["Distance"] / t,
            "axis_distance_ratio": (c["DistanceX"] + c["DistanceY"]) / t,
        }
    total = sum(per_type.values())
    if total == 0:
        return {"n_sketches": 0}
    dist_fam = sum(per_type[t] for t in DISTANCE_FAMILY_TYPES)
    angle_fam = per_type["Angle"]
    key_rel = sum(per_type[t] for t in KEY_RELATIONAL_TYPES)
    n_dim = sum(c for t, c in per_type.items() if t in DIMENSIONAL_TYPES)
    n_rel = total - n_dim
    raw_distance = per_type["Distance"]
    axis_distance = per_type["DistanceX"] + per_type["DistanceY"]
    return {
        "n_sketches": n_sketches,
        "n_constraints": total,
        "rel_ratio": n_rel / total,
        "dimensional_ratio": n_dim / total,
        "distance_family_ratio": dist_fam / total,
        "angle_family_ratio": angle_fam / total,
        "key_relational_ratio": key_rel / total,
        # Over-dim discriminator: raw diagonal Distance vs axis-aligned X/Y split.
        "raw_distance_ratio": raw_distance / total,
        "axis_distance_ratio": axis_distance / total,
        "per_type": dict(per_type.most_common()),
        "per_sketch": per_sketch,
    }


# ===========================================================================
# C. Reference geometry metrics  (pred sketch vs human GT)
# ===========================================================================
def reference_metrics(pred_dir: Path, *, eps: float, geometry_tol: float,
                      limit: int | None) -> dict:
    """Compare each predicted sketch to its human annotations in the canonical frame."""
    import torch  # noqa: F401  (imported lazily; heavy + only needed here)

    from fusion360_annotation.eda_code.config import BASE_POLICY
    from fusion360_annotation.eda_code.canonical import canonical_dir_for
    from fusion360_annotation.eda_code.file_bundle import (
        base_policy_pass,
        build_base_eids,
        build_file_bundle,
    )
    from fusion360_annotation.eda_code.manifest import discover_manifest
    from fusion360_annotation.eda_code.motion_metrics import placement_sims
    from fusion360_annotation.eda_code.progress import progress
    from src.metrics.edit_response_distance import (
        compute_native_response,
        pairwise_distance_observable,
    )
    from src.models.rewards.response_metric_reward import _compute_h_theta_metrics
    from src.sketch_handlers.context import SketchContext
    from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter

    ctx = SketchContext()
    converter = SketchFreecadConverter(ctx, verbose=False)
    manifest, _stats = discover_manifest()
    items = [
        (batch, base, files)
        for batch in sorted(manifest, key=int)
        for base, files in manifest[batch].items()
    ]
    if limit is not None:
        items = items[:limit]

    def prep(sketch, base_eids):
        """Return (native, normalized full-motion dirs, j_phi) for one sketch."""
        native = compute_native_response(sketch, ctx, eps=eps)
        j_phi = pairwise_distance_observable(sketch, base_eids=base_eids).to(
            dtype=native.R.dtype
        )
        if native.R.shape[1] > 0:
            full = j_phi @ native.R
            norms = torch.linalg.norm(full, dim=0)
            keep = norms > 1e-8
            dirs = (full[:, keep] / norms[keep]) if bool(keep.any()) else full[:, :0]
        else:
            dirs = torch.zeros(j_phi.shape[0], 0, dtype=native.R.dtype)
        return native, dirs, j_phi

    ANGLE_KEYS = [
        "motion_full_angle", "motion_relational_angle", "leak_relational_angle",
        "response_full_angle", "response_relational_angle",
    ]
    per_sketch: list[dict] = []
    skipped: Counter = Counter()

    for batch, base, files in progress(items, desc="reference metrics", total=len(items)):
        try:
            bundle = build_file_bundle(
                batch=batch,
                filename=base,
                humans=[(init, str((Path(_gt_root()) / batch))) for init in sorted(files)],
                file_maps={init: {base: actual} for init, actual in files.items()},
                converter=converter,
                geometry_tol=geometry_tol,
                canonical_dir=str(canonical_dir_for(batch, base)),
                model_specs=[("Vitruvion", str(pred_dir))],
                coordinate_scale=1.0,
                bbox_normalize_dst_length=1.0,
            )
        except Exception as exc:  # noqa: BLE001
            skipped[f"bundle_error:{type(exc).__name__}"] += 1
            continue

        if "Vitruvion" not in bundle.sketches:
            skipped[f"pred_{bundle.model_statuses.get('Vitruvion', 'missing')[:20]}"] += 1
            continue

        humans = [n for n in files if n in bundle.sketches]
        if not humans:
            skipped["no_human"] += 1
            continue
        base_eids = build_base_eids(bundle.canonical_sketch, BASE_POLICY)
        if not base_policy_pass(bundle.audits["Vitruvion"], BASE_POLICY)[0]:
            skipped["pred_missing_solid"] += 1
            continue
        humans = [h for h in humans if base_policy_pass(bundle.audits[h], BASE_POLICY)[0]]
        if not humans:
            skipped["humans_missing_solid"] += 1
            continue

        try:
            pred_sketch = bundle.sketches["Vitruvion"]
            p_native, p_dirs, p_jphi = prep(pred_sketch, base_eids)
        except Exception as exc:  # noqa: BLE001
            skipped[f"pred_motion_fail:{type(exc).__name__}"] += 1
            continue

        pair_vals: dict[str, list[float]] = defaultdict(list)
        for h in humans:
            try:
                g_sketch = bundle.sketches[h]
                g_native, g_dirs, g_jphi = prep(g_sketch, base_eids)
                if p_jphi.shape[0] != g_jphi.shape[0] or p_jphi.shape[0] == 0:
                    continue
                hth = _compute_h_theta_metrics(
                    pred_sketch, p_native, g_sketch, g_native, ctx, eps,
                    J_phi_pred=p_jphi, J_phi_gt=g_jphi,
                )
                place = placement_sims(p_dirs, g_dirs)
            except Exception:  # noqa: BLE001
                continue
            for k in ANGLE_KEYS:
                pair_vals[k].append(float(getattr(hth, k)))
            if place.get("asgn_sim") is not None:
                pair_vals["asgn_sim"].append(float(place["asgn_sim"]))

        if not pair_vals.get("motion_full_angle"):
            skipped["no_valid_pair"] += 1
            continue
        # Aggregate the per-reference values into one score per sketch (mean over
        # the human annotators, since intent is multi-valued).
        entry = {k: _mean(vs) for k, vs in pair_vals.items()}
        entry["_id"] = f"{batch}/{base}"
        per_sketch.append(entry)

    out: dict = {"n_scored": len(per_sketch), "skipped": dict(skipped)}
    for k in ANGLE_KEYS + ["asgn_sim"]:
        vals = [s[k] for s in per_sketch if k in s]
        if vals:
            out[k] = {"mean": _mean(vals), "median": _median(vals), "n": len(vals)}
    # Per-sketch values (batch-qualified id) for paired statistics.
    out["per_sketch"] = per_sketch
    return out


# Module-global so reference_metrics' nested lambda can reach the GT root that
# main() resolved (also patched into eda_code.config).
_GT_ROOT: str | None = None


def _gt_root() -> str:
    assert _GT_ROOT is not None
    return _GT_ROOT


# ===========================================================================
# helpers / reporting
# ===========================================================================
def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _median(xs) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) and not math.isnan(x) else str(x)


def report(pred_dir: Path, solver: dict, comp: dict, ref: dict | None) -> None:
    print("=" * 64)
    print(f"Evaluation: {pred_dir.name}")
    print("=" * 64)

    print("\n[A] Solver / population   (n = {})".format(solver.get("n")))
    for k in ("fc_rate", "under_rate", "distorted_rate", "solver_error_rate",
              "valid_dof_mean", "dof_mean"):
        print(f"    {k:<20} {_fmt(solver.get(k))}")
    print(f"    final_status         {solver.get('final_status')}")

    print("\n[B] Composition / over-dim KPI")
    for k in ("rel_ratio", "dimensional_ratio", "distance_family_ratio",
              "angle_family_ratio", "key_relational_ratio", "raw_distance_ratio",
              "axis_distance_ratio"):
        print(f"    {k:<22} {_fmt(comp.get(k))}")
    print(f"    per_type             {comp.get('per_type')}")

    if ref is not None:
        print("\n[C] Reference geometry   (scored {} / skipped {})".format(
            ref.get("n_scored"), sum(ref.get("skipped", {}).values())))
        for k in ("motion_full_angle", "motion_relational_angle",
                  "leak_relational_angle", "response_relational_angle",
                  "response_full_angle", "asgn_sim"):
            v = ref.get(k)
            tag = "  <- pro-over-dim; not for selection" if k == "response_full_angle" else ""
            if v:
                print(f"    {k:<26} mean {_fmt(v['mean'])}  median {_fmt(v['median'])}{tag}")
        if ref.get("skipped"):
            print(f"    skipped breakdown    {ref['skipped']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir", type=Path, help="model_outputs/<run> directory")
    ap.add_argument("gt_dir", type=Path, nargs="?", default=REPO_ROOT / "data/SketchIntent-292",
                    help="annotation root (default: data/SketchIntent-292)")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the FreeCAD-heavy reference geometry metrics (A/B only)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only score the first N sketches (reference metrics)")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--geometry-tol", type=float, default=5e-1)
    ap.add_argument("--json", type=Path, default=None, help="dump full metrics to JSON")
    args = ap.parse_args()

    if not args.pred_dir.exists():
        ap.error(f"pred_dir does not exist: {args.pred_dir}")

    # Point the eda_code pipeline at the requested GT root (data_root() reads this
    # module global at call time, so reassigning it reroutes manifest/canonical).
    global _GT_ROOT
    _GT_ROOT = str(args.gt_dir)
    from fusion360_annotation.eda_code import config as eda_config
    eda_config.DATA_EVAL_ROOT = args.gt_dir.resolve()

    solver = solver_metrics(load_results(args.pred_dir))
    comp = composition_metrics(args.pred_dir)
    ref = None
    if not args.no_reference:
        ref = reference_metrics(
            args.pred_dir, eps=args.eps, geometry_tol=args.geometry_tol,
            limit=args.limit,
        )

    report(args.pred_dir, solver, comp, ref)
    if args.json:
        args.json.write_text(json.dumps(
            {"pred_dir": str(args.pred_dir), "solver": solver,
             "composition": comp, "reference": ref}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
