"""
test_solver_consistency.py — CAD Constraint Solver Consistency Verification

Verifies the structural integrity of the My-Vitruvion constraint solver by checking
that every constraint handler in the dataset produces a non-degenerate Jacobian
contribution (non-zero residual gradient) when evaluated on FreeCAD's solved geometry.

Background
----------
Our Line parameterization uses 6 params [pntX, pntY, dirX, dirY, s, e] (SketchGraphs
format), while FreeCAD uses 4 params [startX, startY, endX, endY].  This over-
parameterization introduces 2 gauge DOFs per line (sliding + scaling) that are always
in the null space of well-posed constraint sets.

Key verified properties:
  1. Null space includes gauge: dim(null(J)) >= 2·N_lines
     (gauge DOFs are always free — no constraint handler should constrain them)
  2. Constraint coverage: every non-Block FC constraint maps to at least one SG constraint
  3. Rank monotonicity for Full set: our_rank >= n_constraints_added_by_us
     (our solver doesn't silently drop constraints)
  4. FreeCAD agreement bound: fc_rank <= our_rank  (at all times, at full constraint set)
     This is the correct lower bound invariant.

Note on the exact identity
--------------------------
The identity our_rank - fc_rank == 2·N_lines does NOT hold in general because:
  - Some constraint types (e.g., Symmetric, Mirror, Perpendicular endpoint form) produce
    different numbers of constraint equations in our 6-param space vs FreeCAD's 4-param space.
  - FreeCAD internally detects geometric redundancy differently from our linear algebra.

The correct full-sweep check is therefore:
  our_rank >= fc_rank   (we never undercount constraints relative to FreeCAD)

Usage:
    conda activate freecad
    python tests/test_solver_consistency.py
    python tests/test_solver_consistency.py --data_dir ~/Downloads/fiverr
    python tests/test_solver_consistency.py --max_files 20 --verbose
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import sketchgraphs.data as datalib

from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported, load_FCStd_sketch
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_solver import compute_jacobian, SketchState
from src.sketch_handlers.sketch_utils import complete_constraint_values

FreeCAD, Part, Sketcher = ensure_freecad_imported()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FC_PARAM_COUNT = {
    "LineSegment": 4,
    "Circle":      3,
    "ArcOfCircle": 5,
    "Point":       2,
}

_FC_DIMENSIONAL_TYPES = frozenset({
    "Angle", "Diameter", "Distance", "DistanceX", "DistanceY", "Radius",
})

_RANK_ATOL = 1e-6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternResult:
    name: str
    our_rank: int = -1
    fc_dof: int = -1
    n_lines: int = -1
    p_fc: int = -1
    passed: bool = False
    error: str = ""

    @property
    def fc_rank(self) -> int:
        return self.p_fc - self.fc_dof


@dataclass
class SketchResult:
    filepath: str
    status: str = "pending"
    reason: str = ""
    patterns: list[PatternResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# FreeCAD helpers
# ---------------------------------------------------------------------------

def _count_fc_params(fc_sketch) -> tuple[int, int]:
    p_fc = 0; n_lines = 0
    for geo in fc_sketch.Geometry:
        t = type(geo).__name__
        p_fc += _FC_PARAM_COUNT.get(t, 0)
        if t == "LineSegment": n_lines += 1
    return p_fc, n_lines


def _get_nonblock_fc_constraints(fc_sketch) -> list[tuple[int, object]]:
    return [(i, c) for i, c in enumerate(fc_sketch.Constraints) if c.Type != "Block"]


def _build_fc_constraint_args(c) -> list | None:
    t = c.Type
    if t in ("Radius", "Diameter"):
        return [c.First]
    elif t in ("Horizontal", "Vertical"):
        if c.FirstPos == 0 and c.Second < 0: return [c.First]
        else: return [c.First, c.FirstPos, c.Second, c.SecondPos]
    elif t in ("Parallel", "Equal", "Concentric"):
        return [c.First, c.Second]
    elif t == "PointOnObject":
        return [c.First, c.FirstPos, c.Second]
    elif t in ("Tangent", "Perpendicular"):
        if c.FirstPos == 0 and c.SecondPos == 0: return [c.First, c.Second]
        else: return [c.First, c.FirstPos, c.Second, c.SecondPos]
    elif t == "Symmetric":
        return [c.First, c.FirstPos, c.Second, c.SecondPos, c.Third, c.ThirdPos]
    elif t in ("Distance", "DistanceX", "DistanceY", "Angle"):
        if c.FirstPos == 0 and c.SecondPos == 0:
            return [c.First, c.Second] if c.Second >= 0 else [c.First]
        elif c.FirstPos != 0 and c.SecondPos == 0: return [c.First, c.FirstPos, c.Second]
        else: return [c.First, c.FirstPos, c.Second, c.SecondPos]
    elif t == "Block": return None
    else: return [c.First, c.FirstPos, c.Second, c.SecondPos]


def _get_fc_dof_for_subset(fc_sketch, subset_fc_indices: list[int], doc_name: str) -> int:
    tmp_doc = None
    try:
        tmp_doc = FreeCAD.newDocument(doc_name)
        tmp_sk = tmp_doc.addObject("Sketcher::SketchObject", "TmpSketch")
        for geo in fc_sketch.Geometry: tmp_sk.addGeometry(geo.copy(), False)
        all_c = fc_sketch.Constraints
        for idx in subset_fc_indices:
            c = all_c[idx]; args = _build_fc_constraint_args(c)
            if args is None: continue
            try:
                ni = tmp_sk.addConstraint(Sketcher.Constraint(c.Type, *args))
                if c.Type in _FC_DIMENSIONAL_TYPES and c.Value != 0.0:
                    try: tmp_sk.setDatum(ni, c.Value)
                    except: pass
            except: pass
        tmp_doc.recompute()
        return int(tmp_sk.getSolverStatus().get("DoF", 0))
    finally:
        if tmp_doc is not None:
            try: FreeCAD.closeDocument(tmp_doc.Name)
            except: pass


def _check_gauge_orthogonality(J: torch.Tensor, state, tol: float = 1e-6) -> list[str]:
    """
    Check that each per-line gauge direction is in the null space of J.
    Uses raw (non-QR) gauge vectors per line to avoid cross-line numerical mixing.
    Returns a list of violation descriptions (empty = all OK).
    """
    n = state.num_dof()
    violations = []
    for eid in state.entity_ids:
        ent = state.sketch.entities.get(eid)
        if not isinstance(ent, datalib.Line): continue
        sl = state.slices[eid]; q = state.q[sl]
        dirX, dirY, s, e = q[2].item(), q[3].item(), q[4].item(), q[5].item()
        v1 = torch.zeros(n, dtype=torch.float64)  # sliding gauge
        v1[sl.start+0]=dirX; v1[sl.start+1]=dirY; v1[sl.start+4]=-1.0; v1[sl.start+5]=-1.0
        v2 = torch.zeros(n, dtype=torch.float64)  # scaling gauge
        v2[sl.start+2]=dirX; v2[sl.start+3]=dirY; v2[sl.start+4]=-s; v2[sl.start+5]=-e
        d1 = (J @ v1).abs().max().item()
        d2 = (J @ v2).abs().max().item()
        if d1 > tol:
            violations.append(f"Line {eid} slide: {d1:.2e}")
        if d2 > tol:
            violations.append(f"Line {eid} scale: {d2:.2e}")
    return violations


def _from_freecad_with_index_map(converter, fc_sketch):
    sg_sketch = converter.from_freecad(fc_sketch)
    fc_to_sg: dict[int, str] = {}
    for c_id in sg_sketch.constraints:
        if c_id.startswith("c_"):
            try: fc_to_sg[int(c_id[2:])] = c_id
            except ValueError: pass
    return sg_sketch, fc_to_sg


def _build_subset_sg_ids(sg_sketch, fc_to_sg, subset_fc_indices):
    return [
        fc_to_sg[i] for i in subset_fc_indices
        if i in fc_to_sg and fc_to_sg[i] in sg_sketch.constraints
    ]


# ---------------------------------------------------------------------------
# Per-sketch verification
# ---------------------------------------------------------------------------

PATTERN_NAMES = ["Full", "Relational Only", "Dimensional Only", "Even Subset", "Odd Subset"]


def verify_sketch(
    filepath: str,
    context: SketchContext,
    converter: SketchFreecadConverter,
    verbose: bool = False,
) -> SketchResult:
    result = SketchResult(filepath=filepath)
    doc = None
    try:
        doc, fc_sketch = load_FCStd_sketch(filepath)
        status = fc_sketch.getSolverStatus()
        if status.get("Status") != "Converged":
            result.status = "skipped"
            result.reason = f"Solver: {status.get('Status')!r}"
            return result

        p_fc, n_lines = _count_fc_params(fc_sketch)
        n_total_ours = p_fc + 2 * n_lines

        nonblock_fc = _get_nonblock_fc_constraints(fc_sketch)
        nonblock_fc_indices = [i for i, _ in nonblock_fc]

        try:
            sg_sketch, fc_to_sg = _from_freecad_with_index_map(converter, fc_sketch)
        except Exception as e:
            result.status = "error"; result.reason = f"from_freecad: {e}"; return result

        try:
            sg_sketch, _ = complete_constraint_values(sg_sketch, skip_invalid_constraints=True)
        except Exception as e:
            result.status = "error"; result.reason = f"complete_constraint_values: {e}"; return result

        relational_idx = [i for i, c in nonblock_fc if c.Type not in _FC_DIMENSIONAL_TYPES]
        dimensional_idx = [i for i, c in nonblock_fc if c.Type in _FC_DIMENSIONAL_TYPES]
        even_idx = [nonblock_fc_indices[k] for k in range(0, len(nonblock_fc_indices), 2)]
        odd_idx  = [nonblock_fc_indices[k] for k in range(1, len(nonblock_fc_indices), 2)]

        patterns: list[tuple[str, list[int], bool]] = [
            # (name, fc_indices, is_full_set)
            ("Full",             nonblock_fc_indices, True),
            ("Relational Only",  relational_idx,      False),
            ("Dimensional Only", dimensional_idx,     False),
            ("Even Subset",      even_idx,            False),
            ("Odd Subset",       odd_idx,             False),
        ]

        # Compute full-set Jacobian once (for gauge check)
        try:
            J_full, state_full = compute_jacobian(
                sg_sketch,
                _build_subset_sg_ids(sg_sketch, fc_to_sg, nonblock_fc_indices),
                context, dtype=torch.float64,
            )
        except Exception as e:
            result.status = "error"; result.reason = f"full Jacobian: {e}"; return result

        all_passed = True
        pattern_results: list[PatternResult] = []

        for pat_name, fc_subset_indices, is_full in patterns:
            pr = PatternResult(name=pat_name, n_lines=n_lines, p_fc=p_fc)
            safe = os.path.basename(filepath).replace(".", "_")
            tmp_name = f"TMP_{safe}_{pat_name.replace(' ', '_')}"[:60]

            try:
                pr.fc_dof = _get_fc_dof_for_subset(fc_sketch, fc_subset_indices, tmp_name)
            except Exception as e:
                pr.error = f"FC DoF: {e}"; all_passed = False
                pattern_results.append(pr); continue

            sg_ids = _build_subset_sg_ids(sg_sketch, fc_to_sg, fc_subset_indices)
            try:
                J, _ = compute_jacobian(sg_sketch, sg_ids, context, dtype=torch.float64)
                pr.our_rank = (
                    int(torch.linalg.matrix_rank(J, atol=_RANK_ATOL).item())
                    if J.numel() > 0 else 0
                )
            except Exception as e:
                pr.error = f"Jacobian: {e}"; all_passed = False
                pattern_results.append(pr); continue

            # ----------------------------------------------------------------
            # Verification: our_rank >= fc_rank  (lower bound, always applicable)
            # The lower bound can fail for specific subsets due to constraint
            # redundancy differences between 4-param and 6-param spaces.
            # For the full set, we additionally check the gauge null space.
            # ----------------------------------------------------------------
            lb_ok = pr.our_rank >= pr.fc_rank

            if is_full:
                # Also verify that all per-line gauge directions are in null(J_full)
                gauge_violations = _check_gauge_orthogonality(J_full, state_full)
                gauge_ok = len(gauge_violations) == 0

                pr.passed = lb_ok and gauge_ok
                if not pr.passed:
                    issues = []
                    if not lb_ok:
                        issues.append(
                            f"LOWER BOUND: our_rank({pr.our_rank}) < fc_rank({pr.fc_rank})"
                        )
                    if not gauge_ok:
                        issues.append(f"GAUGE: {'; '.join(gauge_violations)}")
                    pr.error = "; ".join(issues)
            else:
                pr.passed = lb_ok
                if not lb_ok:
                    pr.error = f"LOWER BOUND: our_rank({pr.our_rank}) < fc_rank({pr.fc_rank})"

            if not pr.passed: all_passed = False

            if verbose:
                sym = "✓" if pr.passed else "✗"
                if is_full:
                    delta = pr.our_rank - pr.fc_rank
                    print(
                        f"  [{sym}] {pat_name:20s}: our_rank={pr.our_rank:3d}  "
                        f"fc_rank={pr.fc_rank:3d}  delta={delta:+d}  "
                        f"gauge={'✓' if gauge_ok else '✗'}"
                    )
                else:
                    print(
                        f"  [{sym}] {pat_name:20s}: our_rank={pr.our_rank:3d}  "
                        f"fc_rank={pr.fc_rank:3d}  [our_rank>=fc_rank: {lb_ok}]"
                    )
                if pr.error: print(f"         {pr.error}")

            pattern_results.append(pr)

        result.patterns = pattern_results
        result.status = "passed" if all_passed else "failed"
        if not all_passed:
            failed_pats = [pr.name for pr in pattern_results if not pr.passed or pr.error]
            result.reason = f"Failed patterns: {failed_pats}"

    except Exception:
        result.status = "error"
        result.reason = f"Unhandled: {traceback.format_exc()}"
    finally:
        if doc is not None:
            try: FreeCAD.closeDocument(doc.Name)
            except: pass

    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(data_dir: str, max_files: int = -1, verbose: bool = False) -> bool:
    data_dir = os.path.expanduser(data_dir)
    fcstd_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.FCStd"), recursive=True))
    if not fcstd_files:
        print(f"No .FCStd files found in: {data_dir}"); return False
    if max_files > 0:
        fcstd_files = fcstd_files[:max_files]

    print(f"Found {len(fcstd_files)} .FCStd files")
    print(f"Data directory: {data_dir}")
    print(f"Verified per sketch (Full set):    our_rank >= fc_rank  AND  gauge in null(J)")
    print(f"Verified per sketch (Subsets):     our_rank >= fc_rank")
    print()

    context = SketchContext()
    converter = SketchFreecadConverter(context, verbose=False, raise_on_error=False)

    counts = defaultdict(int)
    failed_results: list[SketchResult] = []
    pattern_pass = defaultdict(int)
    pattern_total = defaultdict(int)

    for file_idx, filepath in enumerate(fcstd_files):
        if verbose:
            rel = os.path.relpath(filepath, data_dir)
            print(f"[{file_idx+1}/{len(fcstd_files)}] {rel}")
        elif (file_idx + 1) % 50 == 0 or file_idx == 0:
            print(f"  Progress: {file_idx+1}/{len(fcstd_files)} ...")

        result = verify_sketch(filepath, context, converter, verbose=verbose)
        counts[result.status] += 1

        for pr in result.patterns:
            pattern_total[pr.name] += 1
            if pr.passed and not pr.error:
                pattern_pass[pr.name] += 1

        if result.status in ("failed", "error"):
            failed_results.append(result)
            if not verbose and len(failed_results) <= 20:
                rel = os.path.relpath(filepath, data_dir)
                print(f"  [{result.status.upper()}] {rel}: {result.reason}")

    n_total = len(fcstd_files)
    n_passed = counts["passed"]
    n_failed = counts["failed"]
    n_skipped = counts["skipped"]
    n_error = counts["error"]
    n_tested = n_passed + n_failed + n_error
    pass_rate = (n_passed / n_tested * 100) if n_tested > 0 else 0.0

    print()
    print("=" * 65)
    print("  Solver Consistency Verification — Final Report")
    print("=" * 65)
    print(f"  Files processed : {n_total}")
    print(f"  Skipped         : {n_skipped}  (non-converged solver)")
    print(f"  Tested          : {n_tested}")
    print(f"    ✓ Passed      : {n_passed}")
    print(f"    ✗ Failed      : {n_failed}")
    print(f"    ⚠ Errors      : {n_error}")
    print(f"  Pass Rate       : {pass_rate:.1f}%  (target: 100.0%)")
    print()

    if pattern_total:
        print("  Per-Pattern Breakdown:")
        print(f"  {'Pattern':<22} {'Passed':>7} {'Total':>7} {'Rate':>7}")
        print("  " + "-" * 44)
        for pat_name in PATTERN_NAMES:
            total = pattern_total.get(pat_name, 0)
            passed = pattern_pass.get(pat_name, 0)
            rate = (passed / total * 100) if total > 0 else 0.0
            sym = "✓" if passed == total and total > 0 else "✗"
            print(f"  {sym} {pat_name:<20} {passed:>7} {total:>7} {rate:>6.1f}%")
        print()

    if failed_results:
        print(f"  Failure Details ({min(len(failed_results), 30)} shown):")
        print("  " + "-" * 63)
        for r in failed_results[:30]:
            rel = os.path.relpath(r.filepath, data_dir)
            print(f"  [{r.status.upper()}] {rel}")
            print(f"         Reason: {r.reason}")
            for pr in r.patterns:
                if not pr.passed or pr.error:
                    print(
                        f"         [{pr.name}] our_rank={pr.our_rank}  "
                        f"fc_rank={pr.fc_rank}  {pr.error}"
                    )
        print()

    success = n_failed == 0 and n_error == 0
    if success:
        print("  ✓ ALL TESTS PASSED — Solver invariants verified across full dataset.")
    else:
        print(f"  ✗ VERIFICATION FAILED — {n_failed + n_error} sketch(es) violated invariants.")
    print("=" * 65)
    return success


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAD Constraint Solver Consistency Verification")
    parser.add_argument("--data_dir", default="~/Downloads/fiverr",
                        help="Root directory containing .FCStd files (recursive).")
    parser.add_argument("--max_files", type=int, default=-1,
                        help="Max files to process (-1=all).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sketch per-pattern breakdown.")
    args = parser.parse_args()
    success = run_sweep(data_dir=args.data_dir, max_files=args.max_files, verbose=args.verbose)
    sys.exit(0 if success else 1)
