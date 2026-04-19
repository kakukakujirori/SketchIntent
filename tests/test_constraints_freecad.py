"""
Constraint roundtrip verification test.

Two modes:
  1. Unit tests: hand-crafted sketches covering each constraint type (pytest)
  2. Dataset test: load real .py sketch scripts and verify constraint roundtrip (CLI)

Usage:
  pytest tests/verify_constraints_freecad.py -v           # unit tests only
  conda run -n freecad python tests/verify_constraints_freecad.py  # dataset test
"""
from typing import Any
import glob
import math
import os
import sys
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sketchgraphs.data as datalib
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.freecad_utils import ensure_freecad_imported

FreeCAD, Part, Sketcher = ensure_freecad_imported()


def remove_external_refs(fc_sketch: 'Sketcher.SketchObject') -> 'Sketcher.SketchObject':
    invalid_indices = []
    for c_idx, c in enumerate(fc_sketch.Constraints):
        if c.First < 0 or c.Second < 0:
            invalid_indices.append(c_idx)
    for c_idx in invalid_indices[::-1]:
        fc_sketch.delConstraint(c_idx)
    return fc_sketch


def extract_constraints_from_sketch(fc_sketch: 'Sketcher.SketchObject') -> list[dict]:
    """
    Extract constraint info from a FreeCAD SketchObject as comparable dicts.
    """
    constraints = []
    for c in fc_sketch.Constraints:
        info = {
            'type': c.Type,
            'first': c.First,
            'firstPos': c.FirstPos,
            'second': c.Second,
            'secondPos': c.SecondPos,
            'third': c.Third,
            'thirdPos': c.ThirdPos,
            'value': c.Value,
        }
        constraints.append(info)
    return constraints


def compare_constraints(
    orig_constraints: list[dict[str, Any]],
    recon_constraints: list[dict[str, Any]],
    value_tolerance: float = 1e-6,
) -> dict:
    """
    Compare two lists of constraint dicts and return a summary.

    Returns dict with keys:
      - count_match: bool
      - type_match: bool (sorted type lists match)
      - type_count_match: bool (per-type counts match)
      - value_mismatches: list of (index, orig_val, recon_val, delta)
      - missing_types: set of types in orig but not recon
      - extra_types: set of types in recon but not orig
      - orig_types: Counter
      - recon_types: Counter
    """
    from collections import Counter

    result = {}

    # Count match
    result['count_match'] = len(orig_constraints) == len(recon_constraints)
    result['orig_count'] = len(orig_constraints)
    result['recon_count'] = len(recon_constraints)

    # Type counts
    orig_type_counts = Counter(c['type'] for c in orig_constraints)
    recon_type_counts = Counter(c['type'] for c in recon_constraints)
    result['orig_types'] = dict(orig_type_counts)
    result['recon_types'] = dict(recon_type_counts)

    # Sorted type lists
    orig_type_list = sorted(c['type'] for c in orig_constraints)
    recon_type_list = sorted(c['type'] for c in recon_constraints)
    result['type_match'] = orig_type_list == recon_type_list

    # Per-type count match
    result['type_count_match'] = orig_type_counts == recon_type_counts

    # Missing / extra types
    result['missing_types'] = set(orig_type_counts.keys()) - set(recon_type_counts.keys())
    result['extra_types'] = set(recon_type_counts.keys()) - set(orig_type_counts.keys())

    # Value comparison (only for constraints with the same type in order)
    # Group by type, compare values within each group
    value_mismatches = []

    # Build per-type value lists
    orig_values_by_type = {}
    for c in orig_constraints:
        orig_values_by_type.setdefault(c['type'], []).append(c['value'])

    recon_values_by_type = {}
    for c in recon_constraints:
        recon_values_by_type.setdefault(c['type'], []).append(c['value'])

    for ctype in orig_values_by_type:
        if ctype not in recon_values_by_type:
            continue
        orig_vals = sorted(orig_values_by_type[ctype])
        recon_vals = sorted(recon_values_by_type[ctype])
        for i, (ov, rv) in enumerate(zip(orig_vals, recon_vals)):
            delta = abs(ov - rv)
            if delta > value_tolerance:
                value_mismatches.append((ctype, i, ov, rv, delta))

    result['value_mismatches'] = value_mismatches

    return result


# ============================================================
# Dataset Verification (matches verify_primitives_freecad.py style)
# ============================================================

def verify_constraints_freecad(directory: str = 'gt_completed_prim5_py', start_idx: int = 0, process_num: int = -1, verbose: bool = False):
    """
    Load real .py sketch scripts, do FC -> SG -> FC roundtrip,
    and verify constraint preservation.
    """
    print(f"Scanning directory: {directory}...")
    py_files = sorted(glob.glob(os.path.join(directory, "*.py")))

    if not py_files:
        print("No .py files found.")
        return False

    print(f"Found {len(py_files)} files.")
    if process_num < 0:
        process_num = len(py_files)
    end_idx = min(start_idx + process_num, len(py_files))

    context = SketchContext()
    converter = SketchFreecadConverter(context, raise_on_error=True)

    passed_files = 0
    failed_files = 0
    error_files = 0
    total_constraints_orig = 0
    total_constraints_recon = 0

    # Track per-type statistics
    from collections import Counter
    type_pass_counts = Counter()
    type_fail_counts = Counter()
    type_missing_counts = Counter()

    for i in range(start_idx, end_idx):
        py_file = py_files[i]
        basename = os.path.basename(py_file)

        doc_orig = None
        doc_recon = None

        try:
            # ---------------------------------------------------------------
            # 1. Load Original Script -> FreeCAD
            # ---------------------------------------------------------------
            initial_docs = set(FreeCAD.listDocuments().keys())

            exec_globals = {
                "FreeCAD": FreeCAD,
                "Part": Part,
                "Sketcher": Sketcher,
                "Vector": FreeCAD.Vector,
            }
            with open(py_file, 'r') as f:
                script_content = f.read()
            exec(script_content, exec_globals)

            current_docs = set(FreeCAD.listDocuments().keys())
            new_docs = list(current_docs - initial_docs)

            if not new_docs:
                if FreeCAD.ActiveDocument:
                    doc_orig = FreeCAD.ActiveDocument
                else:
                    error_files += 1
                    continue
            else:
                doc_orig = FreeCAD.getDocument(new_docs[0])

            sketches = doc_orig.findObjects(Type="Sketcher::SketchObject")
            if not sketches:
                error_files += 1
                FreeCAD.closeDocument(doc_orig.Name)
                continue

            fc_sketch_orig = sketches[0]
            fc_sketch_orig = remove_external_refs(fc_sketch_orig)
            orig_constraints = extract_constraints_from_sketch(fc_sketch_orig)
            total_constraints_orig += len(orig_constraints)

            if verbose:
                print("[Original Constraints]")
                for constr in orig_constraints:
                    print(constr)
                print()

            # ---------------------------------------------------------------
            # 2. FC -> SketchGraphs
            # ---------------------------------------------------------------
            sg_sketch = converter.from_freecad(fc_sketch_orig)

            if verbose:
                print("[SketchGraphs Constraints]")
                for constr_id, constr in sg_sketch.constraints.items():
                    print(constr)
                print()

            # ---------------------------------------------------------------
            # 3. SketchGraphs -> FC
            # ---------------------------------------------------------------
            doc_recon, fc_sketch_recon, _, _, _ = converter.to_freecad(sg_sketch, doc_name=f"ReconDoc_{i}")
            recon_constraints = extract_constraints_from_sketch(fc_sketch_recon)
            total_constraints_recon += len(recon_constraints)

            if verbose:
                print("[Reconstructed Constraints]")
                for constr in recon_constraints:
                    print(constr)
                print()

            # ---------------------------------------------------------------
            # 4. Compare
            # ---------------------------------------------------------------
            cmp = compare_constraints(orig_constraints, recon_constraints)

            # Count per-type pass/fail
            for ctype, count in cmp['orig_types'].items():
                recon_count = cmp['recon_types'].get(ctype, 0)
                if recon_count >= count:
                    type_pass_counts[ctype] += count
                else:
                    type_pass_counts[ctype] += recon_count
                    type_fail_counts[ctype] += count - recon_count

            for ctype in cmp['missing_types']:
                type_missing_counts[ctype] += cmp['orig_types'][ctype]

            if cmp['type_count_match'] and not cmp['value_mismatches']:
                passed_files += 1
            else:
                failed_files += 1
                if i < 20 or failed_files <= 10:  # Show first few failures
                    print(f"[{i}] {basename}: FAIL")
                    if not cmp['count_match']:
                        print(f"  Count: {cmp['orig_count']} -> {cmp['recon_count']}")
                    if not cmp['type_count_match']:
                        print(f"  Orig types: {cmp['orig_types']}")
                        print(f"  Recon types: {cmp['recon_types']}")
                    if cmp['missing_types']:
                        print(f"  Missing: {cmp['missing_types']}")
                    if cmp['extra_types']:
                        print(f"  Extra: {cmp['extra_types']}")
                    if cmp['value_mismatches']:
                        for vm in cmp['value_mismatches'][:5]:
                            print(f"  Value mismatch: {vm[0]}[{vm[1]}] {vm[2]:.6f} -> {vm[3]:.6f} (Δ={vm[4]:.6f})")

            # Cleanup
            FreeCAD.closeDocument(doc_orig.Name)
            doc_orig = None
            FreeCAD.closeDocument(doc_recon.Name)
            doc_recon = None

        except Exception as e:
            error_files += 1
            if error_files <= 10:
                print(f"[{i}] {basename}: ERROR — {e}")
            try:
                if doc_orig and doc_orig.Name in FreeCAD.listDocuments():
                    FreeCAD.closeDocument(doc_orig.Name)
                if doc_recon and doc_recon.Name in FreeCAD.listDocuments():
                    FreeCAD.closeDocument(doc_recon.Name)
            except:
                pass

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Constraint Roundtrip Verification Complete")
    print(f"{'='*60}")
    print(f"Files Processed: {process_num}")
    print(f"  Passed:  {passed_files}")
    print(f"  Failed:  {failed_files}")
    print(f"  Errors:  {error_files}")
    pass_rate = (passed_files / max(process_num, 1)) * 100
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"\nTotal Constraints: {total_constraints_orig} orig, {total_constraints_recon} recon")

    if type_pass_counts or type_fail_counts or type_missing_counts:
        print(f"\nPer-Type Breakdown:")
        all_types = sorted(set(type_pass_counts) | set(type_fail_counts) | set(type_missing_counts))
        for ctype in all_types:
            p = type_pass_counts.get(ctype, 0)
            f = type_fail_counts.get(ctype, 0)
            m = type_missing_counts.get(ctype, 0)
            total = p + f + m
            rate = (p / total * 100) if total > 0 else 0
            status = "✓" if f == 0 and m == 0 else "✗"
            print(f"  {status} {ctype:20s}: {p:4d}/{total:4d} ({rate:5.1f}%)")

    return failed_files == 0 and error_files == 0


# ============================================================
# Unit Tests (pytest)
# ============================================================

class TestConstraintRoundTrip(unittest.TestCase):
    def setUp(self):
        self.context = SketchContext()
        self.tokenizer = SketchTokenizer(self.context)

    def tearDown(self):
        # Clean up documents
        for name in list(FreeCAD.listDocuments().keys()):
            FreeCAD.closeDocument(name)

    def _roundtrip_check(self, setup_func):
        """
        Helper to run roundtrip check.
        setup_func(sketch_obj) populates the initial sketch.
        """
        doc = FreeCAD.newDocument("TestDoc")
        sketch_orig = doc.addObject("Sketcher::SketchObject", "Sketch")
        setup_func(sketch_orig)
        doc.recompute()

        converter = SketchFreecadConverter(self.context, raise_on_error=True)

        # 1. FC -> SG
        sg_sketch = converter.from_freecad(sketch_orig)

        # 2. SG -> FC
        doc_recon, sketch_recon, _, _, _ = converter.to_freecad(sg_sketch, doc_name="ReconDoc")

        # 3. Compare
        orig_constraints = extract_constraints_from_sketch(sketch_orig)
        recon_constraints = extract_constraints_from_sketch(sketch_recon)

        cmp = compare_constraints(orig_constraints, recon_constraints)

        self.assertTrue(cmp['count_match'],
                        f"Constraint count mismatch: {cmp['orig_count']} -> {cmp['recon_count']}")
        self.assertTrue(cmp['type_match'],
                        f"Constraint types mismatch: {cmp['orig_types']} vs {cmp['recon_types']}")
        self.assertEqual(len(cmp['value_mismatches']), 0,
                         f"Value mismatches: {cmp['value_mismatches']}")

        return sketch_orig, sketch_recon

    def test_coincident(self):
        def setup(sk):
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,0,0)), False)
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(10,0,0), FreeCAD.Vector(10,10,0)), False)
            sk.addConstraint(Sketcher.Constraint('Coincident', 0, 2, 1, 1))
        self._roundtrip_check(setup)

    def test_vertical_horizontal(self):
        def setup(sk):
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,10,0)), False)
            sk.addConstraint(Sketcher.Constraint('Vertical', 0))

            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,0,0)), False)
            sk.addConstraint(Sketcher.Constraint('Horizontal', 1))
        self._roundtrip_check(setup)

    def test_parallel_perpendicular(self):
        def setup(sk):
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,10,0)), False)
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(20,0,0), FreeCAD.Vector(30,10,0)), False)
            sk.addConstraint(Sketcher.Constraint('Parallel', 0, 1))

            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,0,0)), False)
            sk.addConstraint(Sketcher.Constraint('Perpendicular', 1, 2))
        self._roundtrip_check(setup)

    def test_distance(self):
        def setup(sk):
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,0,0)), False)
            sk.addConstraint(Sketcher.Constraint('Distance', 0, 10.0))

            sk.addGeometry(Part.Point(FreeCAD.Vector(0,10,0)))
            sk.addGeometry(Part.Point(FreeCAD.Vector(10,10,0)))
            sk.addConstraint(Sketcher.Constraint('Distance', 1, 1, 2, 1, 10.0))

        self._roundtrip_check(setup)

    def test_radius_diameter(self):
        def setup(sk):
            sk.addGeometry(Part.Circle(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1), 10), False)
            sk.addConstraint(Sketcher.Constraint('Radius', 0, 10.0))

            sk.addGeometry(Part.Circle(FreeCAD.Vector(20,0,0), FreeCAD.Vector(0,0,1), 10), False)
            sk.addConstraint(Sketcher.Constraint('Diameter', 1, 20.0))
        self._roundtrip_check(setup)

    def test_angle(self):
        def setup(sk):
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,0,0)), False)
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,10,0)), False)
            sk.addConstraint(Sketcher.Constraint('Angle', 0, 1, 0.785398))
        self._roundtrip_check(setup)

    def test_midpoint_symmetric(self):
        def setup(sk):
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(0,0,0), FreeCAD.Vector(10,0,0)), False)
            sk.addGeometry(Part.Point(FreeCAD.Vector(5,0,0)))
            sk.addConstraint(Sketcher.Constraint('Symmetric', 0, 1, 0, 2, 1, 1))
        self._roundtrip_check(setup)

    def test_mirror_symmetric(self):
        def setup(sk):
            sk.addGeometry(Part.Point(FreeCAD.Vector(0, 10, 0)))
            sk.addGeometry(Part.Point(FreeCAD.Vector(0, -10, 0)))
            sk.addGeometry(Part.LineSegment(FreeCAD.Vector(-5, 0, 0), FreeCAD.Vector(5, 0, 0)))
            sk.addConstraint(Sketcher.Constraint('Symmetric', 0, 1, 1, 1, 2))
        self._roundtrip_check(setup)

    def test_concentric(self):
        def setup(sk):
            sk.addGeometry(Part.Circle(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1), 10), False)
            sk.addGeometry(Part.Circle(FreeCAD.Vector(0,0,0), FreeCAD.Vector(0,0,1), 5), False)
            sk.addConstraint(Sketcher.Constraint('Coincident', 0, 3, 1, 3))
        self._roundtrip_check(setup)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Constraint roundtrip verification')
    parser.add_argument('--dir', default='data_ryotaro_prim5', help='Directory with .py sketch scripts')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--process_num', type=int, default=-1, help='Number of files to process')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.unit:
        # Run unit tests via unittest
        sys.argv = sys.argv[:1]  # Strip custom args so unittest doesn't complain
        unittest.main()
    else:
        success = verify_constraints_freecad(directory=args.dir, start_idx=args.start_idx, process_num=args.process_num, verbose=args.verbose)
        sys.exit(0 if success else 1)
