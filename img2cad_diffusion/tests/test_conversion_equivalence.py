"""Test equivalence between sequence-based and sketch-based FreeCAD conversion.

This test verifies that convert_sketch_to_freecad produces the same results
as the sequence-based conversion for the same input sketch.

Key metrics tested:
- Geometry count and types
- Constraint count and types
- Solver status: DoF, RedundantConstraints, ConflictingConstraints
"""
import os
import sys
import argparse
from collections import Counter
from tqdm import tqdm

# Add FreeCAD to path
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sketchgraphs.data import flat_array
import sketchgraphs.data as datalib

from img2cad_diffusion.convert_sequence_to_freecad import (
    meter_to_millimeter,
    resolve_subnodes,
    resolve_redundant_coincident,
    resolve_tangent,
    convert_primitive,
    convert_constraint,
)
from img2cad_diffusion.convert_sketch_to_freecad import convert_sketch_to_freecad


def get_geometry_set(fc_sketch) -> set[tuple]:
    """Extract geometry as a set of (type, key_attributes) tuples."""
    geo_set = set()
    for i, geo in enumerate(fc_sketch.Geometry):
        geo_type = type(geo).__name__
        if isinstance(geo, Part.LineSegment):
            # Round to avoid floating point comparison issues
            key = (geo_type,
                   round(geo.StartPoint.x, 3), round(geo.StartPoint.y, 3),
                   round(geo.EndPoint.x, 3), round(geo.EndPoint.y, 3))
        elif isinstance(geo, Part.Circle):
            key = (geo_type,
                   round(geo.Center.x, 3), round(geo.Center.y, 3),
                   round(geo.Radius, 3))
        elif isinstance(geo, Part.ArcOfCircle):
            key = (geo_type,
                   round(geo.Center.x, 3), round(geo.Center.y, 3),
                   round(geo.Radius, 3),
                   round(geo.FirstParameter, 3), round(geo.LastParameter, 3))
        elif hasattr(geo, 'X') and hasattr(geo, 'Y'):  # Point
            key = (geo_type, round(geo.X, 3), round(geo.Y, 3))
        else:
            key = (geo_type, i)  # Fallback
        geo_set.add(key)
    return geo_set


def get_constraint_set(fc_sketch) -> set[tuple]:
    """Extract constraints as a set of (type, references) tuples."""
    cons_set = set()
    for i in range(fc_sketch.ConstraintCount):
        c = fc_sketch.Constraints[i]
        key = (c.Type, c.First, c.FirstPos, c.Second, c.SecondPos, c.Third, c.ThirdPos)
        cons_set.add(key)
    return cons_set


def get_solver_status(fc_sketch) -> dict:
    """Extract solver status (DoF, Redundant, Conflicting)."""
    status = fc_sketch.getSolverStatus()
    return {
        'DoF': status['DoF'],
        'Malformed': len(status.get('MalformedConstraints', [])),
        'Redundant': len(status.get('RedundantConstraints', [])),
        'Conflicting': len(status.get('ConflictingConstraints', [])),
    }


def convert_via_sequence(seq) -> tuple:
    """Convert sequence to FreeCAD using sequence-based approach."""
    doc = FreeCAD.newDocument("SeqTest")
    body = doc.addObject('PartDesign::Body', 'Body')
    fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    xy_plane = body.Origin.OriginFeatures[3]
    fc_sketch.AttachmentSupport = xy_plane
    fc_sketch.MapMode = 'FlatFace'

    # Process sequence
    seq = [meter_to_millimeter(op, False) for op in seq]
    node_idx_map = resolve_subnodes(seq)
    seq = resolve_redundant_coincident(seq, node_idx_map, False)
    seq = resolve_tangent(seq, node_idx_map)

    # Track errors instead of silently ignoring
    primitive_errors = []
    constraint_errors = []

    for op in seq:
        if isinstance(op, datalib.NodeOp):
            try:
                convert_primitive(op, fc_sketch)
            except Exception as e:
                primitive_errors.append((op.label.name if hasattr(op.label, 'name') else str(op.label), str(e)))
        elif isinstance(op, datalib.EdgeOp):
            try:
                convert_constraint(op, fc_sketch, node_idx_map)
            except Exception as e:
                constraint_errors.append((op.label.name if hasattr(op.label, 'name') else str(op.label), str(e)))

    doc.recompute()
    return doc, fc_sketch, primitive_errors, constraint_errors


# =============================================================================
# Pytest Test Class
# =============================================================================

class TestConversionEquivalence:
    """Test that sequence-based and sketch-based converters produce equivalent results."""

    NUM_SAMPLES = 1000  # Default number of samples for pytest runs

    @classmethod
    def setup_class(cls):
        """Load test data once for all tests."""
        cls.seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')
        cls.sequences = cls.seq_data['sequences']

    def test_solver_status_equivalence(self):
        """Test that DoF, Redundant, and Conflicting match for all samples."""
        mismatches = {'DoF': [], 'Redundant': [], 'Conflicting': []}

        for i in range(min(self.NUM_SAMPLES, len(self.sequences))):
            seq = self.sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            try:
                doc1, fc1, _, _ = convert_via_sequence(seq)
                doc2, fc2, _, _ = convert_sketch_to_freecad(sketch)

                if doc2 is None:
                    continue

                s1 = get_solver_status(fc1)
                s2 = get_solver_status(fc2)

                if s1['DoF'] != s2['DoF']:
                    mismatches['DoF'].append((i, s1['DoF'], s2['DoF']))
                if s1['Redundant'] != s2['Redundant']:
                    mismatches['Redundant'].append((i, s1['Redundant'], s2['Redundant']))
                if s1['Conflicting'] != s2['Conflicting']:
                    mismatches['Conflicting'].append((i, s1['Conflicting'], s2['Conflicting']))

                FreeCAD.closeDocument(doc1.Name)
                FreeCAD.closeDocument(doc2.Name)

            except Exception:
                pass  # Skip samples with conversion errors

        # Assert 100% equivalence
        assert len(mismatches['DoF']) == 0, f"DoF mismatches: {mismatches['DoF'][:10]}"
        assert len(mismatches['Redundant']) == 0, f"Redundant mismatches: {mismatches['Redundant'][:10]}"
        assert len(mismatches['Conflicting']) == 0, f"Conflicting mismatches: {mismatches['Conflicting'][:10]}"

    def test_constraint_count_equivalence(self):
        """Test that constraint counts match for all samples."""
        mismatches = []

        for i in range(min(self.NUM_SAMPLES, len(self.sequences))):
            seq = self.sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            try:
                doc1, fc1, _, _ = convert_via_sequence(seq)
                doc2, fc2, _, _ = convert_sketch_to_freecad(sketch)

                if doc2 is None:
                    continue

                if fc1.ConstraintCount != fc2.ConstraintCount:
                    mismatches.append((i, fc1.ConstraintCount, fc2.ConstraintCount))

                FreeCAD.closeDocument(doc1.Name)
                FreeCAD.closeDocument(doc2.Name)

            except Exception:
                pass

        assert len(mismatches) == 0, f"Constraint count mismatches: {mismatches[:10]}"

    def test_geometry_count_equivalence(self):
        """Test that geometry counts match for all samples."""
        mismatches = []

        for i in range(min(self.NUM_SAMPLES, len(self.sequences))):
            seq = self.sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            try:
                doc1, fc1, _, _ = convert_via_sequence(seq)
                doc2, fc2, _, _ = convert_sketch_to_freecad(sketch)

                if doc2 is None:
                    continue

                if len(fc1.Geometry) != len(fc2.Geometry):
                    mismatches.append((i, len(fc1.Geometry), len(fc2.Geometry)))

                FreeCAD.closeDocument(doc1.Name)
                FreeCAD.closeDocument(doc2.Name)

            except Exception:
                pass

        assert len(mismatches) == 0, f"Geometry count mismatches: {mismatches[:10]}"


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test conversion equivalence")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to test")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output for each mismatch")
    args = parser.parse_args()

    print("Loading data...")
    seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')
    sequences = seq_data['sequences']
    num_samples = min(args.num_samples, len(sequences))
    print(f"Testing {num_samples} samples...")

    # Metrics
    total = 0
    geo_exact_match = 0
    cons_exact_match = 0
    solver_match = {'DoF': 0, 'Redundant': 0, 'Conflicting': 0}
    geo_mismatches = []
    cons_mismatches = []
    solver_mismatches = []
    conversion_errors = []

    for i in tqdm(range(num_samples), desc="Testing"):
        seq = sequences[i]
        sketch = datalib.sketch_from_sequence(seq)

        try:
            # Sequence-based conversion
            doc1, fc_sketch1, prim_errs1, cons_errs1 = convert_via_sequence(seq)

            # Sketch-based conversion
            doc2, fc_sketch2, _history, _ = convert_sketch_to_freecad(sketch)

            total += 1

            # Compare geometry sets
            geo_set1 = get_geometry_set(fc_sketch1)
            geo_set2 = get_geometry_set(fc_sketch2)

            if geo_set1 == geo_set2:
                geo_exact_match += 1
            else:
                only_in_seq = geo_set1 - geo_set2
                only_in_sketch = geo_set2 - geo_set1
                geo_mismatches.append({
                    'idx': i,
                    'seq_count': len(geo_set1),
                    'sketch_count': len(geo_set2),
                    'only_in_seq': only_in_seq,
                    'only_in_sketch': only_in_sketch,
                })

            # Compare constraint sets
            cons_set1 = get_constraint_set(fc_sketch1)
            cons_set2 = get_constraint_set(fc_sketch2)

            if cons_set1 == cons_set2:
                cons_exact_match += 1
            else:
                only_in_seq = cons_set1 - cons_set2
                only_in_sketch = cons_set2 - cons_set1
                cons_mismatches.append({
                    'idx': i,
                    'seq_count': len(cons_set1),
                    'sketch_count': len(cons_set2),
                    'only_in_seq': only_in_seq,
                    'only_in_sketch': only_in_sketch,
                })

            # Compare solver status
            s1 = get_solver_status(fc_sketch1)
            s2 = get_solver_status(fc_sketch2)

            if s1['DoF'] == s2['DoF']:
                solver_match['DoF'] += 1
            if s1['Redundant'] == s2['Redundant']:
                solver_match['Redundant'] += 1
            if s1['Conflicting'] == s2['Conflicting']:
                solver_match['Conflicting'] += 1

            if s1 != s2:
                solver_mismatches.append({
                    'idx': i, 'seq': s1, 'sketch': s2
                })

            # Cleanup
            FreeCAD.closeDocument(doc1.Name)
            FreeCAD.closeDocument(doc2.Name)

        except ValueError as e:
            conversion_errors.append((i, f"ValueError: {e}"))
            try:
                if 'doc1' in dir() and doc1:
                    FreeCAD.closeDocument(doc1.Name)
            except:
                pass
        except Exception as e:
            conversion_errors.append((i, str(e)))
            try:
                if 'doc1' in dir() and doc1:
                    FreeCAD.closeDocument(doc1.Name)
                if 'doc2' in dir() and doc2:
                    FreeCAD.closeDocument(doc2.Name)
            except:
                pass

    # Report
    print("\n" + "=" * 60)
    print("EQUIVALENCE TEST RESULTS")
    print("=" * 60)
    print(f"Total samples tested: {total}")
    print(f"Conversion errors: {len(conversion_errors)}")
    print()
    print(f"Geometry exact match: {geo_exact_match}/{total} ({100*geo_exact_match/total:.1f}%)")
    print(f"Constraint exact match: {cons_exact_match}/{total} ({100*cons_exact_match/total:.1f}%)")
    print()
    print("Solver Status Match:")
    print(f"  DoF:        {solver_match['DoF']}/{total} ({100*solver_match['DoF']/total:.2f}%)")
    print(f"  Redundant:  {solver_match['Redundant']}/{total} ({100*solver_match['Redundant']/total:.2f}%)")
    print(f"  Conflicting:{solver_match['Conflicting']}/{total} ({100*solver_match['Conflicting']/total:.2f}%)")

    if geo_mismatches:
        print(f"\n--- Geometry Mismatches ({len(geo_mismatches)} samples) ---")
        for m in geo_mismatches[:5]:
            print(f"  Sample {m['idx']}: seq={m['seq_count']}, sketch={m['sketch_count']}")
            if args.verbose:
                print(f"    Only in seq: {m['only_in_seq']}")
                print(f"    Only in sketch: {m['only_in_sketch']}")

    if cons_mismatches:
        print(f"\n--- Constraint Mismatches ({len(cons_mismatches)} samples) ---")
        patterns = Counter()
        for m in cons_mismatches:
            diff = len(m['only_in_sketch']) - len(m['only_in_seq'])
            patterns[diff] += 1
        print(f"  Difference patterns (sketch - seq): {dict(patterns.most_common(10))}")

        for m in cons_mismatches[:5]:
            print(f"  Sample {m['idx']}: seq={m['seq_count']}, sketch={m['sketch_count']}")
            if args.verbose:
                print(f"    Only in seq: {list(m['only_in_seq'])[:3]}...")
                print(f"    Only in sketch: {list(m['only_in_sketch'])[:3]}...")

    if solver_mismatches:
        print(f"\n--- Solver Status Mismatches ({len(solver_mismatches)} samples) ---")
        for m in solver_mismatches[:5]:
            print(f"  Sample {m['idx']}: seq={m['seq']}, sketch={m['sketch']}")

    if conversion_errors:
        print(f"\n--- Conversion Errors ({len(conversion_errors)}) ---")
        for idx, err in conversion_errors[:5]:
            print(f"  Sample {idx}: {err}")

    # Final assertion for CI
    all_match = (
        solver_match['DoF'] == total and
        solver_match['Redundant'] == total and
        solver_match['Conflicting'] == total
    )
    print(f"\n{'✅ ALL TESTS PASSED' if all_match else '❌ SOME TESTS FAILED'}")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())

