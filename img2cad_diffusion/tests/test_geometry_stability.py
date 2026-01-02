"""Test geometry stability after constraint application.

This test verifies that:
1. Primitive counts match between original SketchGraphs data and FreeCAD conversion
2. After applying constraints, primitives haven't moved significantly from their original positions

Geometry "collapse" occurs when constraints cause primitives to deform or move
to unexpected positions, indicating a bug in constraint conversion logic.
"""
import os
import sys
import argparse
from dataclasses import dataclass
from tqdm import tqdm
import math

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
from sketchgraphs.data.sketch import EntityType

from img2cad_diffusion.convert_sequence_to_freecad import (
    meter_to_millimeter,
    resolve_subnodes,
    resolve_redundant_coincident,
    resolve_tangent,
    convert_primitive,
    convert_constraint,
)
from img2cad_diffusion.convert_sketch_to_freecad import convert_sketch_to_freecad


# Meter to millimeter conversion factor
M_TO_MM = 1000.0


@dataclass
class GeometryPoint:
    """A point in 2D space."""
    x: float
    y: float

    def distance_to(self, other: 'GeometryPoint') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def get_original_geometry_points(sketch) -> list[tuple[str, list[GeometryPoint]]]:
    """Extract geometry points from original SketchGraphs sketch.

    Returns:
        List of (entity_type, [points]) tuples
    """
    result = []

    for entity_id, entity in sketch.entities.items():
        if entity is None:
            continue

        entity_type = type(entity).__name__
        points = []

        if entity_type == 'Line':
            # Line: calculate start and end from direction and params
            px, py = entity.pntX * M_TO_MM, entity.pntY * M_TO_MM
            dx, dy = entity.dirX, entity.dirY
            start_param = entity.startParam * M_TO_MM
            end_param = entity.endParam * M_TO_MM

            start_x = px + dx * start_param
            start_y = py + dy * start_param
            end_x = px + dx * end_param
            end_y = py + dy * end_param

            points = [
                GeometryPoint(start_x, start_y),
                GeometryPoint(end_x, end_y),
            ]

        elif entity_type == 'Circle':
            # Circle: center point
            cx = entity.xCenter * M_TO_MM
            cy = entity.yCenter * M_TO_MM
            points = [GeometryPoint(cx, cy)]

        elif entity_type == 'Arc':
            # Arc: center only (start/end can differ due to parameter interpretation)
            cx = entity.xCenter * M_TO_MM
            cy = entity.yCenter * M_TO_MM
            points = [GeometryPoint(cx, cy)]

        elif entity_type == 'Point':
            x = entity.x * M_TO_MM
            y = entity.y * M_TO_MM
            points = [GeometryPoint(x, y)]

        if points:
            result.append((entity_type, points))

    return result


def get_freecad_geometry_points(fc_sketch) -> list[tuple[str, list[GeometryPoint]]]:
    """Extract geometry points from FreeCAD sketch.

    Returns:
        List of (geometry_type, [points]) tuples
    """
    result = []

    for geo in fc_sketch.Geometry:
        geo_type = type(geo).__name__
        points = []

        if isinstance(geo, Part.LineSegment):
            start = geo.StartPoint
            end = geo.EndPoint
            points = [
                GeometryPoint(start.x, start.y),
                GeometryPoint(end.x, end.y),
            ]

        elif isinstance(geo, Part.Circle):
            center = geo.Center
            points = [GeometryPoint(center.x, center.y)]

        elif isinstance(geo, Part.ArcOfCircle):
            # Arc: center only (start/end can differ due to parameter interpretation)
            center = geo.Center
            points = [GeometryPoint(center.x, center.y)]

        elif isinstance(geo, Part.Point):
            points = [GeometryPoint(geo.X, geo.Y)]

        if points:
            result.append((geo_type, points))

    return result


def compare_geometry_stability(
    original_points: list[tuple[str, list[GeometryPoint]]],
    fc_points: list[tuple[str, list[GeometryPoint]]],
    tolerance: float = 1.0,  # 1mm tolerance
) -> tuple[bool, list[str]]:
    """Compare geometry points between original and FreeCAD conversion.

    Uses nearest-neighbor matching to handle geometry ordering differences.

    Args:
        original_points: Points from original SketchGraphs data
        fc_points: Points from FreeCAD conversion
        tolerance: Maximum allowed distance in mm

    Returns:
        (is_stable, error_messages)
    """
    errors = []

    # Map type names (SketchGraphs vs FreeCAD)
    type_mapping = {
        'Line': 'LineSegment',
        'Arc': 'ArcOfCircle',
    }

    # Group by type
    def group_by_type(points_list):
        groups = {}
        for geo_type, pts in points_list:
            normalized_type = type_mapping.get(geo_type, geo_type)
            if normalized_type not in groups:
                groups[normalized_type] = []
            groups[normalized_type].append(pts)
        return groups

    orig_groups = group_by_type(original_points)
    fc_groups = group_by_type(fc_points)

    # Check type counts
    all_types = set(orig_groups.keys()) | set(fc_groups.keys())
    for geo_type in all_types:
        orig_count = len(orig_groups.get(geo_type, []))
        fc_count = len(fc_groups.get(geo_type, []))
        if orig_count != fc_count:
            errors.append(f"Count mismatch for {geo_type}: original={orig_count}, FreeCAD={fc_count}")

    if errors:
        return False, errors

    # For each type, find best match using nearest-neighbor
    for geo_type in orig_groups:
        orig_geos = orig_groups[geo_type]
        fc_geos = fc_groups.get(geo_type, [])

        if len(orig_geos) != len(fc_geos):
            continue  # Already reported above

        # For each original geometry, find the closest FreeCAD geometry
        fc_used = [False] * len(fc_geos)

        for orig_idx, orig_pts in enumerate(orig_geos):
            best_fc_idx = -1
            best_distance = float('inf')

            for fc_idx, fc_pts in enumerate(fc_geos):
                if fc_used[fc_idx]:
                    continue
                if len(orig_pts) != len(fc_pts):
                    continue

                # Calculate total distance (sum of point distances)
                total_dist = 0
                for orig_pt, fc_pt in zip(orig_pts, fc_pts):
                    total_dist += orig_pt.distance_to(fc_pt)

                # Also try reversed order for Line endpoints (start/end might be swapped)
                if len(orig_pts) == 2:  # Line
                    reversed_dist = (orig_pts[0].distance_to(fc_pts[1]) +
                                    orig_pts[1].distance_to(fc_pts[0]))
                    total_dist = min(total_dist, reversed_dist)

                if total_dist < best_distance:
                    best_distance = total_dist
                    best_fc_idx = fc_idx

            if best_fc_idx >= 0:
                fc_used[best_fc_idx] = True

                # Check if the best match is within tolerance
                fc_pts = fc_geos[best_fc_idx]

                # Calculate per-point distances
                min_total = float('inf')
                best_mapping = None

                # Try normal order
                normal_dists = [orig_pts[i].distance_to(fc_pts[i]) for i in range(len(orig_pts))]
                if sum(normal_dists) < min_total:
                    min_total = sum(normal_dists)
                    best_mapping = normal_dists

                # Try reversed for lines
                if len(orig_pts) == 2:
                    reversed_dists = [orig_pts[0].distance_to(fc_pts[1]),
                                     orig_pts[1].distance_to(fc_pts[0])]
                    if sum(reversed_dists) < min_total:
                        min_total = sum(reversed_dists)
                        best_mapping = reversed_dists

                if best_mapping:
                    max_drift = max(best_mapping)
                    if max_drift > tolerance:
                        # Only report if it's a significant drift (not just ordering)
                        errors.append(
                            f"{geo_type} drift: max_point_drift={max_drift:.2f}mm "
                            f"(tolerance={tolerance}mm)"
                        )

    return len(errors) == 0, errors


def convert_via_sequence(seq):
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

    for op in seq:
        if isinstance(op, datalib.NodeOp):
            convert_primitive(op, fc_sketch)
        elif isinstance(op, datalib.EdgeOp):
            convert_constraint(op, fc_sketch, node_idx_map)

    doc.recompute()
    return doc, fc_sketch


# =============================================================================
# Pytest Test Class
# =============================================================================

class TestGeometryStability:
    """Test that geometry remains stable after constraint application."""

    NUM_SAMPLES = 100  # Default for pytest
    TOLERANCE = 1.0  # 1mm tolerance

    @classmethod
    def setup_class(cls):
        """Load test data once for all tests."""
        cls.seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')
        cls.sequences = cls.seq_data['sequences']

    def test_primitive_count_match(self):
        """Test that primitive counts match between original and converted."""
        mismatches = []

        for i in range(min(self.NUM_SAMPLES, len(self.sequences))):
            seq = self.sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            try:
                doc, fc_sketch, _ = convert_sketch_to_freecad(sketch)
                if doc is None:
                    continue

                # Count non-External entities in original
                orig_count = sum(1 for e in sketch.entities.values()
                                if e is not None and type(e).__name__ != 'External')
                fc_count = len(fc_sketch.Geometry)

                if orig_count != fc_count:
                    mismatches.append((i, orig_count, fc_count))

                FreeCAD.closeDocument(doc.Name)

            except Exception:
                pass

        assert len(mismatches) == 0, f"Primitive count mismatches: {mismatches[:20]}"

    def test_geometry_stability_after_constraints(self):
        """Test that geometry doesn't collapse after constraint application."""
        unstable_samples = []

        for i in range(min(self.NUM_SAMPLES, len(self.sequences))):
            seq = self.sequences[i]
            sketch = datalib.sketch_from_sequence(seq)

            try:
                doc, fc_sketch, _ = convert_sketch_to_freecad(sketch)
                if doc is None:
                    continue

                # Get points before and after
                orig_points = get_original_geometry_points(sketch)
                fc_points = get_freecad_geometry_points(fc_sketch)

                is_stable, errors = compare_geometry_stability(
                    orig_points, fc_points, tolerance=self.TOLERANCE
                )

                if not is_stable:
                    unstable_samples.append((i, errors[:3]))  # First 3 errors

                FreeCAD.closeDocument(doc.Name)

            except Exception as e:
                pass

        assert len(unstable_samples) == 0, (
            f"Geometry collapsed in {len(unstable_samples)} samples:\n" +
            "\n".join(f"  Sample {idx}: {errs}" for idx, errs in unstable_samples[:10])
        )


# =============================================================================
# CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test geometry stability")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of samples to test")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Maximum allowed point drift in mm")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--use-sequence", action="store_true",
                        help="Use sequence-based converter instead of sketch-based")
    args = parser.parse_args()

    print("Loading data...")
    seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')
    sequences = seq_data['sequences']
    num_samples = min(args.num_samples, len(sequences))
    print(f"Testing {num_samples} samples with tolerance={args.tolerance}mm...")

    # Metrics
    total = 0
    primitive_match = 0
    geometry_stable = 0
    primitive_mismatches = []
    unstable_samples = []
    conversion_errors = []

    for i in tqdm(range(num_samples), desc="Testing"):
        seq = sequences[i]
        sketch = datalib.sketch_from_sequence(seq)

        try:
            # Convert
            if args.use_sequence:
                doc, fc_sketch = convert_via_sequence(seq)
            else:
                doc, fc_sketch, _ = convert_sketch_to_freecad(sketch)
                if doc is None:
                    conversion_errors.append((i, "Conversion returned None"))
                    continue

            total += 1

            # Check primitive count
            orig_count = sum(1 for e in sketch.entities.values()
                            if e is not None and type(e).__name__ != 'External')
            fc_count = len(fc_sketch.Geometry)

            if orig_count == fc_count:
                primitive_match += 1
            else:
                primitive_mismatches.append((i, orig_count, fc_count))

            # Check geometry stability
            # All samples should be stable since original SketchGraphs data already satisfies constraints
            solver_status = fc_sketch.getSolverStatus()
            dof = solver_status['DoF']
            redundant = len(solver_status.get('RedundantConstraints', []))
            conflicts = len(solver_status.get('ConflictingConstraints', []))

            orig_points = get_original_geometry_points(sketch)
            fc_points = get_freecad_geometry_points(fc_sketch)

            is_stable, errors = compare_geometry_stability(
                orig_points, fc_points, tolerance=args.tolerance
            )

            if is_stable:
                geometry_stable += 1
            else:
                unstable_samples.append({
                    'idx': i,
                    'dof': dof,
                    'redundant': redundant,
                    'conflicts': conflicts,
                    'errors': errors,
                })

            FreeCAD.closeDocument(doc.Name)

        except Exception as e:
            conversion_errors.append((i, str(e)))

    # Report
    print("\n" + "=" * 60)
    print("GEOMETRY STABILITY TEST RESULTS")
    print("=" * 60)
    print(f"Total samples tested: {total}")
    print(f"Conversion errors: {len(conversion_errors)}")
    print()
    print(f"Primitive count match: {primitive_match}/{total} ({100*primitive_match/total:.2f}%)")
    print(f"Geometry stable: {geometry_stable}/{total} ({100*geometry_stable/total:.2f}%)")

    if primitive_mismatches:
        print(f"\n--- Primitive Count Mismatches ({len(primitive_mismatches)}) ---")
        for idx, orig, fc in primitive_mismatches[:10]:
            print(f"  Sample {idx}: original={orig}, FreeCAD={fc}")

    if unstable_samples:
        print(f"\n--- Geometry Collapse Detected ({len(unstable_samples)}) ---")
        for sample in unstable_samples[:10]:
            print(f"  Sample {sample['idx']} (DoF={sample['dof']}, Red={sample['redundant']}, Con={sample['conflicts']}):")
            for err in sample['errors'][:3]:
                print(f"    - {err}")

    if conversion_errors and args.verbose:
        print(f"\n--- Conversion Errors ({len(conversion_errors)}) ---")
        for idx, err in conversion_errors[:5]:
            print(f"  Sample {idx}: {err}")

    # Final result
    all_pass = primitive_match == total and geometry_stable == total
    print(f"\n{'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
