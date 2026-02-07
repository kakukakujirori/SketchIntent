"""Convert FreeCAD sketches to SketchGraphs Sequence format.

This module provides conversion from FreeCAD Sketcher objects to SketchGraphs
Sequence format (list of NodeOp/EdgeOp) for fine-tuning.

Usage:
    python convert_freecad_to_sequence.py --input-dir /path/to/fcstd/files --output output.npy
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add FreeCAD Libs (conditionally)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part

import sketchgraphs.data as datalib
from sketchgraphs.data import (
    NodeOp, EdgeOp, EntityType, ConstraintType, SubnodeType,
    flat_array,
)


# =============================================================================
# Unit Conversion
# =============================================================================

def _millimeter_to_meter(value: float) -> float:
    """Convert millimeters to meters (FreeCAD uses mm, SketchGraphs uses m)."""
    return value / 1000.0


# =============================================================================
# Entity Conversion: FreeCAD Geometry -> SketchGraphs NodeOp
# =============================================================================

def _convert_line(geo: "Part.LineSegment", is_construction: bool) -> NodeOp:
    """Convert FreeCAD LineSegment to SketchGraphs Line NodeOp."""
    start = geo.StartPoint
    end = geo.EndPoint

    # SketchGraphs Line uses parametric form: pnt + dir * t
    # where t ranges from startParam to endParam
    pnt_x = _millimeter_to_meter(start.x)
    pnt_y = _millimeter_to_meter(start.y)

    dx = end.x - start.x
    dy = end.y - start.y
    length = math.sqrt(dx*dx + dy*dy)

    if length > 1e-9:
        dir_x = dx / length
        dir_y = dy / length
        start_param = 0.0
        end_param = _millimeter_to_meter(length)
    else:
        # Degenerate line
        dir_x, dir_y = 1.0, 0.0
        start_param, end_param = 0.0, 0.0

    return NodeOp(
        label=EntityType.Line,
        parameters={
            'pntX': pnt_x,
            'pntY': pnt_y,
            'dirX': dir_x,
            'dirY': dir_y,
            'startParam': start_param,
            'endParam': end_param,
            'isConstruction': is_construction,
        }
    )


def _convert_circle(geo: "Part.Circle", is_construction: bool) -> NodeOp:
    """Convert FreeCAD Circle to SketchGraphs Circle NodeOp."""
    center = geo.Center
    return NodeOp(
        label=EntityType.Circle,
        parameters={
            'xCenter': _millimeter_to_meter(center.x),
            'yCenter': _millimeter_to_meter(center.y),
            'radius': _millimeter_to_meter(geo.Radius),
            'isConstruction': is_construction,
        }
    )


def _convert_arc(geo: "Part.ArcOfCircle", is_construction: bool) -> NodeOp:
    """Convert FreeCAD ArcOfCircle to SketchGraphs Arc NodeOp.

    FreeCAD stores arcs with FirstParameter and LastParameter (radians).
    SketchGraphs uses xDir, yDir as reference direction and startParam/endParam as offsets.
    """
    center = geo.Center
    radius = geo.Radius

    # FreeCAD arc parameters
    first_param = geo.FirstParameter  # radians, CCW from +X axis
    last_param = geo.LastParameter

    # Determine if clockwise
    # FreeCAD arcs are always CCW internally, but we need to check
    # the actual point positions to determine directionality
    start_pt = geo.StartPoint
    end_pt = geo.EndPoint

    # Calculate angles from center to start/end
    start_angle = math.atan2(start_pt.y - center.y, start_pt.x - center.x)
    end_angle = math.atan2(end_pt.y - center.y, end_pt.x - center.x)

    # SketchGraphs convention: xDir, yDir point to angle=0 reference
    # Use start angle as reference direction
    x_dir = math.cos(start_angle)
    y_dir = math.sin(start_angle)

    # startParam is offset from reference direction to start point (always 0)
    start_param = 0.0

    # endParam is offset from reference direction to end point
    # Calculate CCW angle from start to end
    delta = end_angle - start_angle
    if delta < 0:
        delta += 2 * math.pi

    # Determine clockwise: if arc spans more than half circle in CCW, might be CW
    # FreeCAD arcs are CCW, so if LastParameter < FirstParameter, it wraps around
    clockwise = False
    if last_param < first_param:
        # Wrapped arc - still CCW but crosses 0
        end_param = (2 * math.pi - first_param) + last_param
    else:
        end_param = last_param - first_param

    return NodeOp(
        label=EntityType.Arc,
        parameters={
            'xCenter': _millimeter_to_meter(center.x),
            'yCenter': _millimeter_to_meter(center.y),
            'radius': _millimeter_to_meter(radius),
            'xDir': x_dir,
            'yDir': y_dir,
            'startParam': start_param,
            'endParam': end_param,
            'clockwise': clockwise,
            'isConstruction': is_construction,
        }
    )


def _convert_point(geo: "Part.Point", is_construction: bool) -> NodeOp:
    """Convert FreeCAD Point to SketchGraphs Point NodeOp."""
    return NodeOp(
        label=EntityType.Point,
        parameters={
            'x': _millimeter_to_meter(geo.X),
            'y': _millimeter_to_meter(geo.Y),
            'isConstruction': is_construction,
        }
    )


def convert_geometry_to_nodeops(
    fc_sketch: "Sketcher.SketchObject",
) -> tuple[list[NodeOp], dict[int, int]]:
    """Convert all FreeCAD geometry to SketchGraphs NodeOps.

    Returns:
        nodeops: List of NodeOps (primitives + subnodes)
        geo_to_node_idx: Mapping from FreeCAD GeoId to NodeOp index (for primitive)
    """
    nodeops: list[NodeOp] = []
    # Prepend External node (index 0) to align with SketchGraphs convention
    # sketch_from_sequence skips constraints referencing index 0
    nodeops.append(NodeOp(label=EntityType.External, parameters={}))

    geo_to_node_idx: dict[int, int] = {}  # GeoId -> NodeOp index
    subnode_map: dict[tuple[int, int], int] = {}  # (GeoId, PointPos) -> NodeOp index

    for geo_id in range(len(fc_sketch.Geometry)):
        geo = fc_sketch.Geometry[geo_id]
        is_construction = fc_sketch.getConstruction(geo_id)

        if isinstance(geo, Part.LineSegment):
            node_idx = len(nodeops)
            nodeops.append(_convert_line(geo, is_construction))
            geo_to_node_idx[geo_id] = node_idx

            # Add subnodes for Line (start, end)
            for subnode_type, point_pos in [(SubnodeType.SN_Start, 1), (SubnodeType.SN_End, 2)]:
                sub_idx = len(nodeops)
                nodeops.append(NodeOp(label=subnode_type, parameters={}))
                subnode_map[(geo_id, point_pos)] = sub_idx

        elif isinstance(geo, Part.Circle):
            node_idx = len(nodeops)
            nodeops.append(_convert_circle(geo, is_construction))
            geo_to_node_idx[geo_id] = node_idx

            # Circle has center subnode
            sub_idx = len(nodeops)
            nodeops.append(NodeOp(label=SubnodeType.SN_Center, parameters={}))
            subnode_map[(geo_id, 3)] = sub_idx

        elif isinstance(geo, Part.ArcOfCircle):
            node_idx = len(nodeops)
            nodeops.append(_convert_arc(geo, is_construction))
            geo_to_node_idx[geo_id] = node_idx

            # Arc has center, start, end subnodes
            for subnode_type, point_pos in [
                (SubnodeType.SN_Center, 3),
                (SubnodeType.SN_Start, 1),
                (SubnodeType.SN_End, 2),
            ]:
                sub_idx = len(nodeops)
                nodeops.append(NodeOp(label=subnode_type, parameters={}))
                subnode_map[(geo_id, point_pos)] = sub_idx

        elif hasattr(geo, 'X') and hasattr(geo, 'Y'):  # Part.Point
            node_idx = len(nodeops)
            nodeops.append(_convert_point(geo, is_construction))
            geo_to_node_idx[geo_id] = node_idx
            # Point entity's vertex is at PointPos=1
            subnode_map[(geo_id, 1)] = node_idx

        else:
            # Skip unsupported geometry (Ellipse, Spline, etc.)
            pass

    # Generate Subnode EdgeOps to link subnodes to parents
    subnode_edgeops: list[EdgeOp] = []
    for (geo_id, point_pos), sub_idx in subnode_map.items():
        if geo_id in geo_to_node_idx:
            parent_idx = geo_to_node_idx[geo_id]
            # Check if this is actually a subnode (not a Point entity)
            if nodeops[sub_idx].label in (SubnodeType.SN_Start, SubnodeType.SN_End, SubnodeType.SN_Center):
                subnode_edgeops.append(EdgeOp(
                    label=ConstraintType.Subnode,
                    references=(sub_idx, parent_idx),
                    parameters={},
                ))

    return nodeops, geo_to_node_idx, subnode_map, subnode_edgeops


# =============================================================================
# Constraint Conversion: FreeCAD Constraint -> SketchGraphs EdgeOp
# =============================================================================

def _get_node_ref(
    geo_id: int,
    point_pos: int,
    geo_to_node_idx: dict[int, int],
    subnode_map: dict[tuple[int, int], int],
) -> int | None:
    """Get NodeOp index for a FreeCAD constraint reference.

    Args:
        geo_id: FreeCAD geometry ID
        point_pos: 0=edge, 1=start, 2=end, 3=center
        geo_to_node_idx: Mapping from GeoId to primitive NodeOp index
        subnode_map: Mapping from (GeoId, PointPos) to subnode NodeOp index

    Returns:
        NodeOp index or None if not found
    """
    if geo_id < 0:
        # Check if reference is to an Axis Line (Pos=0) or Root Point (Pos>0)
        if point_pos == 0:
            # Axis Line reference (e.g. X-Axis, Y-Axis)
            # SketchGraphs doesn't support explicit Axis Line nodes.
            # We skip this reference.
            # Note: For DistanceX/Y, skipping axis allows falling back to single-ref implicit origin logic.
            # But for Angle, this must be handled carefully.
            print(f"Warning: Skipping Axis Line reference: GeoId {geo_id}, Pos {point_pos}", file=sys.stderr)
            return None
        else:
            # Root Point / Origin (Pos > 0) -> Node 0
            return 0

    if point_pos == 0:
        # Edge reference -> primitive node
        return geo_to_node_idx.get(geo_id)
    else:
        # Point reference -> subnode
        return subnode_map.get((geo_id, point_pos))


def convert_constraint_to_edgeop(
    constraint: Any,
    geo_to_node_idx: dict[int, int],
    subnode_map: dict[tuple[int, int], int],
    fc_sketch: "Sketcher.SketchObject",
    nodeops: list[NodeOp],
) -> EdgeOp | None:
    """Convert a single FreeCAD constraint to SketchGraphs EdgeOp.

    Returns None if constraint cannot be converted.
    """
    c_type = constraint.Type

    # Get references
    refs = []
    for geo_id, point_pos in [
        (constraint.First, constraint.FirstPos),
        (constraint.Second, constraint.SecondPos),
        (constraint.Third, constraint.ThirdPos),
    ]:
        if geo_id > -1999:
            node_ref = _get_node_ref(geo_id, point_pos, geo_to_node_idx, subnode_map)
            if node_ref is not None:
                refs.append(node_ref)

    if len(refs) == 0:
        return None

    # Map constraint types
    if c_type == 'Coincident':
        if len(refs) >= 2:
            return EdgeOp(label=ConstraintType.Coincident, references=tuple(refs[:2]), parameters={})

    elif c_type == 'PointOnObject':
        if len(refs) >= 2:
            return EdgeOp(label=ConstraintType.Coincident, references=tuple(refs[:2]), parameters={})

    elif c_type == 'Horizontal':
        if len(refs) >= 1:
            return EdgeOp(label=ConstraintType.Horizontal, references=tuple(refs), parameters={})

    elif c_type == 'Vertical':
        if len(refs) >= 1:
            return EdgeOp(label=ConstraintType.Vertical, references=tuple(refs), parameters={})

    elif c_type == 'Parallel':
        if len(refs) >= 2:
            return EdgeOp(label=ConstraintType.Parallel, references=tuple(refs[:2]), parameters={})

    elif c_type == 'Perpendicular':
        if len(refs) >= 2:
            return EdgeOp(label=ConstraintType.Perpendicular, references=tuple(refs[:2]), parameters={})

    elif c_type == 'Tangent':
        if len(refs) >= 2:
            return EdgeOp(label=ConstraintType.Tangent, references=tuple(refs[:2]), parameters={})

    elif c_type == 'Equal':
        if len(refs) >= 2:
            return EdgeOp(label=ConstraintType.Equal, references=tuple(refs[:2]), parameters={})

    elif c_type == 'Symmetric':
        # Symmetric in FreeCAD -> Midpoint in SketchGraphs
        # FreeCAD Symmetric has 3 references: (p1, p2, symmetry_center)

        # We map ALL Symmetric constraints to Midpoint(midpoint, p1, p2)
        # refs[2] is the symmetry element (midpoint/axis)

        if len(refs) >= 3:
            # SketchGraphs Midpoint expects: (midpoint, p1, p2)
            # FreeCAD gives: (p1, p2, midpoint)
            return EdgeOp(
                label=ConstraintType.Midpoint,
                references=(refs[2], refs[0], refs[1]),
                parameters={}
            )
        elif len(refs) == 2:
            # Two-ref case: point on curve midpoint
            return EdgeOp(
                label=ConstraintType.Midpoint,
                references=tuple(refs),
                parameters={}
            )

    elif c_type == 'Distance':
        # Distance constraint - value in string format for SketchGraphs
        value = constraint.Value  # mm
        if len(refs) >= 1:
            return EdgeOp(
                label=ConstraintType.Distance,
                references=tuple(refs),
                parameters={'length': f'{_millimeter_to_meter(value)} METER'}
            )

    elif c_type == 'Radius':
        if len(refs) >= 1:
            value = constraint.Value  # mm
            return EdgeOp(
                label=ConstraintType.Radius,
                references=tuple(refs[:1]),
                parameters={'length': f'{_millimeter_to_meter(value)} METER'}
            )

    elif c_type == 'Diameter':
        if len(refs) >= 1:
            value = constraint.Value  # mm
            return EdgeOp(
                label=ConstraintType.Diameter,
                references=tuple(refs[:1]),
                parameters={'length': f'{_millimeter_to_meter(value)} METER'}
            )

    elif c_type == 'Angle':
        value = constraint.Value  # radians in FreeCAD
        # Convert radians to degrees for SketchGraphs
        value_deg = math.degrees(value)

        if len(refs) >= 2:
            return EdgeOp(
                label=ConstraintType.Angle,
                references=tuple(refs[:2]),
                parameters={'angle': f'{value_deg} DEGREE'}
            )
        elif len(refs) == 1:
            # Single reference Angle (e.g. Arc Span)
            # Must verify that the reference is an Arc/Circle (SPAN), not a Line (Angle with Axis)
            # If it's a Line, "Angle(Line)" is ambiguous or usually means Angle with H/V axis,
            # which SketchGraphs doesn't support directly without changing value semantics.
            ref_node_idx = refs[0]
            if ref_node_idx < len(nodeops):
                node_label = nodeops[ref_node_idx].label
                if node_label in (EntityType.Arc, EntityType.Circle):
                    return EdgeOp(
                        label=ConstraintType.Angle,
                        references=(refs[0],),
                        parameters={'angle': f'{value_deg} DEGREE'}
                    )
                else:
                    # Likely a Line referenced against an Axis (which was skipped)
                    # Skip effectively
                    return None
            return None
    elif c_type in ('DistanceX', 'DistanceY'):
        # DistanceX/DistanceY: use Distance with direction parameter
        # direction: 1 = HORIZONTAL (X), 2 = VERTICAL (Y)
        value = abs(constraint.Value)  # mm, can be negative

        if c_type == 'DistanceX':
            direction = 'HORIZONTAL'
        else:  # DistanceY
            direction = 'VERTICAL'

        if len(refs) >= 2:
            return EdgeOp(
                label=ConstraintType.Distance,
                references=tuple(refs[:2]),
                parameters={
                    'length': f'{_millimeter_to_meter(value)} METER',
                    'direction': direction,
                }
            )
        elif len(refs) == 1:
            # Single reference handling
            # Check if it's an Edge reference (Pos=0) or Point reference (Pos>0)

            # Find the original geo_id/pos that resulted in this ref
            # (Assuming standard usage where First is the ref, or we search)
            valid_geo_pos = None
            for gid, pos in [(constraint.First, constraint.FirstPos),
                             (constraint.Second, constraint.SecondPos),
                             (constraint.Third, constraint.ThirdPos)]:
                if gid > -1999:
                    # Check if this gid maps to refs[0]
                    # This is a bit loose since multiple gs could map to same node if somehow merged,
                    # but usually 1-to-1.
                    # Simpler: just use the first valid one we find.
                    valid_geo_pos = (gid, pos)
                    break

            if valid_geo_pos and valid_geo_pos[1] == 0:
                # Edge reference (e.g. DistanceX(Line)) -> Horizontal Span (Length)
                # Convert to DistanceX between Start and End points
                geo_id = valid_geo_pos[0]
                start_node = subnode_map.get((geo_id, 1))
                end_node = subnode_map.get((geo_id, 2))

                if start_node is not None and end_node is not None:
                    return EdgeOp(
                        label=ConstraintType.Distance,
                        references=(start_node, end_node),
                        parameters={
                            'length': f'{_millimeter_to_meter(value)} METER',
                            'direction': direction,
                        }
                    )

            # Fallback: Point reference or failed to find Start/End
            # Treat as Distance from Origin (Node 0)
            return EdgeOp(
                label=ConstraintType.Distance,
                references=(refs[0], 0),
                parameters={
                    'length': f'{_millimeter_to_meter(value)} METER',
                    'direction': direction,
                }
            )

    # Not supported
    return None


def convert_constraints_to_edgeops(
    fc_sketch: "Sketcher.SketchObject",
    geo_to_node_idx: dict[int, int],
    subnode_map: dict[tuple[int, int], int],
    nodeops: list[NodeOp],
) -> list[EdgeOp]:
    """Convert all FreeCAD constraints to SketchGraphs EdgeOps."""
    edgeops: list[EdgeOp] = []

    for i in range(fc_sketch.ConstraintCount):
        constraint = fc_sketch.Constraints[i]
        edgeop = convert_constraint_to_edgeop(constraint, geo_to_node_idx, subnode_map, fc_sketch, nodeops)
        if edgeop is not None:
            edgeops.append(edgeop)

    return edgeops


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_freecad_sketch_to_sequence(
    fc_sketch: "Sketcher.SketchObject",
) -> list[NodeOp | EdgeOp]:
    """Convert a FreeCAD sketch to SketchGraphs Sequence format.

    Args:
        fc_sketch: FreeCAD Sketcher.SketchObject

    Returns:
        Sequence (list of NodeOp/EdgeOp)
    """
    # Convert geometry
    nodeops, geo_to_node_idx, subnode_map, subnode_edgeops = convert_geometry_to_nodeops(fc_sketch)

    # Convert constraints
    constraint_edgeops = convert_constraints_to_edgeops(fc_sketch, geo_to_node_idx, subnode_map, nodeops)

    # Assemble sequence: primitives first, then subnodes EdgeOps, then constraint EdgeOps
    # Add Stop node at the end
    sequence = list(nodeops) + subnode_edgeops + constraint_edgeops
    sequence.append(NodeOp(label=EntityType.Stop, parameters={}))

    return sequence


def load_freecad_file(filepath: str) -> list["Sketcher.SketchObject"]:
    """Load FreeCAD file and return all sketch objects.

    Supports .FCStd files (opened as documents) and .py scripts (executed).
    """
    sketches = []

    if filepath.endswith('.py'):
        # Execute Python script to generate document/sketch
        # We need to ensure a document is created or active
        try:
            with open(filepath, 'r') as f:
                # Use a fresh document name based on filename to avoid conflicts
                doc_name = os.path.basename(filepath).replace('.py', '')
                # Execute in global namespace or controlled one? Global is risky but standard for FC scripts.
                # But we need to capture the document.
                exec(f.read(), {'__file__': filepath})

            doc = FreeCAD.activeDocument()
            if doc:
                for obj in doc.Objects:
                    if obj.TypeId == 'Sketcher::SketchObject':
                        sketches.append(obj)
            return doc, sketches
        except Exception as e:
            print(f"Error executing script {filepath}: {e}")
            return None, []

    else:
        # Load .FCStd
        try:
            doc = FreeCAD.openDocument(filepath)
            for obj in doc.Objects:
                if obj.TypeId == 'Sketcher::SketchObject':
                    sketches.append(obj)
            return doc, sketches
        except Exception as e:
            print(f"Error opening document {filepath}: {e}")
            return None, []


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert FreeCAD sketches to SketchGraphs Sequence format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input .FCStd file or directory containing .FCStd files"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output .npy file path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        # Look for both .FCStd and .py files
        fcstd_files = list(input_path.glob("**/*.FCStd"))
        py_files = list(input_path.glob("**/*.py"))
        input_files = fcstd_files + py_files
    else:
        input_files = [input_path]

    print(f"Found {len(input_files)} FreeCAD files")

    sequences = []
    errors = []

    for file_path in tqdm(input_files, desc="Converting"):
        try:
            doc, sketches = load_freecad_file(str(file_path))
            if not doc:
                 continue
            if args.verbose:
                if sketches:
                     print(f"  {file_path.name}: {len(sketches)} sketches")

            for sketch in sketches:
                seq = convert_freecad_sketch_to_sequence(sketch)
                sequences.append(seq)
                if args.verbose:
                    print(f"    {sketch.Name}: {len(seq)} ops")

            if doc:
                FreeCAD.closeDocument(doc.Name)
        except Exception as e:
            errors.append((str(file_path), str(e)))
            if args.verbose:
                print(f"  ERROR {fcstd_path}: {e}")

    print(f"Converted {len(sequences)} sketches")
    if errors:
        print(f"Errors: {len(errors)}")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    # Save
    output_data = {'sequences': sequences}
    import numpy as np
    # Pack dictionary and save
    packed_data = flat_array.pack_dictionary_flat(output_data)
    np.save(args.output, packed_data)
    print(f"Saved {args.output}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
