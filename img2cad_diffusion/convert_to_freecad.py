import copy
import os
import re
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Add FreeCAD Libs
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Sketcher
import Part

# # 2. Add Site Packages (Draft, PySide2, etc) AFTER FreeCAD import
# site_packages = os.path.expanduser("~/.FreeCAD/squashfs-root/usr/lib/python3.11/site-packages")
# if site_packages not in sys.path:
#     sys.path.insert(0, site_packages)

# 3. Import SVG stuff properly wrapped
try:
    import importSVG
    from IPython.display import display, SVG
except ImportError as e:
    print(f"Warning: SVG export modules not available: {e}")
    importSVG = None

# 4. Import SketchGraphs
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, NodeOp, EdgeOp, EntityType, ConstraintType


def deduplicate_refs(op: datalib.EdgeOp) -> datalib.EdgeOp:
    """Remove duplicate references from EdgeOp.

    Some Coincident EdgeOps in SketchGraphs data have duplicate references like (5, 5, 3, 3).
    This function deduplicates them. Only applies to 4+ refs cases.

    For refs with <= 3 elements (normal constraints including Midpoint),
    the original EdgeOp is returned unchanged to preserve ordering.

    Note: Sorting is NOT performed here - that responsibility belongs to the caller
    (e.g., tokenize_constraints sorts refs based on gather_idxs).

    Parameters
    ----------
    op : EdgeOp
        The EdgeOp to process.

    Returns
    -------
    EdgeOp
        EdgeOp with deduplicated references for 4+ refs cases, original otherwise.
    """
    refs = op.references
    if len(refs) >= 4:
        # This should only happen for duplicated Coincident constraints
        assert op.label == datalib.ConstraintType.Coincident, \
            f"4+ refs bug should only occur for Coincident, got {op.label}"
        deduped = tuple(set(refs))
        assert len(deduped) == 2, f"Expected 2 unique refs after dedup, got {deduped}"
        return datalib.EdgeOp(op.label, deduped, op.parameters)
    return op


def meter_to_millimeter(op: NodeOp | EdgeOp, debug: bool = False) -> NodeOp | EdgeOp:
    if isinstance(op, NodeOp):
        length_keys = ['x', 'y', 'pntX', 'pntY', 'radius', 'minorRadius', 'xCenter', 'yCenter']
        for k in length_keys:
            if k in op.parameters:
                op.parameters[k] *= 1000

        line_length_key = ['startParam', 'endParam']  # Arc also has these, but scaling shouldn't be applied
        if op.label == EntityType.Line:
            for k in line_length_key:
                if k in op.parameters:
                    op.parameters[k] *= 1000

    elif isinstance(op, EdgeOp):
        if "length" in op.parameters:
            length = op.parameters["length"]
            if isinstance(length, (int, float)):
                op.parameters["length"] *= 1000
            elif length.endswith('METER'):
                op.parameters["length"] = float(length.replace(' METER', '')) * 1000
            else:
                if debug: print(f"[meter_to_millimeter] Unknown length unit ({length}) in {op=}")
                pass  # Skip for now; it won't cause issues unless Diameter/Radius/Length/Distance are used
    else:
        raise ValueError(f"Unknown op type: {type(op)}")
    return op


def convert_primitive(op: NodeOp, sketch):
    """
    Handles NodeOp:
    1. Checks for Subnode/External (returns mapping immediately).
    2. Creates FreeCAD Geometry for primitives.
    3. Returns ((GeoID, VertexID), command_string) or (None, "").

    FreeCAD VertexID Conventions:
    - 0: The Edge/Curve itself
    - 1: Start Point
    - 2: End Point
    - 3: Center Point (for Circle/Arc)
    """
    # Returns ((GeoID, VertexID), command_string) mapping info, or (None, "")
    if isinstance(op.label, datalib.SubnodeType):
        return (('subnode', op.label), "")

    if op.label == EntityType.External:
        # Do not create geometry, map to -1
        return ((-1, 0), "# External Entity Skipped")

    before_count = len(sketch.Geometry)
    cmd = ""

    assert isinstance(op, NodeOp), f"Expected NodeOp, got {type(op)}"

    if op.label == EntityType.Point:
        x = op.parameters['x']
        y = op.parameters['y']
        is_construction = op.parameters['isConstruction']
        sketch.addGeometry(Part.Point(FreeCAD.Vector(x, y, 0)), is_construction)
        cmd = f"sketch.addGeometry(Part.Point(FreeCAD.Vector({x}, {y}, 0)), {is_construction})"

    elif op.label == EntityType.Line:
        pnt = FreeCAD.Vector(op.parameters['pntX'], op.parameters['pntY'], 0)
        dir_vec = FreeCAD.Vector(op.parameters['dirX'], op.parameters['dirY'], 0)
        start = pnt + dir_vec * op.parameters['startParam']
        end = pnt + dir_vec * op.parameters['endParam']
        is_construction = op.parameters['isConstruction']

        # Ensure minimum-length Line (prevent OCC "Both points are equal" error)
        min_length = 1e-3  # 1 micron minimum in mm scale
        if (start - end).Length < min_length:
            # Create a tiny Line in the direction (or arbitrary X direction)
            if dir_vec.Length > 1e-9:
                end = start + dir_vec * min_length
            else:
                end = start + FreeCAD.Vector(min_length, 0, 0)

        sketch.addGeometry(Part.LineSegment(start, end), is_construction)
        cmd = f"sketch.addGeometry(Part.LineSegment({start}, {end}), {is_construction})"

    elif op.label == EntityType.Circle:
        center = FreeCAD.Vector(op.parameters['xCenter'], op.parameters['yCenter'], 0)
        radius = op.parameters['radius']
        is_construction = op.parameters['isConstruction']
        sketch.addGeometry(Part.Circle(center, FreeCAD.Vector(0,0,1), radius), is_construction)
        cmd = f"sketch.addGeometry(Part.Circle({center}, FreeCAD.Vector(0,0,1), {radius}), {is_construction})"

    elif op.label == EntityType.Arc:
        import math
        cx, cy = op.parameters['xCenter'], op.parameters['yCenter']
        rx, ry = op.parameters['xDir'], op.parameters['yDir']
        r = op.parameters['radius']
        is_construction = op.parameters['isConstruction']
        clockwise = op.parameters['clockwise']

        def get_pt(angle_offset):
            base_angle = math.atan2(ry, rx)
            if clockwise:
                angle_offset *= -1
            theta = base_angle + angle_offset
            return FreeCAD.Vector(cx + r*math.cos(theta), cy + r*math.sin(theta), 0)

        p_start = get_pt(op.parameters['startParam'])
        p_end = get_pt(op.parameters['endParam'])

        # Midpoint calculation handling wrap-around
        sp = op.parameters['startParam'] % (2*math.pi)
        ep = op.parameters['endParam'] % (2*math.pi)
        if sp > ep:
            ep += 2*math.pi
        mid_param = (sp + ep) / 2.0
        p_mid = get_pt(mid_param)

        # For clockwise arcs, swap start and end when creating the FreeCAD arc.
        # get_pt() with clockwise=True produces: start at angle -startParam, end at angle -endParam
        # For a semicircle with startParam=-90° and endParam=90°, this gives:
        #   p_start at +90° (top), p_end at -90° (bottom)
        # But FreeCAD Part.ArcOfCircle interprets (start, mid, end) in CCW order,
        # so we swap to get: p_start at bottom, p_end at top (correct CCW orientation)
        if clockwise:
            p_start, p_end = p_end, p_start

        sketch.addGeometry(Part.ArcOfCircle(p_start, p_mid, p_end), is_construction)
        cmd = f"sketch.addGeometry(Part.ArcOfCircle({p_start}, {p_mid}, {p_end}), {is_construction})"

    elif op.label == EntityType.Ellipse:
        import math
        center = FreeCAD.Vector(op.parameters['xCenter'], op.parameters['yCenter'], 0)
        major = op.parameters['radius']
        minor = op.parameters['minorRadius']
        is_construction = op.parameters['isConstruction']

        geo = Part.Ellipse(center, major, minor)
        # Rotate to align with xDir, yDir
        angle = math.atan2(op.parameters['yDir'], op.parameters['xDir'])
        geo.rotate(center, FreeCAD.Vector(0,0,1), math.degrees(angle))
        sketch.addGeometry(geo, is_construction)
        cmd = f"sketch.addGeometry({geo}, {is_construction})"

    elif op.label in (EntityType.Spline, EntityType.Conic):
        cmd = f"# Skipped Geometry: {op.label} (Complex curve)"

    elif op.label in (EntityType.External, EntityType.Stop, EntityType.Unknown):
        cmd = f"# Skipped Entity: {op.label}"

    elif isinstance(op.label, datalib.SubnodeType):
        pass # Implicit geometry, skipped

    else:
        raise ValueError(f"Unknown geometry type: {op.label}")

    # Check mapping
    after_count = len(sketch.Geometry)
    if after_count > before_count:
        geo_id = after_count - 1
        pt_pos = 0
        if op.label == EntityType.Point:
            pt_pos = 1
        return ((geo_id, pt_pos), cmd)
    return (None, cmd)


def convert_constraint(op: EdgeOp, sketch, node_idx_map):
    """
    Handles EdgeOp:
    Interprets EdgeOp as a Constraint application.
    Resolves references using node_idx_map and applies Sketcher constraints.
    Returns command string.
    """
    if op.parameters.get("disable", False): return f"# Disabled: {op}"

    # Deduplicate refs (handles cases like (5, 5, 3, 3) -> (3, 5))
    op = deduplicate_refs(op)
    refs = op.references

    cmd = ""
    try:
        r1 = node_idx_map.get(refs[0]) if len(refs) > 0 else None
        r2 = node_idx_map.get(refs[1]) if len(refs) > 1 else None

        if op.label == ConstraintType.Coincident: # 0
            """ NOTE: Although FreeCAD 1.0.0 and later unifies Coincident and PointOnObject,
                      they need to be differentiated when calling Python API."""
            if r1 and r2:
                if r1[0] == -1 or r2[0] == -1:
                    cmd = f"# Skipped External Coincident: {r1} <-> {r2}"
                elif r1[1] == 0 and r2[1] == 0:
                    cmd = f"# Skipped Edge-Edge Coincident: {r1} <-> {r2}"
                elif r1[1] == 0: # r1 is Edge, r2 is Point
                    sketch.addConstraint(Sketcher.Constraint('PointOnObject', r2[0], r2[1], r1[0]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('PointOnObject', {r2[0]}, {r2[1]}, {r1[0]}))"
                elif r2[1] == 0: # r2 is Edge, r1 is Point
                    sketch.addConstraint(Sketcher.Constraint('PointOnObject', r1[0], r1[1], r2[0]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('PointOnObject', {r1[0]}, {r1[1]}, {r2[0]}))"
                else: # Both are Points
                    sketch.addConstraint(Sketcher.Constraint('Coincident', r1[0], r1[1], r2[0], r2[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Coincident', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"

        elif op.label == ConstraintType.Projected: # 1
            cmd = "# Skipped ConstraintType.Projected (Not implemented)"

        elif op.label == ConstraintType.Mirror: # 2
            cmd = "# Skipped ConstraintType.Mirror (Complexity: Mapping check)"

        elif op.label == ConstraintType.Distance: # 3
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                # Helper to get point coordinates from geometry
                def get_point(geo_id, pt_id):
                    geo = sketch.Geometry[geo_id]
                    if pt_id == 1:  # Start
                        if hasattr(geo, 'StartPoint'):
                            return geo.StartPoint
                        elif hasattr(geo, 'X'):  # Point geometry
                            return FreeCAD.Vector(geo.X, geo.Y, 0)
                    elif pt_id == 2:  # End
                        if hasattr(geo, 'EndPoint'):
                            return geo.EndPoint
                    elif pt_id == 3:  # Center
                        if hasattr(geo, 'Center'):
                            return geo.Center
                    return None

                # Helper to compute perpendicular distance from point to line
                def point_to_line_distance(pt, line_geo):
                    if not hasattr(line_geo, 'StartPoint') or not hasattr(line_geo, 'EndPoint'):
                        return None
                    p1 = line_geo.StartPoint
                    p2 = line_geo.EndPoint
                    # Line direction vector
                    d = p2 - p1
                    d_len = d.Length
                    if d_len < 1e-9:
                        return (pt - p1).Length
                    # Project pt onto line and compute perpendicular distance
                    t = (pt - p1).dot(d) / (d_len * d_len)
                    closest = p1 + d * t
                    return (pt - closest).Length

                if r1[1] > 0 and r2[1] > 0:
                    # Point-to-Point distance
                    p1 = get_point(r1[0], r1[1])
                    p2 = get_point(r2[0], r2[1])
                    if p1 and p2:
                        dist = (p1 - p2).Length
                        sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], r1[1], r2[0], r2[1], dist))
                        cmd = f"sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}, {dist}))"
                    else:
                        cmd = f"# Skipped ConstraintType.Distance: Could not get points ({r1=}, {r2=})"

                elif r1[1] == 0 and r2[1] == 0:
                    # Edge-to-Edge distance (parallel lines)
                    # Use line1's start point to line2 perpendicular distance
                    geo1 = sketch.Geometry[r1[0]]
                    geo2 = sketch.Geometry[r2[0]]
                    if not isinstance(geo1, Part.LineSegment) or not isinstance(geo2, Part.LineSegment):
                        cmd = f"# Skipped ConstraintType.Distance: Not both are LineSegments ({r1=}, {r2=})"
                        return
                    if not hasattr(geo1, 'StartPoint'):
                        cmd = f"# Skipped ConstraintType.Distance: geo1 has no StartPoint ({r1=}, {r2=})"
                        return
                    p1 = geo1.StartPoint
                    dist = point_to_line_distance(p1, geo2)
                    if dist is not None:
                        sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], 1, r2[0], 0, dist))
                        cmd = f"sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, 1, {r2[0]}, 0, {dist}))"
                    else:
                        cmd = f"# Skipped ConstraintType.Distance: Could not compute line distance ({r1=}, {r2=})"

                elif r1[1] > 0 and r2[1] == 0:
                    # Point to Edge distance
                    p1 = get_point(r1[0], r1[1])
                    geo2 = sketch.Geometry[r2[0]]
                    if p1:
                        dist = point_to_line_distance(p1, geo2)
                        if dist is not None:
                            sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], r1[1], r2[0], 0, dist))
                            cmd = f"sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, {r1[1]}, {r2[0]}, 0, {dist}))"
                        else:
                            cmd = f"# Skipped ConstraintType.Distance: Could not compute point-line distance ({r1=}, {r2=})"
                    else:
                        cmd = f"# Skipped ConstraintType.Distance: Could not get point ({r1=})"

                elif r1[1] == 0 and r2[1] > 0:
                    # Edge to Point distance (swap and use Point to Edge)
                    p2 = get_point(r2[0], r2[1])
                    geo1 = sketch.Geometry[r1[0]]
                    if p2:
                        dist = point_to_line_distance(p2, geo1)
                        if dist is not None:
                            sketch.addConstraint(Sketcher.Constraint('Distance', r2[0], r2[1], r1[0], 0, dist))
                            cmd = f"sketch.addConstraint(Sketcher.Constraint('Distance', {r2[0]}, {r2[1]}, {r1[0]}, 0, {dist}))"
                        else:
                            cmd = f"# Skipped ConstraintType.Distance: Could not compute point-line distance ({r1=}, {r2=})"
                    else:
                        cmd = f"# Skipped ConstraintType.Distance: Could not get point ({r2=})"

                else:
                    cmd = f"# Skipped ConstraintType.Distance: Unhandled case ({r1=}, {r2=})"
            else:
                cmd = f"# Skipped ConstraintType.Distance: Invalid references ({r1=}, {r2=})"

        elif op.label == ConstraintType.Horizontal: # 4
            if len(refs) == 1:
                if r1 and r1[0] != -1:
                    sketch.addConstraint(Sketcher.Constraint('Horizontal', r1[0]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Horizontal', {r1[0]}))"
            elif len(refs) == 2:
                if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                    sketch.addConstraint(Sketcher.Constraint('Horizontal', r1[0], r1[1], r2[0], r2[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Horizontal', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"

        elif op.label == ConstraintType.Parallel: # 5
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                sketch.addConstraint(Sketcher.Constraint('Parallel', r1[0], r2[0]))
                cmd = f"sketch.addConstraint(Sketcher.Constraint('Parallel', {r1[0]}, {r2[0]}))"

        elif op.label == ConstraintType.Vertical: # 6
            if len(refs) == 1:
                if r1 and r1[0] != -1:
                    sketch.addConstraint(Sketcher.Constraint('Vertical', r1[0]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Vertical', {r1[0]}))"
            elif len(refs) == 2:
                if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                    sketch.addConstraint(Sketcher.Constraint('Vertical', r1[0], r1[1], r2[0], r2[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Vertical', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"

        elif op.label == ConstraintType.Tangent: # 7
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                if r1[1] > 0 and r2[1] > 0:
                    sketch.addConstraint(Sketcher.Constraint('Tangent', r1[0], r1[1], r2[0], r2[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Tangent', {r1[0]}, {r1[1]}, {r2[0]}, {r2[1]}))"
                elif r1[1] > 0 and r2[1] == 0:
                    sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', r1[0], r2[0], r1[0], r1[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {r1[0]}, {r2[0]}, {r1[0]}, {r1[1]}))"
                elif r1[1] == 0 and r2[1] > 0:
                    sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', r1[0], r2[0], r2[0], r2[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {r1[0]}, {r2[0]}, {r2[0]}, {r2[1]}))"
                else:
                    # Curve-to-curve tangency (both refs are edges, not vertices)
                    # FreeCAD can handle this - it finds the tangent point automatically
                    sketch.addConstraint(Sketcher.Constraint('Tangent', r1[0], r2[0]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Tangent', {r1[0]}, {r2[0]}))"

        elif op.label == ConstraintType.Length: # 8
            if r1 and r1[0] != -1:
                geo = sketch.Geometry[r1[0]]
                # Length constraint only applies to line segments
                if isinstance(geo, Part.LineSegment):
                    length = (geo.EndPoint - geo.StartPoint).Length
                    sketch.addConstraint(Sketcher.Constraint('Distance', r1[0], length))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Distance', {r1[0]}, {length}))  # Line length"
                else:
                    cmd = f"# Skipped ConstraintType.Length: Not a LineSegment ({type(geo).__name__})"
            else:
                cmd = f"# Skipped ConstraintType.Length: Invalid reference ({r1=})"

        elif op.label == ConstraintType.Perpendicular: # 9
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                sketch.addConstraint(Sketcher.Constraint('Perpendicular', r1[0], r2[0]))
                cmd = f"sketch.addConstraint(Sketcher.Constraint('Perpendicular', {r1[0]}, {r2[0]}))"

        elif op.label == ConstraintType.Midpoint: # 10
            r3 = node_idx_map.get(refs[2]) if len(refs) > 2 else None

            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                # Case 1: Three-point Midpoint -> Symmetric
                if r3 and r3[0] != -1:
                    # Midpoint(A, B, C) semantics: A is the midpoint of B and C
                    # Tokenization preserves this order (first ref is midpoint, rest are sorted)
                    # So r1 is always the midpoint, r2 and r3 are the symmetric points
                    mid_ref, p1_ref, p2_ref = r1, r2, r3

                    sketch.addConstraint(Sketcher.Constraint('Symmetric', p1_ref[0], p1_ref[1], p2_ref[0], p2_ref[1], mid_ref[0], mid_ref[1]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Symmetric', {p1_ref[0]}, {p1_ref[1]}, {p2_ref[0]}, {p2_ref[1]}, {mid_ref[0]}, {mid_ref[1]}))"

                # Case 2: Midpoint(Point, Curve) or Midpoint(Curve, Point)
                else:
                    # Identify which is the Curve (Edge) and which is the Point
                    # r[1] == 0 -> Edge, r[1] > 0 -> Point/Vertex
                    c_ref, p_ref = None, None
                    if r1[1] == 0 and r2[1] > 0:
                        c_ref, p_ref = r1, r2
                    elif r2[1] == 0 and r1[1] > 0:
                        c_ref, p_ref = r2, r1

                    if c_ref:
                        # Found Point-Curve pair
                        # Apply Symmetric constraint: Point is symmetric between Curve.Start(1) and Curve.End(2)
                        sketch.addConstraint(Sketcher.Constraint('Symmetric', c_ref[0], 1, c_ref[0], 2, p_ref[0], p_ref[1]))
                        cmd_lines = [f"sketch.addConstraint(Sketcher.Constraint('Symmetric', {c_ref[0]}, 1, {c_ref[0]}, 2, {p_ref[0]}, {p_ref[1]}))"]

                        # If Curve is Arc, also add PointOnObject to ensure it lies on the arc
                        geo_type = sketch.Geometry[c_ref[0]]
                        if isinstance(geo_type, Part.ArcOfCircle):
                            sketch.addConstraint(Sketcher.Constraint('PointOnObject', p_ref[0], p_ref[1], c_ref[0]))
                            cmd_lines.append(f"sketch.addConstraint(Sketcher.Constraint('PointOnObject', {p_ref[0]}, {p_ref[1]}, {c_ref[0]}))")

                        cmd = "\n".join(cmd_lines)
                    else:
                        cmd = f"# Skipped ConstraintType.Midpoint: References not Point-Curve pair or Triple Points ({r1=}, {r2=}, {r3=})"
            else:
                cmd = f"# Skipped ConstraintType.Midpoint: Invalid references ({r1=}, {r2=})"

        elif op.label == ConstraintType.Equal: # 11
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                sketch.addConstraint(Sketcher.Constraint('Equal', r1[0], r2[0]))
                cmd = f"sketch.addConstraint(Sketcher.Constraint('Equal', {r1[0]}, {r2[0]}))"

        elif op.label == ConstraintType.Diameter: # 12
            if r1 and r1[0] != -1:
                src_diameter = float(sketch.Geometry[r1[0]].Radius * 2)
                # tgt_diameter = float(op.parameters['length'])
                # assert abs(src_diameter - tgt_diameter) < 1e-9, f"Diameter Mismatch: {src_diameter} != {tgt_diameter}"
                sketch.addConstraint(Sketcher.Constraint('Diameter', r1[0], src_diameter))
                cmd = f"sketch.addConstraint(Sketcher.Constraint('Diameter', {r1[0]}, {src_diameter}))"

        elif op.label == ConstraintType.Offset: # 13
            # Offset = relationship without distance specification
            # Lines: Parallel, Circles/Arcs: Concentric (distance is implicit in geometry)
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                geo1 = sketch.Geometry[r1[0]]
                geo2 = sketch.Geometry[r2[0]]

                if isinstance(geo1, Part.LineSegment) and isinstance(geo2, Part.LineSegment):
                    # Parallel constraint only (distance is defined by geometry)
                    sketch.addConstraint(Sketcher.Constraint('Parallel', r1[0], r2[0]))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Parallel', {r1[0]}, {r2[0]}))"
                elif hasattr(geo1, 'Center') and hasattr(geo2, 'Center'):
                    # Circles or Arcs: Concentric (radii difference is defined by geometry)
                    sketch.addConstraint(Sketcher.Constraint('Coincident', r1[0], 3, r2[0], 3))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Coincident', {r1[0]}, 3, {r2[0]}, 3))"
                else:
                    cmd = f"# Skipped ConstraintType.Offset: Unsupported geometry types ({type(geo1).__name__}, {type(geo2).__name__})"
            else:
                cmd = f"# Skipped ConstraintType.Offset: Invalid references ({r1=}, {r2=})"

        elif op.label == ConstraintType.Radius: # 14
            if r1 and r1[0] != -1:
                src_radius = float(sketch.Geometry[r1[0]].Radius)
                # tgt_radius = float(op.parameters['length'])
                # assert abs(src_radius - tgt_radius) < 1e-9, f"Radius Mismatch: {src_radius} != {tgt_radius}"
                sketch.addConstraint(Sketcher.Constraint('Radius', r1[0], src_radius))
                cmd = f"sketch.addConstraint(Sketcher.Constraint('Radius', {r1[0]}, {src_radius}))"

        elif op.label == ConstraintType.Concentric: # 15
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                sketch.addConstraint(Sketcher.Constraint('Coincident', r1[0], 3, r2[0], 3))
                cmd = f"sketch.addConstraint(Sketcher.Constraint('Coincident', {r1[0]}, 3, {r2[0]}, 3))"

        elif op.label == ConstraintType.Fix: # 16
            cmd = f"# Skipped ConstraintType.Fix (We should avoid its use)"
            # if r1 and r1[0] != -1:
            #     if r1[1] == 0:
            #         sketch.addConstraint(Sketcher.Constraint('Block', r1[0]))
            #         cmd = f"sketch.addConstraint(Sketcher.Constraint('Block', {r1[0]}))"
            #     else:
            #         cmd = f"# Skipped ConstraintType.Fix (Point Fix is uncommon in practice)"

        elif op.label == ConstraintType.Angle: # 17
            if r1 and r2 and r1[0] != -1 and r2[0] != -1:
                import math
                geo1 = sketch.Geometry[r1[0]]
                geo2 = sketch.Geometry[r2[0]]

                if hasattr(geo1, 'StartPoint') and hasattr(geo1, 'EndPoint') and \
                   hasattr(geo2, 'StartPoint') and hasattr(geo2, 'EndPoint'):
                    d1 = geo1.EndPoint - geo1.StartPoint
                    d2 = geo2.EndPoint - geo2.StartPoint

                    # Compute signed angle from d2 to d1
                    angle1 = math.atan2(d1.y, d1.x)
                    angle2 = math.atan2(d2.y, d2.x)
                    angle_rad = angle2 - angle1

                    # Normalize to [-pi, pi]
                    while angle_rad > math.pi:
                        angle_rad -= 2 * math.pi
                    while angle_rad < -math.pi:
                        angle_rad += 2 * math.pi

                    angle_deg = math.degrees(angle_rad)
                    sketch.addConstraint(Sketcher.Constraint('Angle', r1[0], r2[0], angle_rad))
                    cmd = f"sketch.addConstraint(Sketcher.Constraint('Angle', {r1[0]}, {r2[0]}, {angle_rad}))  # {angle_deg:.2f} deg"
                else:
                    cmd = f"# Skipped ConstraintType.Angle: No direction vectors ({r1=}, {r2=})"
            else:
                cmd = f"# Skipped ConstraintType.Angle: Invalid references ({r1=}, {r2=})"

        elif op.label == ConstraintType.Subnode: # 101
            child_idx = refs[0]
            parent_idx = refs[1]
            child_info = node_idx_map.get(child_idx)
            parent_info = node_idx_map.get(parent_idx)

            if child_info and child_info[0] == 'subnode' and parent_info:
                sub_type = child_info[1]
                parent_geo_id = parent_info[0]
                pt_pos = 0
                if sub_type == datalib.SubnodeType.SN_Start: pt_pos = 1
                elif sub_type == datalib.SubnodeType.SN_End: pt_pos = 2
                elif sub_type == datalib.SubnodeType.SN_Center: pt_pos = 3

                if pt_pos > 0:
                    node_idx_map[child_idx] = (parent_geo_id, pt_pos)
                    cmd = f"# Subnode Resolved: {child_idx} -> ({parent_geo_id}, {pt_pos})"
                else:
                    raise ValueError(f"Unknown Subnode Type: {sub_type}")

    except Exception as e:
        print(f"Warning: Constraint failure ({op.label}): {e}")
        cmd = f"# Error: {e}"
        raise e

    return cmd


def resolve_subnodes(seq: list[NodeOp | EdgeOp]) -> dict[int, tuple[int, int]]:
    node_op_list = [op for op in seq if isinstance(op, NodeOp)]
    node_idx_map = {}
    geo_counter = 0

    for node_idx, op in enumerate(node_op_list):
        if op.label == EntityType.External:
            # Unknown primitives, might be resolved someday?
            node_idx_map[node_idx] = (-1, 0)
        elif isinstance(op.label, datalib.SubnodeType):
            # Placeholder, will be resolved by EdgeOp
            node_idx_map[node_idx] = ('subnode', op.label)
        elif op.label in (EntityType.Stop, EntityType.Unknown):
            # Non-primitives
            pass
        else:
            # Primitives
            node_idx_map[node_idx] = (geo_counter, 1 if op.label == EntityType.Point else 0)
            geo_counter += 1

    # Resolve Subnodes using EdgeOp(label=ConstraintType.Subnode, references=(child, parent))
    for op in seq:
        if isinstance(op, EdgeOp) and op.label == ConstraintType.Subnode:
            op = deduplicate_refs(op)
            child_idx = op.references[0]
            parent_idx = op.references[1]
            assert parent_idx in node_idx_map
            assert child_idx in node_idx_map and node_idx_map[child_idx][0] == 'subnode'

            parent_info = node_idx_map[parent_idx]
            _, sub_type = node_idx_map[child_idx]

            if parent_info[0] != -1:
                # Check for clockwise Arc
                is_clockwise = False
                parent_op = node_op_list[parent_idx]
                if parent_op.label == EntityType.Arc and parent_op.parameters.get('clockwise', False):
                    is_clockwise = True

                # Map subnode type to vertex index
                vtx = 0
                if sub_type == datalib.SubnodeType.SN_Start:
                    vtx = 2 if is_clockwise else 1
                elif sub_type == datalib.SubnodeType.SN_End:
                    vtx = 1 if is_clockwise else 2
                elif sub_type == datalib.SubnodeType.SN_Center:
                    vtx = 3
                assert vtx > 0, f"Subnode must be either of START(1), END(2), or CENTER(3)"

                node_idx_map[child_idx] = (parent_info[0], vtx)

    return node_idx_map


def resolve_redundant_coincident(seq: list[NodeOp | EdgeOp], node_idx_map: dict[int, tuple[int, int]], debug: bool = False) -> list[NodeOp | EdgeOp]:
    """
    PointOnObject(primitive_idx, primitive_point_pos, curve_idx) and Coincident(primitive_idx, primitive_point_pos, curve_idx, curve_point_pos) are deemed redundant.
    This function detects these two pairs and disable the former.
    NOTE: The former is also Coincident in Onshape.
    """
    sort_idx = lambda idx1, pt1, ref1, idx2, pt2, ref2: (idx1, pt1, ref1, idx2, pt2, ref2) if idx1 <= idx2 else (idx2, pt2, ref2, idx1, pt1, ref1)

    seq = copy.deepcopy(seq)

    # Scan for Coincident
    coincident_map = {}
    for coincident_op_idx, op in enumerate(seq):
        if isinstance(op, EdgeOp) and op.label == ConstraintType.Coincident:
            op = deduplicate_refs(op)
            refs = op.references
            if len(refs) != 2:
                raise ValueError(f"Coincident should have 2 refs after dedup, got {refs}")
            ref1, ref2 = refs[0], refs[1]
            idx1, pt1 = node_idx_map[ref1]
            idx2, pt2 = node_idx_map[ref2]
            idx1, pt1, ref1, idx2, pt2, ref2 = sort_idx(idx1, pt1, ref1, idx2, pt2, ref2)
            if idx1 == -1 or idx2 == -1: continue

            # self redundancy
            if idx1 == idx2:
                seq[coincident_op_idx] = EdgeOp(ConstraintType.Coincident, references=(ref1, ref2), parameters={"disable": True})
                continue

            # cross redundancy
            if (idx1, idx2) in coincident_map:
                pt1_prev, ref1_prev, pt2_prev, ref2_prev, coincident_op_idx_prev = coincident_map[(idx1, idx2)]
                if pt1 > 0 and pt2 > 0:
                    coincident_map[(idx1, idx2)] = (pt1, ref1, pt2, ref2, coincident_op_idx)
                    seq[coincident_op_idx_prev] = EdgeOp(ConstraintType.Coincident, references=(ref1_prev, ref2_prev), parameters={"disable": True})
                elif pt1_prev > 0 and pt2_prev > 0:
                    seq[coincident_op_idx] = EdgeOp(ConstraintType.Coincident, references=(ref1, ref2), parameters={"disable": True})
                else:
                    if debug: print(f"[resolve_redundant_coincident] WARNING: Double PointOnLine detected ({idx1=}, {idx2=} / {pt1=}, {pt2=} / {pt1_prev=}, {pt2_prev=})")
            else:
                coincident_map[(idx1, idx2)] = (pt1, ref1, pt2, ref2, coincident_op_idx)

    return seq


def resolve_tangent(seq: list[NodeOp | EdgeOp], node_idx_map: dict[int, tuple[int, int]]):
    """
    FreeCAD doesn't accept Tangent(curve1, curve2) + Coincident(curve1_point1, curve2_point2),
    which is unwillingly marked as redundant.
    This function summarizes these two operations into one: Tangent(curve1, point1, curve2, point2).
    """
    sort_idx = lambda idx1, pt1, ref1, idx2, pt2, ref2: (idx1, pt1, ref1, idx2, pt2, ref2) if idx1 <= idx2 else (idx2, pt2, ref2, idx1, pt1, ref1)

    seq = copy.deepcopy(seq)

    # Scan for Coincident
    coincident_map = {}
    for coincident_op_idx, op in enumerate(seq):
        if isinstance(op, EdgeOp) and op.label == ConstraintType.Coincident:
            op = deduplicate_refs(op)
            refs = op.references
            assert len(refs) == 2, f"Coincident should have 2 refs after dedup, got {refs}"
            ref1, ref2 = refs[0], refs[1]
            idx1, pt1 = node_idx_map[ref1]
            idx2, pt2 = node_idx_map[ref2]
            idx1, pt1, ref1, idx2, pt2, ref2 = sort_idx(idx1, pt1, ref1, idx2, pt2, ref2)
            if idx1 == -1 or idx2 == -1: continue
            coincident_map[(idx1, idx2)] = (pt1, ref1, pt2, ref2, coincident_op_idx)

    # Scan for Tangent
    for tangent_op_idx, op in enumerate(seq):
        if isinstance(op, EdgeOp) and op.label == ConstraintType.Tangent:
            # Check references (pt1_dummy == pt2_dummy == 0 in most cases)
            op = deduplicate_refs(op)
            refs = op.references
            assert len(refs) == 2, f"Tangent should have 2 refs after dedup, got {refs}"
            ref1_dummy, ref2_dummy = refs[0], refs[1]
            idx1, pt1_dummy = node_idx_map[ref1_dummy]
            idx2, pt2_dummy = node_idx_map[ref2_dummy]
            idx1, pt1_dummy, ref1_dummy, idx2, pt2_dummy, ref2_dummy = sort_idx(idx1, pt1_dummy, ref1_dummy, idx2, pt2_dummy, ref2_dummy)

            if (idx1, idx2) in coincident_map:
                pt1, ref1, pt2, ref2, coincident_op_idx = coincident_map[(idx1, idx2)]
                if pt1_dummy > 0:
                    assert pt1_dummy == pt1
                if pt2_dummy > 0:
                    assert pt2_dummy == pt2

                seq[tangent_op_idx] = EdgeOp(ConstraintType.Tangent, references=(ref1, ref2), parameters={})
                seq[coincident_op_idx] = EdgeOp(ConstraintType.Coincident, references=(ref1, ref2), parameters={"disable": True})

    return seq


def draw_svg(doc, sketch, output_svg_path: str, debug: bool = False):
    """
    Generate SVG from sketch geometry.
    - Construction lines are rendered as dashed lines.
    - Regular geometry is rendered as solid lines.
    """
    import math

    # Collect geometry data and compute bounding box
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
    svg_elements = []

    for i, geo in enumerate(sketch.Geometry):
        is_construction = sketch.getConstruction(i)

        # Extract geometry coordinates
        try:
            # Check for Circle/Arc FIRST (they also have StartPoint/EndPoint)
            if hasattr(geo, 'Center') and hasattr(geo, 'Radius'):
                # Circle or Arc
                c = geo.Center
                r = geo.Radius
                x_min = min(x_min, c.x - r)
                y_min = min(y_min, c.y - r)
                x_max = max(x_max, c.x + r)
                y_max = max(y_max, c.y + r)

                # Distinguish full circle from arc by geometry type or angle span
                geo_type_name = type(geo).__name__
                is_full_circle = 'Circle' in geo_type_name and 'Arc' not in geo_type_name

                if not is_full_circle and hasattr(geo, 'FirstParameter') and hasattr(geo, 'LastParameter'):
                    # Check if it's nearly a full circle (angle span ≈ 2π)
                    angle_span = abs(geo.LastParameter - geo.FirstParameter)
                    if angle_span >= 2 * math.pi - 0.01:
                        is_full_circle = True

                if is_full_circle:
                    svg_elements.append(('circle', c.x, c.y, r, is_construction))
                else:
                    # Calculate base angle from StartPoint (most reliable)
                    p_start = geo.StartPoint
                    base_angle = math.atan2(p_start.y - c.y, p_start.x - c.x)

                    # Arc span
                    arc_span = geo.LastParameter - geo.FirstParameter

                    # Direction.z tells us CCW (+1) or CW (-1)
                    is_ccw = geo.Axis.z > 0

                    # For CW arcs, we go backwards from start
                    if is_ccw:
                        start_angle = base_angle
                        end_angle = base_angle + arc_span
                    else:
                        start_angle = base_angle
                        end_angle = base_angle - arc_span

                    svg_elements.append(('arc', c.x, c.y, r, start_angle, end_angle, is_ccw, is_construction))

            elif hasattr(geo, 'StartPoint') and hasattr(geo, 'EndPoint'):
                # Line segment (check after Circle/Arc)
                p1 = geo.StartPoint
                p2 = geo.EndPoint
                x_min = min(x_min, p1.x, p2.x)
                y_min = min(y_min, p1.y, p2.y)
                x_max = max(x_max, p1.x, p2.x)
                y_max = max(y_max, p1.y, p2.y)
                svg_elements.append(('line', p1.x, p1.y, p2.x, p2.y, is_construction))

            elif hasattr(geo, 'X') and hasattr(geo, 'Y'):
                # Point
                x_min = min(x_min, geo.X)
                y_min = min(y_min, geo.Y)
                x_max = max(x_max, geo.X)
                y_max = max(y_max, geo.Y)
                svg_elements.append(('point', geo.X, geo.Y, is_construction))

        except Exception as e:
            if debug:
                print(f"Skipping geometry {i}: {e}")

    if x_min >= float('inf'):
        print("SVG Export failed: No valid geometry")
        return

    # Calculate viewBox and stroke width
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    vb_x = x_min - margin
    vb_y = -(y_max + margin)  # SVG Y is inverted
    vb_w = (x_max - x_min) + 2 * margin
    vb_h = (y_max - y_min) + 2 * margin

    diag = (vb_w**2 + vb_h**2) ** 0.5
    stroke_width = max(diag * 0.001, 0.5)
    point_radius = stroke_width * 1.5

    # Build SVG content
    svg_lines = [
        '<?xml version="1.0"?>',
        f'<svg viewBox="{vb_x:.4f} {vb_y:.4f} {vb_w:.4f} {vb_h:.4f}" xmlns="http://www.w3.org/2000/svg">',
        f'<g transform="scale(1,-1)">',  # Flip Y axis
    ]

    for elem in svg_elements:
        if elem[0] == 'line':
            _, x1, y1, x2, y2, is_const = elem
            color = "#0088FF" if is_const else "#FF0000"
            dash = f'stroke-dasharray="{stroke_width*3:.2f},{stroke_width*2:.2f}"' if is_const else ''
            svg_lines.append(
                f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" {dash} stroke-linecap="round"/>'
            )

        elif elem[0] == 'circle':
            _, cx, cy, r, is_const = elem
            color = "#0088FF" if is_const else "#FF0000"
            dash = f'stroke-dasharray="{stroke_width*3:.2f},{stroke_width*2:.2f}"' if is_const else ''
            svg_lines.append(
                f'<circle cx="{cx:.4f}" cy="{cy:.4f}" r="{r:.4f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" fill="none" {dash}/>'
            )

        elif elem[0] == 'arc':
            _, cx, cy, r, start_angle, end_angle, is_ccw, is_const = elem

            # Calculate start and end points
            x1 = cx + r * math.cos(start_angle)
            y1 = cy + r * math.sin(start_angle)
            x2 = cx + r * math.cos(end_angle)
            y2 = cy + r * math.sin(end_angle)

            # Arc span determines large_arc flag
            arc_span = abs(end_angle - start_angle)
            large_arc = 1 if arc_span > math.pi else 0

            # Sweep: CCW arcs need sweep=1 (after Y-flip appears CCW), CW arcs need sweep=0
            sweep = 1 if is_ccw else 0

            color = "#0088FF" if is_const else "#FF0000"
            dash = f'stroke-dasharray="{stroke_width*3:.2f},{stroke_width*2:.2f}"' if is_const else ''
            svg_lines.append(
                f'<path d="M {x1:.4f},{y1:.4f} A {r:.4f},{r:.4f} 0 {large_arc},{sweep} {x2:.4f},{y2:.4f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.2f}" fill="none" {dash} stroke-linecap="round"/>'
            )

        elif elem[0] == 'point':
            _, x, y, is_const = elem
            color = "#0088FF" if is_const else "#FF0000"
            svg_lines.append(
                f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{point_radius:.4f}" fill="{color}"/>'
            )

    svg_lines.append('</g>')
    svg_lines.append('</svg>')

    content = '\n'.join(svg_lines)

    with open(output_svg_path, 'w') as f:
        f.write(content)

    if debug:
        display(SVG(data=content))
        print(f"SVG exported to {output_svg_path}")


def process_sequence(
    seq: list[NodeOp | EdgeOp],
    output_py_path: str = "sketch_history.py",
    output_svg_path: str = "sketch_export.svg",
    debug: bool = False,
):
    doc = FreeCAD.newDocument("MyDoc")
    body = doc.addObject('PartDesign::Body', 'Body')
    sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    xy_plane = body.Origin.OriginFeatures[3]
    sketch.AttachmentSupport = xy_plane
    sketch.MapMode = 'FlatFace'

    history_commands = []
    history_commands.append("import FreeCAD")
    history_commands.append("import Part")
    history_commands.append("import Sketcher")
    history_commands.append("doc = FreeCAD.newDocument('MyDoc')")
    history_commands.append("body = doc.addObject('PartDesign::Body', 'Body')")
    history_commands.append("sketch = body.newObject('Sketcher::SketchObject', 'Sketch')")
    history_commands.append("# ... attachment setup skipped for brevity ...")

    # rescale
    seq = [meter_to_millimeter(op, debug) for op in seq]
    if debug:
        geom_cnt = 0
        for i, op in enumerate(seq):
            if isinstance(op, NodeOp):
                print(f"{i:02d} {geom_cnt:2d}", op)
                geom_cnt += 1
            else:
                print(f"{i:02d}    {op}")

    # Pass 1: Map Building (node_idx_map = {node_id: (primitive_id, 0~3), ...} where node_id includes both primitives and subnodes)
    node_idx_map = resolve_subnodes(seq)
    if debug: print(f"{node_idx_map=}")

    # Pass 2: Constraint Refinement
    seq = resolve_redundant_coincident(seq, node_idx_map, debug)
    seq = resolve_tangent(seq, node_idx_map)

    # Pass 3: Generation
    for i, op in enumerate(seq):

        if isinstance(op, NodeOp):
            # Primitive Creation
            mapping, cmd = convert_primitive(op, sketch)
            if cmd:
                cmd = cmd.replace("Vector (", "FreeCAD.Vector(")
                history_commands.append(cmd)

        elif isinstance(op, EdgeOp):
            # Constraint Application
            cmd = convert_constraint(op, sketch, node_idx_map)
            if cmd:
                history_commands.append(cmd)

    doc.recompute()

    # Dump History to File
    with open(output_py_path, "w") as f:
        history_commands.append("doc.recompute()")
        f.write("\n".join(history_commands))
    if debug: print(f"History saved to {output_py_path}")

    # Dump SVG
    draw_svg(doc, sketch, output_svg_path, debug)


if __name__ == '__main__':
    seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')

    output_dir = "sg_filtered_unique"
    os.makedirs(output_dir, exist_ok=True)

    error_log = os.path.join(output_dir, "error_log.txt")
    if os.path.isfile(error_log):
        os.remove(error_log)

    for i, seq in enumerate(tqdm(seq_data['sequences'])):

        if os.path.isfile(f"{output_dir}/{i}.svg"):
            print(f"{output_dir}/{i}.svg already exists. Skipping...")
            continue

        sketch = datalib.sketch_from_sequence(seq)
        fig = datalib.render_sketch(sketch)
        fig.savefig(f"{output_dir}/{i}_gt.svg")
        plt.close(fig)

        try:
            process_sequence(
                seq,
                output_py_path=f"{output_dir}/{i}.py",
                output_svg_path=f"{output_dir}/{i}.svg",
                debug=False)
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            with open(error_log, "a") as f:
                f.write(f"Error processing sequence {i}: {e}\n\n")
            continue


    # seq = seq_data['sequences'][20]
    # sketch = datalib.sketch_from_sequence(seq)
    # fig = datalib.render_sketch(sketch)
    # fig.savefig('sketch_gt.svg')
    # process_sequence(seq, debug=True)
