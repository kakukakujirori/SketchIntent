"""
Treat solver_output as user prompts
"""
from typing import Any
import argparse
import base64
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from io import BytesIO

import cairosvg
import traceback
import PIL.Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler

# Add FreeCAD Libs (conditionally)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part
import Sketcher

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array


from img2cad_diffusion.convert_sketch_to_freecad import convert_sketch_to_freecad, draw_svg
from v20260116_draw_constraints import visualize_sketch_for_vlm, load_and_extract_from_file


def parse_vector(vec_str):
    # Extracts numbers from "Vector(x, y, z)"
    # Pattern supports scientific notation like 8.326e-14, 1.5E+10
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", vec_str)
    # Return x, y as float
    return [float(n) for n in nums[:2]]


def parse_freecad_script(filepath_or_commands: str | list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primitives = []
    constraints_gt = []

    if isinstance(filepath_or_commands, str):
        with open(filepath_or_commands, 'r') as f:
            content = f.read()
        lines = content.split('\n')
    else:
        lines = filepath_or_commands

    for line in lines:
        line = line.strip()
        if not line: continue

        # --- Primitives ---
        if line.startswith('fc_sketch.addGeometry('):
            is_construction = 'True' in line.split(',')[-1]

            if 'Part.LineSegment' in line:
                # Part.LineSegment(Vector(...), Vector(...))
                vecs = re.findall(r'Vector\s*\([^)]+\)', line)
                if len(vecs) >= 2:
                    p1 = parse_vector(vecs[0])
                    p2 = parse_vector(vecs[1])
                    primitives.append({
                        "type": "LineSegment",
                        "start": p1,
                        "end": p2,
                        "is_construction": is_construction
                    })
            elif 'Part.Point' in line:
                # Part.Point(Vector(...))
                vecs = re.findall(r'Vector\s*\([^)]+\)', line)
                if len(vecs) >= 1:
                    p1 = parse_vector(vecs[0])
                    primitives.append({
                        "type": "Point",
                        "x": p1[0],
                        "y": p1[1],
                        "is_construction": is_construction
                    })
            elif 'Part.Circle' in line:
                # Part.Circle(Vector(...), Vector(...), Radius)
                inner_match = re.search(r'Part\.Circle\s*\((.*)\)', line)
                if inner_match:
                    inner_content = inner_match.group(1)
                    vecs = re.findall(r'Vector\s*\([^)]+\)', inner_content)
                    if len(vecs) >= 1:
                        center = parse_vector(vecs[0])
                        # Extract radius: find number after second Vector
                        v2_end_idx = inner_content.rfind('Vector')
                        if v2_end_idx != -1:
                            v2_close = inner_content.find(')', v2_end_idx)
                            if v2_close != -1:
                                remainder = inner_content[v2_close+1:]
                                radius_match = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", remainder)
                                if radius_match:
                                    radius = float(radius_match.group(0))
                                    primitives.append({
                                        "type": "Circle",
                                        "center": center,
                                        "radius": radius,
                                        "is_construction": is_construction,
                                        "is_clockwise": False
                                    })
            elif 'Part.ArcOfCircle' in line:
                # Part.ArcOfCircle(Start, Mid, End) or similar 3-point in this context
                vecs = re.findall(r'Vector\s*\([^)]+\)', line)
                if len(vecs) >= 3:
                    p1 = parse_vector(vecs[0])
                    p2 = parse_vector(vecs[1])
                    p3 = parse_vector(vecs[2])
                    primitives.append({
                        "type": "Arc",
                        "start": p1,
                        "mid": p2,
                        "end": p3,
                        "is_construction": is_construction
                    })

        # --- Constraints ---
        elif line.startswith('fc_sketch.addConstraint('):
            # Example: fc_sketch.addConstraint(Sketcher.Constraint('Tangent', 3, 2, 6, 2))
            inner = re.search(r"Sketcher\.Constraint\((.*)\)", line)
            if inner:
                args_str = inner.group(1).rstrip(')')
                # Naive CSV split (assuming no commas in strings)
                raw_args = [a.strip().strip("'").strip('"') for a in args_str.split(',')]
                if not raw_args: continue

                c_type = raw_args[0]
                c_args = []
                for a in raw_args[1:]:
                    try:
                        # Convert to float if it has dot, else int
                        if '.' in a:
                            c_args.append(float(a))
                        else:
                            c_args.append(int(a))
                    except:
                        c_args.append(a)

                constraints_gt.append({"type": c_type, "args": c_args})

    return primitives, constraints_gt


def compute_arc_center(start: list, mid: list, end: list) -> tuple | None:
    """
    Compute the center of a circle passing through three points (start, mid, end).
    Returns (cx, cy, radius) or None if points are collinear.
    """
    ax, ay = start
    bx, by = mid
    cx, cy = end

    # Calculate the determinant to check if points are collinear
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None  # Points are collinear

    # Calculate center coordinates
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

    # Calculate radius
    radius = math.sqrt((ax - ux)**2 + (ay - uy)**2)

    return (ux, uy, radius)


def get_point_from_primitive(primitives: list, entity_idx: int, subnode: int) -> list | None:
    """
    Get a point [x, y] from a primitive based on entity index and subnode.

    Subnode values:
        0: Entity itself (for Point primitive only)
        1: StartPoint
        2: EndPoint
        3: CenterPoint (Circle or Arc)
    """
    if entity_idx < 0 or entity_idx >= len(primitives):
        return None

    prim = primitives[entity_idx]
    ptype = prim["type"]

    if subnode == 0:  # Entity itself
        if ptype == "Point":
            return [prim["x"], prim["y"]]
        return None

    elif subnode == 1:  # StartPoint
        if ptype == "LineSegment":
            return prim["start"]
        elif ptype == "Arc":
            return prim["start"]

    elif subnode == 2:  # EndPoint
        if ptype == "LineSegment":
            return prim["end"]
        elif ptype == "Arc":
            return prim["end"]

    elif subnode == 3:  # CenterPoint
        if ptype == "Circle":
            return prim["center"]
        elif ptype == "Arc":
            result = compute_arc_center(prim["start"], prim["mid"], prim["end"])
            if result:
                return [result[0], result[1]]

    return None


def compute_constraint_value(ctype: str, primitives: list, e1: int, s1: int | None = None,
                              e2: int | None = None, s2: int | None = None) -> float | None:
    """
    Compute the value for a dimensional constraint based on primitive coordinates.

    Args:
        ctype: Constraint type ('Distance', 'Angle', 'Radius', 'Diameter')
        primitives: List of primitive dicts
        e1, s1: First entity index and subnode
        e2, s2: Second entity index and subnode (optional)

    Returns:
        Computed value or None if unable to compute
    """
    if e1 is None or e1 < 0 or e1 >= len(primitives):
        return None

    prim1 = primitives[e1]

    if ctype == "Distance":
        if e2 is not None and s1 is not None and s2 is not None:
            # Point-to-point distance
            p1 = get_point_from_primitive(primitives, e1, s1)
            p2 = get_point_from_primitive(primitives, e2, s2)
            if p1 and p2:
                return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        elif e2 is not None and s1 is not None and s2 is None:
            # Point-to-edge distance: compute perpendicular distance from point to line
            p1 = get_point_from_primitive(primitives, e1, s1)
            prim2 = primitives[e2] if e2 < len(primitives) else None
            if p1 and prim2 and prim2["type"] == "LineSegment":
                lx1, ly1 = prim2["start"]
                lx2, ly2 = prim2["end"]
                # Vector from line start to point
                dx, dy = lx2 - lx1, ly2 - ly1
                line_len_sq = dx**2 + dy**2
                if line_len_sq > 1e-9:
                    # Project point onto line and compute perpendicular dist
                    t = ((p1[0] - lx1) * dx + (p1[1] - ly1) * dy) / line_len_sq
                    closest_x = lx1 + t * dx
                    closest_y = ly1 + t * dy
                    return math.sqrt((p1[0] - closest_x)**2 + (p1[1] - closest_y)**2)
        elif e2 is None or (s1 is None and s2 is None):
            # Line length (single entity)
            if prim1["type"] == "LineSegment":
                sx, sy = prim1["start"]
                ex, ey = prim1["end"]
                return math.sqrt((ex - sx)**2 + (ey - sy)**2)

    elif ctype == "Radius":
        if prim1["type"] == "Circle":
            return prim1.get("radius")
        elif prim1["type"] == "Arc":
            result = compute_arc_center(prim1["start"], prim1["mid"], prim1["end"])
            if result:
                return result[2]  # radius

    elif ctype == "Diameter":
        if prim1["type"] == "Circle":
            r = prim1.get("radius")
            return r * 2 if r else None
        elif prim1["type"] == "Arc":
            result = compute_arc_center(prim1["start"], prim1["mid"], prim1["end"])
            if result:
                return result[2] * 2

    elif ctype == "Angle":
        if e2 is not None:
            # Angle between two lines
            prim2 = primitives[e2] if e2 < len(primitives) else None
            if prim1["type"] == "LineSegment" and prim2 and prim2["type"] == "LineSegment":
                dx1 = prim1["end"][0] - prim1["start"][0]
                dy1 = prim1["end"][1] - prim1["start"][1]
                dx2 = prim2["end"][0] - prim2["start"][0]
                dy2 = prim2["end"][1] - prim2["start"][1]
                angle1 = math.atan2(dy1, dx1)
                angle2 = math.atan2(dy2, dx2)
                return angle2 - angle1  # radian
        else:
            # Single line angle (slope angle from horizontal)
            if prim1["type"] == "LineSegment":
                dx = prim1["end"][0] - prim1["start"][0]
                dy = prim1["end"][1] - prim1["start"][1]
                return math.atan2(dy, dx)  # radian

    return None


def export_to_freecad_script(primitives: list[dict[str, Any]], constraints: list[dict[str, Any]], output_path: str):
    """
    Convert parsed primitives and inferred constraints JSON to a FreeCAD Python script.

    Args:
        primitives: List of primitive dicts (from parse_freecad_script)
        constraints: List of constraint dicts (from extract_json_from_thinking_output)
        output_path: Path to save the generated .py file
    """
    lines = []
    lines.append("import FreeCAD")
    lines.append("import Part")
    lines.append("import Sketcher")
    lines.append("from FreeCAD import Vector")
    lines.append("")
    lines.append("# Create a new document")
    lines.append("doc = FreeCAD.newDocument('ConstraintTest')")
    lines.append("body = doc.addObject('PartDesign::Body', 'Body')")
    lines.append("fc_sketch = doc.addObject('Sketcher::SketchObject', 'Sketch')")
    lines.append("body.addObject(fc_sketch)")
    lines.append("")
    lines.append("# --- Add Primitives ---")

    for i, prim in enumerate(primitives):
        ptype = prim.get("type")
        is_const = prim.get("is_construction", False)

        if ptype == "LineSegment":
            sx, sy = prim["start"]
            ex, ey = prim["end"]
            lines.append(f"fc_sketch.addGeometry(Part.LineSegment(Vector({sx}, {sy}, 0), Vector({ex}, {ey}, 0)), {is_const})")
        elif ptype == "Point":
            x, y = prim["x"], prim["y"]
            lines.append(f"fc_sketch.addGeometry(Part.Point(Vector({x}, {y}, 0)), {is_const})")
        elif ptype == "Circle":
            cx, cy = prim["center"]
            r = prim["radius"]
            lines.append(f"fc_sketch.addGeometry(Part.Circle(Vector({cx}, {cy}, 0), Vector(0, 0, 1), {r}), {is_const})")
        elif ptype == "Arc":
            sx, sy = prim["start"]
            mx, my = prim["mid"]
            ex, ey = prim["end"]
            lines.append(f"fc_sketch.addGeometry(Part.ArcOfCircle(Vector({sx}, {sy}, 0), Vector({mx}, {my}, 0), Vector({ex}, {ey}, 0)), {is_const})")

    lines.append("")
    lines.append("# --- Add Constraints ---")

    for c in constraints:
        ctype = c.get("type", "")
        e1 = c.get("entity1")
        s1 = c.get("subnode1")
        e2 = c.get("entity2")
        s2 = c.get("subnode2")
        e3 = c.get("entity3")
        s3 = c.get("subnode3")

        # Build constraint arguments based on type
        if ctype == "Coincident":
            # 4-arg: entity1, subnode1, entity2, subnode2
            if s1 is not None and s2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {e1}, {s1}, {e2}, {s2}))")
            else:
                lines.append(f"# Skipped Coincident: missing subnode info")
        elif ctype == "Tangent":
            # Can be 4-arg (point-to-point) or 2-arg (edge-to-edge)
            if s1 is not None and s2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {e1}, {s1}, {e2}, {s2}))")
            elif e1 is not None and e2 is not None:
                # Edge-to-edge tangent (no subnode required)
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {e1}, {e2}))")
            else:
                lines.append(f"# Skipped Tangent: missing required args")
        elif ctype == "TangentViaPoint":
            # TangentViaPoint(e1, e2, ePoint, sPoint) - 4 args
            # entity3 and subnode3 hold the point reference
            if all(v is not None for v in [e1, e2, e3, s3]):
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', {e1}, {e2}, {e3}, {s3}))")
            else:
                lines.append(f"# Skipped TangentViaPoint: missing args")
        elif ctype == "Concentric":
            # Concentric means center points coincide: Coincident(e1, 3, e2, 3)
            if e1 is not None and e2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {e1}, 3, {e2}, 3))  # Concentric")
            else:
                lines.append(f"# Skipped Concentric: missing args")
        elif ctype in ["Horizontal", "Vertical"]:
            # Can be 1-arg (line) or 4-arg (two points)
            if e2 is not None and s1 is not None and s2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{ctype}', {e1}, {s1}, {e2}, {s2}))")
            else:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{ctype}', {e1}))")
        elif ctype in ["Parallel", "Perpendicular", "Equal"]:
            # 2-arg: entity1, entity2
            if e2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{ctype}', {e1}, {e2}))")
            else:
                lines.append(f"# Skipped {ctype}: missing entity2")
        elif ctype == "PointOnObject":
            # 3-arg: entity1, subnode1, entity2
            if s1 is not None and e2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', {e1}, {s1}, {e2}))")
            else:
                lines.append(f"# Skipped PointOnObject: missing args")
        elif ctype == "Symmetric":
            # 6-arg: entity1, subnode1, entity2, subnode2, entity3, subnode3
            if all(v is not None for v in [s1, e2, s2, e3, s3]):
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {e1}, {s1}, {e2}, {s2}, {e3}, {s3}))")
            else:
                lines.append(f"# Skipped Symmetric: missing args")
        elif ctype == "Distance":
            # Use value from Qwen3 output if available, otherwise compute
            value = c.get("value")
            if value is None:
                value = compute_constraint_value("Distance", primitives, e1, s1, e2, s2)
            if value is not None:
                if e2 is not None and s1 is not None and s2 is not None:
                    # Point-to-point distance
                    lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {e1}, {s1}, {e2}, {s2}, {value}))")
                elif e2 is not None and s1 is not None:
                    # Point-to-edge distance (s2 is None or 0)
                    lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {e1}, {s1}, {e2}, 0, {value}))")
                else:
                    # Line length
                    lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {e1}, {value}))")
            else:
                lines.append(f"# Skipped Distance: could not compute value")
        elif ctype == "Angle":
            # Use value from Qwen3 output if available, otherwise compute
            value = c.get("value")
            if value is None:
                value = compute_constraint_value("Angle", primitives, e1, s1, e2, s2)
            if value is not None:
                if e2 is not None:
                    lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Angle', {e1}, {e2}, {value}))")
                else:
                    lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Angle', {e1}, {value}))")
            else:
                lines.append(f"# Skipped Angle: could not compute value")
        elif ctype == "Radius":
            value = c.get("value")
            if value is None:
                value = compute_constraint_value("Radius", primitives, e1)
            if value is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Radius', {e1}, {value}))")
            else:
                lines.append(f"# Skipped Radius: could not compute value")
        elif ctype == "Diameter":
            value = c.get("value")
            if value is None:
                value = compute_constraint_value("Diameter", primitives, e1)
            if value is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Diameter', {e1}, {value}))")
            else:
                lines.append(f"# Skipped Diameter: could not compute value")
        else:
            lines.append(f"# Unknown constraint type: {ctype}")

    lines.append("")
    lines.append("# Recompute")
    lines.append("doc.recompute()")

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Exported to {output_path}")


def format_constraint_to_json(ctype: str, args: list) -> dict:
    """Convert raw FreeCAD constraint arguments to structured JSON format.

    Based on FreeCAD constraint API signatures from convert_sketch_to_freecad.py.

    Args:
        ctype: Constraint type name (e.g., 'Coincident', 'Horizontal', 'Tangent')
        args: List of constraint arguments from parse_freecad_script

    Returns:
        dict with keys: type, entity1, subnode1, entity2, subnode2, entity3, subnode3
    """
    c_out = {
        "type": ctype,
        "entity1": None, "subnode1": None,
        "entity2": None, "subnode2": None,
        "entity3": None, "subnode3": None
    }

    def get_arg(idx):
        return args[idx] if idx < len(args) else None

    # Helper to check if arg is numeric (not a value like distance)
    def is_int_arg(val):
        return isinstance(val, int)

    if ctype == "Coincident":
        # Coincident(e1, s1, e2, s2) - 4 args
        if len(args) >= 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    elif ctype == "PointOnObject":
        # PointOnObject(e1, s1, e2) - 3 args
        c_out["entity1"] = get_arg(0)
        c_out["subnode1"] = get_arg(1)
        c_out["entity2"] = get_arg(2)

    elif ctype in ["Horizontal", "Vertical"]:
        # Horizontal(e1) - 1 arg for line
        # Horizontal(e1, s1, e2, s2) - 4 args for two points
        if len(args) == 1:
            c_out["entity1"] = get_arg(0)
        elif len(args) >= 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    elif ctype in ["Parallel", "Perpendicular", "Equal"]:
        # Parallel(e1, e2) - 2 args
        c_out["entity1"] = get_arg(0)
        c_out["entity2"] = get_arg(1)

    elif ctype == "Tangent":
        # Tangent(e1, e2) - 2 args (edge-edge)
        # Tangent(e1, s1, e2, s2) - 4 args (point-point)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
        elif len(args) >= 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    elif ctype == "TangentViaPoint":
        # TangentViaPoint(e1, e2, e3, s3) - 4 args
        c_out["entity1"] = get_arg(0)
        c_out["entity2"] = get_arg(1)
        c_out["entity3"] = get_arg(2)
        c_out["subnode3"] = get_arg(3)

    elif ctype == "Concentric":
        # Concentric(e1, e2) - 2 args (centers coincide)
        c_out["entity1"] = get_arg(0)
        c_out["entity2"] = get_arg(1)

    elif ctype == "Symmetric":
        # Symmetric(e1, s1, e2, s2, e3, s3) - 6 args
        c_out["entity1"] = get_arg(0)
        c_out["subnode1"] = get_arg(1)
        c_out["entity2"] = get_arg(2)
        c_out["subnode2"] = get_arg(3)
        c_out["entity3"] = get_arg(4)
        c_out["subnode3"] = get_arg(5)

    elif ctype == "Distance":
        # Distance(e1, value) - 2 args (line length)
        # Distance(e1, s1, e2, s2, value) - 5 args (point-point or point-edge)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["value"] = get_arg(1)  # Preserve the value
        elif len(args) >= 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["value"] = get_arg(4)  # Preserve the value

    elif ctype in ["Diameter", "Radius"]:
        # Diameter(e1, value) - 2 args
        # Radius(e1, value) - 2 args
        c_out["entity1"] = get_arg(0)
        c_out["value"] = get_arg(1)  # Preserve the value

    elif ctype == "Angle":
        # Angle(e1, value) - 2 args (single line angle)
        # Angle(e1, e2, value) - 3 args (two lines)
        # Angle(e1, s1, e2, s2, value) - 5 args (two points)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["value"] = get_arg(1)  # Preserve the value
        elif len(args) == 3:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
            c_out["value"] = get_arg(2)  # Preserve the value
        elif len(args) >= 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["value"] = get_arg(4)  # Preserve the value

    else:
        raise NotImplementedError(f"Unknown constraint type: {ctype}")

    # Return only non-null fields
    return {k: v for k, v in c_out.items() if v is not None}


def prepare_prompts(primitives_list: list, constraints_list: list, image_path_list: str | list[str], max_turns: int):

    if isinstance(image_path_list, str):
        image_path_list = [image_path_list]

    encoded_images = []
    for image_path in image_path_list:
        if image_path.endswith(".svg"):
            png_bytes = cairosvg.svg2png(url=image_path, scale=1.0)
            encoded_data = base64.b64encode(png_bytes).decode("utf-8")
        else:
            with open(image_path, "rb") as img_f:
                encoded_data = base64.b64encode(img_f.read()).decode("utf-8")
        encoded_images.append(encoded_data)

    primitives_text = json.dumps(primitives_list, indent=2)

    # Format constraints text for prompt (convert to JSON format if needed)
    constraints_json_list = []
    for c in constraints_list:
        if "args" not in c:
            raise ValueError(f"prepare_prompts expects constraints with 'args' key, got: {c}")
        constraints_json_list.append(format_constraint_to_json(c["type"], c["args"]))
    constraints_text = json.dumps(constraints_json_list, indent=2)

    # Run solver on partial constraints to get initial status
    initial_solver_output = run_solver_tool(primitives_list, constraints_list)

    system_prompt_text = """You are an expert CAD engineer.
Analyze the provided 2D sketch image, the list of geometric primitives, and the list of partial geometric constraints.

Your task is to **add missing constraints** to the provided partial constraint list until the sketch is fully constrained.
MOST CRUCIALLY, THESE CONSTRAINTS MUST REFLECT THE DESIGN INTENT OF THE ORIGINAL SKETCH.
Therefore, the constraints you infer must be intuitive, easy for post-editing, and human-friendly.

The output must include ALL constraints: both the given partial constraints AND the new constraints you infer.
Note that we define that the sketch is fully constrained if its Degree of Freedom is 2, i.e., leaving only the global xy translation.
It is also important that the constraints are not malformed, conflicting, or redundant.

The constraint types and grammar follow that of FreeCAD. The primitives are assigned reference numbers in the defined order,
and they are used to specify the position on which the constraints are applied.

[Inputs]
The input consists of a list of geometric primitives, a list of partial geometric constraints, and an image of the sketch.
The supported primitives and their arguments are:
- Point(x, y, is_construction)
- LineSegment((start_x, start_y), (end_x, end_y), is_construction)
- Circle((center_x, center_y), is_clockwise, radius, is_construction)
- Arc((start_x, start_y), (mid_x, mid_y), (end_x, end_y), is_construction)
* Note: mid_x/y for Arc is a point *on the arc* (roughly halfway), not the center or the mid-coord point.
The primitives are implicitly indexed starting from 0 (0-indexed) based on their order in the input list.
The partial constraints format is described in the following section.

[Outputs]
Output the **complete** list of constraints, which consists of:
1. The given partial constraints (copy them exactly as provided)
2. Additional constraints you infer from the visual arrangement, geometric properties, and the design intent.

General Argument Structure follows the pattern below:
{Type, Entity1, Subnode1, Entity2, Subnode2, Entity3, Subnode3, Value}

- Type: The type of the constraint (e.g., "Coincident", "Horizontal", "Parallel", "Tangent", etc. See [Constraint Types and Signatures] below).
- EntityN: The 0-based index of the primitive.
- SubnodeN: One of the following numbers, specifying which part of the primitive is being referenced.
    - 0: The Entity itself (e.g., the edge of a line or circle)
    - 1: StartPoint (Vertex)
    - 2: EndPoint (Vertex)
    - 3: CenterPoint (Strictly the center of a Circle or Arc, NOT the midpoint of a line or an arc)
- Value: The parameter value of dimension constraints (e.g., length, angle, etc.)

Depending on the Type, some arguments may be omitted. For example, Horizontal constraint only requires Entity1.

[Constraint Types and Signatures]
Strictly adhere to the following signatures for each constraint type. Arguments not listed for a type should be omitted.

NOTE: For INPUT partial dimension constraints, we show the parameter `Value` such as length or angle.
However, for OUTPUT complete constraints, you may omit the inference of these parameter `Value` for dimensional constraints.
Even though you infer the parameter `Value`, we discard them and instead calculate the `Value` from the original primitive configuration.

1. Relational Constraints:
- Coincident(Entity1, Subnode1, Entity2, Subnode2)
    * Fixes two points to be the same.
    * Example: Coincident(Line1, 2, Line2, 1) connects end of Line1 to start of Line2.
- PointOnObject(Entity1, Subnode1, Entity2)
    * Fixes a point (Entity1.Subnode1) to lie on an object (Entity2).
    * Use this instead of Coincident when constraining a point to a line or curve edge (Subnode 0).
- Vertical(Entity1)
    * Constrains a LineSegment (Entity1) to be vertical.
- Vertical(Entity1, Subnode1, Entity2, Subnode2)
    * Constrains two points to have the same X coordinate (vertically aligned).
- Horizontal(Entity1)
    * Constrains a LineSegment (Entity1) to be horizontal.
- Horizontal(Entity1, Subnode1, Entity2, Subnode2)
    * Constrains two points to have the same Y coordinate (horizontally aligned).
- Parallel(Entity1, Entity2)
    * Constrains two lines to be parallel.
- Perpendicular(Entity1, Entity2)
    * Constrains two lines to be perpendicular.
- Tangent(Entity1, Subnode1, Entity2, Subnode2)
    * Tangency between two curves or a curve and a line.
    * If constraining two circles/arcs, use Subnode 0 for both.
    * If constraining a line end to a circle, specify the point subnode (1 or 2) for the line and subnode 0 for the circle.
- Equal(Entity1, Entity2)
    * Constrains two segments to have equal length, or two circles/arcs to have equal radius.
- Symmetric(Entity1, Subnode1, Entity2, Subnode2, Entity3, Subnode3)
    * Constrains Entity1.Subnode1 and Entity2.Subnode2 to be symmetric with respect to Entity3.Subnode3.
    * Entity3 can be a Point (symmetry point) or a Line (symmetry axis).

2. Dimensional Constraints (Use only when necessary for fully constraining the sketch):
- Distance(Entity1, Subnode1, Entity2, Subnode2, Value)
    * Fixes distance between two points.
- Diameter(Entity1, Value)
    * Fixes diameter of Circle or Arc.
- Radius(Entity1, Value)
    * Fixes radius of Circle or Arc.
- Angle(Entity1, Value)
    * Fixes slope angle of a LineSegment or angular span of an Arc (Subnode1 should be 0).
- Angle(Entity1, Subnode1, Entity2, Subnode2, Value)
    * Fixes angle between two lines.
- AngleViaPoint(Entity1, Entity2, Entity3, Subnode3, Value)
    * Fixes angle between two curves at a specific point.

[Input/Output Format]
All inputs (except for the sketch image) and outputs are in JSON format. After the thinking session (i.e., after the </think> tag),
you MUST output the COMPLETE JSON array, including the given input partial constraints.
DO NOT include any comments, texts, or abbreviations (such as "//" or "...") in your JSON output.

Example inputs:
```
# Primitives Information
[
  {
    "type": "LineSegment",
    "start": [
      -200.0,
      250.0
    ],
    "end": [
      -200.0,
      -250.0
    ],
    "is_construction": false
  },
  {
    "type": "LineSegment",
    "start": [
      -200.0,
      -250.0
    ],
    "end": [
      200.0,
      -250.0
    ],
    "is_construction": false
  },
  {
    "type": "LineSegment",
    "start": [
      200.0,
      -250.0
    ],
    "end": [
      200.0,
      250.0
    ],
    "is_construction": false
  },
  {
    "type": "LineSegment",
    "start": [
      200.0,
      250.0
    ],
    "end": [
      -200.0,
      250.0
    ],
    "is_construction": false
  }
]

# Partial Constraints Information
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
]
```

Example output (Copy the given input partial constraints and add your constraints)
```
<think>
................
</think>
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
  {"type": "Vertical", "entity1": 0},
  {"type": "Vertical", "entity1": 2},
  {"type": "Perpendicular", "entity1": 1, "entity2": 2},
  {"type": "Tangent", "entity1": 0, "subnode1": 2, "entity2": 1, "subnode2": 1}
]
```

[Tool Use: CAD Solver]
This task is a multi-turn conversation, and you can refine your solution by using the CAD solver tool.
At every end of your turn, we will call the solver tool to verify your current constraints guess,
and return the following JSON format:
```
{
  'DoF': int,                         # Current Degrees of Freedom
  'Status': str,                      # "Converged", "Conflicting", "Redundant", "Failed", etc.
  'ConflictingConstraints': [int...], # List of indices causing conflicts (0-based)
  'RedundantConstraints': [int...],   # List of indices that are redundant (0-based)
  'ConflictGroups': [[int...], ...],  # Groups of constraints causing issues (0-based)
  'MalformedConstraints': [int...],   # Malformed constraints (0-based)
  'GeometryValidation': {             # Geometry impact check
      'MaxDeviation_mm': float,       # Maximum deviation of primitives (mm)
      'MaxDeviationPrimitive': int,   # 0-based index of primitive with max deviation (-1 if none)
      'Passed': bool,                 # True if deviation < Threshold_mm. False means the geometry was moved unacceptably yoo much.
      'Threshold_mm': float           # Threshold value used for judgement
  }
}
```

The desirable solver status is DoF=2, Status=Converged, no Conflicting/Redundant/Malformed constraints, and GeometryValidation.Passed=true.
- If the solver reports 'Status' as 'Failed', it means that the solver failed to converge, indicating that some constraints are geometrically unachievable. As the problem identification is usually very difficult, revert to a point where the solver reports Converged.
- If the solver reports 'RedundantConstraints', those constraints are deduced from other constraints. Try removing them (prioritize keeping the given partial constraints).
- If the solver reports 'ConflictingConstraints', there is a contradiction among constraints shown there. Analyze the logic and remove or modify the incorrect constraints.
- If the solver reports 'MalformedConstraints', those constraints are syntactically incorrect (e.g., missing parameters or referring to non-existent entities). Remove the constraints or modify their syntax.
- If the solver reports 'GeometryValidation.Passed' is False, your constraints are moving the geometry too much. You MUST remove or revise the problematic constraints (check 'MaxDeviationPrimitive' for the affected primitive index).
- The criticalness is Malformed > Conflicting > GeometryValidation > Redundant.
- The solver output indices are 0-based and correspond to the order in your <solver_call> list.

Along with the input primitives and partial constraints, we also provide the initial solver output for your reference.

IMPORTANT: If the output doesn't conform to the JSON format, we return an error message.
Immediately fix the error based on the message and re-submit your answer.

IMPORTANT: If you achieve the desirable solver status, or judge that it's impossible to improve your solution further,
encircle your solution JSON array with the <final> tag:
```
<final>
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
  {"type": "Vertical", "entity1": 0},
  {"type": "Vertical", "entity1": 2},
  {"type": "Perpendicular", "entity1": 1, "entity2": 2},
  {"type": "Tangent", "entity1": 0, "subnode1": 2, "entity2": 1, "subnode2": 1}
]
</final>
```

Example Tool Outputs (the input primitives and partial constraints are the same as the previous example):
```
# Initial Solver Status
{
  "DoF": 8,
  "Status": "Converged",
  "ConflictingConstraints": [],
  "RedundantConstraints": [],
  "ConflictGroups": [],
  "MalformedConstraints": [],
  "GeometryValidation": {
    "MaxDeviation_mm": 0.0,
    "MaxDeviationPrimitive": -1,
    "Passed": true,
    "Threshold_mm": 50.0
  }
}

# Example 1: Redundant Constraints (Vertical(0), Vertical(2), Perpendicular(1,2) implies Perpendicular(0,1)):
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
  {"type": "Vertical", "entity1": 0},
  {"type": "Vertical", "entity1": 2},
  {"type": "Perpendicular", "entity1": 1, "entity2": 2},
  {"type": "Perpendicular", "entity1": 0, "entity2": 1}
]
# Example 1 Solver Output (You should remove the constraint at index 7.)
{
  "DoF": 5,
  "Status": "Redundant",
  "ConflictingConstraints": [],
  "RedundantConstraints": [7],
  "ConflictGroups": [[4, 7, 5, 6]],
  "MalformedConstraints": [],
  "GeometryValidation": {
    "MaxDeviation_mm": 0.0,
    "MaxDeviationPrimitive": -1,
    "Passed": true,
    "Threshold_mm": 50.0
  }
}

# Example 2: Conflicting Constraints
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
  {"type": "Vertical", "entity1": 0},
  {"type": "Vertical", "entity1": 2},
  {"type": "Perpendicular", "entity1": 1, "entity2": 2},
  {"type": "Tangent", "entity1": 0, "subnode1": 2, "entity2": 1, "subnode2": 1}
]
# Example 2 Solver Output (You should remove some constraints in [4,5,6,7]. Since 7 involves many subgroups in ConflictGroups, you should remove 7 first.)
{
  "DoF": 5,
  "Status": "Conflicting",
  "ConflictingConstraints": [4, 5, 6, 7],
  "RedundantConstraints": [0],
  "ConflictGroups": [[0, 7], [5, 4, 7, 6], [0, 7]],
  "MalformedConstraints": [],
  "GeometryValidation": {
    "MaxDeviation_mm": 0.0,
    "MaxDeviationPrimitive": -1,
    "Passed": true,
    "Threshold_mm": 50.0
  }
}

# Example 3: Valid Constraints (NOTE: You can omit "Value" in Distance constraints)
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
  {"type": "Vertical", "entity1": 0},
  {"type": "Vertical", "entity1": 2},
  {"type": "Perpendicular", "entity1": 1, "entity2": 2},
  {"type": "Perpendicular", "entity1": 0, "entity2": 3},
  {"type": "Distance", "entity1": 2, "Subnode1": 1, "Entity2": 2, "Subnode2": 2},
  {"type": "Distance", "entity1": 1, "Subnode1": 1, "Entity2": 1, "Subnode2": 2}
]
# Example 3 Solver Output (Since it is verified as fully constrained, you must re-submit the answer constraints with <final></final> tag in the next turn.)
{
  "DoF": 2,
  "Status": "Converged",
  "ConflictingConstraints": [],
  "RedundantConstraints": [],
  "ConflictGroups": [],
  "MalformedConstraints": [],
  "GeometryValidation": {
    "MaxDeviation_mm": 0.0,
    "MaxDeviationPrimitive": -1,
    "Passed": true,
    "Threshold_mm": 50.0
  }
}
```

[Rules]
1. Identify geometric relationships based on the visual appearance in the image and the provided primitive IDs.
2. Achieve fully constrained (DoF=2) CAD sketches with constraints maintaining the design intent. As a rule of thumb,
    a. Fewer constraints are better.
    b. Prioritize geometric relationships over dimension constraints as much as possible.
    c. When you have to use dimension constraints, follow these rules:
        i. Avoid using angles to the maximum extent possible.
        ii. When defining Distance constraints, prioritize horizontal and vertical directions.
        iii. Chain dimensions horizontally or vertically. Chaining means defining length incrementally, for example, A---B and B---C, rather than A---B and A---C.
        iv. When a sketch has concave regions, define length in the positive sketch area: the line defining that length should be inside the valid sketch area.
    d. Always consider that the enforced constraints are intuitive, aligned with the design intent, and easy to understand/modify for post-editing.
3. You must submit your answer without the <final> tag several rounds to reflect the solver feedback.
4. If you cannot name all the constraints to achieve DoF=2, output the constraints you have identified so far to achieve the lowest DoF without any malformed or conflicting constraints.
5. Make sure to add the <final> tag to your final answer.
6. The final JSON output (after </think>) must be a valid JSON array starting with '[' and ending with ']'.
    a. NEVER use abbreviations like `[... Existing constraints ...]`. You MUST list ALL constraints explicitly, including the given input constraints.
    b. NEVER include comments (e.g., `// Original Constraints`). JSON does not support comments.
    c. If the solver returns an ERROR message (e.g., invalid JSON), you MUST immediately correct your output and re-submit it.
"""

    # 6. Construct User Message Content
    user_messages = []

    # Add all images (Examples + Target)
    for encoded_data in encoded_images:
        encoded_text = f"data:image/png;base64,{encoded_data}"
        user_messages.append({"type": "image_url", "image_url": {"url": encoded_text}})

    # Add final text prompt
    user_prompt_text = f"""
Complete the constraint list by adding missing constraints to achieve DoF=2.
At each turn, submit your answer in the specified JSON format, and integrate the feedback of the CAD solver to improve your constraints.
You can submit your answer up to {max_turns} times.
Your final output must:
1. Start with the exact partial constraints provided below (in the same order)
2. Append any additional constraints needed to fully constrain the sketch
Use the 2D sketch image as your visual reference.

# Primitives Information
{primitives_text}

# Partial Constraints Information
{constraints_text}

# Initial Solver Status
{initial_solver_output}
"""
    user_messages.append({"type": "text", "text": user_prompt_text})

    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_messages}
    ]

    return messages


def parse_solver_call_json(call_content: str) -> tuple[list, str | None]:
    """Parse JSON for CAD solver call.

    Checks for common invalid patterns before parsing:
    - Abbreviations like '...'
    - JS-style comments like '// comment'

    Args:
        call_content: Raw content extracted after </think> tags

    Returns:
        Tuple of (constraints_list, None) on success, or (None, error_msg) on failure
    """
    # Check for abbreviations like "..."
    if re.search(r'\.\.\.|…', call_content):
        return [], "You MUST output ALL constraints explicitly. Abbreviations '...' are NOT allowed."

    # Check for JS-style comments (lines starting with //)
    if re.search(r'^\s*//', call_content, re.MULTILINE):
        return [], "JSON does NOT support comments. Remove all '//' comment lines from your output."

    # Attempt JSON parse
    try:
        constraints = json.loads(call_content)
        if not isinstance(constraints, list):
            return [], f"Expected a JSON array, got {type(constraints).__name__}."
        return constraints, None
    except json.JSONDecodeError as e:
        error_detail = f"Invalid JSON syntax: {str(e)[:50]}. Check for trailing commas, missing quotes, or other syntax errors."
        return [], error_detail


def run_solver_tool(primitives_list: list, constraints_json_list: list, geo_threshold_mm: float = 50.0) -> str:
    """Run FreeCAD solver for the provided constraints and return status JSON string."""
    doc = None
    try:
        # Create temporary document
        doc = FreeCAD.newDocument("SolverToolDoc")
        body = doc.addObject('PartDesign::Body', 'Body')
        fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
        xy_plane = body.Origin.OriginFeatures[3]
        fc_sketch.AttachmentSupport = xy_plane
        fc_sketch.MapMode = 'FlatFace'

        # Add primitives and constraints
        add_primitives_to_sketch(primitives_list, fc_sketch)
        doc.recompute()

        # We need the mapping to convert 1-based FC indices back to 0-based input indices
        constraint_mapping, failed_indices = add_constraints_to_sketch(primitives_list, constraints_json_list, fc_sketch)
        doc.recompute()

        # Solve
        fc_sketch.solve()
        status = fc_sketch.getSolverStatus()

        # Helper to convert FC 1-based indices to 0-based input indices
        fc_to_orig = constraint_mapping

        def convert_indices(fc_indices):
            return [fc_to_orig.get(idx - 1, -1) for idx in fc_indices if (idx - 1) in fc_to_orig]

        def convert_groups(groups):
            return [convert_indices(group) for group in groups]

        # Combine malformed indices (both reported by solver and failed to add)
        malformed_from_solver = convert_indices(status.get('MalformedConstraints', []))
        all_malformed = list(set(malformed_from_solver + failed_indices))
        all_malformed.sort()

        # Construct response dict
        response = {
            'DoF': status['DoF'],
            'Status': status['Status'],
            'ConflictingConstraints': convert_indices(status.get('ConflictingConstraints', [])),
            'RedundantConstraints': convert_indices(status.get('RedundantConstraints', [])),
            'ConflictGroups': convert_groups(status.get('ConflictGroups', [])),
            'MalformedConstraints': all_malformed
        }

        # --- Geometry Validation ---
        max_dev, max_dev_primitive_idx, geo_passed, geo_error = validate_geometry_from_lists(
            primitives_list, constraints_json_list, geo_threshold_mm
        )

        response['GeometryValidation'] = {
            'MaxDeviation_mm': round(max_dev, 2),
            'MaxDeviationPrimitive': max_dev_primitive_idx,
            'Passed': geo_passed,
            'Threshold_mm': geo_threshold_mm
        }
        if geo_error:
            response['GeometryValidation']['Error'] = geo_error

    except Exception:
        response = {'Error': traceback.format_exc()}

    finally:
        if doc:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    return json.dumps(response, indent=2)


@dataclass
class VerificationResult:
    """Verification result from FreeCAD solver."""
    success: bool = False
    dof: int = -1
    status: str = "Unknown"
    malformed_count: int = 0
    conflicting_count: int = 0
    redundant_count: int = 0
    solver_converged: bool = True  # True if solver found a solution
    dof_decreased: bool = True  # True if Qwen3 DoF <= GT DoF (improved or same)
    geo_max_deviation: float = -1.0
    geo_passed: bool = False
    error: str | None = None

    def is_valid(self) -> bool:
        """Check if constraints are valid (no issues and geo passed)."""
        return (
            self.success and
            self.malformed_count == 0 and
            self.conflicting_count == 0 and
            self.redundant_count == 0 and
            self.solver_converged and
            self.dof_decreased and
            self.geo_passed
        )

    def summary(self) -> str:
        """Return a short summary string."""
        if self.error:
            return f"ERROR: {self.error[:50]}"
        issues = []
        if self.malformed_count > 0:
            issues.append(f"malformed={self.malformed_count}")
        if self.conflicting_count > 0:
            issues.append(f"conflicting={self.conflicting_count}")
        if self.redundant_count > 0:
            issues.append(f"redundant={self.redundant_count}")
        if not self.solver_converged:
            issues.append("solver_fail")
        if not self.geo_passed:
            issues.append(f"geo_fail(dev={self.geo_max_deviation:.1f}mm)")
        if not self.dof_decreased:
            issues.append("dof_increased")
        issue_str = ", ".join(issues) if issues else "OK"
        return f"DoF={self.dof}, {issue_str}"


def extract_geometry_coords(fc_sketch: "Sketcher.SketchObject") -> dict[int, list[tuple[float, float]]]:
    """Extract coordinates from all geometries in a FreeCAD sketch.

    Returns dict mapping geo_id -> list of (x, y) points.
    """
    coords = {}
    for geo_id, geo in enumerate(fc_sketch.Geometry):
        geo_type = type(geo).__name__
        points = []

        if geo_type == 'Point':
            points.append((geo.X, geo.Y))
        elif geo_type == 'LineSegment':
            sp = geo.StartPoint
            ep = geo.EndPoint
            points.append((sp.x, sp.y))
            points.append((ep.x, ep.y))
        elif geo_type in ('Circle', 'ArcOfCircle'):
            if hasattr(geo, 'StartPoint'):
                sp = geo.StartPoint
                points.append((sp.x, sp.y))
            if hasattr(geo, 'EndPoint'):
                ep = geo.EndPoint
                points.append((ep.x, ep.y))
            c = geo.Center
            points.append((c.x, c.y))

        coords[geo_id] = points

    return coords


def add_primitives_to_sketch(primitives: list, fc_sketch: "Sketcher.SketchObject") -> None:
    """Add primitives to FreeCAD sketch object.

    Args:
        primitives: List of primitive dicts
        fc_sketch: FreeCAD sketch object to add primitives to
    """
    for prim in primitives:
        ptype = prim.get("type")
        is_const = prim.get("is_construction", False)

        if ptype == "LineSegment":
            sx, sy = prim["start"]
            ex, ey = prim["end"]
            fc_sketch.addGeometry(
                Part.LineSegment(FreeCAD.Vector(sx, sy, 0), FreeCAD.Vector(ex, ey, 0)),
                is_const
            )
        elif ptype == "Point":
            x, y = prim["x"], prim["y"]
            fc_sketch.addGeometry(Part.Point(FreeCAD.Vector(x, y, 0)), is_const)
        elif ptype == "Circle":
            cx, cy = prim["center"]
            r = prim["radius"]
            fc_sketch.addGeometry(
                Part.Circle(FreeCAD.Vector(cx, cy, 0), FreeCAD.Vector(0, 0, 1), r),
                is_const
            )
        elif ptype == "Arc":
            # ArcOfCircle requires special handling - use 3 points
            sx, sy = prim["start"]
            mx, my = prim["mid"]
            ex, ey = prim["end"]
            try:
                arc = Part.ArcOfCircle(
                    FreeCAD.Vector(sx, sy, 0),
                    FreeCAD.Vector(mx, my, 0),
                    FreeCAD.Vector(ex, ey, 0)
                )
                fc_sketch.addGeometry(arc, is_const)
            except Exception:
                print(f"ERROR: Failed to add arc: {prim}")
                pass  # Skip invalid arcs


def add_constraints_to_sketch(
    primitives: list,
    constraints: list,
    fc_sketch: "Sketcher.SketchObject"
) -> tuple[dict[int, int], list[int]]:
    """Add constraints to FreeCAD sketch object.

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts from Qwen3 output
        fc_sketch: FreeCAD sketch object to add constraints to

    Returns:
        Tuple of:
          - Mapping from FreeCAD constraint index to original constraint index
          - List of indices of constraints that failed to add (malformed/error)
    """
    constraint_mapping = {}
    failed_indices = []

    for idx, c in enumerate(constraints):
        ctype = c.get("type", "")
        count_before = int(fc_sketch.ConstraintCount)

        try:
            if "args" in c:
                # GT format: use args directly based on constraint type
                args = c.get("args", [])

                if ctype == "Coincident":
                    # args: [e1, s1, e2, s2]
                    if len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', int(args[0]), int(args[1]), int(args[2]), int(args[3])))
                elif ctype == "Tangent":
                    # args: [e1, e2] or [e1, s1, e2, s2]
                    if len(args) == 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', int(args[0]), int(args[1])))
                    elif len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', int(args[0]), int(args[1]), int(args[2]), int(args[3])))
                elif ctype == "TangentViaPoint":
                    # args: [e1, e2, e3, s3]
                    if len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', int(args[0]), int(args[1]), int(args[2]), int(args[3])))
                elif ctype == "Concentric":
                    # args: [e1, e2] -> Coincident(e1, 3, e2, 3)
                    if len(args) >= 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', int(args[0]), 3, int(args[1]), 3))
                elif ctype in ["Horizontal", "Vertical"]:
                    # args: [e1] or [e1, s1, e2, s2]
                    if len(args) == 1:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(args[0])))
                    elif len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(args[0]), int(args[1]), int(args[2]), int(args[3])))
                elif ctype in ["Parallel", "Perpendicular", "Equal"]:
                    # args: [e1, e2]
                    if len(args) >= 2:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(args[0]), int(args[1])))
                elif ctype == "PointOnObject":
                    # args: [e1, s1, e2]
                    if len(args) >= 3:
                        fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', int(args[0]), int(args[1]), int(args[2])))
                elif ctype == "Symmetric":
                    # args: [e1, s1, e2, s2, e3, s3]
                    if len(args) >= 6:
                        fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', int(args[0]), int(args[1]), int(args[2]), int(args[3]), int(args[4]), int(args[5])))
                elif ctype == "Distance":
                    # args: [e1, value] or [e1, s1, e2, s2, value]
                    if len(args) == 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Distance', int(args[0]), float(args[1])))
                    elif len(args) == 5:
                        fc_sketch.addConstraint(Sketcher.Constraint('Distance', int(args[0]), int(args[1]), int(args[2]), int(args[3]), float(args[4])))
                elif ctype == "Angle":
                    # args: [e1, value] or [e1, e2, value]
                    if len(args) == 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Angle', int(args[0]), float(args[1])))
                    elif len(args) == 3:
                        fc_sketch.addConstraint(Sketcher.Constraint('Angle', int(args[0]), int(args[1]), float(args[2])))
                elif ctype in ["Radius", "Diameter"]:
                    # args: [e1, value]
                    if len(args) >= 2:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(args[0]), float(args[1])))
            else:
                # Qwen3 format: entity1/subnode1/entity2/subnode2/...
                e1 = c.get("entity1")
                s1 = c.get("subnode1")
                e2 = c.get("entity2")
                s2 = c.get("subnode2")
                e3 = c.get("entity3")
                s3 = c.get("subnode3")

                if ctype == "Coincident":
                    if e1 is not None and s1 is not None and e2 is not None and s2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', int(e1), int(s1), int(e2), int(s2)))
                elif ctype == "Tangent":
                    # Can be 4-arg (point-to-point) or 2-arg (edge-to-edge)
                    if s1 is not None and s2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', int(e1), int(s1), int(e2), int(s2)))
                    elif e1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', int(e1), int(e2)))
                elif ctype == "TangentViaPoint":
                    # TangentViaPoint(e1, e2, e3, s3)
                    if all(v is not None for v in [e1, e2, e3, s3]):
                        fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', int(e1), int(e2), int(e3), int(s3)))
                elif ctype == "Concentric":
                    # Concentric means center points coincide
                    if e1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', int(e1), 3, int(e2), 3))
                elif ctype in ["Horizontal", "Vertical"]:
                    if e2 is not None and s1 is not None and s2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(e1), int(s1), int(e2), int(s2)))
                    elif e1 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(e1)))
                elif ctype in ["Parallel", "Perpendicular", "Equal"]:
                    if e1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(e1), int(e2)))
                elif ctype == "PointOnObject":
                    if e1 is not None and s1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', int(e1), int(s1), int(e2)))
                elif ctype == "Symmetric":
                    if all(v is not None for v in [e1, s1, e2, s2, e3, s3]):
                        fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', int(e1), int(s1), int(e2), int(s2), int(e3), int(s3)))
                elif ctype == "Distance":
                    value = compute_constraint_value("Distance", primitives, e1, s1, e2, s2)
                    if value is not None:
                        if e2 is not None and s1 is not None and s2 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', int(e1), int(s1), int(e2), int(s2), float(value)))
                        elif e2 is not None and s1 is not None:
                            # Point-to-edge distance
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', int(e1), int(s1), int(e2), 0, float(value)))
                        elif e1 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', int(e1), float(value)))
                elif ctype == "Angle":
                    value = compute_constraint_value("Angle", primitives, e1, s1, e2, s2)
                    if value is not None:
                        if e2 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Angle', int(e1), int(e2), float(value)))
                        elif e1 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Angle', int(e1), float(value)))
                elif ctype in ["Radius", "Diameter"]:
                    value = compute_constraint_value(ctype, primitives, e1)
                    if value is not None and e1 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, int(e1), float(value)))
        except Exception:
            print(f"ERROR: FAILED to add constraint {idx}: {c}")
            failed_indices.append(idx)
            pass  # Skip constraints that fail to add


        count_after = int(fc_sketch.ConstraintCount)
        # If count increased, map the new indices
        if count_after > count_before:
            for i in range(count_before, count_after):
                constraint_mapping[i] = idx
        else:
            # If count didn't increase but it wasn't caught by exception (e.g. logic error), treat as failed if not already added
            # Note: Some constraints might merge or be ignored silently, but usually logic errors throw.
            # Here we assume if exception didn't fire, it succeeded or did nothing harmful.
            # But if we strictly want 1:1, we might check. For now, exception catching is the main one.
            pass

    return constraint_mapping, failed_indices



def validate_geometry_from_lists(
    primitives: list,
    constraints: list,
    threshold_mm: float = 50.0,
) -> tuple[float, int, bool, str]:
    """
    Validate geometry impact of constraints by comparing before/after.

    This function:
    1. Add primitives only to FreeCAD → capture coordinates
    2. Add constraints → capture coordinates
    3. Compare the two to see how much constraints move geometry

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts from Qwen3 output
        threshold_mm: Maximum allowed deviation in mm

    Returns:
        Tuple of (max_deviation, max_deviation_primitive_idx, all_passed, error_message)
    """
    doc = None
    max_deviation = 0.0
    max_deviation_primitive_idx = -1
    all_passed = True
    error_msg = ""

    try:
        # Create FreeCAD document
        doc = FreeCAD.newDocument("GeoValidationDoc")
        body = doc.addObject('PartDesign::Body', 'Body')
        fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
        xy_plane = body.Origin.OriginFeatures[3]
        fc_sketch.AttachmentSupport = xy_plane
        fc_sketch.MapMode = 'FlatFace'

        # Step 1: Add primitives only
        add_primitives_to_sketch(primitives, fc_sketch)
        doc.recompute()

        # Capture geometry BEFORE constraints
        coords_before = extract_geometry_coords(fc_sketch)
        geo_types = {i: type(geo).__name__ for i, geo in enumerate(fc_sketch.Geometry)}

        # Step 2: Add constraints
        add_constraints_to_sketch(primitives, constraints, fc_sketch)
        doc.recompute()

        # Capture geometry AFTER constraints
        coords_after = extract_geometry_coords(fc_sketch)

        # Step 3: Compare before and after
        for geo_id in coords_before:
            before = coords_before[geo_id]
            after = coords_after.get(geo_id, [])

            if len(before) != len(after):
                continue

            for (bx, by), (ax, ay) in zip(before, after):
                deviation = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_deviation_primitive_idx = geo_id

        all_passed = max_deviation < threshold_mm

    except Exception:
        error_msg = "ERROR: " + traceback.format_exc()
        all_passed = False

    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    return max_deviation, max_deviation_primitive_idx, all_passed, error_msg


def verify_constraints(
    primitives: list,
    constraints: list,
    geo_threshold_mm: float = 50.0,
) -> VerificationResult:
    """
    Verify constraints using FreeCAD solver and geometry validation.

    This function performs two checks:
    1. Solver status check: malformed/conflicting/redundant constraints
    2. Geometry preservation check: primitives don't move too much after constraints

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts from Qwen3 output
        geo_threshold_mm: Maximum allowed geometry deviation in mm

    Returns:
        VerificationResult with detailed status
    """
    result = VerificationResult()
    doc = None

    try:
        # Step 1: Create FreeCAD document and add primitives + constraints
        doc = FreeCAD.newDocument("VerificationDoc")
        body = doc.addObject('PartDesign::Body', 'Body')
        fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
        xy_plane = body.Origin.OriginFeatures[3]
        fc_sketch.AttachmentSupport = xy_plane
        fc_sketch.MapMode = 'FlatFace'

        # Add primitives
        add_primitives_to_sketch(primitives, fc_sketch)
        doc.recompute()

        # Add constraints
        _mapping, failed_indices = add_constraints_to_sketch(primitives, constraints, fc_sketch)
        doc.recompute()

        # Step 2: Get solver status
        fc_sketch.solve()
        status = fc_sketch.getSolverStatus()

        result.success = True
        result.dof = status['DoF']
        result.status = status['Status']
        # Malformed = Reported by solver + Failed to add
        result.malformed_count = len(status.get('MalformedConstraints', [])) + len(failed_indices)
        result.conflicting_count = len(status.get('ConflictingConstraints', []))
        result.redundant_count = len(status.get('RedundantConstraints', []))
        # Check if solver converged (Status = 'Converged' means OK)
        result.solver_converged = (status.get('Status', '') == 'Converged')

    except Exception:
        result.error = "ERROR: " + traceback.format_exc()
        result.success = False
        result.solver_converged = False
        result.geo_passed = False
        result.geo_max_deviation = -1.0

    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    # Step 3: Geometry validation (only if solver check passed)
    if result.success and result.malformed_count == 0 and result.conflicting_count == 0:
        max_dev, _max_dev_prim_idx, geo_passed, geo_error = validate_geometry_from_lists(
            primitives, constraints, geo_threshold_mm
        )
        result.geo_max_deviation = max_dev
        result.geo_passed = geo_passed
        if geo_error:
            result.error = geo_error
    else:
        result.geo_passed = False
        result.geo_max_deviation = -1.0

    return result


def cleanup_constraints_from_lists(
    primitives: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    max_cleanup_iters: int = 10,
    prefix_size: int = 0,
) -> tuple[list[dict[str, Any]], VerificationResult]:
    """
    Remove malformed/conflicting/redundant constraints iteratively.

    This function uses FreeCAD solver to detect problematic constraints
    and removes them one by one until no issues remain or max iterations reached.

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts
        max_cleanup_iters: Maximum cleanup iterations
        prefix_size: Number of constraints at the start that should NOT be removed (GT prefix)

    Returns:
        Tuple of (cleaned_constraints, final_verification_result)
    """
    current_constraints = constraints.copy()
    final_result = VerificationResult()

    for iter_idx in range(max_cleanup_iters):
        doc = None
        try:
            # Create FreeCAD document
            doc = FreeCAD.newDocument(f"CleanupDoc{iter_idx}")
            body = doc.addObject('PartDesign::Body', 'Body')
            fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
            xy_plane = body.Origin.OriginFeatures[3]
            fc_sketch.AttachmentSupport = xy_plane
            fc_sketch.MapMode = 'FlatFace'

            # Add primitives
            add_primitives_to_sketch(primitives, fc_sketch)
            doc.recompute()

            # Add constraints with mapping
            constraint_mapping, failed_indices = add_constraints_to_sketch(primitives, current_constraints, fc_sketch)
            doc.recompute()

            # Get solver status
            fc_sketch.solve()
            status = fc_sketch.getSolverStatus()

            # Build reverse mapping: original constraint index -> FC constraint indices
            fc_to_orig: dict[int, int] = constraint_mapping  # FC index -> original index

            # Collect problem indices (original constraint indices)
            malformed_fc = status.get('MalformedConstraints', [])
            conflicting_fc = status.get('ConflictingConstraints', [])
            redundant_fc = status.get('RedundantConstraints', [])

            # Map FC indices back to original constraint indices
            # FC getSolverStatus returns 1-based indices

            # Malformed: remove all (these are syntax errors)
            malformed_orig_indices = set()
            for idx in failed_indices:
                if idx >= prefix_size:
                    malformed_orig_indices.add(idx)

            for fc_idx in malformed_fc:
                orig_idx = fc_to_orig.get(fc_idx - 1)  # Convert 1-based to 0-based
                if orig_idx is not None and orig_idx >= prefix_size:
                    malformed_orig_indices.add(orig_idx)

            # Conflicting: collect all, but will remove only last one
            conflicting_orig_indices = set()
            for fc_idx in conflicting_fc:
                orig_idx = fc_to_orig.get(fc_idx - 1)
                if orig_idx is not None and orig_idx >= prefix_size:
                    conflicting_orig_indices.add(orig_idx)

            # Redundant: collect all, but will remove only last one
            redundant_orig_indices = set()
            for fc_idx in redundant_fc:
                orig_idx = fc_to_orig.get(fc_idx - 1)
                if orig_idx is not None and orig_idx >= prefix_size:
                    redundant_orig_indices.add(orig_idx)

            # Build final result
            final_result = VerificationResult(
                success=True,
                dof=status['DoF'],
                status=status['Status'],
                malformed_count=len(malformed_fc) + len(failed_indices),
                conflicting_count=len(conflicting_fc),
                redundant_count=len(redundant_fc),
            )

            FreeCAD.closeDocument(doc.Name)
            doc = None

            # Prioritize: malformed > conflicting > redundant
            # Malformed: remove all at once (syntax errors)
            # Conflicting/Redundant: remove only highest index (one at a time)
            if malformed_orig_indices:
                indices_to_remove = malformed_orig_indices
            elif conflicting_orig_indices:
                # Remove only the highest index conflicting constraint
                indices_to_remove = {max(conflicting_orig_indices)}
            elif redundant_orig_indices:
                # Remove only the highest index redundant constraint
                indices_to_remove = {max(redundant_orig_indices)}
            else:
                # No problems, done
                break

            # Remove problematic constraints
            current_constraints = [
                c for i, c in enumerate(current_constraints)
                if i not in indices_to_remove
            ]

        except Exception:
            if final_result is None:
                final_result = VerificationResult(error="ERROR: " + traceback.format_exc())
            break

        finally:
            if doc is not None:
                try:
                    FreeCAD.closeDocument(doc.Name)
                except:
                    pass

    return current_constraints, final_result


def comment_out_file(file_path: str) -> None:
    """Comment out all lines in a Python file.

    For lines that are already comments, add another # to make # # [Comment].
    """
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    commented_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty line - keep as is
            commented_lines.append(line)
        else:
            # Add # prefix to all non-empty lines (including existing comments)
            commented_lines.append('# ' + line)

    with open(file_path, 'w') as f:
        f.write('\n'.join(commented_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save FC GT and predictions that pass all validation criteria")
    parser.add_argument('--geo_threshold', type=float, default=50.0, help="Geometry validation threshold in mm")
    parser.add_argument('--output_dir', type=str, default='qwen3_refined_data_with_solver', help="Output directory")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of sequences to process")
    parser.add_argument('--end_idx', type=int, default=1000, help="End index of sequences to process")
    parser.add_argument('--max_turns', type=int, default=10, help="Maximum solver calls before accepting answer")
    args = parser.parse_args()

    # --- Model Configuration ---
    # MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-235B-A22B-Thinking-Q4_K_M-split-00001-of-00003.gguf"
    # MMPROJ_PATH = "/home/ryotaro/github/llama.cpp/models/mmproj-Qwen3VL-235B-A22B-Thinking-F16.gguf"

    MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf"
    MMPROJ_PATH = "/home/ryotaro/github/llama.cpp/models/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf"

    # Initialize LLM
    llm = Llama(
        model_path=MODEL_PATH,
        chat_handler=Qwen3VLChatHandler(
            clip_model_path=MMPROJ_PATH,
            force_reasoning=True,
            image_min_tokens=1024,
            verbose=False,
        ),
        n_gpu_layers=-1,
        n_ctx=65536,
        swa_full=True,
        verbose=False,
    )
    llm.reset()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/images", exist_ok=True)
    os.makedirs(args.output_dir + "/commands", exist_ok=True)

    error_log = os.path.join(args.output_dir, "error_log.txt")

    # load data
    sequence_path = "/home/ryotaro/my_works/My-Vitruvion/data_processed/sg_filtered_unique.npy"
    seq_data = flat_array.load_dictionary_flat(sequence_path)
    sequences = seq_data['sequences']

    for i, seq in enumerate(sequences[args.start_idx : args.end_idx]):
        i += args.start_idx

        # output file paths
        output_history_original = f"{args.output_dir}/commands/{i:07d}_original.py"
        output_history_refined = f"{args.output_dir}/commands/{i:07d}_refined.py"
        output_image_path = f"{args.output_dir}/images/{i:07d}_original.png"

        # Skip if refined file already exists (resume support)
        if os.path.isfile(output_history_refined):
            print(f"=== Skipping sample {i} (refined already exists) ===")
            continue
        print(f"=== Processing sample {i} ===")

        # load and convert original data to commands
        sketch = datalib.sketch_from_sequence(seq)
        doc = None
        try:
            doc, fc_sketch, history_commands, entity_to_geoid, mapping = convert_sketch_to_freecad(sketch)
            sample_primitives, sample_constraints = parse_freecad_script(history_commands)
        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            with open(error_log, "a") as f:
                f.write(f"Error processing sequence {i}: {e}\n")
            continue
        finally:
            if doc is not None:
                FreeCAD.closeDocument(doc.Name)

        # ======== GT PROCESSING ========

        # GT Verification (with cleanup)
        print("\n--- GT Verification (with cleanup) ---")
        sample_constraints_cleaned, sample_cleanup_result = cleanup_constraints_from_lists(
            sample_primitives, sample_constraints
        )
        if sample_cleanup_result.error:
            print(f"  ✗ GT CLEANUP ERROR: {sample_cleanup_result.error.splitlines()[-1]}")
            with open(error_log, "a") as f:
                f.write(f"GT Cleanup Error for sample {i}:\n{sample_cleanup_result.error}\n")
            continue

        print(f"  Cleanup: {len(sample_constraints)} -> {len(sample_constraints_cleaned)} constraints")

        # Convert sample_constraints_cleaned to Qwen3 JSON format for both export and Qwen3 input
        sample_constraints_cleaned_json = []
        for c in sample_constraints_cleaned:
            if "args" in c:
                sample_constraints_cleaned_json.append(format_constraint_to_json(c["type"], c["args"]))
            else:
                sample_constraints_cleaned_json.append(c)

        # Export GT commands with cleaned constraints
        export_to_freecad_script(sample_primitives, sample_constraints_cleaned_json, output_history_original)

        # Export GT sketch image with cleaned constraints
        sketch_data = load_and_extract_from_file(output_history_original)
        visualize_sketch_for_vlm(sketch_data, save_path=output_image_path, title=None)

        # Sanity check
        sample_verification = verify_constraints(sample_primitives, sample_constraints_cleaned, args.geo_threshold)
        sample_summary = sample_verification.summary()
        if not sample_verification.is_valid():
            print("  ✗ GT INVALID: Issues detected - commenting out GT file")
            os.remove(output_history_original)
            os.remove(output_image_path)
            continue

        # ======== VLM PROCESSING ========

        print("\n--- Qwen3 Generation (Interactive) ---")
        messages = prepare_prompts(sample_primitives, sample_constraints_cleaned, output_image_path, args.max_turns)
        predicted_constraints_list = []

        for solver_call_count in range(args.max_turns):

            # Notify remaining turns
            remaining_times_notification = f"Turn {solver_call_count + 1}: You can attempt submission up to {args.max_turns - solver_call_count} more times."
            messages.append({"role": "user", "content": remaining_times_notification})
            print("User: " + remaining_times_notification + "\n")

            # Run VLM
            try:
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=32768,
                    temperature=0.7,
                    stream=True
                )

                current_turn_text = ""
                for chunk in stream:
                    delta = chunk['choices'][0]['delta']
                    content = delta.get('content', '')
                    current_turn_text += content
                    if content:
                        sys.stdout.write(content)
                        sys.stdout.flush()

            except Exception as e:
                print(f"Error in VLM inference: {e}")
                with open(error_log, "a") as f:
                    f.write(f"Error in VLM inference: {e}\n")
                break

            messages.append({"role": "assistant", "content": current_turn_text})

            # 1. Check for <final> tag
            final_match = re.search(r"<final>\n(.*?)\n</final>", current_turn_text, re.DOTALL)
            if final_match:
                predicted_constraints, parse_error = parse_solver_call_json(final_match.group(1))
                predicted_constraints_list.append(predicted_constraints)
                if parse_error:
                    solver_output = json.dumps({"DoF": "null", "Error": f"JSON Parse Error: {parse_error}. Fix the syntax and try again."}, indent=2)
                    messages.append({"role": "user", "content": solver_output})
                    print("User: " + solver_output + "\n")
                    continue
                else:
                    break

            # 2. Check for </think> and extract JSON
            parts = current_turn_text.split("</think>")
            content_to_parse = parts[-1] if len(parts) > 1 else current_turn_text
            json_match = re.search(r'\[[\s\S]*?\]', content_to_parse)
            if json_match:
                predicted_constraints, parse_error = parse_solver_call_json(json_match.group(0))
                predicted_constraints_list.append(predicted_constraints)

                if parse_error:
                    solver_output = json.dumps({"DoF": "null", "Error": parse_error}, indent=2)
                else:
                    # Run solver with predicted constraints
                    solver_output = run_solver_tool(sample_primitives, predicted_constraints)
            else:
                solver_output = json.dumps({"DoF": "null", "Error": "You MUST output the JSON array of constraints."}, indent=2)

            # Check for solver error (Top-level or GeometryValidation) and log/truncate
            # Note: We expect run_solver_tool to ALWAYS return valid JSON. If it fails (JSONDecodeError),
            # it indicates an implementation bug in run_solver_tool, so we let it raise.
            solver_resp = json.loads(solver_output)

            # Case 1: Top-level Error (e.g. Exception)
            if "Error" in solver_resp:
                with open(error_log, "a") as f:
                    f.write(f"Solver Error (Top-level) for sample {i} turn {solver_call_count}:\n{solver_resp['Error']}\n")
                solver_resp['Error'] = solver_resp['Error'][:200]

            # Case 2: Geometry Validation Error
            if "GeometryValidation" in solver_resp and "Error" in solver_resp["GeometryValidation"]:
                with open(error_log, "a") as f:
                    f.write(f"Solver Error (Geometry) for sample {i} turn {solver_call_count}:\n{solver_resp['GeometryValidation']['Error']}\n")
                solver_resp["GeometryValidation"]["Error"] = solver_resp["GeometryValidation"]["Error"][:200]

            solver_output = json.dumps(solver_resp, indent=2)

            messages.append({"role": "user", "content": solver_output})
            print("User: " + solver_output + "\n")

        # Select the best solution so far
        if len(predicted_constraints_list) == 0:
            print("ERROR: No predicted constraints found.")
            with open(error_log, "a") as f:
                f.write(f"Error: No predicted constraints found for sequence {i}\n")
            continue

        valid_prediction_found = False

        for pred_validation_version_id, predicted_constraints in enumerate(predicted_constraints_list[::-1]):

            print(f"\n~~~~~~~~ Qwen3 Output Version {pred_validation_version_id + 1} ~~~~~~~~")

            # Cleanup the final result
            print("\n--- Qwen3 Output Cleanup ---")
            predicted_constraints_cleaned, predicted_constraints_cleanup_result = cleanup_constraints_from_lists(
                sample_primitives, predicted_constraints
            )
            if predicted_constraints_cleanup_result.error:
                 with open(error_log, "a") as f:
                    f.write(f"Prediction Cleanup Error for sample {i} version {pred_validation_version_id}:\n{predicted_constraints_cleanup_result.error}\n")

            print(f"  Cleanup: {len(predicted_constraints)} -> {len(predicted_constraints_cleaned)} constraints")

            # Prediction verification
            print("\n--- Qwen3 Verification ---")
            predicted_verification = verify_constraints(sample_primitives, predicted_constraints_cleaned, args.geo_threshold)

            # Check if Qwen3 made DoF worse (more DoF = less constrained = worse)
            if predicted_verification.dof > sample_verification.dof:
                predicted_verification.dof_decreased = False
                print(f"  ⚠ DOF INCREASED: VLM DoF ({predicted_verification.dof}) > Original DoF ({sample_verification.dof})")

            predicted_summary = predicted_verification.summary()
            print(f"  Result: {predicted_summary}")
            if predicted_verification.geo_max_deviation >= 0:
                print(f"  Geometry deviation: {predicted_verification.geo_max_deviation:.2f}mm")

            if predicted_verification.is_valid():
                print("  ✓ VALID: Constraints are well-formed and geometry preserved")
                valid_prediction_found = True
                # Export cleaned constraints
                export_to_freecad_script(sample_primitives, predicted_constraints_cleaned, output_history_refined)
                break
            else:
                print("  ✗ INVALID: Issues detected - commenting out output file")
                if predicted_verification.error:
                    print(f"  Error: {predicted_verification.error}")

        if not valid_prediction_found:
            # Take the last prediction, and comment out the output file
            predicted_constraints = predicted_constraints_list[-1]
            predicted_constraints_cleaned, predicted_constraints_cleanup_result = cleanup_constraints_from_lists(sample_primitives, predicted_constraints)
            export_to_freecad_script(sample_primitives, predicted_constraints_cleaned, output_history_refined)
            comment_out_file(output_history_refined)
            predicted_verification = verify_constraints(sample_primitives, predicted_constraints_cleaned, args.geo_threshold)
            predicted_verification.dof_decreased = predicted_verification.dof <= sample_verification.dof
            predicted_summary = predicted_verification.summary()

        # Log verification result (GT vs Qwen3 comparison)
        with open(os.path.join(args.output_dir, "verification_log.txt"), "a") as f:
            valid_str = "PASS" if predicted_verification.is_valid() else "FAIL"
            f.write(f"{i:07d}: [GT] {sample_summary} | [Qwen3] {predicted_summary} | solver_calls={solver_call_count} [{valid_str}]\n")
