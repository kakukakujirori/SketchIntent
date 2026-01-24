"""
Treat solver_output as user prompts
"""
from absl.logging import flush
import argparse
import base64
import glob
import json
import math
import os
import re
import sys
from io import BytesIO

import cairosvg
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



# --- Model Configuration ---
# # Define the path to the main model file
# MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-235B-A22B-Thinking-Q4_K_M-split-00001-of-00003.gguf"
# # Define the path to the multi-modal projector file
# MMPROJ_PATH = "/home/ryotaro/github/llama.cpp/models/mmproj-Qwen3VL-235B-A22B-Thinking-F16.gguf"

MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf"
MMPROJ_PATH = "/home/ryotaro/github/llama.cpp/models/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf"



# --- Initialize the Llama Model ---
# Model initialization is deferred to main execution block
llm = None



def parse_vector(vec_str):
    # Extracts numbers from "Vector(x, y, z)"
    # Pattern supports scientific notation like 8.326e-14, 1.5E+10
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", vec_str)
    # Return x, y as float
    return [float(n) for n in nums[:2]]


def parse_freecad_script(filepath):
    primitives = []
    constraints_gt = []

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return [], []

    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
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


def export_to_freecad_script(primitives: list, constraints: list, output_path: str):
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


def prepare_prompts(primitives_list: list, constraints_list: list, image_path_list: str | list[str]):

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

    system_prompt_text = """
You are an expert CAD engineer.
Analyze the provided 2D sketch image, the list of geometric primitives, and the list of partial geometric constraints.

Your task is to **add missing constraints** to the provided partial constraint list until the sketch is fully constrained.
The output must include ALL constraints: both the given partial constraints AND the new constraints you infer.
More specifically, the sketch is fully constrained if its degree-of-freedom is 2, i.e., leaving only the global xy translation.
It is also important that the constraints are not malformed, conflicting, or redundant.

The grammar follows that of FreeCAD, i.e., the primitives are assigned reference numbers in the defined order,
and they are used to specify the position on which the constraints are applied.

[Input Format]
The input consists of a list of geometric primitives and a list of partial geometric constraints.
The supported primitives and their arguments are:
- Point(x, y, is_construction)
- LineSegment((start_x, start_y), (end_x, end_y), is_construction)
- Circle((center_x, center_y), is_clockwise, radius, is_construction)
- Arc((start_x, start_y), (mid_x, mid_y), (end_x, end_y), is_construction)
* Note: mid_x/y for Arc is a point *on the arc* (roughly halfway), not the center or the mid-coord point.
The primitives are implicitly indexed starting from 0 (0-indexed) based on their order in the input list.
The partial constraints format is described in the following section.

[Output Format]
Output the **complete** list of constraints, which consists of:
1. The given partial constraints (copy them exactly as provided)
2. Additional constraints you infer from the visual arrangement and geometric properties
Each constraint should follow the FreeCAD internal constraint syntax, tailored to the specific constraint type.

General Argument Structure:
Type, Entity1, Subnode1, Entity2, Subnode2, Entity3, Subnode3, Value

- EntityN: The 0-based index of the primitive.
- SubnodeN: Specifies which part of the primitive is being referenced.
    - 0: The Entity itself (e.g., the edge of a line or circle)
    - 1: StartPoint (Vertex)
    - 2: EndPoint (Vertex)
    - 3: CenterPoint (Strictly the center of a Circle or Arc, NOT the midpoint of a line or an arc)


[Constraint Types and Signatures]
Please strictly adhere to the following signatures for each constraint type. Arguments not listed for a type should be omitted.
Note: We omit the inference of the parameter values for dimensional constraints (e.g., xxx meters or xxx degrees).

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

2. Dimensional Constraints (Use only if geometry implies specific dimensions):
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

[Tool Use: Interactive Solver]
You can verify your current constraints guess using the FreeCAD solver tool during your thought process.
To call the solver, you MUST literally output the opening tag <solver_call>, followed by a JSON array of constraints, followed by the closing tag </solver_call>.
The system will detect these tags and execute the solver, then return the result in <solver_output> tags.

IMPORTANT: The JSON inside <solver_call> must be a COMPLETE, VALID JSON array.
Do NOT use abbreviations like "..." or "[... existing constraints]". Every constraint must be written out fully.

Example Tool Call:
```
<solver_call>
[
  {
    "type":"Coincident",
    "entity1":0,
    "subnode1":2,
    "entity2":1,
    "subnode2":1
  },
  {
    "type":"Horizontal",
    "entity1":0
  }
]
</solver_call>
```

[Tool Output Format]
The tool will return a JSON object with the solver status:
```
<solver_output>
{
  'DoF': int,                         # Current Degrees of Freedom
  'Status': str,                      # "Converged", "Conflicting", "Redundant", etc.
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
</solver_output>
```

[Solver Feedback Interpretation]
- DoF: Current system's degree of freedom.
- ConflictGroups: Lists of constraint groups that contradict each other or form redundancy.
- RedundantConstraints: Constraints that are automatically deduced from existing constraints. You MUST remove them.
- ConflictingConstraints: Constraints that mathematically contradict others. You MUST revise your logic to resolve them.
- MalformedConstraints: Constraints that are syntactically incorrect, e.g., referring to non-existing primitives, or too many/few primitives are specified in a particular constraint. You MUST review the constraint command.
- GeometryValidation: Checks if applying constraints moves the geometry too much (bad). If Passed=False, it means the constraints are distorting the sketch significantly (e.g., flipping lines, moving points far away). You should avoid such constraints.

[Behavioral Guidelines]
1. You MUST use the <solver_call> tool to verify your constraints with the solver BEFORE outputting your final answer.
2. Continue calling the solver iteratively until you achieve DoF=2 or cannot reduce further.
3. If the solver reports 'RedundantConstraints', try removing them (prioritize keeping the given partial constraints).
4. If the solver reports 'ConflictingConstraints', analyze the logic and remove or modify the incorrect constraints.
5. If the solver reports 'MalformedConstraints', remove the constraints or modify their syntax.
6. Keep the position of the primitives almost the same by applying the constraints; they are assumed to be correctly configured from the input.
7. The solver output indices are 0-based and correspond to the order in your <solver_call> list.
8. If 'GeometryValidation.Passed' is False, your constraints are moving the geometry too much. You MUST remove or revise the problematic constraints (check 'MaxDeviationPrimitive' for the affected primitive index).

**Desirable solver status**: DoF=2, Status=Converged, no Conflicting/Redundant/Malformed constraints, and GeometryValidation.Passed=true.

[Tool Use Interaction Examples]
Here are examples of how the solver responds to different constraint sets.
In all the following examples, the input primitives and partial constraints are as follows:

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
# Initial Solver Status
```
<solver_output>
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
</solver_output>
```

Example 1: Redundant Constraints
If you submit a set where one constraint is implied by others (e.g., Vertical(0), Vertical(2), Perpendicular(1,2) implies Perpendicular(0,1)):
```
<solver_call>
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
</solver_call>
<solver_output>
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
</solver_output>
```
(Action: You should remove the constraint at index 7)

Example 2: Conflicting Constraints
If you submit a set with contradictions (e.g., expecting a sharp corner to be Tangent):
```
<solver_call>
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
</solver_call>
<solver_output>
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
</solver_output>
```
(Action: You must remove or fix the conflicting constraints; in this case, remove 8)

Example 3: Valid Constraints
When the set is consistent:
```
<solver_call>
[
  {"type": "Coincident", "entity1": 1, "subnode1": 1, "entity2": 0, "subnode2": 2},
  {"type": "Coincident", "entity1": 2, "subnode1": 1, "entity2": 1, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 1, "entity2": 2, "subnode2": 2},
  {"type": "Coincident", "entity1": 3, "subnode1": 2, "entity2": 0, "subnode2": 1},
  {"type": "Vertical", "entity1": 0},
  {"type": "Vertical", "entity1": 2},
  {"type": "Perpendicular", "entity1": 1, "entity2": 2}
]
</solver_call>
<solver_output>
{
  "DoF": 5,
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
</solver_output>
```

[Rules]
1. Identify relationships strictly based on the visual appearance in the image and the provided IDs.
2. Prioritize geometric relationships (Coincident, Horizontal, Vertical, Tangent, Parallel, Perpendicular, etc.).
3. Only use dimensional constraints (Length, Radius, etc.) if the geometry cannot be fixed by relationships alone. Also, the position to add dimension constraints must reflect design intent of the CAD sketch.
4. Use the exact IDs provided in the Primitives Information.
5. If you cannot name all the constraints to achieve DoF=2, output the constraints you have identified so far to achieve the lowest DoF.
6. **CRITICAL**: You MUST call <solver_call> AT LEAST ONCE before outputting your final answer. Answers without solver verification are INVALID and will be rejected.
7. **NEVER** use abbreviations like `[... Existing constraints ...]` in your JSON output. You MUST list ALL constraints explicitly.
8. **NEVER** include comments (e.g., `// Original Constraints`) in your JSON output. JSON does not support comments.
9. If the solver returns an ERROR message (e.g., invalid JSON), you MUST immediately correct your output and call <solver_call> again.

[Process and Output Instructions]
1. **Thinking Process**: You MUST start with a <think> block. Analyze the geometry and iteratively use the <solver_call> tool to verify your constraints until reaching DoF=2 or being unable to be reduced further.
2. **Final Output**: Once you have determined the correct set of constraints, close the </think> block and output the final result.

[Final Answer Format]
The final answer (outside the thinking block) MUST be a valid JSON array, including all the given partial constraints and your inferred additional constraints.
Example output for two constraints:
```
[
  {
    "type":"Coincident",
    "entity1":0,
    "subnode1":2,
    "entity2":1,
    "subnode2":1
  },
  {
    "type":"Horizontal",
    "entity1":0
  }
]
```
Field descriptions:
- "type": String, the constraint type (e.g., "Coincident", "Horizontal", "Parallel", "Tangent", etc.)
- "entity1", "entity2", "entity3": Integer (0-based primitive index). Omit if not applicable.
- "subnode1", "subnode2", "subnode3": Integer (0=Entity, 1=Start, 2=End, 3=Center). Omit if not applicable.

The final JSON output (after </think>) must be a valid JSON array starting with '[' and ending with ']'."""

    # 6. Construct User Message Content
    user_messages = []

    # Add all images (Examples + Target)
    for encoded_data in encoded_images:
        encoded_text = f"data:image/png;base64,{encoded_data}"
        user_messages.append({"type": "image_url", "image_url": {"url": encoded_text}})

    # Add final text prompt
    user_prompt_text = f"""
Complete the constraint list by adding missing constraints to achieve DoF=2.
You must use solver_call tool while thinking to verify your constraints.
Your final output must:
1. Start with the exact partial constraints provided below (in the same order)
2. Append any additional constraints needed to fully constrain the sketch
Use the 2D sketch image as your visual reference.

[Primitives Information]
```
{primitives_text}
```

[Partial Constraints Information]
```
{constraints_text}
```

[Initial Solver Status]
{initial_solver_output}
"""
    user_messages.append({"type": "text", "text": user_prompt_text})

    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_messages}
    ]

    return messages


def extract_json_from_thinking_output(text: str) -> str:
        """Thinkingモデルの出力から思考部分を除去してJSON配列を抽出する"""
        # </think> タグの後の部分を取得
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        # JSON配列を正規表現で抽出（[...]の形式）
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            return json_match.group(0)

        return text


def parse_solver_call_json(call_content: str) -> tuple[list, str | None]:
    """Parse JSON from <solver_call> content with validation.

    Checks for common invalid patterns before parsing:
    - Abbreviations like '...'
    - JS-style comments like '// comment'

    Args:
        call_content: Raw content extracted from <solver_call> tags

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
        constraint_mapping = add_constraints_to_sketch(primitives_list, constraints_json_list, fc_sketch)
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

        # Construct response dict
        response = {
            'DoF': status['DoF'],
            'Status': status['Status'],
            'ConflictingConstraints': convert_indices(status.get('ConflictingConstraints', [])),
            'RedundantConstraints': convert_indices(status.get('RedundantConstraints', [])),
            'ConflictGroups': convert_groups(status.get('ConflictGroups', [])),
            'MalformedConstraints': convert_indices(status.get('MalformedConstraints', []))
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

    except Exception as e:
        response = {'Error': str(e)}

    finally:
        if doc:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    return f"<solver_output>\n{json.dumps(response, indent=2)}\n</solver_output>"


from dataclasses import dataclass, field


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
    prefix_matched: bool = True  # True if Qwen3 output contains input constraints as prefix
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
            self.prefix_matched and
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
        if not self.prefix_matched:
            issues.append("prefix_mismatch")
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
) -> dict[int, int]:
    """Add constraints to FreeCAD sketch object.

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts from Qwen3 output
        fc_sketch: FreeCAD sketch object to add constraints to

    Returns:
        Mapping from FreeCAD constraint index to original constraint index
    """
    constraint_mapping = {}

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
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', args[0], args[1], args[2], args[3]))
                elif ctype == "Tangent":
                    # args: [e1, e2] or [e1, s1, e2, s2]
                    if len(args) == 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', args[0], args[1]))
                    elif len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', args[0], args[1], args[2], args[3]))
                elif ctype == "TangentViaPoint":
                    # args: [e1, e2, e3, s3]
                    if len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', args[0], args[1], args[2], args[3]))
                elif ctype == "Concentric":
                    # args: [e1, e2] -> Coincident(e1, 3, e2, 3)
                    if len(args) >= 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', args[0], 3, args[1], 3))
                elif ctype in ["Horizontal", "Vertical"]:
                    # args: [e1] or [e1, s1, e2, s2]
                    if len(args) == 1:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, args[0]))
                    elif len(args) >= 4:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, args[0], args[1], args[2], args[3]))
                elif ctype in ["Parallel", "Perpendicular", "Equal"]:
                    # args: [e1, e2]
                    if len(args) >= 2:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, args[0], args[1]))
                elif ctype == "PointOnObject":
                    # args: [e1, s1, e2]
                    if len(args) >= 3:
                        fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', args[0], args[1], args[2]))
                elif ctype == "Symmetric":
                    # args: [e1, s1, e2, s2, e3, s3]
                    if len(args) >= 6:
                        fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', args[0], args[1], args[2], args[3], args[4], args[5]))
                elif ctype == "Distance":
                    # args: [e1, value] or [e1, s1, e2, s2, value]
                    if len(args) == 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Distance', args[0], args[1]))
                    elif len(args) == 5:
                        fc_sketch.addConstraint(Sketcher.Constraint('Distance', args[0], args[1], args[2], args[3], args[4]))
                elif ctype == "Angle":
                    # args: [e1, value] or [e1, e2, value]
                    if len(args) == 2:
                        fc_sketch.addConstraint(Sketcher.Constraint('Angle', args[0], args[1]))
                    elif len(args) == 3:
                        fc_sketch.addConstraint(Sketcher.Constraint('Angle', args[0], args[1], args[2]))
                elif ctype in ["Radius", "Diameter"]:
                    # args: [e1, value]
                    if len(args) >= 2:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, args[0], args[1]))
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
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', e1, s1, e2, s2))
                elif ctype == "Tangent":
                    # Can be 4-arg (point-to-point) or 2-arg (edge-to-edge)
                    if s1 is not None and s2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', e1, s1, e2, s2))
                    elif e1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Tangent', e1, e2))
                elif ctype == "TangentViaPoint":
                    # TangentViaPoint(e1, e2, e3, s3)
                    if all(v is not None for v in [e1, e2, e3, s3]):
                        fc_sketch.addConstraint(Sketcher.Constraint('TangentViaPoint', e1, e2, e3, s3))
                elif ctype == "Concentric":
                    # Concentric means center points coincide
                    if e1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('Coincident', e1, 3, e2, 3))
                elif ctype in ["Horizontal", "Vertical"]:
                    if e2 is not None and s1 is not None and s2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, e1, s1, e2, s2))
                    elif e1 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, e1))
                elif ctype in ["Parallel", "Perpendicular", "Equal"]:
                    if e1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, e1, e2))
                elif ctype == "PointOnObject":
                    if e1 is not None and s1 is not None and e2 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', e1, s1, e2))
                elif ctype == "Symmetric":
                    if all(v is not None for v in [e1, s1, e2, s2, e3, s3]):
                        fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', e1, s1, e2, s2, e3, s3))
                elif ctype == "Distance":
                    value = compute_constraint_value("Distance", primitives, e1, s1, e2, s2)
                    if value is not None:
                        if e2 is not None and s1 is not None and s2 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', e1, s1, e2, s2, value))
                        elif e2 is not None and s1 is not None:
                            # Point-to-edge distance
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', e1, s1, e2, 0, value))
                        elif e1 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Distance', e1, value))
                elif ctype == "Angle":
                    value = compute_constraint_value("Angle", primitives, e1, s1, e2, s2)
                    if value is not None:
                        if e2 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Angle', e1, e2, value))
                        elif e1 is not None:
                            fc_sketch.addConstraint(Sketcher.Constraint('Angle', e1, value))
                elif ctype in ["Radius", "Diameter"]:
                    value = compute_constraint_value(ctype, primitives, e1)
                    if value is not None and e1 is not None:
                        fc_sketch.addConstraint(Sketcher.Constraint(ctype, e1, value))
        except Exception:
            print(f"\n\n\nERROR: Failed to add constraint {idx}: {c}\n\n\n")
            pass  # Skip constraints that fail to add


        count_after = int(fc_sketch.ConstraintCount)
        for i in range(count_before, count_after):
            constraint_mapping[i] = idx

    return constraint_mapping



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

    except Exception as e:
        error_msg = str(e)[:100]
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
        add_constraints_to_sketch(primitives, constraints, fc_sketch)
        doc.recompute()

        # Step 2: Get solver status
        fc_sketch.solve()
        status = fc_sketch.getSolverStatus()

        result.success = True
        result.dof = status['DoF']
        result.status = status['Status']
        result.malformed_count = len(status.get('MalformedConstraints', []))
        result.conflicting_count = len(status.get('ConflictingConstraints', []))
        result.redundant_count = len(status.get('RedundantConstraints', []))
        # Check if solver converged (Status = 'Converged' means OK)
        result.solver_converged = (status.get('Status', '') == 'Converged')

    except Exception as e:
        result.error = str(e)[:100]

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
    primitives: list,
    constraints: list,
    max_cleanup_iters: int = 10,
    prefix_size: int = 0,
) -> tuple[list, VerificationResult | None]:
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
    final_result = None

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
            constraint_mapping = add_constraints_to_sketch(primitives, current_constraints, fc_sketch)
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
                malformed_count=len(malformed_fc),
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

        except Exception as e:
            if final_result is None:
                final_result = VerificationResult(error=str(e)[:100])
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


def constraints_prefix_matched(input_constraints: list, output_constraints: list) -> bool:
    """Check if output constraints contain input constraints as a prefix.

    Args:
        input_constraints: Original partial constraints given to Qwen3 (GT format with args)
        output_constraints: Constraints returned by Qwen3 (entity1/subnode1 format)

    Returns:
        True if output starts with input constraints, False otherwise
    """
    if len(output_constraints) < len(input_constraints):
        return False

    for i, input_c in enumerate(input_constraints):
        output_c = output_constraints[i]

        # Compare constraint type
        if input_c.get("type") != output_c.get("type"):
            return False

        # Normalize both to args format for comparison
        def to_args(c: dict) -> list:
            """Convert constraint to args list format (excluding values for dimensional constraints)."""
            ctype = c.get("type", "")

            if "args" in c:
                raw_args = c["args"]
                # For dimensional constraints, exclude the value (last element is the value)
                if ctype in ["Radius", "Diameter"]:
                    # args: [entity, value] -> [entity]
                    return raw_args[:1] if len(raw_args) >= 1 else []
                elif ctype == "Distance":
                    # args: [entity, value] or [e1, s1, e2, s2, value]
                    if len(raw_args) == 2:
                        return raw_args[:1]  # [entity]
                    elif len(raw_args) == 5:
                        return raw_args[:4]  # [e1, s1, e2, s2]
                    return raw_args
                elif ctype == "Angle":
                    # args: [entity, value] or [e1, e2, value]
                    if len(raw_args) == 2:
                        return raw_args[:1]
                    elif len(raw_args) == 3:
                        return raw_args[:2]
                    return raw_args
                else:
                    return raw_args

            # Convert from entity1/subnode1 format
            args = []
            if ctype in ["Horizontal", "Vertical"]:
                # These can be 1-arg (line only) or 4-arg (point-to-point)
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
                    if c.get("subnode1") is not None and c.get("entity2") is not None:
                        args.append(c.get("subnode1"))
                        args.append(c.get("entity2"))
                        args.append(c.get("subnode2"))
            elif ctype in ["Parallel", "Perpendicular", "Equal"]:
                # 2-arg: entity1, entity2
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
                if c.get("entity2") is not None:
                    args.append(c.get("entity2"))
            elif ctype in ["Coincident", "Tangent"]:
                # 4-arg: entity1, subnode1, entity2, subnode2
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
                if c.get("subnode1") is not None:
                    args.append(c.get("subnode1"))
                if c.get("entity2") is not None:
                    args.append(c.get("entity2"))
                if c.get("subnode2") is not None:
                    args.append(c.get("subnode2"))
            elif ctype == "PointOnObject":
                # 3-arg: entity1, subnode1, entity2
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
                if c.get("subnode1") is not None:
                    args.append(c.get("subnode1"))
                if c.get("entity2") is not None:
                    args.append(c.get("entity2"))
            elif ctype == "Symmetric":
                # 6-arg: e1, s1, e2, s2, e3, s3
                for key in ["entity1", "subnode1", "entity2", "subnode2", "entity3", "subnode3"]:
                    if c.get(key) is not None:
                        args.append(c.get(key))
            elif ctype == "Distance":
                # Distance can be [e1] (line length) or [e1, s1, e2, s2] (point-to-point)
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
                    if c.get("subnode1") is not None:
                        args.append(c.get("subnode1"))
                        if c.get("entity2") is not None:
                            args.append(c.get("entity2"))
                            if c.get("subnode2") is not None:
                                args.append(c.get("subnode2"))
                            else:
                                args.append(0)  # Point-to-edge uses 0 for edge
            elif ctype == "Angle":
                # Angle can be [e1] (single line) or [e1, e2] (two lines)
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
                    if c.get("entity2") is not None:
                        args.append(c.get("entity2"))
            elif ctype in ["Radius", "Diameter"]:
                # entity1 only (ignore value for prefix matching)
                if c.get("entity1") is not None:
                    args.append(c.get("entity1"))
            return args

        input_args = to_args(input_c)
        output_args = to_args(output_c)

        if input_args != output_args:
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save FC GT and predictions that pass all validation criteria")
    parser.add_argument('--geo_threshold', type=float, default=50.0, help="Geometry validation threshold in mm")
    parser.add_argument('--keep_gt', action="store_true", help="Keep GT constraints")
    parser.add_argument('--output_dir', type=str, default='qwen3_refined_data_with_solver', help="Output directory")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of sequences to process")
    parser.add_argument('--end_idx', type=int, default=1000, help="End index of sequences to process")
    parser.add_argument('--min_solver_calls', type=int, default=1, help="Minimum solver calls required before accepting answer")
    args = parser.parse_args()

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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/svg", exist_ok=True)
    os.makedirs(args.output_dir + "/commands", exist_ok=True)

    error_log = os.path.join(args.output_dir, "error_log.txt")

    # load data
    sequence_path = "/home/ryotaro/my_works/My-Vitruvion/data_processed/sg_filtered_unique.npy"
    seq_data = flat_array.load_dictionary_flat(sequence_path)
    sequences = seq_data['sequences']

    for i, seq in enumerate(sequences[args.start_idx : args.end_idx]):
        i += args.start_idx

        # Skip if refined file already exists (resume support)
        output_vlm_path = f"{args.output_dir}/commands/{i:07d}_refined.py"
        if os.path.exists(output_vlm_path):
            print(f"=== Skipping sample {i} (refined already exists) ===")
            continue

        sketch = datalib.sketch_from_sequence(seq)

        # original data
        doc = None
        try:
            doc, fc_sketch, history_commands, entity_to_geoid, mapping = convert_sketch_to_freecad(sketch)

            # Dump History to File
            output_cmd_path = f"{args.output_dir}/commands/{i:07d}_original.py"
            with open(output_cmd_path, "w") as f:
                f.write("\n".join(history_commands))

            # Dump Sketch to PNG
            output_png_path = f"{args.output_dir}/svg/{i:07d}_original.png"
            output_vlm_image_path = output_png_path

            # Extract data from the saved script
            sketch_data = load_and_extract_from_file(output_cmd_path)
            visualize_sketch_for_vlm(sketch_data, save_path=output_png_path, title=None)

        except Exception as e:
            print(f"Error processing sequence {i}: {e}")
            with open(error_log, "a") as f:
                f.write(f"Error processing sequence {i}: {e}\n")
            continue

        finally:
            if doc is not None:
                FreeCAD.closeDocument(doc.Name)

        # refine by Qwen3
        sample_primitives, sample_constraints = parse_freecad_script(output_cmd_path)

        print(f"=== Processing sample {i} ===")
        print(f"File: {output_cmd_path}")
        print(f"Primitives: {len(sample_primitives)} items")
        print(f"GT Constraints: {len(sample_constraints)} items")
        print("=" * 50)

        # === GT Verification (with cleanup) ===
        # Note: We verify using cleaned constraints, but keep the original GT file
        # since export_to_freecad_script is designed for Qwen3 JSON format
        print("\n--- GT Verification (with cleanup) ---")
        gt_cleaned, gt_cleanup_result = cleanup_constraints_from_lists(
            sample_primitives, sample_constraints
        )
        print(f"  Cleanup: {len(sample_constraints)} -> {len(gt_cleaned)} constraints")

        # Convert gt_cleaned to Qwen3 JSON format for both export and Qwen3 input
        gt_cleaned_json = []
        for c in gt_cleaned:
            if "args" in c:
                gt_cleaned_json.append(format_constraint_to_json(c["type"], c["args"]))
            else:
                gt_cleaned_json.append(c)

        # Re-export GT file with cleaned constraints
        export_to_freecad_script(sample_primitives, gt_cleaned_json, output_cmd_path)
        print(f"  Re-exported GT with cleaned constraints: {output_cmd_path}")

        gt_verification = verify_constraints(sample_primitives, gt_cleaned, args.geo_threshold)
        gt_summary = gt_verification.summary()
        print(f"  Result: {gt_summary}")
        if gt_verification.geo_max_deviation >= 0:
            print(f"  Geometry deviation: {gt_verification.geo_max_deviation:.2f}mm")

        # Comment out GT file if verification fails
        if not gt_verification.is_valid():
            print("  ✗ GT INVALID: Issues detected - commenting out GT file")
            comment_out_file(output_cmd_path)


        # ストリーミングモードで実行
        # Use cleaned GT constraints as input to Qwen3
        messages = prepare_prompts(sample_primitives, gt_cleaned, output_vlm_image_path)
        llm.reset()

        # Main Inference Loop with Tool Use
        print("\n--- Qwen3 Generation (Interactive) ---")
        solver_call_count = 0

        while True:
            # Call LLM with streaming
            try:
                # Debug: show messages state before calling LLM
                # msg_summary = [(m.get("role"), len(str(m.get("content", "")))) for m in messages]
                # print(f"\n[DEBUG] Calling LLM with {len(messages)} messages: {msg_summary}", flush=True)

                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=32768,
                    temperature=0.7,
                    stream=True
                )

                current_turn_text = ""
                tool_call_detected = False

                for chunk in stream:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta and delta['content']:
                        content = delta['content']
                        current_turn_text += content
                        sys.stdout.write(content)
                        sys.stdout.flush()

                        # Check for tool call end tag
                        if "</solver_call>" in current_turn_text:
                            tool_call_detected = True
                            break # Stop generation to run tool

                # Note: We don't try to consume remaining stream chunks as it would block
                # until generation completes, causing multi-minute hangs

                # Append the model's output for this turn (keep as-is, no manipulation)
                messages.append({"role": "assistant", "content": current_turn_text})

                if tool_call_detected:
                    # Extract content between <solver_call> tags
                    call_content = current_turn_text.split("<solver_call>")[-1].split("</solver_call>")[0].strip()

                    # Parse and validate JSON
                    constraints_to_verify, parse_error = parse_solver_call_json(call_content)

                    if parse_error:
                        # Return structured error
                        tool_output = f'<solver_output>\n{{\n  "DoF": "null",\n  "Error": "{parse_error}"\n}}\n</solver_output>'
                        print(f"{tool_output}\n", flush=True)
                        messages.append({"role": "user", "content": tool_output})
                        continue

                    # Run solver with validated constraints
                    try:
                        solver_call_count += 1

                        tool_output = run_solver_tool(sample_primitives, constraints_to_verify)
                        print(f"{tool_output}\n", flush=True)

                        # Append Tool Output to messages
                        messages.append({"role": "user", "content": tool_output})

                        # Remove image from original user message after first successful tool call
                        # to avoid re-encoding the image on subsequent iterations
                        if solver_call_count == 1:
                            for msg in messages:
                                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                                    # Filter out image_url items, keep only text
                                    msg["content"] = [item for item in msg["content"] if item.get("type") != "image_url"]
                                    if not msg["content"]:
                                        msg["content"] = "[Image context already processed]"

                        continue

                    except Exception as e:
                        print(f"\n  [Tool Error]: {e}")
                        error_detail = f"Solver error: {str(e)[:80]}"
                        tool_output = f'<solver_output>\n{{\n  "DoF": "null",\n  "Error": "{error_detail}"\n}}\n</solver_output>'
                        messages.append({"role": "user", "content": tool_output})
                        continue

                else:
                    # Normal finish (no tool call in this chunk, likely finished)
                    # But first check: does model need more solver calls?
                    if solver_call_count < args.min_solver_calls and "</think>" in current_turn_text:
                        print(f"\n\n[ENFORCEMENT] Solver calls: {solver_call_count}/{args.min_solver_calls}. Running solver on proposed answer...", flush=True)

                        # Keep original assistant text as-is (no manipulation)
                        # messages.append already done above at L.2068

                        # Extract JSON from the final answer
                        json_text = extract_json_from_thinking_output(current_turn_text)

                        if json_text:
                            # Parse and run solver
                            constraints_to_verify, parse_error = parse_solver_call_json(json_text)

                            if parse_error:
                                tool_output = f'<solver_output>\n{{\n  "DoF": "null",\n  "Error": "{parse_error}"\n}}\n</solver_output>'
                            else:
                                tool_output = run_solver_tool(sample_primitives, constraints_to_verify)

                            # Add explanatory user message with solver output
                            explanation = f"""NOTE: You output </think> without calling <solver_call> first. The system has run the solver on your proposed constraints.
{tool_output}
Please review the result and either:
1. If there are issues, use <solver_call> to verify your revised constraints
2. If the result is satisfactory, you may output your final answer"""

                            print(f"{tool_output}\n", flush=True)
                            messages.append({"role": "user", "content": explanation})
                            solver_call_count += 1
                            continue
                        else:
                            # Couldn't extract JSON
                            error_msg = f'<solver_output>\n{{\n  "DoF": "null",\n  "Error": "Could not extract JSON from your final answer. Please use <solver_call> to verify your constraints."\n}}\n</solver_output>'
                            messages.append({"role": "user", "content": error_msg})
                            print(f"{error_msg}", flush=True)
                            continue

                    print("\n\n=== Generation Complete ===")
                    break

            except Exception as e:
                print(f"\nCritical Error during inference: {e}")
                break

        # Extract final JSON from the last assistant message
        output_text = messages[-1]['content']
        json_text = extract_json_from_thinking_output(output_text)
        print(f"\nExtracted JSON:\n{json_text}")

        try:
                output_vlm_path = f"{args.output_dir}/commands/{i:07d}_refined.py"
                constraints_json = json.loads(json_text)

                # === Qwen3 Output Cleanup ===
                print("\n--- Qwen3 Output Cleanup ---")

                # Protect GT prefix constraints only if keep_gt is True
                prefix_size = len(gt_cleaned) if args.keep_gt else 0

                qwen3_cleaned, qwen3_cleanup_result = cleanup_constraints_from_lists(
                    sample_primitives, constraints_json, prefix_size=prefix_size
                )
                print(f"  Cleanup: {len(constraints_json)} -> {len(qwen3_cleaned)} constraints (prefix_size={prefix_size})")

                # Check prefix matching (Qwen3 output should start with cleaned GT constraints)
                prefix_ok = constraints_prefix_matched(gt_cleaned, constraints_json)
                if not prefix_ok:
                    msg = "  ⚠ PREFIX MISMATCH: Qwen3 output does not contain cleaned GT constraints as prefix"
                    if args.keep_gt:
                        print(msg + " [STRICT MODE: FAILURE]")
                    else:
                        print(msg + " [IGNORED]")

                # Export cleaned constraints
                export_to_freecad_script(sample_primitives, qwen3_cleaned, output_vlm_path)

                # === Qwen3 Verification ===
                print("\n--- Qwen3 Verification ---")
                qwen3_verification = verify_constraints(sample_primitives, qwen3_cleaned, args.geo_threshold)

                # Check strict adherence only if keep_gt is True
                if args.keep_gt and not prefix_ok:
                    qwen3_verification.prefix_matched = False
                else:
                    qwen3_verification.prefix_matched = True


                # Check if Qwen3 made DoF worse (more DoF = less constrained = worse)
                if qwen3_verification.dof > gt_verification.dof:
                    qwen3_verification.dof_decreased = False
                    print(f"  ⚠ DOF INCREASED: Qwen3 DoF ({qwen3_verification.dof}) > GT DoF ({gt_verification.dof})")

                qwen3_summary = qwen3_verification.summary()
                print(f"  Result: {qwen3_summary}")
                if qwen3_verification.geo_max_deviation >= 0:
                    print(f"  Geometry deviation: {qwen3_verification.geo_max_deviation:.2f}mm")

                if qwen3_verification.is_valid():
                    print("  ✓ VALID: Constraints are well-formed and geometry preserved")
                else:
                    print("  ✗ INVALID: Issues detected - commenting out output file")
                    if qwen3_verification.error:
                        print(f"  Error: {qwen3_verification.error}")
                    # Comment out the output file
                    comment_out_file(output_vlm_path)

                # Log verification result (GT vs Qwen3 comparison)
                with open(os.path.join(args.output_dir, "verification_log.txt"), "a") as f:
                    valid_str = "PASS" if qwen3_verification.is_valid() else "FAIL"
                    f.write(f"{i:07d}: [GT] {gt_summary} | [Qwen3] {qwen3_summary} | solver_calls={solver_call_count} [{valid_str}]\n")

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            with open(error_log, "a") as f:
                f.write(f"Failed to parse JSON for sequence {i}: {e}\n")
            # Delete original file since refined file was not generated
            if os.path.exists(output_cmd_path):
                try:
                    os.remove(output_cmd_path)
                except:
                    pass
                print(f"  Deleted {output_cmd_path} (no valid refined output)")
