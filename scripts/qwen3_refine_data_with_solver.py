from typing import Any
import argparse
import base64
import json
import math
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass

import cairosvg
import rootutils
import traceback
import PIL.Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen35ChatHandler

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import dump_freecad_commands, ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_utils import cleanse_sketch_primitives, cleanse_sketch_constraints, complete_constraint_values, normalize_sketch
from scripts.draw_constraints import visualize_sketch_for_vlm, load_and_extract_from_file, export_sketch_to_dict


FreeCAD, Part, Sketcher = ensure_freecad_imported()


# ============================================================
# Geometry & Math Utils
# ============================================================

def parse_vector(vec_str: str) -> list[float]:
    # Extracts numbers from "Vector(x, y, z)"
    # Pattern supports scientific notation like 8.326e-14, 1.5E+10
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", vec_str)
    # Return x, y as float
    return [float(n) for n in nums[:2]]


def compute_arc_center(
    start: tuple[float, float],
    mid: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float, float]:
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
        raise ValueError(f"Points are collinear: {start}, {mid}, {end}")

    # Calculate center coordinates
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

    # Calculate radius
    radius = math.sqrt((ax - ux)**2 + (ay - uy)**2)

    return (ux, uy, radius)


def get_point_from_primitive(
    primitives: list[dict[str, Any]],
    entity_idx: int,
    subnode: int | None,
) -> tuple[float, float] | None:
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

    if subnode == 0 or subnode is None:  # Entity itself
        if ptype == "Point" and "x" in prim and "y" in prim:
            return (prim["x"], prim["y"])
        return None

    elif subnode == 1:  # StartPoint (or Point Vertex 1)
        if ptype == "Point" and "x" in prim and "y" in prim:
            return (prim["x"], prim["y"])
        else:
            return prim.get("start")

    elif subnode == 2:  # EndPoint
        if ptype == "Point" and "x" in prim and "y" in prim:
            return (prim["x"], prim["y"])
        else:
            return prim.get("end")

    elif subnode == 3:  # CenterPoint
        return prim.get("center")

    return None


def oscillate_primitives(primitives: list[dict[str, Any]], noise_std: float = 0.01) -> list[dict[str, Any]]:
    assert noise_std >= 0, f"{noise_std=}"

    oscillated_primitives = []
    for prim in primitives:
        prim_oscillated = prim.copy()

        for key, val in prim.items():
            if key in ["id", "type", "is_construction"]:
                continue
            if isinstance(val, bool):
                continue  # Just in case, because isinstance(True/False, bool) = True

            if isinstance(val, float):
                prim_oscillated[key] = val + random.random() * noise_std
            elif isinstance(val, (list, tuple)) and all(isinstance(x, float) for x in val):
                prim_oscillated[key] = [x + random.random() * noise_std for x in val]

        oscillated_primitives.append(prim_oscillated)

    return oscillated_primitives


def compute_constraint_value(
    ctype: str,
    primitives: list[dict[str, Any]],
    e1: int,
    s1: int | None = None,
    e2: int | None = None,
    s2: int | None = None,
) -> float:
    """
    Compute the value for a dimensional constraint based on primitive coordinates.

    Args:
        ctype: Constraint type ('Distance', 'Angle', 'Radius', 'Diameter')
        primitives: List of primitive dicts
        e1, s1: First entity index and subnode
        e2, s2: Second entity index and subnode (optional)

    Returns:
        Computed value
    """
    if e1 is None or e1 < 0 or e1 >= len(primitives):
        raise ValueError(f"Entity index invalid or out of range: {e1} ({primitives=})")

    prim1 = primitives[e1]
    prim2 = primitives[e2] if e2 is not None else None

    if ctype == "Distance":
        # Case 1: Single entity (e.g. Line Length)
        if e2 is None:
            if prim1["type"] == "LineSegment":
                sx, sy = prim1["start"]
                ex, ey = prim1["end"]
                return math.sqrt((ex - sx)**2 + (ey - sy)**2)
            else:
                raise ValueError(f"Unsupported primitive type: {prim1['type']}")

        # Case 2: Two entities (Point-Point or Point-Line)
        if prim2 is None:
            raise ValueError(f"Entity index out of range: {e2} ({primitives=})")

        def compute_dist_point_to_obj(p1: tuple[float, float], prim2: dict[str, Any]) -> float:
            if prim2["type"] == "Point":
                p2 = get_point_from_primitive(primitives, e2, s2)
                if p2 is None:
                    raise ValueError(f"Entity index {e2} not found: {primitives=}")
                return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if prim2["type"] == "LineSegment":
                lx1, ly1 = prim2["start"]
                lx2, ly2 = prim2["end"]
                dx, dy = lx2 - lx1, ly2 - ly1
                tx, ty = p1[0] - lx1, p1[1] - ly1
                area = abs(tx * dy - ty * dx)
                norm = math.sqrt(dx**2 + dy**2)
                return area / norm if norm > 0 else 0.0
            if prim2["type"] in ("Circle", "Arc"):
                cx, cy = prim2["center"]
                r = prim2["radius"]
                d_centers = math.sqrt((p1[0] - cx)**2 + (p1[1] - cy)**2)
                return abs(d_centers - r)  # For Point-Circle, FreeCAD expects non-negative values.
            raise ValueError(f"Unsupported primitive type: {prim2['type']}")

        def compute_dist_line_to_obj(line: dict[str, Any], prim2: dict[str, Any]) -> float:
            if line["type"] != "LineSegment":
                raise ValueError(f"`line` is not a Line primitive: {line=}")

            if prim2["type"] == "Point":
                p2 = get_point_from_primitive(primitives, e2, s2)
                if p2 is None:
                    raise ValueError(f"Entity index {e2} not found: {primitives=}")
                return compute_dist_point_to_obj(p2, line)
            if prim2["type"] == "LineSegment":
                dx1, dy1 = line["end"][0] - line["start"][0], line["end"][1] - line["start"][1]
                dx2, dy2 = prim2["end"][0] - prim2["start"][0], prim2["end"][1] - prim2["start"][1]
                # Check parallelism: |v1 x v2| should be close to 0
                cross_prod = abs(dx1 * dy2 - dy1 * dx2)
                norm1 = math.sqrt(dx1**2 + dy1**2)
                norm2 = math.sqrt(dx2**2 + dy2**2)
                if cross_prod / (norm1 * norm2) > 1e-3:
                    raise ValueError(f"Lines are not parallel: {line}, {prim2}")
                # Calculate distance from l1.start to l2
                return compute_dist_point_to_obj(line["start"], prim2)
            if prim2["type"] in ("Circle", "Arc"):
                d_center_to_line = compute_dist_point_to_obj(prim2["center"], line)
                return d_center_to_line - prim2["radius"]
            raise ValueError(f"Unsupported primitive type: {prim2['type']}")

        def compute_dist_circle_to_circle(prim1: dict[str, Any], prim2: dict[str, Any]) -> float:
            if prim1["type"] not in ("Circle", "Arc") or prim2["type"] not in ("Circle", "Arc"):
                raise ValueError(f"Both primitives must be a Circle or an Arc: {prim1=}, {prim2=}")
            cx1, cy1 = prim1["center"]
            cx2, cy2 = prim2["center"]
            d_centers = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if d_centers < abs(prim1["radius"] - prim2["radius"]):
                # whole circle is included by the other (positive)
                return abs(prim1["radius"] - prim2["radius"]) - d_centers
            else:
                # circles are disjoint (positive) or partially overlapping (negative)
                return d_centers - (prim1["radius"] + prim2["radius"])

        p1 = get_point_from_primitive(primitives, e1, s1)
        p2 = get_point_from_primitive(primitives, e2, s2)

        # Case2.1: Point-XXX (XXX = Point, Line, Circle, Arc)
        if p1 is not None:
            return compute_dist_point_to_obj(p1, prim2)
        elif p2 is not None:
            return compute_dist_point_to_obj(p2, prim1)
        # Case 2.2: Line-XXX (XXX = Line, Circle, Arc)
        elif prim1["type"] == "LineSegment":
            return compute_dist_line_to_obj(prim1, prim2)
        elif prim2["type"] == "LineSegment":
            return compute_dist_line_to_obj(prim2, prim1)
        # Case 2.3: Circle/Arc-Circle/Arc
        else:
            return compute_dist_circle_to_circle(prim1, prim2)

    elif ctype == "DistanceX":
        # Case 1: Single entity (horizontal span of a line)
        if e2 is None:
            if prim1["type"] == "LineSegment":
                sx, sy = prim1["start"]
                ex, ey = prim1["end"]
                return ex - sx
            else:
                raise ValueError(f"Unsupported primitive type: {prim1['type']}")

        # Case 2: Two points
        if prim2 is None:
            raise ValueError(f"Entity index out of range: {e2} ({primitives=})")

        p1 = get_point_from_primitive(primitives, e1, s1)
        p2 = get_point_from_primitive(primitives, e2, s2)

        if p1 is not None and p2 is not None:
            return p2[0] - p1[0]

        raise ValueError(f"DistanceX only accepts Line length and Point-Point distance: got {prim1=}, {prim2=}")

    elif ctype == "DistanceY":
        # Case 1: Single entity (vertical span of a line)
        if e2 is None:
            if prim1["type"] == "LineSegment":
                sx, sy = prim1["start"]
                ex, ey = prim1["end"]
                return ey - sy
            else:
                raise ValueError(f"Unsupported primitive type: {prim1['type']}")

        # Case 2: Two points
        if prim2 is None:
            raise ValueError(f"Entity index out of range: {e2} ({primitives=})")

        p1 = get_point_from_primitive(primitives, e1, s1)
        p2 = get_point_from_primitive(primitives, e2, s2)

        if p1 is not None and p2 is not None:
            return p2[1] - p1[1]

        raise ValueError(f"DistanceY only accepts Line length and Point-Point distance: got {prim1=}, {prim2=}")

    elif ctype == "Radius":
        return prim1["radius"]

    elif ctype == "Diameter":
        return prim1["radius"] * 2

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
                raise ValueError(f"Unsupported primitive type: {prim1=}, {prim2=}")
        else:
            # Single line angle (slope angle from horizontal)
            if prim1["type"] == "LineSegment":
                dx = prim1["end"][0] - prim1["start"][0]
                dy = prim1["end"][1] - prim1["start"][1]
                return math.atan2(dy, dx)  # radian
            else:
                raise ValueError(f"Unsupported primitive type: {prim1=}")

    raise ValueError(f"Unsupported constraint type: {ctype}")


# ============================================================
# Two-Pass Constraint Helpers
# ============================================================

DIMENSIONAL_CONSTRAINT_TYPES = {"Distance", "DistanceX", "DistanceY", "Angle", "Radius", "Diameter"}


def classify_constraints(constraints: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Classify constraints into relational and dimensional.

    Returns:
        Tuple of (relational_constraints, dimensional_constraints)
    """
    relational = []
    dimensional = []

    for c in constraints:
        ctype = c.get("type", "")
        if ctype in DIMENSIONAL_CONSTRAINT_TYPES:
            dimensional.append(c)
        else:
            relational.append(c)

    return relational, dimensional


def extract_primitives_from_sketch(fc_sketch: "Sketcher.SketchObject") -> list[dict[str, Any]]:
    """
    Extract primitive coordinates from FreeCAD sketch geometry.

    Derives type and construction flag directly from fc_sketch.

    Args:
        fc_sketch: FreeCAD sketch object

    Returns:
        List of primitive dicts with coordinates from fc_sketch.Geometry.
    """
    # Map FreeCAD geometry type names to our internal type names
    FC_TYPE_TO_PTYPE = {
        "Point": "Point",
        "LineSegment": "LineSegment",
        "Circle": "Circle",
        "ArcOfCircle": "Arc",
    }

    updated_primitives = []

    for geo_id, geo in enumerate(fc_sketch.Geometry):
        geo_type = type(geo).__name__

        # Determine primitive type
        ptype = FC_TYPE_TO_PTYPE.get(geo_type)
        if ptype is None:
            raise NotImplementedError(f"Unsupported geometry type: {geo_type}")

        new_prim: dict[str, Any] = {"type": ptype}

        # Get construction flag from fc_sketch
        try:
            is_construction = fc_sketch.getConstruction(geo_id)
            if is_construction:
                new_prim["is_construction"] = True
        except Exception:
            pass  # Not a construction line

        # Extract coordinates based on geometry type
        if geo_type == "Point":
            new_prim["x"] = geo.X
            new_prim["y"] = geo.Y
        elif geo_type == "LineSegment":
            sp = geo.StartPoint
            ep = geo.EndPoint
            new_prim["start"] = [sp.x, sp.y]
            new_prim["end"] = [ep.x, ep.y]
        elif geo_type == "Circle":
            c = geo.Center
            new_prim["center"] = [c.x, c.y]
            new_prim["radius"] = geo.Radius
        elif geo_type == "ArcOfCircle":
            sp = geo.StartPoint
            ep = geo.EndPoint
            # Get midpoint on arc curve using center and angles
            # This handles wrap-around cases correctly
            cx, cy = geo.Center.x, geo.Center.y
            r = geo.Radius
            # Compute angles from center to start/end points
            start_angle = math.atan2(sp.y - cy, sp.x - cx)
            end_angle = math.atan2(ep.y - cy, ep.x - cx)
            # Normalize and handle wrap-around
            if end_angle < start_angle:
                end_angle += 2 * math.pi
            mid_angle = (start_angle + end_angle) / 2.0
            mx = cx + r * math.cos(mid_angle)
            my = cy + r * math.sin(mid_angle)
            new_prim["start"] = [sp.x, sp.y]
            new_prim["mid"] = [mx, my]
            new_prim["end"] = [ep.x, ep.y]
            new_prim["center"] = [cx, cy]
            new_prim["radius"] = r
        else:
            # Unknown geometry type, skip
            continue

        updated_primitives.append(new_prim)

    return updated_primitives


def compute_centroid_aligned_deviation(
    primitives_before: list, primitives_after: list
) -> tuple[float, float, int]:
    """
    Compute max deviation between two primitive sets after aligning their centroids.

    Also computes length/radius deviations for lines, circles, and arcs.

    Returns:
        (max_coord_deviation, max_length_deviation, max_deviation_primitive_idx)
        - max_coord_deviation: Maximum point deviation in mm after centroid alignment
        - max_length_deviation: Maximum length/radius deviation in mm
        - max_deviation_primitive_idx: Index of primitive with largest deviation (-1 if none)
    """
    def get_all_points(primitives):
        points = []
        for p in primitives:
            ptype = p["type"]
            if ptype == "Point":
                points.append([p["x"], p["y"]])
            elif ptype == "LineSegment":
                points.append(list(p["start"]))
                points.append(list(p["end"]))
            elif ptype == "Circle":
                points.append(list(p["center"]))
            elif ptype == "Arc":
                points.append(list(p["start"]))
                points.append(list(p["mid"]))
                points.append(list(p["end"]))
        return points

    pts_before = get_all_points(primitives_before)
    pts_after = get_all_points(primitives_after)

    if len(pts_before) != len(pts_after) or len(pts_before) == 0:
        return float('inf'), float('inf'), -1

    # Compute centroids
    cx_b = sum(p[0] for p in pts_before) / len(pts_before)
    cy_b = sum(p[1] for p in pts_before) / len(pts_before)
    cx_a = sum(p[0] for p in pts_after) / len(pts_after)
    cy_a = sum(p[1] for p in pts_after) / len(pts_after)

    # Offset to align centroids
    offset_x = cx_b - cx_a
    offset_y = cy_b - cy_a

    # Compute coordinate deviation per primitive and track which has max
    max_coord_deviation = 0.0
    max_coord_primitive_idx = -1
    point_idx = 0

    for geo_id, prim in enumerate(primitives_before):
        ptype = prim["type"]
        if ptype == "Point":
            num_pts = 1
        elif ptype == "LineSegment":
            num_pts = 2
        elif ptype == "Circle":
            num_pts = 1
        elif ptype == "Arc":
            num_pts = 3
        else:
            continue

        prim_max_dev = 0.0
        for _ in range(num_pts):
            if point_idx >= len(pts_before):
                break
            bx, by = pts_before[point_idx]
            ax, ay = pts_after[point_idx]
            ax_aligned = ax + offset_x
            ay_aligned = ay + offset_y
            dev = math.sqrt((bx - ax_aligned)**2 + (by - ay_aligned)**2)
            prim_max_dev = max(prim_max_dev, dev)
            point_idx += 1

        if prim_max_dev > max_coord_deviation:
            max_coord_deviation = prim_max_dev
            max_coord_primitive_idx = geo_id

    # Compute length/radius deviation per primitive
    max_length_deviation = 0.0
    max_length_primitive_idx = -1

    for geo_id, (prim_b, prim_a) in enumerate(zip(primitives_before, primitives_after)):
        ptype = prim_b["type"]
        len_dev = 0.0

        if ptype == "LineSegment":
            len_b = math.sqrt((prim_b["end"][0] - prim_b["start"][0])**2 +
                              (prim_b["end"][1] - prim_b["start"][1])**2)
            len_a = math.sqrt((prim_a["end"][0] - prim_a["start"][0])**2 +
                              (prim_a["end"][1] - prim_a["start"][1])**2)
            len_dev = abs(len_b - len_a)

        elif ptype == "Circle" or ptype == "Arc":
            r_b = prim_b.get("radius", 0)
            r_a = prim_a.get("radius", 0)
            len_dev = abs(r_b - r_a)

        if len_dev > max_length_deviation:
            max_length_deviation = len_dev
            max_length_primitive_idx = geo_id

    # Determine which primitive has the overall maximum deviation
    if max_coord_deviation >= max_length_deviation:
        max_deviation_primitive_idx = max_coord_primitive_idx
    else:
        max_deviation_primitive_idx = max_length_primitive_idx

    return max_coord_deviation, max_length_deviation, max_deviation_primitive_idx


# ============================================================
# Parsing & JSON Parsing
# ============================================================

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

    def get_arg(idx: int):
        return args[idx] if idx < len(args) else None

    # Helper to check if arg is numeric (not a value like distance)
    def is_int_arg(val):
        return isinstance(val, int)

    if ctype == "Coincident":
        # Coincident(e1, s1, e2, s2) - 4 args
        if len(args) == 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    elif ctype == "PointOnObject":
        # PointOnObject(e1, s1, e2) - 3 args
        if len(args) == 3:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)

    elif ctype in ["Horizontal", "Vertical"]:
        # Horizontal(e1) - 1 arg for line
        # Horizontal(e1, s1, e2, s2) - 4 args for two points
        if len(args) == 1:
            c_out["entity1"] = get_arg(0)
        elif len(args) == 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    elif ctype in ["Parallel", "Equal"]:
        # Parallel(e1, e2) - 2 args
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)

    elif ctype in ["Tangent", "Perpendicular"]:
        # Tangent(e1, e2) - 2 args (edge-edge)
        # Tangent(e1, s1, e2) - 3 args (point-endge)
        # Tangent(e1, s1, e2, s2) - 4 args (point-point)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
        elif len(args) == 3:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
        elif len(args) == 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    elif ctype == "TangentViaPoint":
        # TangentViaPoint(e1, e2, e3, s3) - 4 args
        if len(args) == 4:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
            c_out["entity3"] = get_arg(2)
            c_out["subnode3"] = get_arg(3)

    elif ctype == "Concentric":
        # Concentric(e1, e2) - 2 args (centers coincide)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)

    elif ctype == "Symmetric":
        # Symmetric(e1, s1, e2, s2, e3, s3) - 6 args
        if len(args) == 6:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["entity3"] = get_arg(4)
            c_out["subnode3"] = get_arg(5)

    elif ctype in ["Distance", "DistanceX", "DistanceY"]:
        # Distance/DistanceX/DistanceY share the same argument signatures:
        # (e1, value) - 2 args (line length / horizontal or vertical span)
        # (e1, e2, value) - 3 args (distance between two entities)
        # (e1, s1, e2, value) - 4 args (point-entity distance)
        # (e1, s1, e2, s2, value) - 5 args (point-point distance)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["value"] = get_arg(1)
        elif len(args) == 3:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
            c_out["value"] = get_arg(2)
        elif len(args) == 4:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["value"] = get_arg(3)
        elif len(args) == 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["value"] = get_arg(4)

    elif ctype in ["Diameter", "Radius"]:
        # Diameter(e1, value) - 2 args
        # Radius(e1, value) - 2 args
        if len(args) == 2:
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
        elif len(args) == 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["value"] = get_arg(4)  # Preserve the value

    if c_out["entity1"] is None:
        raise ValueError(f"Unhandled arguments: {ctype=}, {args=}")

    # Return only non-null fields
    return {k: v for k, v in c_out.items() if v is not None}


def parse_freecad_script(filepath_or_commands: str | list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primitives = []
    constraints_gt = []
    primitive_cnt = -1

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
            primitive_cnt += 1

            if 'Part.LineSegment' in line:
                # Part.LineSegment(Vector(...), Vector(...))
                vecs = re.findall(r'Vector\s*\([^)]+\)', line)
                if len(vecs) >= 2:
                    p1 = parse_vector(vecs[0])
                    p2 = parse_vector(vecs[1])
                    primitives.append({
                        "id": primitive_cnt,
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
                        "id": primitive_cnt,
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
                                        "id": primitive_cnt,
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
                    cx, cy, radius = compute_arc_center(p1, p2, p3)
                    primitives.append({
                        "id": primitive_cnt,
                        "type": "Arc",
                        "start": p1,
                        "mid": p2,
                        "end": p3,
                        "center": [cx, cy],
                        "radius": radius,
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

                c_dict = format_constraint_to_json(c_type, c_args)
                constraints_gt.append(c_dict)

    return primitives, constraints_gt


def parse_qwen_output_json(call_content: str) -> tuple[list, str | None]:
    """Parse Qwen output JSON.

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


# ============================================================
# FreeCAD Command Generators
# ============================================================

def generate_freecad_primitives(primitives: list):
    """
    Generator that parses primitives and yields normalized FreeCAD arguments.

    Yields:
        Tuple of (primitive_type, args_list, is_construction, error_msg)
    """
    for prim in primitives:
        ptype = prim.get("type")
        is_const = prim.get("is_construction", False)

        try:
            if ptype == "LineSegment":
                if "start" in prim and "end" in prim:
                    sx, sy = prim["start"]
                    ex, ey = prim["end"]
                    yield "LineSegment", [sx, sy, 0, ex, ey, 0], is_const, None
                else:
                    yield "LineSegment", None, is_const, "Missing start/end for LineSegment"

            elif ptype == "Point":
                if "x" in prim and "y" in prim:
                    x, y = prim["x"], prim["y"]
                    yield "Point", [x, y, 0], is_const, None
                else:
                    yield "Point", None, is_const, "Missing x/y for Point"

            elif ptype == "Circle":
                if "center" in prim and "radius" in prim:
                    cx, cy = prim["center"]
                    r = prim["radius"]
                    yield "Circle", [cx, cy, 0, 0, 0, 1, r], is_const, None
                else:
                    yield "Circle", None, is_const, "Missing center/radius for Circle"

            elif ptype == "Arc":
                if "start" in prim and "mid" in prim and "end" in prim:
                    sx, sy = prim["start"]
                    mx, my = prim["mid"]
                    ex, ey = prim["end"]
                    yield "ArcOfCircle", [sx, sy, 0, mx, my, 0, ex, ey, 0], is_const, None
                else:
                    yield "ArcOfCircle", None, is_const, "Missing start/mid/end for Arc"

            else:
                yield ptype, None, is_const, f"Unknown primitive type: {ptype}"

        except Exception as e:
            # Report exception as error
            yield ptype, None, is_const, f"Exception parsing primitive: {e}"


def generate_freecad_constraints(primitives: list[dict[str, Any]], constraints: list[dict[str, Any]]):
    """
    Generator that parses constraints and yields normalized FreeCAD arguments.

    Yields:
        Tuple of (original_index, constraint_type, args_list, error_msg)

    Note:
        - Handles both Qwen3 format (keys) and GT format (args list)
        - Computes/Recomputes values for Distance, Angle, Radius, Diameter
        - Normalizes arguments (e.g. s2=0 for Point-Line Distance)
        - Yields only valid constraints (invalid ones report errors)
    """
    for idx, c in enumerate(constraints):
        ctype = c.get("type", "")

        try:
            # --- 1. Input Source Resolution & Normalization ---
            final_args = []

            # Qwen3 format: resolve from keys
            e1, s1 = c.get("entity1"), c.get("subnode1", 0)
            e2, s2 = c.get("entity2"), c.get("subnode2", 0)
            e3, s3 = c.get("entity3"), c.get("subnode3", 0)

            # Safeguard against explicitly 'null' subnodes from LLM output
            s1 = 0 if s1 is None else s1
            s2 = 0 if s2 is None else s2
            s3 = 0 if s3 is None else s3
            prim1 = primitives[e1] if e1 is not None else None
            prim2 = primitives[e2] if e2 is not None else None
            prim3 = primitives[e3] if e3 is not None else None

            if e1 is None:
                yield idx, ctype, None, f"First reference is None for {ctype}"
                continue

            # Preprocessing: Point entity should have subnode=1, not 0
            if prim1 and prim1["type"] == "Point":
                s1 = 1
            if prim2 and prim2["type"] == "Point":
                s2 = 1
            if prim3 and prim3["type"] == "Point":
                s3 = 1

            # --- Value Recomputation for Dimensional Constraints ---
            value = None
            if ctype in ["Distance", "DistanceX", "DistanceY", "Angle", "Radius", "Diameter"]:
                # ALWAYS recompute value
                try:
                    value = compute_constraint_value(ctype, primitives, e1, s1, e2, s2)
                except ValueError as e:
                    yield idx, ctype, None, f"Failed to compute value for {ctype}"
                    continue # Skip if computation fails

            # --- Type Logic ---
            if ctype == "Coincident":
                if e1 is not None and s1 is not None and e2 is not None and s2 is not None:
                    final_args = [int(e1), int(s1), int(e2), int(s2)]
                else:
                    yield idx, ctype, None, "Missing args for Coincident"
                    continue

            elif ctype in ["Tangent", "Perpendicular"]:
                if s1 == 0 and s2 == 0:
                    if e1 is not None and e2 is not None:
                        final_args = [int(e1), int(e2)]
                    else:
                        yield idx, ctype, None, f"Missing args for {ctype}"
                        continue
                else:
                    if e1 is not None and e2 is not None:
                        final_args = [int(e1), int(s1), int(e2), int(s2)]
                    else:
                        yield idx, ctype, None, f"Missing args for {ctype}"
                        continue

            elif ctype == "TangentViaPoint":
                if all(v is not None for v in [e1, e2, e3, s3]):
                    final_args = [int(e1), int(e2), int(e3), int(s3)]
                else:
                    yield idx, ctype, None, "Missing args for TangentViaPoint"
                    continue

            elif ctype == "Concentric":
                if e1 is not None and e2 is not None:
                    final_args = [int(e1), 3, int(e2), 3]
                else:
                    yield idx, ctype, None, "Missing args for Concentric"
                    continue

            elif ctype in ["Horizontal", "Vertical"]:
                if e2 is not None and s1 is not None and s2 is not None:
                    final_args = [int(e1), int(s1), int(e2), int(s2)]
                elif e1 is not None:
                    final_args = [int(e1)]
                else:
                    yield idx, ctype, None, f"Missing args for {ctype}"
                    continue

            elif ctype in ["Parallel", "Equal"]:
                if e1 is not None and e2 is not None:
                    final_args = [int(e1), int(e2)]
                else:
                    yield idx, ctype, None, f"Missing args for {ctype}"
                    continue

            elif ctype == "PointOnObject":
                if e1 is not None and s1 is not None and e2 is not None:
                    final_args = [int(e1), int(s1), int(e2)]
                else:
                    yield idx, ctype, None, "Missing args for PointOnObject"
                    continue

            elif ctype == "Symmetric":
                if all(v is not None for v in [e1, s1, e2, s2, e3]):
                    final_args = [int(e1), int(s1), int(e2), int(s2), int(e3)]
                    if s3 is not None:
                        final_args.append(int(s3))
                else:
                    yield idx, ctype, None, "Missing args for Symmetric"
                    continue

            elif ctype in ["Distance", "DistanceX", "DistanceY"]:
                assert value is not None
                if e2 is None:
                    # Line span (e1, value)
                    final_args = [int(e1), value]
                else:
                    if prim1 and prim1["type"] == "LineSegment" and s1 == 0 and s2 > 0:
                        # FreeCAD doesn't accept Line-XXX (XXX = Point/Circle/Arc). XXX-Line is okay.
                        final_args = [int(e2), int(s2), int(e1), int(s1), value]
                    else:
                        final_args = [int(e1), int(s1), int(e2), int(s2), value]

            elif ctype == "Angle":
                assert value is not None
                if e2 is not None:
                    final_args = [int(e1), int(e2), value]
                elif e1 is not None:
                    final_args = [int(e1), value]
                else:
                    yield idx, ctype, None, "Missing args for Angle"
                    continue

            elif ctype in ["Radius", "Diameter"]:
                assert value is not None
                if e1 is not None:
                    final_args = [int(e1), value]
                else:
                    yield idx, ctype, None, f"Missing args for {ctype}"
                    continue

            else:
                # Unknown type
                yield idx, ctype, None, f"Unknown constraint type: {ctype}"
                continue

            # --- 2. Yield Result ---
            yield idx, ctype, final_args, None

        except Exception as e:
            # If any parsing fails, skip this constraint but log/track failure if needed
            print(traceback.format_exc())
            yield idx, ctype, None, f"Exception parsing constraint: {e}"



# ============================================================
# FreeCAD Interaction (Builders)
# ============================================================

def add_primitives_to_sketch(primitives: list[dict[str, Any]], fc_sketch: "Sketcher.SketchObject") -> None:
    """Add primitives to FreeCAD sketch object.

    Args:
        primitives: List of primitive dicts
        fc_sketch: FreeCAD sketch object to add primitives to
    """
    for ptype, args, is_const, error_msg in generate_freecad_primitives(primitives):
        if error_msg:
            print(f"ERROR: Invalid primitive: {error_msg}")
            continue

        try:
            if ptype == "LineSegment":
                # args: [sx, sy, 0, ex, ey, 0]
                sx, sy, sz, ex, ey, ez = args
                fc_sketch.addGeometry(
                    Part.LineSegment(FreeCAD.Vector(sx, sy, sz), FreeCAD.Vector(ex, ey, ez)),
                    is_const
                )
            elif ptype == "Point":
                # args: [x, y, 0]
                x, y, z = args
                fc_sketch.addGeometry(Part.Point(FreeCAD.Vector(x, y, z)), is_const)
            elif ptype == "Circle":
                # args: [cx, cy, 0, 0, 0, 1, r]
                cx, cy, cz, nx, ny, nz, r = args
                fc_sketch.addGeometry(
                    Part.Circle(FreeCAD.Vector(cx, cy, cz), FreeCAD.Vector(nx, ny, nz), r),
                    is_const
                )
            elif ptype == "ArcOfCircle":
                # args: [sx, sy, 0, mx, my, 0, ex, ey, 0]
                sx, sy, sz, mx, my, mz, ex, ey, ez = args
                arc = Part.ArcOfCircle(
                    FreeCAD.Vector(sx, sy, sz),
                    FreeCAD.Vector(mx, my, mz),
                    FreeCAD.Vector(ex, ey, ez)
                )
                fc_sketch.addGeometry(arc, is_const)
        except Exception:
            print(f"ERROR: Failed to add primitive: {ptype}, {args}")
            pass


def add_constraints_to_sketch(
    primitives: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    fc_sketch: "Sketcher.SketchObject"
) -> tuple[dict[int, int], list[int], list]:
    """Add constraints to FreeCAD sketch object.

    1. Pre-calculate all the missing dimensional constraint values
    2. Apply all the constraints

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts from Qwen3 output
        fc_sketch: FreeCAD sketch object to add constraints to

    Returns:
        Tuple of:
          - Mapping from FreeCAD constraint index to original constraint index
          - List of indices of constraints that failed to add (malformed/error)
          - Updated primitives list (after all constraints applied)
    """
    constraint_mapping = {}
    failed_indices = []

    # Classify constraints into relational and dimensional
    relational_constraints, dimensional_constraints = classify_constraints(constraints)

    # 1. Determine dimensional constraint values (values are computed within generate_freecad_constraints)
    relational_constraints_for_freecad = list(generate_freecad_constraints(primitives, relational_constraints))
    dimensional_constraints_for_freecad = list(generate_freecad_constraints(primitives, dimensional_constraints))

    for pass_constraints, source_list in [
        (relational_constraints_for_freecad, relational_constraints),
        (dimensional_constraints_for_freecad, dimensional_constraints),
    ]:
        for idx, ctype, args, error_msg in pass_constraints:
            orig_idx = constraints.index(source_list[idx])
            if error_msg:
                print(f"[add_constraints_to_sketch] Error: {idx=}, {ctype=}, {error_msg=}", flush=True)
                failed_indices.append(orig_idx)
                continue

            fc_idx = int(fc_sketch.ConstraintCount)
            try:
                fc_sketch.addConstraint(Sketcher.Constraint(ctype, *args))
                constraint_mapping[fc_idx] = orig_idx
            except Exception:
                failed_indices.append(orig_idx)

    fc_sketch.solve()
    final_primitives = extract_primitives_from_sketch(fc_sketch)

    return constraint_mapping, failed_indices, final_primitives


def _write_freecad_script(primitives: list[dict[str, Any]], constraints: list[dict[str, Any]]) -> list[str]:
    """Write primitive and constraint commands into lines list.

    Uses original primitives for constraint value computation.

    Args:
        primitives: Original (pre-constraint) primitive list, used for constraint value computation
        constraints: Constraint list
        lines: Output list to append script lines into
    """

    lines = [
        "import FreeCAD",
        "import Part",
        "import Sketcher",
        "",
        "doc = FreeCAD.newDocument('SketchDoc')",
        "body = doc.addObject('PartDesign::Body', 'Body')",
        "fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')",
        "xy_plane = body.Origin.OriginFeatures[3]",
        "fc_sketch.AttachmentSupport = xy_plane",
        "fc_sketch.MapMode = 'FlatFace'",
        "",
    ]

    lines.append("# --- Add Primitives ---")
    for ptype, args, is_const, error_msg in generate_freecad_primitives(primitives):
        if error_msg:
            lines.append(f"# ERROR: Invalid primitive: {error_msg}, Type: {ptype}")
            continue
        if ptype == "LineSegment":
            sx, sy, sz, ex, ey, ez = args
            lines.append(f"fc_sketch.addGeometry(Part.LineSegment(FreeCAD.Vector({sx}, {sy}, {sz}), FreeCAD.Vector({ex}, {ey}, {ez})), {is_const})")
        elif ptype == "Point":
            x, y, z = args
            lines.append(f"fc_sketch.addGeometry(Part.Point(FreeCAD.Vector({x}, {y}, {z})), {is_const})")
        elif ptype == "Circle":
            cx, cy, cz, nx, ny, nz, r = args
            lines.append(f"fc_sketch.addGeometry(Part.Circle(FreeCAD.Vector({cx}, {cy}, {cz}), FreeCAD.Vector({nx}, {ny}, {nz}), {r}), {is_const})")
        elif ptype == "ArcOfCircle":
            sx, sy, sz, mx, my, mz, ex, ey, ez = args
            lines.append(f"fc_sketch.addGeometry(Part.ArcOfCircle(FreeCAD.Vector({sx}, {sy}, {sz}), FreeCAD.Vector({mx}, {my}, {mz}), FreeCAD.Vector({ex}, {ey}, {ez})), {is_const})")

    lines.append("")
    lines.append("# --- Add Constraints ---")
    for _, ctype, args, error_msg in generate_freecad_constraints(primitives, constraints):
        if error_msg:
            lines.append(f"# ERROR: Invalid constraint: {error_msg}")
            continue
        args_str = ", ".join(map(str, args))
        lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{ctype}', {args_str}))")

    lines += [
        "",
        "doc.recompute()",
        "FreeCAD.ActiveDocument.recompute()",
    ]

    return lines


def export_to_freecad_script(primitives: list, constraints: list, output_path: str) -> str | None:
    """Generate a FreeCAD Python script to create a sketch with given primitives and constraints.

    Applies constraints via add_constraints_to_sketch to obtain final solved primitive
    coordinates, then exports a script using those coordinates with constraint values
    computed from the original primitives.

    Args:
        primitives: Original primitive list
        constraints: Constraint list
        output_path: Path to write the output script

    Returns:
        None on success, error string on failure
    """
    doc = None
    final_primitives = primitives  # fallback if FreeCAD fails

    try:
        doc = FreeCAD.newDocument("ExportTempDoc")
        body = doc.addObject('PartDesign::Body', 'Body')
        fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
        xy_plane = body.Origin.OriginFeatures[3]
        fc_sketch.AttachmentSupport = xy_plane
        fc_sketch.MapMode = 'FlatFace'

        add_primitives_to_sketch(primitives, fc_sketch)
        doc.recompute()

        _, _, final_primitives = add_constraints_to_sketch(primitives, constraints, fc_sketch)
        doc.recompute()

    except Exception as e:
        return str(e)
    finally:
        FreeCAD.closeDocument(doc.Name) if doc else None

    # Use final_primitives for geometry coordinates (solved positions),
    # but original primitives for constraint value computation.
    lines = _write_freecad_script(final_primitives, constraints)

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Exported to {output_path}")
    return None


# ============================================================
# Verification & Validation
# ============================================================

def extract_geometry_coords(fc_sketch: "Sketcher.SketchObject") -> dict[int, dict]:
    """Extract coordinates and length from all geometries in a FreeCAD sketch.

    Returns dict mapping geo_id -> {'points': [(x,y), ...], 'length': float}.
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

        # safely get length (some primitives might use Length or length)
        fn_len = getattr(geo, 'Length', getattr(geo, 'length', 0.0))
        if callable(fn_len):
            fn_len = fn_len()

        coords[geo_id] = {
            'points': points,
            'length': float(fn_len)
        }

    return coords


def validate_geometry_preservation(
    primitives: list,
    constraints: list,
    threshold_mm: float = 10.0,
) -> tuple[list[int], str]:
    """
    Validate geometry impact of constraints greedily, filtering out those that cause deviation or solver failure.

    Args:
        primitives: List of primitive dicts
        constraints: List of constraint dicts
        threshold_mm: Maximum allowed deviation in mm

    Returns:
        Tuple of (distorting_indices, error_msg)
        - distorting_indices: List of original indices of constraints that were filtered out
    """
    doc = None
    distorting_indices = []
    error_msg = ""

    try:
        # Create FreeCAD document for base state
        doc = FreeCAD.newDocument("GeoValidationDoc")
        body = doc.addObject('PartDesign::Body', 'Body')
        fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
        fc_sketch.AttachmentSupport = body.Origin.OriginFeatures[3]
        fc_sketch.MapMode = 'FlatFace'

        # Base State: Add primitives only
        add_primitives_to_sketch(primitives, fc_sketch)
        doc.recompute()
        primitives_before = extract_primitives_from_sketch(fc_sketch)

        # Prepare indexed constraints for iteration
        # Note: We process Relational then Dimensional, but must track original indices
        indexed_constraints = list(enumerate(constraints))

        dim_types = {"Distance", "DistanceX", "DistanceY", "Angle", "Radius", "Diameter"}
        rel_indexed = [(i, c) for i, c in indexed_constraints if c.get("type") not in dim_types]
        dim_indexed = [(i, c) for i, c in indexed_constraints if c.get("type") in dim_types]

        kept_constraints = []

        # Iterate and filter using Incremental Rebuild Strategy (O(N^2))
        # This is necessary because FreeCAD's abortTransaction() is unreliable for geometry reverts.
        for constr_idx, constr in rel_indexed + dim_indexed:
            try:
                # 1. Clear Sketch
                fc_sketch.deleteAllGeometry()

                # 2. Add Primitives (Clean State)
                add_primitives_to_sketch(primitives, fc_sketch)
                doc.recompute() # Ensure geometry handles are fresh

                # 3. Add Kept Constraints + Candidate
                # constraint_mapping works on the original primitives list which matches current geometry order
                test_constraints = kept_constraints + [constr]
                add_constraints_to_sketch(primitives, test_constraints, fc_sketch)

                # 4. Solve and Check
                fc_sketch.solve()
                doc.recompute()

                status = fc_sketch.getSolverStatus()

                is_solver_broken = (
                    status.get('Status', '') != 'Converged' or
                    len(status.get('ConflictingConstraints', [])) > 0 or
                    len(status.get('MalformedConstraints', [])) > 0 or
                    len(status.get('RedundantConstraints', [])) > 0
                )

                # Check deviation
                primitives_now = extract_primitives_from_sketch(fc_sketch)
                m_c, m_l, _ = compute_centroid_aligned_deviation(primitives_before, primitives_now)

                is_distorting = (m_c > threshold_mm or m_l > threshold_mm)

                if is_distorting:
                    distorting_indices.append(constr_idx)
                    # Do NOT append to kept_constraints
                elif is_solver_broken:
                    # Broken but not distorting? Or Broken AND Distorting?
                    # Priority: Distorting > Broken. If dist check passed, then it's just broken solver.
                    pass # Do NOT append
                else:
                    # Valid
                    kept_constraints.append(constr)

            except Exception as e:
                # If unexpected error, treat as skipping this constraint
                error_msg += f"Warning: Error validating constraint {constr_idx}: {e}\n"
                pass


    except Exception:
        error_msg += f"ERROR: {traceback.format_exc()}"

    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    return distorting_indices, error_msg


@dataclass
class VerificationResult:
    """Verification result from FreeCAD solver."""
    dof: int = -1
    status: str = "Unknown"
    malformed_count: int = 0
    conflicting_count: int = 0
    redundant_count: int = 0
    distorting_count: int = 0
    dof_decreased: bool = True  # True if Qwen3 DoF <= GT DoF (improved or same)
    error: str | None = None


    def is_valid(self) -> bool:
        """Check if constraints are valid (no issues and no distorting constraints)."""
        return (
            self.status == "Converged" and
            self.malformed_count == 0 and
            self.conflicting_count == 0 and
            self.redundant_count == 0 and
            self.distorting_count == 0 and
            self.dof_decreased and
            (self.error is None or self.error == "")
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
        if self.status != "Converged":
            issues.append(self.status)
        if self.distorting_count > 0:
            issues.append(f"distorting={self.distorting_count}")
        if not self.dof_decreased:
            issues.append("dof_increased")
        issue_str = ", ".join(issues) if issues else "OK"
        return f"DoF={self.dof}, {issue_str}"


def verify_constraints(
    primitives: list,
    constraints: list,
    geo_threshold_mm: float = 10.0,
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
        _mapping, failed_indices, _ = add_constraints_to_sketch(primitives, constraints, fc_sketch)
        doc.recompute()

        # Step 2: Get solver status
        fc_sketch.solve()
        status = fc_sketch.getSolverStatus()

        result.dof = status['DoF']
        result.status = "Malformed" if failed_indices else status['Status']
        # Malformed = Reported by solver + Failed to add
        result.malformed_count = len(status.get('MalformedConstraints', [])) + len(failed_indices)
        result.conflicting_count = len(status.get('ConflictingConstraints', []))
        result.redundant_count = len(status.get('RedundantConstraints', []))

    except Exception:
        result.error = "ERROR: " + traceback.format_exc()
        result.status = "Error"

    finally:
        if doc is not None:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    # Step 3: Geometry validation (Greedy check to find specific bad constraints)
    distorting_indices, geo_error = validate_geometry_preservation(
        primitives, constraints, geo_threshold_mm
    )
    result.distorting_count = len(distorting_indices)
    if geo_error:
        result.error = geo_error
    if result.status == "Converged" and result.distorting_count > 0:
        result.status = "Distorted"

    return result


def static_constraints_diagnosis(
    primitives_list: list[dict[str, Any]],
    constraints_json_list: list[dict[str, Any]],
):
    warnings = []

    for i, c in enumerate(constraints_json_list):

        e1, s1 = c.get("entity1"), c.get("subnode1")
        e2, s2 = c.get("entity2"), c.get("subnode2")
        e3, s3 = c.get("entity3"), c.get("subnode3")
        p1 = get_point_from_primitive(primitives_list, e1, s1) if e1 is not None else None
        p2 = get_point_from_primitive(primitives_list, e2, s2) if e2 is not None else None
        p3 = get_point_from_primitive(primitives_list, e3, s3) if e3 is not None else None
        constraint_name = f"{c['type']}(" + ", ".join(str(x) for x in [e1, s1, e2, s2, e3, s3] if x is not None) + ")"

        # Check for non-existing primitive references (allow external entity IDs)
        if e1 is not None and (e1 < 0 or len(primitives_list) <= e1):
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: entity1={e1} is out of bounds.")
        if e2 is not None and (e2 < 0 or len(primitives_list) <= e2):
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: entity2={e2} is out of bounds.")
        if e3 is not None and (e3 < 0 or len(primitives_list) <= e3):
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: entity3={e3} is out of bounds.")

        # Check if point entity subnode is set to 1
        if e1 is not None and 0 <= e1 < len(primitives_list) and primitives_list[e1]["type"] == "Point" and s1 != 1:
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: point entity subnode must be 1.")
        if e2 is not None and 0 <= e2 < len(primitives_list) and primitives_list[e2]["type"] == "Point" and s2 != 1:
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: point entity subnode must be 1.")
        if e3 is not None and 0 <= e3 < len(primitives_list) and primitives_list[e3]["type"] == "Point" and s3 != 1:
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: point entity subnode must be 1.")

        # Check if line entity subnode <= 2
        if e1 is not None and 0 <= e1 < len(primitives_list) and primitives_list[e1]["type"] == "LineSegment" and s1 == 3:
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: line entity subnode must be <= 2.")
        if e2 is not None and 0 <= e2 < len(primitives_list) and primitives_list[e2]["type"] == "LineSegment" and s2 == 3:
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: line entity subnode must be <= 2.")
        if e3 is not None and 0 <= e3 < len(primitives_list) and primitives_list[e3]["type"] == "LineSegment" and s3 == 3:
            warnings.append(f"Constraint {i} ({constraint_name}) is malformed: line entity subnode must be <= 2.")

        # Check for malformed Coincident constraints
        if c["type"] == "Coincident":
            if p1 is None or p2 is None:
                warnings.append(f"Constraint {i} ({constraint_name}) is malformed: Coincident applies between points, so subnode1/2 must be >= 1.")

        # Check for malformed PointsOnObject constraints
        if c["type"] == "PointOnObject":
            if p1 is None:
                warnings.append(f"Constraint {i} ({constraint_name}) is malformed: entity1.subnode1 must be a point.")

        # Check for malformed Symmetric constraints
        if c["type"] == "Symmetric":
            if p1 is None or p2 is None:
                warnings.append(f"Constraint {i} ({constraint_name}) is malformed: subnode1 and subnode2 must be >= 1.")

        # Check for malformed Tangent and Perpendicular constraints
        if c["type"] in ["Tangent", "Perpendicular"]:
            if (s1 == 0 or s1 is None) and s2 is not None and s2 > 0:
                warnings.append(f"Constraint {i} ({constraint_name}) is malformed: When combining an edge and a point for {c['type']}, the point must be specified first (Subnode1 > 0) due to CAD solver specifications. Please swap Entity1 and Entity2.")

        # Check for malformed/diagonal Distance constraints
        if c["type"] in ["Distance", "DistanceX", "DistanceY"]:
            if (e1 is not None and primitives_list[e1]["type"] == "LineSegment" and p1 is None) and (e2 is not None and primitives_list[e2]["type"] == "LineSegment" and p2 is None):
                warnings.append(f"Constraint {i} ({constraint_name}) is malformed: You cannot define a distance between lines. Define Point-Line distance instead by setting either Subnode1 >= 1 or Subnode2 >= 1.")
            elif c["type"] in ["DistanceX", "DistanceY"] and (e2 is not None) and (e1 != e2) and (p1 is None or p2 is None):
                warnings.append(f"Constraint {i} ({constraint_name}) is malformed: You cannot define a horizontal/vertical distance between a point and a line. Use a normal Distance instead, or define Point-Point distance by setting both Subnode1 and Subnode2 >= 1.")
            elif c["type"] == "Distance" and p1 and p2:
                # Diagonal check only applies to Distance, not DistanceX/Y (which are axis-aligned by definition)
                # p1, p2 are [x, y] lists
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])

                # Threshold for "Diagonal": neither H nor V (e.g. > 0.1mm deviation in both axes)
                if dx > 0.1 and dy > 0.1:
                    warnings.append(f"Constraint {i} ({constraint_name}) is diagonally placed. Consider using DistanceX (horizontal) or DistanceY (vertical) constraint instead.")

    # # DEBUG
    # if warnings:
    #     warnings.append(primitives_list)

    return warnings


def run_solver_tool(
    primitives_list: list,
    constraints_json_list: list,
    geo_threshold_mm: float = 10.0,
    viz_image_path: str | None = None,
) -> str:
    """Run FreeCAD solver for the provided constraints and return status JSON string."""
    doc = None
    base_sketch_data = None
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

        # Capture Pre-Solve State (Primitives Only) for visualization
        if viz_image_path:
            try:
                base_sketch_data = export_sketch_to_dict(fc_sketch)
                # Ensure no constraints exist (should be empty anyway at this stage)
                base_sketch_data["constraints"] = []
            except:
                pass

        constraint_mapping, failed_indices, _ = add_constraints_to_sketch(primitives_list, constraints_json_list, fc_sketch)
        doc.recompute()

        # Solve
        fc_sketch.solve()
        status = fc_sketch.getSolverStatus()

        fc_to_orig = constraint_mapping

        def convert_indices(fc_indices):
            return [fc_to_orig[idx] for idx in fc_indices if idx in fc_to_orig]

        def convert_groups(groups):
            return [convert_indices(set(group)) for group in groups]

        # Combine malformed indices (both reported by solver and failed to add)
        malformed_from_solver = convert_indices(status.get('MalformedConstraints', []))
        all_malformed = list(set(malformed_from_solver + failed_indices))
        all_malformed.sort()

        # Construct response dict
        response = {
            'DoF': status['DoF'],
            'Status': "Malformed" if all_malformed else status['Status'],
            'ConflictingConstraints': convert_indices(status.get('ConflictingConstraints', [])),
            'RedundantConstraints': convert_indices(status.get('RedundantConstraints', [])),
            'ConflictGroups': convert_groups(status.get('ConflictGroups', [])),
            'MalformedConstraints': all_malformed
        }

        # safety net (critical level: high to low)
        if response['Status'] == 'Converged' and response['MalformedConstraints']:
            response['Status'] = 'Malformed'
        elif response['Status'] == 'Converged' and response['ConflictingConstraints']:
            response['Status'] = 'Conflicting'
        elif response['Status'] == 'Converged' and response['RedundantConstraints']:
            response['Status'] = 'Redundant'

        # --- Geometry Validation ---
        distorting_indices, geo_error = validate_geometry_preservation(
            primitives_list, constraints_json_list, geo_threshold_mm
        )
        response['DistortingConstraints'] = distorting_indices
        if geo_error:
            print("[run_solver_tool] geo_error: ", geo_error)

        # reflect distortion to solver status
        if response['Status'] == "Converged" and distorting_indices:
            response['Status'] = 'Distorted'

        # --- Diagnostic Warnings (Diagonal Distance, Syntax Error) ---
        warnings = static_constraints_diagnosis(primitives_list, constraints_json_list)
        if warnings:
            response["Warnings"] = warnings

    except Exception:
        response = {'Error': traceback.format_exc()}

    finally:
        # --- Image Generation for Feedback ---
        if doc and viz_image_path:
            try:
                # Export sketch data with constraints
                sketch_data_with = export_sketch_to_dict(fc_sketch)

                # 1. Generate "With Constraints" (Right)
                img_with = visualize_sketch_for_vlm(
                    sketch_data_with,
                    save_path=None,
                    title="With Constraints",
                    title_fontsize=12,
                    icon_fontsize=10,
                    figsize=(640, 640),
                )

                # 2. Generate "Primitives Only" (Left)
                if not base_sketch_data:
                    raise ValueError("base_sketch_data is None")
                img_base = visualize_sketch_for_vlm(
                    base_sketch_data,
                    save_path=None,
                    title="Primitives Only",
                    title_fontsize=12,
                    icon_fontsize=10,
                    figsize=(640, 640),
                )

                # 3. Stitch Images
                if img_base and img_with:
                    # Canvas: Double Width
                    total_width = img_base.width + img_with.width
                    max_height = max(img_base.height, img_with.height)

                    img_concat = PIL.Image.new("RGB", (total_width, max_height), color='white')

                    # Layout: [Primitives Only] [With Constraints]
                    img_concat.paste(img_base, (0, 0))
                    img_concat.paste(img_with, (img_base.width, 0))

                    img_concat.save(viz_image_path)

            except Exception as e:
                print(f"Failed to generate feedback image: {e}")

        try:
            FreeCAD.closeDocument(doc.Name) if doc else None
        except:
            pass

    return json.dumps(response, indent=2)


def oscillate_sketch(primitives: list[dict[str, Any]], constraints_json_list: list, noise_std: float = 1.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    doc = None
    oscillated_primitives_final = []
    try:
        # Create temporary document
        doc = FreeCAD.newDocument("SolverToolDoc")
        body = doc.addObject('PartDesign::Body', 'Body')
        fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
        xy_plane = body.Origin.OriginFeatures[3]
        fc_sketch.AttachmentSupport = xy_plane
        fc_sketch.MapMode = 'FlatFace'

        # Add primitives and constraints
        oscillated_primitives = oscillate_primitives(primitives, noise_std=noise_std)
        add_primitives_to_sketch(oscillated_primitives, fc_sketch)
        constraint_mapping, failed_indices, _ = add_constraints_to_sketch(oscillated_primitives, constraints_json_list, fc_sketch)
        doc.recompute()
        status = fc_sketch.getSolverStatus()

        # extract primitives from the oscillated sketch
        history_commands = dump_freecad_commands(fc_sketch)
        oscillated_primitives_final, _ = parse_freecad_script(history_commands)

    except Exception:
        status = {'Status': 'Error', 'Error': traceback.format_exc()}

    finally:
        if doc:
            try:
                FreeCAD.closeDocument(doc.Name)
            except:
                pass

    return oscillated_primitives_final, status


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
            constraint_mapping, failed_indices, _ = add_constraints_to_sketch(primitives, current_constraints, fc_sketch)
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

            # Malformed: remove all (these are syntax errors)
            malformed_orig_indices = set()
            for idx in failed_indices:
                if idx >= prefix_size:
                    malformed_orig_indices.add(idx)

            for fc_idx in malformed_fc:
                orig_idx = fc_to_orig.get(fc_idx)
                if orig_idx is not None and orig_idx >= prefix_size:
                    malformed_orig_indices.add(orig_idx)

            # Conflicting: collect all, but will remove only last one
            conflicting_orig_indices = set()
            for fc_idx in conflicting_fc:
                orig_idx = fc_to_orig.get(fc_idx)
                if orig_idx is not None and orig_idx >= prefix_size:
                    conflicting_orig_indices.add(orig_idx)

            # Redundant: collect all, but will remove only last one
            redundant_orig_indices = set()
            for fc_idx in redundant_fc:
                orig_idx = fc_to_orig.get(fc_idx)
                if orig_idx is not None and orig_idx >= prefix_size:
                    redundant_orig_indices.add(orig_idx)

            # Build final result
            final_result = VerificationResult(
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



# ============================================================
# Main Workflow
# ============================================================

def prepare_prompts(
    primitives_list: list,
    constraints_list: list,
    image_path_list: str | list[str],
    max_turns: int,
    geo_threshold_mm: float = 10.0,
):

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

    constraints_list_value_stripped = []
    for constraint in constraints_list:
        constraint_copy = constraint.copy()
        constraint_copy.pop("value", None)
        constraints_list_value_stripped.append(constraint_copy)
    constraints_text = json.dumps(constraints_list_value_stripped, indent=2)

    # Run solver on partial constraints to get initial status
    initial_solver_output = run_solver_tool(primitives_list, constraints_list, geo_threshold_mm)

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
* Note: mid_x/y for Arc is a point *on the arc*, not the center or the mid-coord point.
The primitives are implicitly indexed starting from 0 (0-indexed) based on their order in the input list.
The partial constraints format is described in the following section.

[Outputs]
Output the **complete** list of constraints, which consists of:
1. The given partial constraints (copy them exactly as provided)
2. Additional constraints you infer from the visual arrangement, geometric properties, and the design intent.

General Argument Structure follows the pattern below:
{Type, Entity1, Subnode1, Entity2, Subnode2, Entity3, Subnode3}

- Type: The type of the constraint (e.g., "Coincident", "Horizontal", "Parallel", "Tangent", etc. See [Constraint Types and Signatures] below).
- EntityN: The 0-based index of the primitive.
- SubnodeN: One of the following numbers, specifying which part of the primitive is being referenced.
    - 0: The Entity itself (e.g., the edge of a line or circle)
    - 1: StartPoint (Vertex)
    - 2: EndPoint (Vertex)
    - 3: CenterPoint (Strictly the center of a Circle or Arc, NOT the midpoint of a line or an arc)

NOTE: Depending on the Type, some arguments may be omitted. For example, Horizontal constraint only requires Entity1.
NOTE: For "Point" primitives, always use subnode=1 instead of 0.

[Constraint Types and Signatures]
Strictly adhere to the following signatures for each constraint type. Arguments not listed for a type should be omitted.

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
- Perpendicular(Entity1, Subnode1, Entity2, Subnode2)
    * Perpendicularity between two curves.
    * If Subnode1 = Subnode2 = 0, the the two curves intersect at a right angle, abbreviate as Perpendicular(Entity1, Entity2).
    * If Subnode1 > 0, Entity1 and Entity2 are perpendicular at Entity1.Subnode1.
    * If Subnode1 > 0 and Subnode2 > 0, then it is equivalent to Perpendicular(Entity1, Entity2) and Coincident(Entity1, Subnode1, Entity2, Subnode2).
    * WARNING: Subnode1 = 0 and Subnode2 > 0 are regarded malformed.
- Tangent(Entity1, Subnode1, Entity2, Subnode2)
    * Tangency between two curves.
    * If Subnode1 = Subnode2 = 0, the the two curves are tangent to each other, abbreviate as Tangent(Entity1, Entity2).
    * If Subnode1 > 0, Entity1 and Entity2 are tangent at Entity1.Subnode1.
    * If Subnode1 > 0 and Subnode2 > 0, then it is equivalent to Tangent(Entity1, Entity2) and Coincident(Entity1, Subnode1, Entity2, Subnode2).
    * WARNING: Subnode1 = 0 and Subnode2 > 0 are regarded malformed.
    * TIPS: If Entity1 and Entity2 are both Lines, then this achieves Collinear.
- Equal(Entity1, Entity2)
    * Constrains two segments to have equal length, or two circles/arcs to have equal radius.
- Symmetric(Entity1, Subnode1, Entity2, Subnode2, Entity3, Subnode3)
    * Constrains Entity1.Subnode1 and Entity2.Subnode2 to be symmetric with respect to Entity3.Subnode3.
    * Entity3 can be a Point (symmetry point) or a Line (symmetry axis).
    * Examples:
        - Two points and a symmetry line (you can omit Subnode3 = 0 in this case): Symmetric(Line1, PointOfLine1, Line2, PointOfLine2, SymmetryLine)
        - Two points and a symmetry point: Symmetric(Line1, PointOfLine1, Line2, PointOfLine2, LineS, PointOfLineS)
        - A line and a symmetry point: Symmetric(Line, 1, Line, 2, LineS, PointOfLineS)
    * NOTE: Entity1.Subnode1 and Entity2.Subnode2 must be points, so Subnode1/2 must be >= 1.

2. Dimensional Constraints (Use only when necessary for fully constraining the sketch):
- Distance(Entity1, Subnode1, Entity2, Subnode2)
    * Examples:
        - Fixes distance between two points: Distance(Entity1, Subnode1, Entity2, Subnode2)
        - Fixes distance between a point and a line (you can omit Subnode2 = 0 in this case): Distance(Entity1, Subnode1, Line2)
        - Fixes length of a line: Distance(Line, 1, Line, 2), or Distance(Line) for shorthand notation.
    * NOTE: You CANNOT define the distance between two lines: Distance(Line1, 0, Line2, 0) is judged as malformed.
- DistanceX(Entity1, Subnode1, Entity2, Subnode2)
    * Fixes the HORIZONTAL (X-axis) distance between two points. This constrains only the X-component of the displacement.
    * Examples:
        - Horizontal distance between two points: DistanceX(Entity1, Subnode1, Entity2, Subnode2)
        - Horizontal span of a line: DistanceX(Line, 1, Line, 2), or DistanceX(Line) for shorthand.
    * PREFERRED over diagonal Distance constraints when constraining horizontal positioning.
    * NOTE: You CANNOT define the horizontal distance between a point and a line: Either Subnode1=0 or Subnode2=0 is judged as malformed.
- DistanceY(Entity1, Subnode1, Entity2, Subnode2)
    * Fixes the VERTICAL (Y-axis) distance between two points. This constrains only the Y-component of the displacement.
    * Examples:
        - Vertical distance between two points: DistanceY(Entity1, Subnode1, Entity2, Subnode2)
        - Vertical span of a line: DistanceY(Line, 1, Line, 2), or DistanceY(Line) for shorthand.
    * PREFERRED over diagonal Distance constraints when constraining vertical positioning.
    * NOTE: You CANNOT define the vertical distance between a point and a line: Either Subnode1=0 or Subnode2=0 is judged as malformed.
- Diameter(Entity1)
    * Fixes diameter of Circle or Arc.
- Radius(Entity1)
    * Fixes radius of Circle or Arc.
- Angle(Entity1)
    * Fixes slope angle of a LineSegment or angular span of an Arc (Subnode1 should be 0).
- Angle(Entity1, Entity2)
    * Fixes the angle between two lines (Subnode1 and Subnode2 are ignored).

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
  'DoF': int,                          # Current Degrees of Freedom
  'Status': str,                       # "Converged", "Conflicting", "Redundant", "Failed", etc.
  'ConflictingConstraints': [int...],  # List of indices causing conflicts (0-based)
  'RedundantConstraints': [int...],    # List of indices that are redundant (0-based)
  'ConflictGroups': [[int...], ...],   # Groups of constraints causing issues (0-based)
  'MalformedConstraints': [int...],    # Malformed constraints (0-based)
  'MalformedConstraints': [int...],    # Malformed constraints (0-based)
  'DistortingConstraints': [int...],   # List of indices that are valid but cause severe geometry distortion (0-based).
  'Warnings': [str...],                # Other recommendations to note, e.g., replace diagonally defined distance constraints with horizontal/vertical ones.
  }
}
```

The desirable solver status is DoF=2, Status=Converged, no Conflicting/Redundant/Malformed/Distorting constraints.
- If the solver reports 'Status' as 'Failed', it means that the solver failed to converge, indicating that some constraints are geometrically unachievable. As the problem identification is usually very difficult, revert to a point where the solver reports Converged.
- If the solver reports 'RedundantConstraints', those constraints are deduced from other constraints. Try removing them (prioritize keeping the given partial constraints).
- If the solver reports 'ConflictingConstraints', there is a contradiction among constraints shown there. Analyze the logic and remove or modify the incorrect constraints.
- If the solver reports 'MalformedConstraints', those constraints are syntactically incorrect (e.g., missing parameters or referring to non-existent entities). Remove the constraints or modify their syntax.
- If the solver reports 'DistortingConstraints', these constraints are causing the geometry to distort from the original position. You MUST remove or revise these constraints.
- The criticalness is Malformed > Conflicting > Distorting > Redundant.
- The solver output indices are 0-based and correspond to the order in your <solver_call> list.

Along with the input primitives and partial constraints, we also provide the initial solver output for your reference.
In addition, we provide the 2D sketch image with your inferred constraints applied at every turn. Refer to it for understanding geometric relationships and your solution's integrity.

IMPORTANT: If the output doesn't conform to the JSON format, we return an error message.
Immediately fix the error based on the message and re-submit your answer.

IMPORTANT: Even if you achieve DoF=2, we reject your solution if one or more constraints are judged as malformed, conflicting, or redundant.
Therefore, prioritize no malformed, conflicting, or redundant status. Upon that, focus on reducing DoF.
If you see a lot of conflicting constraints, first identify the ones that most frequently appear in the ConflictGroups, and try deleting it.
If you still don't see any improvements, it may be recommended that you start over from the initial constraints.
Also, solver status can be 'Failed' without any invalid constraints or geometric deviation. This is a solver's limitation and it is extremely hard to recover from here. Consider reverting to a previous 'Converged' state.

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
  "DistortingConstraints": []
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
  "DistortingConstraints": []
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
  "DistortingConstraints": []
}

# Example 3: Valid Constraints
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
  "DistortingConstraints": []
}
```

[IMPORTANT RULES]
1. Identify geometric relationships based on the visual appearance in the image and the provided primitive IDs.
2. Achieve fully constrained (DoF=2) CAD sketches with constraints maintaining the design intent.
    a. Try to achieve it with fewer constraints.
    b. Always prioritize relational constraints over dimensional constraints.
    c. When you have to use dimensional constraints, follow these rules:
        i. Avoid using angles to the maximum extent possible unless you think appropriate.
        ii. When defining Distance constraints, prioritize HORIZONTAL and VERTICAL directions, and avoid diagonal or slanted directions.
        iii. Try to define horizontal or vertical distances ***incrementally without overlaps***. For example, A---B and B---C, rather than A---B and A---C.
        iv. In concave sketches, avoid defining distance constraints across gaps; instead, define them between primitives within the sketch geometry.
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
    user_prompt_text = f"""Complete the constraint list by adding missing constraints to achieve DoF=2.
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


# ============================================================
# Main Execution
# ============================================================

def main(args):
    # Initialize LLM
    if "Qwen3VL" in os.path.basename(args.qwen_model_path):
        llm = Llama(
            model_path=args.qwen_model_path,
            chat_handler=Qwen3VLChatHandler(
                clip_model_path=args.qwen_mmproj_path,
                force_reasoning=True,
                image_min_tokens=1024,
                verbose=False,
            ),
            n_gpu_layers=-1,
            n_ctx=81920, #65536,
            swa_full=True,
            verbose=False,
        )
    elif "Qwen3.5" in os.path.basename(args.qwen_model_path):
        llm = Llama(
            model_path=args.qwen_model_path,
            chat_handler=Qwen35ChatHandler(
                clip_model_path=args.qwen_mmproj_path,
                image_min_tokens=1024,
                verbose=False,
            ),
            n_gpu_layers=-1,
            n_ctx=81920, #65536,
            swa_full=True,
            verbose=False,
        )
    else:
        raise NotImplementedError(f"Unhandled model: {args.model_path}")
    llm.reset()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/images", exist_ok=True)
    os.makedirs(args.output_dir + "/commands", exist_ok=True)

    error_log = os.path.join(args.output_dir, "error_log.txt")

    # initialize a converter
    context = SketchContext()  # NOTE: No context attributes are used, so arguments can be blank
    converter = SketchFreecadConverter(context, raise_on_error=False)

    # load data
    seq_data = flat_array.load_dictionary_flat(args.sequence_path)
    sequences = seq_data['sequences']

    for i in range(args.start_idx, args.start_idx + args.process_num):
        # output file paths
        output_history_original = f"{args.output_dir}/commands/{i:07d}_original.py"
        output_history_refined = f"{args.output_dir}/commands/{i:07d}_refined.py"
        output_image_path_original = f"{args.output_dir}/images/{i:07d}_original.png"
        output_image_path_refined = f"{args.output_dir}/images/{i:07d}_original_refined.png"

        # Skip if refined file already exists (resume support)
        if os.path.isfile(output_history_refined):
            print(f"=== Skipping sample {i} (refined already exists) ===")
            continue
        print(f"\n\n====== Processing sample {i} ======")

        # load and convert original data to commands
        seq = sequences[i]
        sketch = datalib.sketch_from_sequence(seq)
        try:
            _ = normalize_sketch(sketch, dst_length=1000)
            sketch = cleanse_sketch_primitives(sketch)
            sketch = cleanse_sketch_constraints(sketch)
            sketch, _ = complete_constraint_values(sketch)
        except Exception as e:
            print(f"Error processing sequence {i}: {traceback.format_exc()}")
            with open(error_log, "a") as f:
                f.write(f"Error processing sequence {i}: {traceback.format_exc()}\n")
            continue

        # ======== GT PROCESSING ========

        # Check the solver status
        doc = None
        try:
            doc, fc_sketch, _, _, _ = converter.to_freecad(sketch)
            initial_status = fc_sketch.getSolverStatus()
            while initial_status['Status'] == "Redundant":
                for cid in sorted(initial_status['RedundantConstraints'], reverse=True):
                    fc_sketch.delConstraint(cid)
                doc.recompute()
                initial_status = fc_sketch.getSolverStatus()

            if initial_status['Status'] != "Converged" or len(initial_status['ConflictingConstraints']) > 0 or len(initial_status['MalformedConstraints']) > 0:
                print(f"Initial status is not Converged: {initial_status}")
                with open(error_log, "a") as f:
                    f.write(f"Initial status is not Converged for sequence {i}: {initial_status}\n")
                continue

            history_commands = dump_freecad_commands(fc_sketch)
            sample_primitives, sample_constraints = parse_freecad_script(history_commands)

            if not (len(sample_primitives) == fc_sketch.GeometryCount and len(sample_constraints) == fc_sketch.ConstraintCount):
                error_msg = f"Number of constraints does not match with sketch {i}:\n\n"
                error_msg += "\n".join(history_commands) + "\n\n"
                for p in sample_primitives:
                    error_msg += f"{p}\n"
                for c in sample_constraints:
                    error_msg += f"{c}\n"
                raise ValueError(error_msg)

        except Exception as e:
            print(f"Error processing sequence {i}: {traceback.format_exc()}")
            with open(error_log, "a") as f:
                f.write(f"Error processing sequence {i}: {traceback.format_exc()}\n")
            continue
        finally:
            FreeCAD.closeDocument(doc.Name) if doc else None

        # Export GT commands
        error = export_to_freecad_script(sample_primitives, sample_constraints, output_history_original)
        if error:
            print(f"Error exporting GT commands: {error}")
            with open(error_log, "a") as f:
                f.write(f"Error exporting GT commands for sample {i}:\n{error}\n")
            continue

        # Export GT sketch image
        sketch_data = load_and_extract_from_file(output_history_original)
        if sketch_data is None:
            print(f"Error loading GT sketch for sample {i}")
            with open(error_log, "a") as f:
                f.write(f"Error loading GT sketch for sample {i}\n")
            continue
        visualize_sketch_for_vlm(
            sketch_data,
            save_path=output_image_path_original,
            title=None,
            icon_fontsize=10,
            figsize=(640, 640),
        )

        # Sanity check
        sample_verification = verify_constraints(sample_primitives, sample_constraints, args.geo_threshold)
        sample_summary = sample_verification.summary()
        if not sample_verification.is_valid():
            print(f"GT INVALID: Issues detected in sample {i}: {sample_summary}")
            comment_out_file(output_history_original)
            continue

        # ======== VLM PROCESSING ========

        predicted_constraints_list = []
        solver_call_count = 0

        if sample_verification.dof == 2:
            print("\n--- GT DoF is already 2. Skipping Qwen3 Generation. ---")
            predicted_constraints_list.append(sample_constraints)
            turn_range = []
        else:
            print("\n--- Qwen3 Generation (Interactive) ---")
            messages = prepare_prompts(sample_primitives, sample_constraints, output_image_path_original, args.max_turns, args.geo_threshold)
            print(messages[0]["content"])
            for msg in messages[1]["content"]:
                if "text" in msg:
                    print(msg["text"])
            turn_range = range(args.max_turns)

        for solver_call_count in turn_range:

            # Notify remaining turns
            remaining_times_notification = f"Turn {solver_call_count}: You can attempt submission up to {args.max_turns - solver_call_count} more times."
            messages.append({"role": "user", "content": remaining_times_notification})
            print("User: " + remaining_times_notification + "\n")

            # Run VLM
            try:
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=None,  # 65536
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

            # Define image path at start
            FEEDBACK_IMAGE_PATH = f"{args.output_dir}/images/{i:07d}_feedback_{solver_call_count:02d}.png"

            # 1. Check for <final> tag
            final_match = re.search(r"<final>\n(.*?)\n</final>", current_turn_text, re.DOTALL)
            if final_match:
                predicted_constraints, parse_error = parse_qwen_output_json(final_match.group(1))
                predicted_constraints_list.append(predicted_constraints)
                if parse_error:
                    solver_output = json.dumps({"DoF": "null", "Error": f"JSON Parse Error: {parse_error}. Fix the syntax and try again."}, indent=2)
                    messages.append({"role": "user", "content": solver_output})
                    print("User: " + solver_output + "\n")
                    continue
                else:
                    # Run solver to provide feedback for the final answer (to terminal)
                    solver_output = run_solver_tool(sample_primitives, predicted_constraints, args.geo_threshold, FEEDBACK_IMAGE_PATH)
                    print("User: " + solver_output + "\n")
                    if os.path.isfile(FEEDBACK_IMAGE_PATH):
                        print(f"[Visual Feedback] Image appended: {FEEDBACK_IMAGE_PATH}\n")
                    break

            # 2. Check for </think> and extract JSON
            parts = current_turn_text.split("</think>")
            content_to_parse = parts[-1] if len(parts) > 1 else current_turn_text
            json_match = re.search(r'\[[\s\S]*?\]', content_to_parse)
            if json_match:
                predicted_constraints, parse_error = parse_qwen_output_json(json_match.group(0))
                predicted_constraints_list.append(predicted_constraints)



                # Dump intermediate results
                output_history_intermediate = f"{args.output_dir}/commands/{i:07d}_round{solver_call_count}.py"
                error = export_to_freecad_script(sample_primitives, predicted_constraints, output_history_intermediate)
                if error:
                    print(f"Error exporting intermediate results: {error}")
                    with open(error_log, "a") as f:
                        f.write(f"Error exporting intermediate results for sample {i} turn {solver_call_count}:\n{error}\n")


                if parse_error:
                    solver_output = json.dumps({"DoF": "null", "Error": parse_error}, indent=2)
                    FEEDBACK_IMAGE_PATH = None
                else:
                    # Run solver with predicted constraints
                    solver_output = run_solver_tool(sample_primitives, predicted_constraints, args.geo_threshold, FEEDBACK_IMAGE_PATH)
            else:
                solver_output = json.dumps({"DoF": "null", "Error": "You MUST output the JSON array of constraints."}, indent=2)
                FEEDBACK_IMAGE_PATH = None

            # Check for solver error (Top-level or GeometryValidation) and log/truncate
            # Note: We expect run_solver_tool to ALWAYS return valid JSON. If it fails (JSONDecodeError),
            # it indicates an implementation bug in run_solver_tool, so we let it raise.
            solver_resp = json.loads(solver_output)




            # if solver_resp.get("MalformedConstraints", []):
            #     assert solver_resp["Status"] == "Malformed",f"{i=}: {solver_resp}"
            # elif solver_resp.get("ConflictingConstraints", []):
            #     assert solver_resp["Status"] == "Conflicting", f"{i=}: {solver_resp}"
            # elif solver_resp.get("RedundantConstraints", []):
            #     assert solver_resp["Status"] == "Redundant", f"{i=}: {solver_resp}"





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

            # Construct Visual Feedback Message
            user_content = []

            if FEEDBACK_IMAGE_PATH and os.path.isfile(FEEDBACK_IMAGE_PATH):
                try:
                    with open(FEEDBACK_IMAGE_PATH, "rb") as img_f:
                        feedback_b64 = base64.b64encode(img_f.read()).decode("utf-8")
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{feedback_b64}"}})
                    solver_output += "\n\nAttached is a sketch with your constraints applied. Identify and fix any invalid constraints from it if any, then find missing constraints."
                except Exception as e:
                    print(f"Error encoding feedback image: {e}")

            # # Show an oscillation result to help the model find missing constraints easily
            # if solver_resp.get("Status", "") in ["Converged"]:
            #     oscillated_primitives, oscillated_status = oscillate_sketch(sample_primitives, predicted_constraints, noise_std=1.0)
            #     if oscillated_status.get("Status", "") in ["Converged"]:
            #         solver_output += "\n\n[HINT] Below are the before-and-after results obtained by slightly perturbing the primitive coordinates and applying the predicted constraints. Regions where the original geometric relations are broken indicate missing constraints."
            #         solver_output += "\nBefore:\n"
            #         solver_output += json.dumps(sample_primitives, indent=2)
            #         solver_output += "\nAfter:\n"
            #         solver_output += json.dumps(oscillated_primitives, indent=2)

            user_content.append({"type": "text", "text": solver_output})
            messages.append({"role": "user", "content": user_content})

            print("User: " + solver_output + "\n")
            if FEEDBACK_IMAGE_PATH:
                print(f"[Visual Feedback] Image appended: {FEEDBACK_IMAGE_PATH}\n")

        # Select the best solution so far
        if len(predicted_constraints_list) == 0:
            print("ERROR: No predicted constraints found.")
            with open(error_log, "a") as f:
                f.write(f"Error: No predicted constraints found for sequence {i}\n")
            continue

        valid_candidates = []

        # Iterate through all predictions
        for pred_validation_version_id, predicted_constraints in enumerate(predicted_constraints_list):
            print(f"\n~~~~~~~~ Qwen3 Output Version {pred_validation_version_id} ~~~~~~~~")

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

            # Attempt to repair if invalid ONLY due to distortion
            if not predicted_verification.is_valid() and predicted_verification.distorting_count > 0:
                # Filter out distorting constraints
                print(f"  ⚠ Distortion detected. Checking indices for repair...")
                distorting_indices, _ = validate_geometry_preservation(sample_primitives, predicted_constraints_cleaned, args.geo_threshold)
                indices_to_remove = set(distorting_indices)
                repaired_constraints = [c for idx, c in enumerate(predicted_constraints_cleaned) if idx not in indices_to_remove]

                print(f"  ⚠ Attempting repair: Removing {len(indices_to_remove)} distorting constraints...")
                repaired_verification = verify_constraints(sample_primitives, repaired_constraints, args.geo_threshold)

                if repaired_verification.is_valid():
                    print(f"  ✓ REPAIRED VALID: {repaired_verification.summary()}")
                    valid_candidates.append({
                        'dof': repaired_verification.dof,
                        'constraints': repaired_constraints,
                        'version': pred_validation_version_id,
                        'verification': repaired_verification,
                    })

            if predicted_verification.is_valid():
                print("  ✓ VALID: Constraints are well-formed and geometry preserved")
                valid_candidates.append({
                    'dof': predicted_verification.dof,
                    'constraints': predicted_constraints_cleaned,
                    'version': pred_validation_version_id,
                    'verification': predicted_verification
                })
            else:
                print("  ✗ INVALID: Issues detected")
                if predicted_verification.error:
                    print(f"  Error: {predicted_verification.error}")

        if valid_candidates:
            # Sort by DoF (ascending), then by version (descending) to prefer later refined versions if DoF is same
            valid_candidates.sort(key=lambda x: (x['dof'], -x['version']))
            best_candidate = valid_candidates[0]
            print(f"\n★ SELECTED BEST CANDIDATE: Version {best_candidate['version']} with DoF={best_candidate['dof']}")
            error = export_to_freecad_script(sample_primitives, best_candidate['constraints'], output_history_refined)
            if error:
                print(f"Error exporting refined sketch data: {error}")
                with open(error_log, "a") as f:
                    f.write(f"Error exporting refined sketch data for sample {i}:\n{error}\n")
                continue

            # Update variables for logging
            predicted_verification = best_candidate['verification']
            predicted_summary = predicted_verification.summary()
        else:
            print("\n✗ NO VALID CANDIDATES FOUND. Exporting last prediction (commented out).")
            # Take the last prediction, and comment out the output file
            predicted_constraints = predicted_constraints_list[-1]
            predicted_constraints_cleaned, _ = cleanup_constraints_from_lists(sample_primitives, predicted_constraints)
            error = export_to_freecad_script(sample_primitives, predicted_constraints_cleaned, output_history_refined)
            if error:
                print(f"Error exporting refined sketch data: {error}")
                with open(error_log, "a") as f:
                    f.write(f"Error exporting refined sketch data for sample {i}:\n{error}\n")
                continue
            comment_out_file(output_history_refined)
            predicted_verification = verify_constraints(sample_primitives, predicted_constraints_cleaned, args.geo_threshold)
            predicted_verification.dof_decreased = predicted_verification.dof <= sample_verification.dof
            predicted_summary = predicted_verification.summary()

        # Log verification result (GT vs Qwen3 comparison)
        with open(os.path.join(args.output_dir, f"verification_log_{args.start_idx}_{args.start_idx + args.process_num}.txt"), "a") as f:
            valid_str = "PASS" if predicted_verification.is_valid() else "FAIL"
            f.write(f"{i:07d}: [GT] {sample_summary} | [Qwen3] {predicted_summary} | solver_calls={solver_call_count} [{valid_str}]\n")

        # visualize the sketch after refinement
        if not args.skip_viz_comparison:
            sketch_data = load_and_extract_from_file(output_history_refined)
            visualize_sketch_for_vlm(
                sketch_data,
                save_path=output_image_path_refined,
                title="Refined",
                title_fontsize=12,
                icon_fontsize=10,
                figsize=(640, 640),
            )

            # combine
            if os.path.isfile(output_image_path_original) and os.path.isfile(output_image_path_refined):
                original_image = PIL.Image.open(output_image_path_original)
                refined_image = PIL.Image.open(output_image_path_refined)

                max_height = max(original_image.height, refined_image.height)
                total_width = original_image.width + refined_image.width

                combined_image = PIL.Image.new('RGB', (total_width, max_height), (255, 255, 255))
                combined_image.paste(original_image, (0, 0))
                combined_image.paste(refined_image, (original_image.width, 0))
                combined_image.save(output_image_path_refined)
                os.remove(output_image_path_original)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save FC GT and predictions that pass all validation criteria")
    parser.add_argument('--geo_threshold', type=float, default=1.0, help="Geometry validation threshold in mm")
    parser.add_argument('--output_dir', type=str, default='data_qwen3_refined_with_solver', help="Output directory")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index of sequences to process")
    parser.add_argument('--process_num', type=int, default=1000, help="Number of sequences to process")
    parser.add_argument('--max_turns', type=int, default=10, help="Maximum solver calls before accepting answer")
    parser.add_argument('--skip_viz_comparison', action='store_true', help="Skip visual comparison before and after refinement")
    parser.add_argument(
        '--sequence_path',
        type=str,
        default="~/my_works/My-Vitruvion/data_processed/sg_filtered_unique.npy",
        help="Path to the sequence data",
    )
    parser.add_argument(
        '--qwen_model_path',
        type=str,
        default="qwen_weights/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf",
        help="Path to the Qwen3VL or Qwen3.5 model weight",
    )
    parser.add_argument(
        '--qwen_mmproj_path',
        type=str,
        default="qwen_weights/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf",
        help="Path to the Qwen3VL or Qwen3.5 projection module weight",
    )
    args = parser.parse_args()
    args.sequence_path = os.path.expanduser(args.sequence_path)
    main(args)
