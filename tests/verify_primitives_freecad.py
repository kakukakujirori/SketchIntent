from typing import Any
import glob
import math
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sketchgraphs.data as datalib
from src.sketch_tokenization.constants import PrimitiveToken
from src.sketch_tokenization.context import SketchContext
from src.sketch_tokenization.tokenizer import SketchTokenizer
from src.sketch_tokenization.freecad_utils import ensure_freecad_imported

FreeCAD, Part, Sketcher = ensure_freecad_imported()

# ============================================================
# Helper Functions from notebooks/v20260124_qwen3_refine_data_with_solver.py
# ============================================================

def compute_arc_center(start: list[float], mid: list[float], end: list[float]) -> tuple[float, float, float] | None:
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


def extract_primitives_from_sketch(fc_sketch: 'Sketcher.SketchObject') -> list[dict[str, Any]]:
    """
    Extract primitive coordinates from FreeCAD sketch geometry.
    Derives type and construction flag directly from fc_sketch.
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
            continue # Skip unsupported

        new_prim = {"type": ptype}

        # Get construction flag from fc_sketch
        try:
            is_construction = fc_sketch.getConstruction(geo_id)
            if is_construction:
                new_prim["construction"] = True
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

        updated_primitives.append(new_prim)

    return updated_primitives


def compute_centroid_aligned_deviation(
    primitives_before: list[dict[str, Any]], primitives_after: list[dict[str, Any]]
) -> tuple[float, float, int]:
    """
    Compute max deviation between two primitive sets after aligning their centroids.
    Returns (max_coord_deviation, max_length_deviation, max_deviation_primitive_idx)
    """
    def get_all_points(primitives: list[dict[str, Any]]):
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

        elif ptype == "Circle":
            r_b = prim_b.get("radius", 0)
            r_a = prim_a.get("radius", 0)
            len_dev = abs(r_b - r_a)

        elif ptype == "Arc":
            arc_result_b = compute_arc_center(prim_b["start"], prim_b["mid"], prim_b["end"])
            arc_result_a = compute_arc_center(prim_a["start"], prim_a["mid"], prim_a["end"])
            if arc_result_b and arc_result_a:
                r_b = arc_result_b[2]
                r_a = arc_result_a[2]
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


def freecad_to_sketchgraphs(sketch_obj, tokenizer: SketchTokenizer):
    """
    Extracts a datalib.Sketch object from a FreeCAD SketchObject using registered handlers.
    """
    entities = {}

    # Iterate over geometry in the FreeCAD sketch
    for i, geo in enumerate(sketch_obj.Geometry):
        geo_type = type(geo).__name__
        handler = None

        # Mapping FreeCAD types to our primitive names
        if isinstance(geo, Part.LineSegment):
            handler = tokenizer.context.get_primitive_handler("Line")
        elif isinstance(geo, Part.Circle):
            handler = tokenizer.context.get_primitive_handler("Circle")
        elif isinstance(geo, Part.ArcOfCircle):
            handler = tokenizer.context.get_primitive_handler("Arc")
        elif isinstance(geo, Part.Point):
            handler = tokenizer.context.get_primitive_handler("Point")

        if handler:
            # from_freecad returns a datalib Entity
            try:
                # We use string ID based on index
                entity = handler.from_freecad(geo, id=str(i))
                if entity:
                    entities[entity.entityId] = entity
            except Exception as e:
                print(f"  Error converting geometry {i} ({geo_type}): {e}")
        else:
            print(f"  Warning: No handler for geometry type {geo_type}")

    # For this verification, we are ignoring constraints as per user request
    return datalib.Sketch(entities=entities)


def sketchgraphs_to_freecad(sg_sketch: datalib.Sketch, tokenizer: SketchTokenizer, doc_name="ReconDoc"):
    """
    Converts a datalib.Sketch back to a FreeCAD SketchObject.
    Creates a new document and sketch for this purpose.
    """
    doc = FreeCAD.newDocument(doc_name)
    sketch_obj = doc.addObject("Sketcher::SketchObject", "Sketch")

    sorted_entities = sorted(sg_sketch.entities.values(), key=lambda e: int(e.entityId) if e.entityId.isdigit() else e.entityId)

    for entity in sorted_entities:
        handler = None
        if isinstance(entity, datalib.Line):
            handler = tokenizer.context.get_primitive_handler("Line")
        elif isinstance(entity, datalib.Circle):
            handler = tokenizer.context.get_primitive_handler("Circle")
        elif isinstance(entity, datalib.Arc):
            handler = tokenizer.context.get_primitive_handler("Arc")
        elif isinstance(entity, datalib.Point):
            handler = tokenizer.context.get_primitive_handler("Point")

        if handler:
            try:
                handler.to_freecad(entity, sketch_obj)
            except Exception as e:
                print(f"  Error creating FreeCAD geometry for {entity}: {e}")
        else:
            print(f"  Warning: No handler for entity {entity}")

    doc.recompute()
    return doc, sketch_obj


def verify_primitives_freecad(directory: str = 'gt_completed_prim5_py', limit: int = 10000):

    print(f"Scanning directory: {directory}...")
    py_files = sorted(glob.glob(os.path.join(directory, "*.py")))

    if not py_files:
        print("No .py files found.")
        return False

    num_files = min(len(py_files), limit)
    print(f"Found {len(py_files)} files. Testing {num_files}...")

    context = SketchContext()
    tokenizer = SketchTokenizer(context)

    passed_files = 0

    for i in range(num_files):
        py_file = py_files[i]

        doc_name_orig = f"VerifyDoc_Orig_{i}"
        doc_name_recon = f"VerifyDoc_Recon_{i}"

        # Cleanup potential previous docs with same name
        if doc_name_orig in FreeCAD.listDocuments(): FreeCAD.closeDocument(doc_name_orig)
        if doc_name_recon in FreeCAD.listDocuments(): FreeCAD.closeDocument(doc_name_recon)

        try:
            # -----------------------------------------------------------------
            # 1. Load Original Script -> FreeCAD (Orig)
            # -----------------------------------------------------------------
            # We intercept newDocument to name it `doc_name_orig`

            # Since we can't easily change the script's `newDocument` call without complex parsing,
            # we'll just run it, find the doc, and rename it or just use it.
            # Most scripts do `doc = FreeCAD.newDocument()`.

            # We'll monitor open docs.
            initial_docs = set(FreeCAD.listDocuments().keys())

            exec_globals = {
                "FreeCAD": FreeCAD,
                "Part": Part,
                "Sketcher": Sketcher,
                "Vector": FreeCAD.Vector
            }
            with open(py_file, 'r') as f:
                script_content = f.read()

            exec(script_content, exec_globals)

            # Find the new document
            current_docs = set(FreeCAD.listDocuments().keys())
            new_docs = list(current_docs - initial_docs)

            if not new_docs:
                # Maybe it used ActiveDocument?
                if FreeCAD.ActiveDocument:
                    doc_orig = FreeCAD.ActiveDocument
                else:
                    print(f"[{i}] {os.path.basename(py_file)}: No document created.")
                    continue
            else:
                doc_orig = FreeCAD.getDocument(new_docs[0])

            # Find sketch in Orig
            sketches_orig = doc_orig.findObjects(Type="Sketcher::SketchObject")
            if not sketches_orig:
                print(f"[{i}] {os.path.basename(py_file)}: No sketch found.")
                FreeCAD.closeDocument(doc_orig.Name)
                continue

            fc_sketch_orig = sketches_orig[0]

            # -----------------------------------------------------------------
            # 2. FreeCAD (Orig) -> SketchGraphs (Orig)
            # -----------------------------------------------------------------
            sg_sketch_orig = freecad_to_sketchgraphs(fc_sketch_orig, tokenizer)

            # -----------------------------------------------------------------
            # 3. SketchGraphs (Orig) -> FreeCAD (Recon)
            # -----------------------------------------------------------------
            doc_recon, fc_sketch_recon = sketchgraphs_to_freecad(sg_sketch_orig, tokenizer, doc_name=doc_name_recon)

            # -----------------------------------------------------------------
            # 4. Compare FreeCAD (Orig) vs FreeCAD (Recon)
            # -----------------------------------------------------------------
            # Extract primitives from both sketches as dicts
            primitives_orig = extract_primitives_from_sketch(fc_sketch_orig)
            primitives_recon = extract_primitives_from_sketch(fc_sketch_recon)

            # Compute deviation
            max_coord_dev, max_len_dev, max_dev_idx = compute_centroid_aligned_deviation(primitives_orig, primitives_recon)

            threshold_mm = 1e-3 # 1 micron tolerance

            if max_coord_dev > threshold_mm or max_len_dev > threshold_mm:
                print(f"[{i}] {os.path.basename(py_file)}: Mismatch (Deviation > {threshold_mm}mm)")
                print(f"  Max Coord Dev: {max_coord_dev:.6f}")
                print(f"  Max Len Dev:   {max_len_dev:.6f}")
                print(f"  Max Dev Idx:   {max_dev_idx}")
            else:
                passed_files += 1

            # Cleanup
            FreeCAD.closeDocument(doc_orig.Name)
            FreeCAD.closeDocument(doc_recon.Name)


        except Exception as e:
            print(f"[{i}] Failed {os.path.basename(py_file)}: {e}")
            import traceback
            traceback.print_exc()
            # Cleanup
            try:
                if 'doc_orig' in locals() and doc_orig: FreeCAD.closeDocument(doc_orig.Name)
                if 'doc_recon' in locals() and doc_recon: FreeCAD.closeDocument(doc_recon.Name)
            except: pass

            continue

    print(f"\nVerification Complete.")
    print(f"Total Processed: {num_files}")
    print(f"Total Passed: {passed_files}")
    pass_rate = (passed_files / num_files * 100) if num_files > 0 else 0
    print(f"Pass Rate: {pass_rate:.2f}%")

    return passed_files == num_files

if __name__ == "__main__":
    if verify_primitives_freecad():
        sys.exit(0)
    else:
        sys.exit(1)
