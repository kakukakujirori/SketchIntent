import argparse
import copy
import csv
import glob
import json
import os
import shutil
import traceback
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.draw_constraints import export_sketch_to_dict, visualize_sketch_for_vlm
from src.models.components.solver_feedback import SolverFeedback
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import cleanse_sketch_constraints, complete_constraint_values, normalize_sketch, _get_entity_bbox, rotate_sketch, normalize_sketch_circumradius
from sketchgraphs.data import EntityType

FreeCAD, _, _ = ensure_freecad_imported()


def get_sketch_hash(sketch, n_bins=5):
    """
    Robust geometric hash (Finalized v3):
    1. Filter construction/points
    2. Centroid-based canonical orientation
    3. Max-dim square grid quantization
    4. Aspect ratio stabilization (<1.3 -> 1.0)
    """
    all_pts = []
    primitive_types = []
    # Filter out construction
    to_delete_ids = [eid for eid, ent in sketch.entities.items()
                     if getattr(ent, 'is_construction', False) or ent.type == EntityType.Point]

    for eid, ent in sketch.entities.items():
        if eid in to_delete_ids: continue
        primitive_types.append(ent.type.name)
        if ent.type == EntityType.Line:
            p, d = np.array([ent.pntX, ent.pntY]), np.array([ent.dirX, ent.dirY])
            all_pts.extend([p + ent.startParam * d, p + ent.endParam * d])
        elif ent.type in (EntityType.Circle, EntityType.Arc):
            all_pts.append(np.array([ent.xCenter, ent.yCenter]))
            r = ent.radius
            all_pts.append(np.array([ent.xCenter + r, ent.yCenter]))
            if ent.type == EntityType.Arc:
                all_pts.append(np.array([ent.xCenter + r*np.cos(ent.startAngle), ent.yCenter + r*np.sin(ent.startAngle)]))
                all_pts.append(np.array([ent.xCenter + r*np.cos(ent.endAngle), ent.yCenter + r*np.sin(ent.endAngle)]))

    if not all_pts: return None
    all_pts = np.array(all_pts)

    # 1. Translation Invariance (Centroid)
    centroid = np.mean(all_pts, axis=0)
    all_pts -= centroid

    # 2. Rotation Invariant Aspect Ratio using PCA
    if len(all_pts) > 1:
        cov = np.cov(all_pts.T)
        eigvals = np.linalg.eigvals(cov)
        raw_aspect = np.sqrt(max(eigvals) / max(1e-9, min(eigvals)))
    else:
        raw_aspect = 1.0

    # Stabilize nearly-regular polygons (hexagons, etc. -> 1.0)
    aspect = 1.0 if raw_aspect < 1.3 else round(raw_aspect, 1)

    # 3. Canonical Orientation (all vertices)
    possible_hashes = []
    for idx in range(len(all_pts)):
        angle = np.arctan2(all_pts[idx, 1], all_pts[idx, 0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)

        rot_pts = []
        for p in all_pts:
            rx = p[0] * cos_a - p[1] * sin_a
            ry = p[0] * sin_a + p[1] * cos_a
            rot_pts.append(np.array([rx, ry]))
        rot_pts = np.array(rot_pts)

        # Scale isotropically so that max dimension is 1.0
        min_p, max_p = np.min(rot_pts, axis=0), np.max(rot_pts, axis=0)
        W, H = max_p[0] - min_p[0], max_p[1] - min_p[1]
        max_dim = max(1e-6, W, H)
        rot_pts /= max_dim

        # Re-center perfectly in the [-0.5, 0.5] grid
        min_p, max_p = np.min(rot_pts, axis=0), np.max(rot_pts, axis=0)
        rot_pts -= (min_p + max_p) / 2.0

        q_pts = []
        for p in rot_pts:
            qx = int(round((p[0] + 0.5) * (n_bins - 1)))
            qy = int(round((p[1] + 0.5) * (n_bins - 1)))
            q_pts.append((qx, qy))

        possible_hashes.append(tuple(sorted(list(set(q_pts)))))

    canonical_pts = min(possible_hashes)
    type_counts = tuple(sorted(Counter(primitive_types).items()))
    return ((canonical_pts, aspect), type_counts)


def extract_sketches(json_path: str) -> list[dict]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities = data["entities"]
    sketches =[]

    for entity_id, entity_data in entities.items():
        if entity_data["type"] == "Sketch":
            sketch_info = {
                "uuid": entity_id,
                "name": entity_data.get("name", "Sketch"),
                "points": entity_data.get("points", {}),
                "curves": entity_data.get("curves", {}),
                "constraints": entity_data.get("constraints", {}) ,
                "dimensions": entity_data.get("dimensions", {}),
            }
            sketches.append(sketch_info)
        else:
            pass

    return sketches


def generate_freecad_script(sketch: dict[str, Any], remove_constraints: bool = False) -> list[str]:
    sketch_name = sketch["name"]
    script =[
        "import math",
        "import FreeCAD",
        "import Part",
        "import Sketcher",
        "",
        "doc = FreeCAD.newDocument('SketchDoc')",
        "body = doc.addObject('PartDesign::Body', 'Body')",
        f"fc_sketch = body.newObject('Sketcher::SketchObject', '{sketch_name}')",
        "xy_plane = body.Origin.OriginFeatures[3]",
        "fc_sketch.AttachmentSupport = xy_plane",
        "fc_sketch.MapMode = 'FlatFace'",
        "",
    ]

    def get_pt(pt_uuid: str):
        pt = sketch["points"][pt_uuid]
        return f"FreeCAD.Vector({pt['x']}, {pt['y']}, 0)"

    uuid_to_fcid = {}
    fcid_to_uuid = {}
    point_connections = defaultdict(list)
    next_fc_id = 0

    script.append("# --- Add Primitives ---")
    for curve_uuid, curve in sketch["curves"].items():
        c_type = curve["type"]
        is_const = "True" if curve["construction_geom"] else "False"

        if c_type == "SketchLine":
            sp_id = curve["start_point"]
            ep_id = curve["end_point"]
            script.append(f"fc_sketch.addGeometry(Part.LineSegment({get_pt(sp_id)}, {get_pt(ep_id)}), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            point_connections[sp_id].append((curve_uuid, 1))
            point_connections[ep_id].append((curve_uuid, 2))
            next_fc_id += 1

        elif c_type == "SketchCircle":
            cp_id = curve["center_point"]
            r = curve["radius"]
            script.append(f"fc_sketch.addGeometry(Part.Circle({get_pt(cp_id)}, FreeCAD.Vector(0,0,1), {r}), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            point_connections[cp_id].append((curve_uuid, 3))
            next_fc_id += 1

        elif c_type == "SketchArc":
            sp_id = curve["start_point"]
            ep_id = curve["end_point"]
            cp_id = curve["center_point"]
            r = curve["radius"]
            sa = curve["start_angle"]
            ea = curve["end_angle"]
            ref_vec = curve["reference_vector"]

            base_a = np.arctan2(ref_vec['y'], ref_vec['x'])
            diff = ea - sa
            if diff < 0:
                diff += 2 * np.pi
            mid_a = sa + base_a + diff / 2.0

            cp = sketch["points"][cp_id]
            mid_x = cp['x'] + r * np.cos(mid_a)
            mid_y = cp['y'] + r * np.sin(mid_a)

            script.append(f"fc_sketch.addGeometry(Part.ArcOfCircle({get_pt(sp_id)}, FreeCAD.Vector({mid_x}, {mid_y}, 0), {get_pt(ep_id)}), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            point_connections[sp_id].append((curve_uuid, 1))
            point_connections[ep_id].append((curve_uuid, 2))
            point_connections[cp_id].append((curve_uuid, 3))
            next_fc_id += 1

        elif c_type == "SketchEllipse":
            cp_id = curve["center_point"]
            major_r = curve["major_axis_radius"]
            minor_r = curve["minor_axis_radius"]
            major_axis = curve["major_axis"]

            cp = sketch["points"][cp_id]
            maj_len = np.hypot(major_axis['x'], major_axis['y'])
            maj_ux, maj_uy = major_axis['x'] / maj_len, major_axis['y'] / maj_len
            s1_x = cp['x'] + maj_ux * major_r
            s1_y = cp['y'] + maj_uy * major_r
            s2_x = cp['x'] + (-maj_uy) * minor_r
            s2_y = cp['y'] + maj_ux * minor_r
            script.append(f"fc_sketch.addGeometry(Part.Ellipse(FreeCAD.Vector({s1_x}, {s1_y}, 0), FreeCAD.Vector({s2_x}, {s2_y}, 0), {get_pt(cp_id)}), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            point_connections[cp_id].append((curve_uuid, 3))
            next_fc_id += 1

        elif c_type == "SketchEllipticalArc":
            sp_id = curve["start_point"]
            ep_id = curve["end_point"]
            cp_id = curve["center_point"]
            major_r = curve["major_axis_radius"]
            minor_r = curve["minor_axis_radius"]
            major_axis = curve["major_axis"]

            cp = sketch["points"][cp_id]
            maj_len = np.hypot(major_axis['x'], major_axis['y'])
            maj_ux, maj_uy = major_axis['x'] / maj_len, major_axis['y'] / maj_len
            s1_x = cp['x'] + maj_ux * major_r
            s1_y = cp['y'] + maj_uy * major_r
            s2_x = cp['x'] + (-maj_uy) * minor_r
            s2_y = cp['y'] + maj_ux * minor_r
            script.append(f"fc_sketch.addGeometry(Part.ArcOfEllipse(Part.Ellipse(FreeCAD.Vector({s1_x}, {s1_y}, 0), FreeCAD.Vector({s2_x}, {s2_y}, 0), {get_pt(cp_id)}), {get_pt(sp_id)}, {get_pt(ep_id)}), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            point_connections[sp_id].append((curve_uuid, 1))
            point_connections[ep_id].append((curve_uuid, 2))
            point_connections[cp_id].append((curve_uuid, 3))
            next_fc_id += 1

        elif c_type == "SketchFittedSpline" or c_type == "SketchFixedSpline":
            if "fit_points" in curve:
                fit_points = curve["fit_points"]
            elif "control_points" in curve:
                fit_points = curve["control_points"]
            else:
                raise KeyError(f"{c_type} missing both fit_points and control_points")

            pts_str = ", ".join([f"FreeCAD.Vector({p['x']}, {p['y']}, 0)" for p in fit_points])
            script.append(f"fc_sketch.addGeometry(Part.BSplineCurve([{pts_str}]), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            next_fc_id += 1

        elif c_type == "SketchConicCurve":
            sp_id = curve["start_point"]
            ep_id = curve["end_point"]
            ap_id = curve["apex_point"]
            pts_str = f"{get_pt(sp_id)}, {get_pt(ap_id)}, {get_pt(ep_id)}"
            script.append(f"fc_sketch.addGeometry(Part.BSplineCurve([{pts_str}], None, None, False, 2), {is_const})")

            uuid_to_fcid[curve_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = curve_uuid
            point_connections[sp_id].append((curve_uuid, 1))
            point_connections[ep_id].append((curve_uuid, 2))
            next_fc_id += 1

        else:
            raise NotImplementedError(f"Warning: Unhandled curve type {c_type}")

    all_points = set(sketch["points"].keys())
    points_in_constraints = set()
    for const in sketch["constraints"].values():
        for key in ["point", "point_one", "point_two", "entity", "entity_one", "entity_two", "entity_three"]:
            if key in const:
                val = const[key]
                if val in all_points: points_in_constraints.add(val)
    for dim in sketch["dimensions"].values():
        for key in ["entity_one", "entity_two", "line"]:
            if key in dim:
                val = dim[key]
                if val in all_points: points_in_constraints.add(val)

    points_to_add = points_in_constraints - set(point_connections.keys())
    if points_to_add:
        script.append("# --- Add Independent Points ---")
        for pt_uuid in points_to_add:
            script.append(f"fc_sketch.addGeometry(Part.Point({get_pt(pt_uuid)}), True)")
            uuid_to_fcid[pt_uuid] = next_fc_id
            fcid_to_uuid[next_fc_id] = pt_uuid
            point_connections[pt_uuid].append((pt_uuid, 1))
            next_fc_id += 1

    script.append(f"fcid_to_uuid = {json.dumps(fcid_to_uuid)}")


    if remove_constraints:
        script += [
            "",
            "doc.recompute()",
            "FreeCAD.ActiveDocument.recompute()",
        ]
        return script


    # script.append("# --- Anchor to Origin ---")
    # anchored_one = False
    # for pt_uuid, pt in sketch["points"].items():
    #     if abs(pt['x']) < 1e-8 and abs(pt['y']) < 1e-8:
    #         if pt_uuid in point_connections:
    #             curve_uuid, vtx = point_connections[pt_uuid][0]
    #             script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {uuid_to_fcid[curve_uuid]}, {vtx}, -1, 1))")
    #             anchored_one = True
    #             break
    #         elif pt_uuid in uuid_to_fcid:
    #             script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {uuid_to_fcid[pt_uuid]}, 1, -1, 1))")
    #             anchored_one = True
    #             break
    #         else:
    #             pass

    script.append("# --- Add Implicit Coincident Constraints ---")
    for pt_uuid, connections in point_connections.items():
        if len(connections) >= 2:
            base_curve, base_vtx = connections[0]
            base_curve_id = uuid_to_fcid[base_curve]
            for target_curve, target_vtx in connections[1:]:
                target_curve_id = uuid_to_fcid[target_curve]
                script.append(
                    f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {base_curve_id}, {base_vtx}, {target_curve_id}, {target_vtx}))"
                )

    script.append("# --- Add Relational Constraints ---")
    for const_uuid, const in sketch["constraints"].items():
        c_type = const["type"]

        if c_type == "HorizontalConstraint":
            line_uuid = const['line']
            if line_uuid is None or line_uuid not in uuid_to_fcid:
                print(f"Warning: HorizontalConstraint line_uuid {line_uuid} not found, skipping")
                continue
            line_fcid = uuid_to_fcid[line_uuid]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Horizontal', {line_fcid}))")

        elif c_type == "VerticalConstraint":
            line_uuid = const['line']
            if line_uuid is None or line_uuid not in uuid_to_fcid:
                print(f"Warning: VerticalConstraint line_uuid {line_uuid} not found, skipping")
                continue
            line_fcid = uuid_to_fcid[line_uuid]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Vertical', {line_fcid}))")

        elif c_type == "ParallelConstraint":
            l1 = const['line_one']
            l2 = const['line_two']
            if l1 is None or l2 is None or l1 not in uuid_to_fcid or l2 not in uuid_to_fcid:
                print(f"Warning: ParallelConstraint l1={l1}, l2={l2} not fully resolved, skipping")
                continue
            l1_fcid = uuid_to_fcid[l1]
            l2_fcid = uuid_to_fcid[l2]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Parallel', {l1_fcid}, {l2_fcid}))")

        elif c_type == "PerpendicularConstraint":
            l1 = const['line_one']
            l2 = const['line_two']
            if l1 is None or l2 is None or l1 not in uuid_to_fcid or l2 not in uuid_to_fcid:
                print(f"Warning: PerpendicularConstraint l1={l1}, l2={l2} not fully resolved, skipping")
                continue
            l1_fcid = uuid_to_fcid[l1]
            l2_fcid = uuid_to_fcid[l2]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Perpendicular', {l1_fcid}, {l2_fcid}))")

        elif c_type == "TangentConstraint":
            c1 = const["curve_one"]
            c2 = const["curve_two"]
            if c1 is None or c2 is None or c1 not in uuid_to_fcid or c2 not in uuid_to_fcid:
                print(f"Warning: TangentConstraint c1={c1}, c2={c2} not fully resolved, skipping")
                continue
            c1_fcid = uuid_to_fcid[c1]
            c2_fcid = uuid_to_fcid[c2]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {c1_fcid}, {c2_fcid}))")

        elif c_type == "CoincidentConstraint":
            pt_uuid = const["point"]
            entity_uuid = const["entity"]

            if pt_uuid is None or entity_uuid is None:
                print(f"Warning: CoincidentConstraint has None value, skipping")
                continue
            if pt_uuid in point_connections and entity_uuid in point_connections:
                pt_conns = point_connections[pt_uuid]
                en_conns = point_connections[entity_uuid]
                c1, v1 = pt_conns[0]
                c2, v2 = en_conns[0]
                if pt_uuid != entity_uuid:
                    script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[c2]}, {v2}))")
                else:
                    raise ValueError(f"CoincidentConstraint targets same point: {pt_uuid}")

            elif pt_uuid in point_connections and entity_uuid in uuid_to_fcid:
                pt_conns = point_connections[pt_uuid]
                c1, v1 = pt_conns[0]
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[entity_uuid]}))")

            elif entity_uuid in point_connections and pt_uuid in uuid_to_fcid:
                en_conns = point_connections[entity_uuid]
                c2, v2 = en_conns[0]
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('PointOnObject', {uuid_to_fcid[c2]}, {v2}, {uuid_to_fcid[pt_uuid]}))")
            else:
                print(f"Warning: CoincidentConstraint targets unresolved: pt_uuid={pt_uuid}, entity_uuid={entity_uuid}, skipping")
                continue

        elif c_type == "ConcentricConstraint":
            c1 = const["curve_one"]
            c2 = const["curve_two"]
            if c1 is None or c2 is None or c1 not in uuid_to_fcid or c2 not in uuid_to_fcid:
                print(f"Warning: ConcentricConstraint c1={c1}, c2={c2} not fully resolved, skipping")
                continue
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Coincident', {uuid_to_fcid[c1]}, 3, {uuid_to_fcid[c2]}, 3))")

        elif c_type == "EqualConstraint":
            c1 = const["curve_one"]
            c2 = const["curve_two"]
            if c1 is None or c2 is None or c1 not in uuid_to_fcid or c2 not in uuid_to_fcid:
                print(f"Warning: EqualConstraint c1={c1}, c2={c2} not fully resolved, skipping")
                continue
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Equal', {uuid_to_fcid[c1]}, {uuid_to_fcid[c2]}))")

        elif c_type == "CollinearConstraint":
            l1 = const["line_one"]
            l2 = const["line_two"]
            if l1 is None or l2 is None or l1 not in uuid_to_fcid or l2 not in uuid_to_fcid:
                print(f"Warning: CollinearConstraint l1={l1}, l2={l2} not fully resolved, skipping")
                continue
            l1_fcid = uuid_to_fcid[l1]
            l2_fcid = uuid_to_fcid[l2]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Tangent', {l1_fcid}, {l2_fcid}))")

        elif c_type == "MidPointConstraint":
            pt_uuid = const["point"]
            curve_uuid = const["mid_point_curve"]
            if pt_uuid not in point_connections:
                print(f"Warning: MidPointConstraint pt_uuid {pt_uuid} not found in point_connections, skipping")
                continue
            if curve_uuid not in uuid_to_fcid:
                print(f"Warning: MidPointConstraint curve_uuid {curve_uuid} not found in uuid_to_fcid, skipping")
                continue
            c1, v1 = point_connections[pt_uuid][0]
            if sketch["curves"][curve_uuid]["type"] == "SketchLine":
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {uuid_to_fcid[curve_uuid]}, 1, {uuid_to_fcid[curve_uuid]}, 2, {uuid_to_fcid[c1]}, {v1}))")
            else:
                print(f"Warning: MidPointConstraint curve_uuid {curve_uuid} is not a SketchLine, Skipping")
                continue

        elif c_type == "SymmetryConstraint":
            e1 = const["entity_one"]
            e2 = const["entity_two"]
            sym_line = const["symmetry_line"]
            if sym_line not in uuid_to_fcid:
                print(f"Warning: SymmetryConstraint sym_line {sym_line} not found in uuid_to_fcid, skipping")
                continue
            if e1 in point_connections and e2 in point_connections:
                c1, v1 = point_connections[e1][0]
                c2, v2 = point_connections[e2][0]
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[c2]}, {v2}, {uuid_to_fcid[sym_line]}))")
            elif e1 in uuid_to_fcid and e2 in uuid_to_fcid:
                type1 = sketch["curves"][e1]["type"]
                type2 = sketch["curves"][e2]["type"]
                if type1 != type2:
                    print(f"Warning: SymmetryConstraint between different types {type1} and {type2} unsupported, skipping")
                    continue

                fc_id_1 = uuid_to_fcid[e1]
                fc_id_2 = uuid_to_fcid[e2]
                fc_sym = uuid_to_fcid[sym_line]

                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Equal', {fc_id_1}, {fc_id_2}))")

                if type1 == "SketchLine":
                    sym_c = sketch["curves"].get(sym_line)
                    reversed_pts = False
                    if sym_c and sym_c.get("type") == "SketchLine":
                        s_s = sketch["points"][sym_c["start_point"]]
                        s_e = sketch["points"][sym_c["end_point"]]
                        l1_s = sketch["points"][sketch["curves"][e1]["start_point"]]
                        l1_e = sketch["points"][sketch["curves"][e1]["end_point"]]
                        l2_s = sketch["points"][sketch["curves"][e2]["start_point"]]
                        l2_e = sketch["points"][sketch["curves"][e2]["end_point"]]

                        ux = s_e['x'] - s_s['x']
                        uy = s_e['y'] - s_s['y']
                        alen = (ux**2 + uy**2)**0.5
                        if alen > 1e-6:
                            ux /= alen
                            uy /= alen
                        else:
                            ux, uy = 1.0, 0.0

                        def pair_error(px, py):
                            # vector connecting l1_s to p
                            cx, cy = px - l1_s['x'], py - l1_s['y']
                            # Dot product with unit axis vector (parallel deviation)
                            dp = abs(cx * ux + cy * uy)

                            # Midpoint distance to axis (perpendicular deviation)
                            mx, my = (l1_s['x'] + px)/2, (l1_s['y'] + py)/2
                            dist = abs((mx - s_s['x']) * uy - (my - s_s['y']) * ux)

                            return dp + dist

                        err1 = pair_error(l2_s['x'], l2_s['y'])
                        err2 = pair_error(l2_e['x'], l2_e['y'])

                        if err2 < err1:
                            reversed_pts = True

                    if reversed_pts:
                        script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {fc_id_1}, 1, {fc_id_2}, 2, {fc_sym}))")
                        script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {fc_id_1}, 2, {fc_id_2}, 1, {fc_sym}))")
                    else:
                        script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {fc_id_1}, 1, {fc_id_2}, 1, {fc_sym}))")
                        script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {fc_id_1}, 2, {fc_id_2}, 2, {fc_sym}))")
                elif type1 == "SketchCircle":
                    script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Symmetric', {fc_id_1}, 3, {fc_id_2}, 3, {fc_sym}))")
                elif type1 == "SketchArc":
                    # NOTE: We do not add explicit endpoint symmetries for Arcs here because Fusion 360
                    # JSON explicitly includes point symmetries for Arcs to preserve their directionality.
                    pass
                else:
                    print(f"Warning: SymmetryConstraint between non-points of type {type1} are unsupported in FreeCAD, skipping")
                    continue
            else:
                print(f"Warning: SymmetryConstraint targets unresolved: e1={e1}, e2={e2}, sym_line={sym_line}, skipping")
                continue

        elif c_type == "HorizontalPointsConstraint":
            p1 = const["point_one"]
            p2 = const["point_two"]
            if p1 not in point_connections or p2 not in point_connections:
                raise ValueError(f"Points not found for {c_type}: p1={p1}, p2={p2}")
            c1, v1 = point_connections[p1][0]
            c2, v2 = point_connections[p2][0]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Horizontal', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[c2]}, {v2}))")

        elif c_type == "VerticalPointsConstraint":
            p1 = const["point_one"]
            p2 = const["point_two"]
            if p1 not in point_connections or p2 not in point_connections:
                raise ValueError(f"Points not found for {c_type}: p1={p1}, p2={p2}")
            c1, v1 = point_connections[p1][0]
            c2, v2 = point_connections[p2][0]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Vertical', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[c2]}, {v2}))")

        elif c_type in ("OffsetConstraint", "PolygonConstraint"):
            print(f"Unhandled constraint type {c_type}")

        else:
            raise NotImplementedError(f"Warning: Unhandled constraint/dimension type {c_type}")

    script.append("# --- Add Dimensional Constraints ---")
    for const_uuid, const in sketch["dimensions"].items():
        c_type = const["type"]
        param = const['parameter']
        value = param['value']

        if c_type == "SketchAngularDimension":
            l1 = const['line_one']
            l2 = const['line_two']

            # Fusion360 data doesn't hold the angle direction?
            # So we ignore the value in json and calculate it from the points
            c1_info = sketch["curves"].get(l1)
            c2_info = sketch["curves"].get(l2)

            if c1_info is None or c2_info is None:
                print(f"Warning: SketchAngularDimension curves not found: l1={l1}, l2={l2}")
                continue

            p1_start = sketch["points"][c1_info["start_point"]]
            p1_end = sketch["points"][c1_info["end_point"]]
            p2_start = sketch["points"][c2_info["start_point"]]
            p2_end = sketch["points"][c2_info["end_point"]]

            v1_x = p1_end['x'] - p1_start['x']
            v1_y = p1_end['y'] - p1_start['y']
            v2_x = p2_end['x'] - p2_start['x']
            v2_y = p2_end['y'] - p2_start['y']

            angle1 = np.arctan2(v1_y, v1_x)
            angle2 = np.arctan2(v2_y, v2_x)

            diff = float(angle2 - angle1)
            while diff < 0:
                diff += 2 * np.pi
            while diff >= 2 * np.pi:
                diff -= 2 * np.pi

            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Angle', {uuid_to_fcid[l1]}, {uuid_to_fcid[l2]}, {diff}))")

        elif c_type == "SketchLinearDimension" or c_type == "SketchOffsetDimension":
            if "entity_one" in const:
                e1 = const["entity_one"]
            elif "line" in const:
                e1 = const["line"]
            else:
                raise KeyError(f'{c_type} missing both "entity_one" and "line"')

            e2 = const["entity_two"]
            orientation = const.get("orientation")

            fc_type = 'Distance'
            if orientation == 'HorizontalDimensionOrientation':
                fc_type = 'DistanceX'
            elif orientation == 'VerticalDimensionOrientation':
                fc_type = 'DistanceY'
            elif orientation is None or orientation == 'AlignedDimensionOrientation':
                pass
            else:
                raise ValueError(f"Unknown dimension orientation: {orientation}")

            conns1 = point_connections.get(e1, [])
            conns2 = point_connections.get(e2, [])

            if conns1 and conns2:
                c1, v1 = conns1[0]
                c2, v2 = conns2[0]
                val = value
                if fc_type in ('DistanceX', 'DistanceY'):
                    x1, y1 = sketch["points"][e1]["x"], sketch["points"][e1]["y"]
                    x2, y2 = sketch["points"][e2]["x"], sketch["points"][e2]["y"]
                    if fc_type == 'DistanceX' and x2 < x1:
                        val = -val
                    elif fc_type == 'DistanceY' and y2 < y1:
                        val = -val
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{fc_type}', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[c2]}, {v2}, {val}))")
            elif conns1 and e2 in uuid_to_fcid:
                c1, v1 = conns1[0]
                c2_in_curves = e2 in sketch["curves"]
                c2_type = sketch["curves"][e2]["type"] if c2_in_curves else "SketchPoint"
                v2 = 1 if c2_type == "SketchPoint" else 0
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{fc_type}', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[e2]}, {v2}, {value}))")
            elif e1 in uuid_to_fcid and conns2:
                c2, v2 = conns2[0]
                c1_in_curves = e1 in sketch["curves"]
                c1_type = sketch["curves"][e1]["type"] if c1_in_curves else "SketchPoint"
                v1 = 1 if c1_type == "SketchPoint" else 0
                script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{fc_type}', {uuid_to_fcid[c2]}, {v2}, {uuid_to_fcid[e1]}, {v1}, {value}))")
            elif e1 in uuid_to_fcid and e2 in uuid_to_fcid:
                c1_in_curves = e1 in sketch["curves"]
                c2_in_curves = e2 in sketch["curves"]
                c1_type = sketch["curves"][e1]["type"] if c1_in_curves else "SketchPoint"
                c2_type = sketch["curves"][e2]["type"] if c2_in_curves else "SketchPoint"

                if c1_type == "SketchLine" and c2_type == "SketchLine" and fc_type == "Distance":
                    p1_start = sketch["points"][sketch["curves"][e1]["start_point"]]
                    p1_end   = sketch["points"][sketch["curves"][e1]["end_point"]]
                    p2_start = sketch["points"][sketch["curves"][e2]["start_point"]]
                    p2_end   = sketch["points"][sketch["curves"][e2]["end_point"]]

                    v1x, v1y = p1_end['x'] - p1_start['x'], p1_end['y'] - p1_start['y']
                    v2x, v2y = p2_end['x'] - p2_start['x'], p2_end['y'] - p2_start['y']

                    cross = abs(v1x * v2y - v1y * v2x)
                    len1 = (v1x**2 + v1y**2)**0.5
                    len2 = (v2x**2 + v2y**2)**0.5

                    if len1 > 1e-6 and len2 > 1e-6 and cross / (len1 * len2) < 1e-4:
                        script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Parallel', {uuid_to_fcid[e1]}, {uuid_to_fcid[e2]}))")
                        script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {uuid_to_fcid[e1]}, 1, {uuid_to_fcid[e2]}, 0, {value}))")
                    else:
                        raise ValueError(f"Distance constraint between non-parallel lines {e1} and {e2} is unsupported.")
                else:
                    v1 = 1 if c1_type == "SketchPoint" else 0
                    v2 = 1 if c2_type == "SketchPoint" else 0
                    script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{fc_type}', {uuid_to_fcid[e1]}, {v1}, {uuid_to_fcid[e2]}, {v2}, {value}))")
            else:
                raise ValueError(f"Linear/Offset Dimension targets unresolved: e1={e1}, e2={e2}")

        elif c_type == "SketchDiameterDimension":
            c_uuid = const["curve"]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Diameter', {uuid_to_fcid[c_uuid]}, {value}))")

        elif c_type == "SketchRadialDimension":
            c_uuid = const["curve"]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Radius', {uuid_to_fcid[c_uuid]}, {value}))")

        elif c_type == "SketchEllipseMajorRadiusDimension":
            c_uuid = const["ellipse"]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {uuid_to_fcid[c_uuid]}, 3, {uuid_to_fcid[c_uuid]}, 1, {value}))")

        elif c_type == "SketchEllipseMinorRadiusDimension":
            c_uuid = const["ellipse"]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {uuid_to_fcid[c_uuid]}, 3, {uuid_to_fcid[c_uuid]}, 4, {value}))")

        elif c_type == "SketchPointDistanceDimension":
            e1 = const["entity_one"]
            e2 = const["entity_two"]
            if e1 not in point_connections or e2 not in point_connections:
                raise ValueError(f"Point Distance Points not found: e1={e1}, e2={e2}")
            c1, v1 = point_connections[e1][0]
            c2, v2 = point_connections[e2][0]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {uuid_to_fcid[c1]}, {v1}, {uuid_to_fcid[c2]}, {v2}, {value}))")

        elif c_type == "SketchConcentricCircleDimension":
            c1 = const["curve_one"]
            c2 = const["curve_two"]
            script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('Distance', {uuid_to_fcid[c1]}, {uuid_to_fcid[c2]}, {value}))")

        elif c_type in ("SketchOffsetCurvesDimension"):
            print(f"Unhandled dimension type {c_type}")

        else:
            raise NotImplementedError(f"Unhandled dimension type {c_type}")

    # script.append("# --- Handle Fixed Entities ---")
    # for curve_uuid, curve in sketch["curves"].items():
    #     if "fixed" in curve and curve["fixed"]:
    #         if curve_uuid not in uuid_to_fcid:
    #             raise KeyError(f"Fixed curve {curve_uuid} not found in uuid_to_fcid")
    #         fcid = uuid_to_fcid[curve_uuid]
    #         c_type = curve["type"]
    #         # Helper to lock a specific point index of a geometry
    #         def lock_pt(geom_id, pt_idx, x, y):
    #             script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('DistanceX', {geom_id}, {pt_idx}, -1, 1, {x}))")
    #             script.append(f"fc_sketch.addConstraint(Sketcher.Constraint('DistanceY', {geom_id}, {pt_idx}, -1, 1, {y}))")

    #         if c_type == "SketchLine":
    #             p1 = sketch["points"][curve["start_point"]]
    #             p2 = sketch["points"][curve["end_point"]]
    #             lock_pt(fcid, 1, p1['x'], p1['y'])
    #             lock_pt(fcid, 2, p2['x'], p2['y'])
    #         elif c_type in ["SketchCircle", "SketchArc", "SketchEllipse", "SketchEllipticalArc"]:
    #             cp = sketch["points"][curve["center_point"]]
    #             lock_pt(fcid, 3, cp['x'], cp['y'])
    #         elif c_type in ("SketchFittedSpline", "SketchFixedSpline"):
    #             pass
    #         else:
    #             raise NotImplementedError(f"Fixing curve type {c_type} is not implemented")

    # Deduplicate constraints
    script = deduplicate_constraints(script)

    script += [
        "",
        "doc.recompute()",
        "FreeCAD.ActiveDocument.recompute()",
    ]
    return script


def deduplicate_constraints(script: list[str]) -> list[str]:
    script_deduplicated = []
    seen_constraints = set()
    for line in script:
        if line.startswith("fc_sketch.addConstraint(Sketcher.Constraint("):
            if line in seen_constraints:
                continue
            seen_constraints.add(line)
            script_deduplicated.append(line)
        else:
            script_deduplicated.append(line)
    return script_deduplicated


def count_freecad_commands(freecad_script: list[str]) -> tuple[Counter[str], Counter[str]]:
    primitive_counter = Counter()
    constraint_counter = Counter()
    for line in freecad_script:
        if line.startswith("fc_sketch.addGeometry(Part."):
            prim = line[len("fc_sketch.addGeometry(Part."):].split("(")[0]
            primitive_counter[prim] += 1
        elif line.startswith("fc_sketch.addConstraint(Sketcher.Constraint('"):
            cons = line[len("fc_sketch.addConstraint(Sketcher.Constraint('"):].split("'")[0]
            constraint_counter[cons] += 1
    return primitive_counter, constraint_counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="disk2/Fusion360Gallery/r1.0.1/reconstruction")
    parser.add_argument("--out_dir", type=str, default="disk2/Fusion360Gallery/r1.0.1/freecad_commands")
    parser.add_argument("--sketch_dump_dir", type=str, default="disk2/Fusion360Gallery/r1.0.1/freecad_images")
    parser.add_argument("--log_path", type=str, default="disk2/Fusion360Gallery/r1.0.1/freecad_conversion_log.csv")
    parser.add_argument("--test_data_only", action='store_true', help="If True, process only test data")
    parser.add_argument("--deduplicate_bins", type=int, default=5, help="If set, deduplicate sketches using this number of quantization bins")
    parser.add_argument("--prim_num_range", type=int, nargs=2, default=(6, 16), help="Only convert sketches with number of primitives in this range")
    parser.add_argument("--cons_num_range", type=int, nargs=2, default=(0, 64),help="Only convert sketches with number of constraints in this range" )
    args = parser.parse_args()

    UNHANDLED_PRIMITIVES = {"Ellipse", "ArcOfEllipse", "BSplineCurve"}
    UNHANDLED_CONSTRAINTS = {"Offset"}

    # Output file/folder organization
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    if os.path.isdir(args.sketch_dump_dir):
        shutil.rmtree(args.sketch_dump_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.sketch_dump_dir, exist_ok=True)

    # Use specified deduplicate bins if provided
    ctx_kwargs = {}
    if args.deduplicate_bins:
        ctx_kwargs['num_quantization_bins'] = args.deduplicate_bins
    context = SketchContext(**ctx_kwargs)
    sketch_freecad_converter = SketchFreecadConverter(context, raise_on_error=False)
    tokenizer = SketchTokenizer(context)
    solver_feedback = SolverFeedback(context)
    solver_summary = Counter()
    seen_hashes = set()

    with open(args.log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "status", "dof", "redundant_constraints"])

    if args.test_data_only:
        train_test_json = os.path.join(os.path.dirname(args.data_dir), "train_test.json")
        assert os.path.isfile(train_test_json)
        with open(train_test_json, "r") as f:
            train_test_ids = json.load(f)
        test_ids = train_test_ids["test"]
        json_path_list = sorted([os.path.join(args.data_dir, f"{test_id}.json") for test_id in test_ids])
    else:
        json_path_list = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))

    for json_path in tqdm(json_path_list):
        for sketch_id, sketch in enumerate(extract_sketches(json_path)):
            doc = None
            try:
                # 1. Convert
                sketch['name'] = f"{os.path.splitext(os.path.basename(json_path))[0]}_{sketch['name']}"
                freecad_script = generate_freecad_script(sketch, remove_constraints=False)

                # 2, Filter
                primitives_counter, constraints_counter = count_freecad_commands(freecad_script)
                if not (args.prim_num_range[0] <= sum(primitives_counter.values()) <= args.prim_num_range[1] and
                        args.cons_num_range[0] <= sum(constraints_counter.values()) <= args.cons_num_range[1]):
                    continue
                if primitives_counter.keys() & UNHANDLED_PRIMITIVES or constraints_counter.keys() & UNHANDLED_CONSTRAINTS:
                    continue

                # 3. Resolve redundancy
                sg_sketch = sketch_freecad_converter.from_freecad_commands(freecad_script)
                sg_sketch = cleanse_sketch_constraints(sg_sketch, resolve_redundancy=True)
                sg_sketch, _ = complete_constraint_values(sg_sketch, skip_invalid_constraints=True)

                # 3.5 Deduplicate
                if args.deduplicate_bins is not None:
                    sketch_hash = get_sketch_hash(sg_sketch, n_bins=args.deduplicate_bins)

                    if sketch_hash in seen_hashes:
                        solver_summary["Duplicate"] += 1
                        continue
                    seen_hashes.add(sketch_hash)

                doc, fc_sketch, _, _, _  = sketch_freecad_converter.to_freecad(sg_sketch)

                # 4. Test-run the script
                solver_status = solver_feedback.get_solver_status(sg_sketch)
                if solver_status.DistortingConstraints:
                    solver_status.Status = "Distorting"
                solver_summary[solver_status.Status] += 1
                with open(args.log_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([sketch['name'], solver_status.Status, solver_status.DoF, str(solver_status.RedundantConstraints)])

                # 5. Export the commands and FCStd
                command_file_path = os.path.join(args.out_dir, f"{os.path.basename(json_path).replace('.json', f'_sketch{sketch_id:02d}.py')}")
                with open(command_file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(freecad_script))
                doc.saveAs(command_file_path.replace('.py', '.FCStd'))

                # 4. Export the sketch
                # sketch_file_path = os.path.join(args.sketch_dump_dir, f"{os.path.basename(json_path).replace('.json', f'_sketch{sketch_id:02d}.jpg')}")
                # visualize_sketch_for_vlm(
                #     export_sketch_to_dict(fc_sketch),
                #     save_path=sketch_file_path,
                #     title=None,
                #     icon_fontsize=10,
                #     figsize=(640, 640),
                # )
                # plt.close()

            except KeyError:
                # NOTE: Ellipses are not supported in visualize_sketch_for_vlm, and emit key errors. We skip them.
                continue
            except Exception as e:
                print(f"Error processing {sketch_id}-th sketch in {json_path}:\n{traceback.format_exc()}")
                continue
            finally:
                FreeCAD.closeDocument(doc.Name) if doc is not None else None

    print("=== Summary ===")
    total_found = sum(solver_summary.values())
    duplicates = solver_summary.get("Duplicate", 0)
    total_processed = total_found - duplicates

    if total_processed > 0:
        for key, num in solver_summary.items():
            if key == "Duplicate":
                print(f"{key}: {num} (Excluded from percentages)")
            else:
                print(f"{key}: {num} ({num / total_processed * 100:.2f}%)")
    else:
        print("No sketches processed.")
