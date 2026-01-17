from collections import defaultdict
import glob
import json
import os
import math
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add FreeCAD Libs (conditionally)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part
import Sketcher
from FreeCAD import Vector


# ==========================================
# 1. データ抽出ロジック (Extraction)
# ==========================================

def export_sketch_to_dict(sketch_obj):
    drawing_data = {
        "primitives": [],
        "constraints": []
    }

    # --- A. Primitives ---
    for i, geom in enumerate(sketch_obj.Geometry):
        p_id = str(i)
        prim_data = {"id": p_id}

        if isinstance(geom, Part.LineSegment):
            prim_data["type"] = "Line"
            prim_data["start"] = (geom.StartPoint.x, geom.StartPoint.y)
            prim_data["end"] = (geom.EndPoint.x, geom.EndPoint.y)

        elif isinstance(geom, Part.Point):
            prim_data["type"] = "Point"
            prim_data["coords"] = (geom.X, geom.Y)

        elif isinstance(geom, Part.ArcOfCircle):
            prim_data["type"] = "Arc"
            prim_data["center"] = (geom.Center.x, geom.Center.y)
            prim_data["radius"] = geom.Radius
            prim_data["start"] = (geom.StartPoint.x, geom.StartPoint.y)
            prim_data["end"] = (geom.EndPoint.x, geom.EndPoint.y)

            cx, cy = geom.Center.x, geom.Center.y
            sx, sy = geom.StartPoint.x, geom.StartPoint.y
            ex, ey = geom.EndPoint.x, geom.EndPoint.y
            prim_data["start_angle"] = math.degrees(math.atan2(sy - cy, sx - cx))
            prim_data["end_angle"] = math.degrees(math.atan2(ey - cy, ex - cx))

        elif isinstance(geom, Part.Circle):
            prim_data["type"] = "Circle"
            prim_data["center"] = (geom.Center.x, geom.Center.y)
            prim_data["radius"] = geom.Radius

        # Check if this is construction geometry
        prim_data["construction"] = sketch_obj.getConstruction(i)

        drawing_data["primitives"].append(prim_data)

    # --- B. Constraints ---
    for const in sketch_obj.Constraints:
        c_data = {
            "type": const.Type,
            "target": [] # リストの中身を {"id": "1", "pos": 1} のような辞書にする
        }

        # Helper to unpack id and pos
        # FreeCAD Constraint: First, FirstPos, Second, SecondPos, Third, ThirdPos
        # PosId: 0=Edge itself, 1=Start/Center, 2=End, 3=Center(Circle) etc.

        targets = []
        if const.First > -1:
            targets.append({"id": str(const.First), "pos": const.FirstPos})
        if const.Second > -1:
            targets.append({"id": str(const.Second), "pos": const.SecondPos})
        if const.Third > -1:
            targets.append({"id": str(const.Third), "pos": const.ThirdPos})

        c_data["target"] = targets

        if const.Type in ["Distance", "DistanceX", "DistanceY", "Radius", "Diameter", "Angle", "Length"]:
            c_data["value"] = const.Value

        drawing_data["constraints"].append(c_data)

    return drawing_data

# ==========================================
# 2. ファイルロード & 実行ロジック (Loading)
# ==========================================

def load_and_extract_from_file(file_path: str):
    """
    指定されたPythonファイルを実行し、生成されたSketchからデータを抽出し、ドキュメントを閉じる
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # 実行用のコンテキスト辞書
    exec_globals = {
        "FreeCAD": FreeCAD,
        "Part": Part,
        "Sketcher": Sketcher,
        "Vector": FreeCAD.Vector
    }

    # ファイル読み込み
    with open(file_path, 'r') as f:
        script_content = f.read()

    # --- スクリプト実行 ---
    try:
        exec(script_content, exec_globals)
    except Exception as e:
        print(f"Error executing script: {e}")
        return None

    # --- Sketchオブジェクトの探索 ---
    target_sketch = None

    # 1. exec_globals の変数から探す
    for key, val in exec_globals.items():
        if hasattr(val, "TypeId") and val.TypeId == "Sketcher::SketchObject":
            target_sketch = val
            break

    # 2. 見つからない場合、ActiveDocumentから探す
    if target_sketch is None and FreeCAD.ActiveDocument:
        for obj in FreeCAD.ActiveDocument.Objects:
            if obj.TypeId == "Sketcher::SketchObject":
                target_sketch = obj
                break

    if target_sketch is None:
        print("Error: No Sketch object found after script execution.")
        # ドキュメントが生成されている可能性があるので念のため閉じる処理を試みる
        if FreeCAD.ActiveDocument:
            FreeCAD.closeDocument(FreeCAD.ActiveDocument.Name)
        return None

    # --- データ抽出 ---
    extracted_data = export_sketch_to_dict(target_sketch)

    # --- メモリ解放 (closeDocument) ---
    doc_name = target_sketch.Document.Name
    FreeCAD.closeDocument(doc_name)

    return extracted_data


# ==========================================
# 3. 可視化ロジック (Visualizing)
# ==========================================


def visualize_sketch_for_vlm(data, save_path: str = "vlm_input.png", title: str | None = None, title_fontsize: int = 24):
    """
    Visualize sketch with primitives and constraints.

    Args:
        data: Sketch data dictionary with primitives and constraints
        save_path: Path to save the output image
        title: Optional label (e.g., "Sketch A") to display in top-left corner
        title_fontsize: Font size for the title label
    """
    if not data: return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    all_x, all_y = [], []
    prim_map = {p["id"]: p for p in data["primitives"]}

    # --- 1. Primitives (Geometry) ---
    STYLE_GEO = dict(color='black', linewidth=2.5, zorder=10, solid_capstyle='round')
    STYLE_GEO_CONSTR = dict(color='black', linewidth=2.0, linestyle='--', zorder=9)  # Construction: dashed
    STYLE_PNT = dict(color='black', s=60, zorder=11)

    for prim in data["primitives"]:
        is_construction = prim.get("construction", False)
        line_style = STYLE_GEO_CONSTR if is_construction else STYLE_GEO

        if prim["type"] == "Line":
            s, e = prim["start"], prim["end"]
            ax.plot([s[0], e[0]], [s[1], e[1]], **line_style)
            ax.scatter([s[0], e[0]], [s[1], e[1]], **STYLE_PNT)
            all_x.extend([s[0], e[0]])
            all_y.extend([s[1], e[1]])
        elif prim["type"] == "Circle":
            c, r = prim["center"], prim["radius"]
            ls = '--' if is_construction else '-'
            circ = patches.Circle(c, r, edgecolor='black', facecolor='none', linewidth=2.5 if not is_construction else 2.0, linestyle=ls, zorder=10 if not is_construction else 9)
            ax.add_patch(circ)
            ax.scatter([c[0]], [c[1]], color='black', s=50, marker='x', zorder=11)
            all_x.extend([c[0]-r, c[0]+r])
            all_y.extend([c[1]-r, c[1]+r])
        elif prim["type"] == "Arc":
            c, r = prim["center"], prim["radius"]
            ls = '--' if is_construction else '-'
            arc = patches.Arc(c, 2*r, 2*r, angle=0,
                              theta1=prim["start_angle"], theta2=prim["end_angle"],
                              linewidth=2.5 if not is_construction else 2.0, linestyle=ls, color='black', zorder=10 if not is_construction else 9)
            ax.add_patch(arc)
            s, e = prim["start"], prim["end"]
            ax.scatter([s[0], e[0]], [s[1], e[1]], **STYLE_PNT)
            ax.scatter([c[0]], [c[1]], color='black', s=40, marker='+', zorder=11)
            all_x.extend([c[0]-r, c[0]+r])
            all_y.extend([c[1]-r, c[1]+r])
        elif prim["type"] == "Point":
            c = prim["coords"]
            ax.scatter([c[0]], [c[1]], **STYLE_PNT)
            all_x.append(c[0])
            all_y.append(c[1])

    if not all_x: return
    center_x, center_y = np.mean(all_x), np.mean(all_y)
    bbox_w = max(all_x) - min(all_x)
    bbox_h = max(all_y) - min(all_y)
    scale_ref = max(bbox_w, bbox_h) * 0.1

    margin = scale_ref * 4
    view_min_x, view_max_x = min(all_x)-margin, max(all_x)+margin
    view_min_y, view_max_y = min(all_y)-margin, max(all_y)+margin

    # --- Helper Functions ---
    def get_pos_coord(target_info):
        tid, pos = target_info["id"], target_info["pos"]
        prim = prim_map.get(tid)
        if not prim: return (0,0)
        if prim["type"] == "Line":
            if pos == 1: return prim["start"]
            if pos == 2: return prim["end"]
            return ((prim["start"][0]+prim["end"][0])/2, (prim["start"][1]+prim["end"][1])/2)
        if prim["type"] == "Arc":
            if pos == 1: return prim["start"]
            if pos == 2: return prim["end"]
            if pos == 3: return prim["center"]
            mid = math.radians((prim["start_angle"] + prim["end_angle"])/2)
            return (prim["center"][0] + prim["radius"]*math.cos(mid), prim["center"][1] + prim["radius"]*math.sin(mid))
        if prim["type"] == "Circle":
            if pos == 3: return prim["center"]
            return (prim["center"][0] + prim["radius"], prim["center"][1])
        if prim["type"] == "Point": return prim["coords"]
        return (0,0)

    def get_outward_normal(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0: return (0, 1)
        nx, ny = -dy/dist, dx/dist
        mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        cx, cy = mx - center_x, my - center_y
        if nx*cx + ny*cy < 0: nx, ny = -nx, -ny
        return (nx, ny)

    def project_point_onto_line(pt, line_start, line_end):
        x, y = pt; x1, y1 = line_start; x2, y2 = line_end
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return (x1, y1)
        t = ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)
        return (x1 + t * dx, y1 + t * dy)

    def find_shared_vertex(targets):
        endpoints = []
        for t in targets:
            tid = t["id"]
            p = prim_map.get(tid)
            if not p: continue
            if p["type"] == "Line": endpoints.extend([p["start"], p["end"]])
            if p["type"] == "Arc": endpoints.extend([p["start"], p["end"]])
            if p["type"] == "Point": endpoints.append(p["coords"])
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                p1, p2 = endpoints[i], endpoints[j]
                if (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 < 1e-6:
                    return p1
        return None

    # --- Grid size for position-based offset calculation ---
    pos_grid_size = scale_ref * 0.5

    # --- Draw Functions ---
    def draw_linear_dimension(p1, p2, text):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-6: return
        nx, ny = get_outward_normal(p1, p2)
        gap, off_dist, over = scale_ref*0.3, scale_ref*2.5, scale_ref*0.3

        p1_gap = (p1[0] + nx*gap, p1[1] + ny*gap)
        p1_ext = (p1[0] + nx*(off_dist+over), p1[1] + ny*(off_dist+over))
        ax.plot([p1_gap[0], p1_ext[0]], [p1_gap[1], p1_ext[1]], color='black', lw=0.6, zorder=5, alpha=0.7)
        p2_gap = (p2[0] + nx*gap, p2[1] + ny*gap)
        p2_ext = (p2[0] + nx*(off_dist+over), p2[1] + ny*(off_dist+over))
        ax.plot([p2_gap[0], p2_ext[0]], [p2_gap[1], p2_ext[1]], color='black', lw=0.6, zorder=5, alpha=0.7)

        dim_start = (p1[0] + nx*off_dist, p1[1] + ny*off_dist)
        dim_end = (p2[0] + nx*off_dist, p2[1] + ny*off_dist)
        ax.plot([dim_start[0], dim_end[0]], [dim_start[1], dim_end[1]], color='blue', lw=1.0, zorder=6)
        ax.scatter([dim_start[0], dim_end[0]], [dim_start[1], dim_end[1]], color='blue', s=20, zorder=7)

        mx, my = (dim_start[0] + dim_end[0])/2, (dim_start[1] + dim_end[1])/2
        ax.text(mx, my, text, color='blue', fontsize=11, weight='bold', ha='center', va='center',
                bbox=dict(facecolor='#f0f8ff', edgecolor='blue', boxstyle='round,pad=0.2', alpha=1.0), zorder=20)

    # Angles ordered to maximize separation, starting with 45 degrees to avoid overlap with horizontal/vertical primitives
    radial_angles = [45, 225, 135, 315, 60, 240, 120, 300,
                     30, 210, 150, 330, 15, 195, 75, 255,
                     105, 285, 165, 345, 0, 180, 90, 270]
    radial_counter_by_center = defaultdict(int)  # Track per center position

    # Track label count by position (rounded to grid) for offset calculation
    # This handles cases where different primitives have the same position
    label_count_by_pos = defaultdict(int)

    def get_label_offset(base_x, base_y):
        """Calculate offset position for label to avoid overlap based on position"""
        # Round position to grid to group nearby labels
        grid_key = (round(base_x / pos_grid_size), round(base_y / pos_grid_size))
        offset_idx = label_count_by_pos[grid_key]
        label_count_by_pos[grid_key] += 1
        if offset_idx == 0:
            return base_x, base_y
        # Offset in different directions around the point
        angle = offset_idx * (2 * math.pi / 6)  # 6方向に分散
        offset_dist = scale_ref * 0.6 * (1 + (offset_idx - 1) // 6)
        return base_x + offset_dist * math.cos(angle), base_y + offset_dist * math.sin(angle)

    def draw_radial_dimension(prim, val, is_diameter=False):
        c, r = prim["center"], prim["radius"]
        prim_type = prim.get("type", "Circle")

        if prim_type == "Arc":
            # For Arc: use the arc's midpoint angle, draw leader line from edge only
            start_ang = prim["start_angle"]
            end_ang = prim["end_angle"]
            # Calculate midpoint angle (handle wraparound)
            if end_ang < start_ang:
                mid_ang = (start_ang + end_ang + 360) / 2
                if mid_ang >= 360: mid_ang -= 360
            else:
                mid_ang = (start_ang + end_ang) / 2
            angle = math.radians(mid_ang)
            dx, dy = math.cos(angle), math.sin(angle)

            p_edge = (c[0] + dx*r, c[1] + dy*r)
            p_outer = (c[0] + dx*(r + scale_ref*2.5), c[1] + dy*(r + scale_ref*2.5))

            # Draw only leader line from arc edge (no center/diameter line)
            ax.plot([p_edge[0], p_outer[0]], [p_edge[1], p_outer[1]], color='blue', lw=1.2, zorder=6)
            ax.scatter([p_edge[0]], [p_edge[1]], color='blue', s=25, zorder=7)
            label = f"⌀{val:.1f}" if is_diameter else f"R{val:.1f}"
        else:
            # For Circle: use position-based angle selection
            center_key = (round(c[0] / pos_grid_size), round(c[1] / pos_grid_size))
            angle_idx = radial_counter_by_center[center_key]
            radial_counter_by_center[center_key] += 1
            angle_deg = radial_angles[angle_idx % len(radial_angles)]
            angle = math.radians(angle_deg)
            dx, dy = math.cos(angle), math.sin(angle)

            p_edge = (c[0] + dx*r, c[1] + dy*r)
            p_outer = (c[0] + dx*(r + scale_ref*2.5), c[1] + dy*(r + scale_ref*2.5))

            # Use leader line only (no full diameter line through center)
            ax.plot([p_edge[0], p_outer[0]], [p_edge[1], p_outer[1]], color='blue', lw=1.2, zorder=6)
            ax.scatter([p_edge[0]], [p_edge[1]], color='blue', s=25, zorder=7)
            label = f"⌀{val:.1f}" if is_diameter else f"R{val:.1f}"

        # Apply position-based offset to label to avoid overlapping with other labels
        label_x, label_y = get_label_offset(p_outer[0], p_outer[1])
        ax.text(label_x, label_y, label, color='blue', fontsize=11, weight='bold', ha='left', va='center',
                bbox=dict(facecolor='#f0f8ff', edgecolor='blue', boxstyle='round,pad=0.2', alpha=1.0), zorder=20)

    align_counter = 0
    def draw_alignment_line(p1, p2, type_str):
        nonlocal align_counter
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linestyle=':', lw=1.2, zorder=5)
        ratio = 0.35 if align_counter % 2 == 0 else 0.65; align_counter += 1
        mx = p1[0] + (p2[0]-p1[0]) * ratio
        my = p1[1] + (p2[1]-p1[1]) * ratio
        label = "H" if type_str == "Horizontal" else "V"
        ax.text(mx, my, label, color='red', fontsize=11, weight='bold', ha='center', va='center', zorder=30,
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='square,pad=0.1', alpha=1.0))


    # Counter for symmetry markers to vary symbols
    symmetry_marker_counter = 0
    # Use standard matplotlib markers: circle, x, square, diamond, triangle_up, triangle_down
    SYMMETRY_MARKERS = ['o', 's', 'D', '^', 'v']

    def draw_symmetry_axis(targets):
        nonlocal symmetry_marker_counter
        AXIS_COLOR = '#800080'

        # Get symmetry axis/point
        axis_target = targets[2]
        axis_id = axis_target["id"]
        prim = prim_map.get(axis_id)
        if not prim: return

        # Get the two symmetric points
        t1, t2 = targets[0], targets[1]
        p1 = get_pos_coord(t1)
        p2 = get_pos_coord(t2)

        # Draw line between points
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=AXIS_COLOR, linestyle='-', lw=1.0, zorder=5, alpha=0.8)

        # Draw markers at 1/4 and 3/4 positions
        marker = SYMMETRY_MARKERS[symmetry_marker_counter % len(SYMMETRY_MARKERS)]
        symmetry_marker_counter += 1

        # Calculate positions
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        m1 = (p1[0] + dx*0.25, p1[1] + dy*0.25)
        m2 = (p1[0] + dx*0.75, p1[1] + dy*0.75)

        # Draw markers with white background for visibility
        for mx, my in [m1, m2]:
            ax.scatter([mx], [my], marker=marker, color=AXIS_COLOR, s=60, zorder=6,
                        edgecolors=AXIS_COLOR, facecolors='white', linewidth=1.5)
        return

    # --- Constraints Loop ---
    text_props = dict(ha='center', va='center', weight='bold', zorder=30)
    constraint_counters = {"Parallel": 0, "Equal": 0}


    for const in data["constraints"]:
        c_type = const["type"]
        targets = const["target"]
        if not targets: continue

        # 1. Dimensions
        if c_type in ["Distance", "Length", "Radius", "Diameter", "DistanceX", "DistanceY"]:
            val = const.get("value", "?")
            val_str = f"{val:.1f}"
            if c_type in ["Radius", "Diameter"] and len(targets) == 1:
                tid = targets[0]["id"]
                prim = prim_map.get(tid)
                if prim["type"] in ["Circle", "Arc"]:
                    draw_radial_dimension(prim, val, is_diameter=(c_type=="Diameter"))
                    continue
            p1, p2 = None, None
            if len(targets) == 1 and prim_map[targets[0]["id"]]["type"] == "Line":
                tid = targets[0]["id"]; p1 = prim_map[tid]["start"]; p2 = prim_map[tid]["end"]
            elif len(targets) == 2:
                t1, t2 = targets[0], targets[1]
                type1 = prim_map[t1["id"]]["type"]; type2 = prim_map[t2["id"]]["type"]
                c1 = get_pos_coord(t1); c2 = get_pos_coord(t2)
                if type1 == "Line" and t1["pos"] == 0 and t2["pos"] != 0:
                    l_start = prim_map[t1["id"]]["start"]; l_end = prim_map[t1["id"]]["end"]
                    c1 = project_point_onto_line(c2, l_start, l_end)
                elif type2 == "Line" and t2["pos"] == 0 and t1["pos"] != 0:
                    l_start = prim_map[t2["id"]]["start"]; l_end = prim_map[t2["id"]]["end"]
                    c2 = project_point_onto_line(c1, l_start, l_end)
                elif type1 == "Line" and type2 == "Line" and t1["pos"] == 0 and t2["pos"] == 0:
                    l2_start = prim_map[t2["id"]]["start"]; l2_end = prim_map[t2["id"]]["end"]
                    c2 = project_point_onto_line(c1, l2_start, l2_end)
                p1, p2 = c1, c2
            if p1 and p2: draw_linear_dimension(p1, p2, val_str)
            continue

        # 2. Symmetric
        if c_type == "Symmetric" and len(targets) == 3:
            draw_symmetry_axis(targets)
            continue

        # 3. Equal
        if c_type == "Equal":
            constraint_counters["Equal"] += 1
            label = f"={constraint_counters['Equal']}"
            for t in targets:
                tid = t["id"]
                mx, my = get_pos_coord({"id": tid, "pos": 0})
                mx, my = get_label_offset(mx, my)
                ax.text(mx, my, label, color='red', fontsize=12, **text_props,
                        bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
            continue

        # 4. Parallel
        if c_type == "Parallel":
            constraint_counters["Parallel"] += 1
            label = f"//{constraint_counters['Parallel']}"
            for t in targets:
                tid = t["id"]
                mx, my = get_pos_coord({"id": tid, "pos": 0})
                mx, my = get_label_offset(mx, my)
                ax.text(mx, my, label, color='red', fontsize=12, **text_props,
                        bbox=dict(facecolor='white', edgecolor='none', pad=0, alpha=0.8))
            continue

        # 5. PointOnObject
        if c_type == "PointOnObject":
            pt_target = targets[0] if prim_map[targets[0]["id"]]["type"] == "Point" else targets[1]
            c = get_pos_coord(pt_target)
            ax.scatter([c[0]], [c[1]], color='red', s=80, zorder=15, edgecolors='white', linewidth=1.0)
            continue

        # 6. Tangent
        if c_type == "Tangent":
            # ターゲットが2つあるはず
            if len(targets) == 2:
                # 幾何学的な接点を計算する（簡易版：円の中心から直線への投影）
                t1, t2 = targets[0], targets[1]
                type1 = prim_map[t1["id"]]["type"]
                type2 = prim_map[t2["id"]]["type"]

                tangent_point = None

                # Case: Circle/Arc vs Line
                circle_id, line_id = None, None
                if type1 in ["Circle", "Arc"] and type2 == "Line":
                    circle_id, line_id = t1["id"], t2["id"]
                elif type2 in ["Circle", "Arc"] and type1 == "Line":
                    circle_id, line_id = t2["id"], t1["id"]

                if circle_id and line_id:
                    c = prim_map[circle_id]["center"]
                    l_start = prim_map[line_id]["start"]
                    l_end = prim_map[line_id]["end"]
                    # 中心を直線に投影した点が接点
                    tangent_point = project_point_onto_line(c, l_start, l_end)

                # Case: Circle vs Circle (連線上の点)
                elif type1 in ["Circle", "Arc"] and type2 in ["Circle", "Arc"]:
                    c1 = prim_map[t1["id"]]["center"]
                    c2 = prim_map[t2["id"]]["center"]
                    r1 = prim_map[t1["id"]]["radius"]
                    # C1からC2に向かうベクトル
                    vx, vy = c2[0]-c1[0], c2[1]-c1[1]
                    dist = math.sqrt(vx*vx + vy*vy)
                    if dist > 0:
                        tangent_point = (c1[0] + vx/dist*r1, c1[1] + vy/dist*r1)

                if tangent_point:
                    tx, ty = tangent_point
                    # 赤丸
                    ax.scatter([tx], [ty], color='red', s=60, zorder=15, edgecolors='white', linewidth=1.0)
                    # [T]マーク (法線方向に少しずらす)
                    vx, vy = tx - center_x, ty - center_y
                    dist = math.sqrt(vx*vx + vy*vy)
                    if dist > 0:
                        off = scale_ref * 0.4
                        ax.text(tx + vx/dist*off, ty + vy/dist*off, "T", color='red', fontsize=11, weight='bold', ha='center', va='center',
                                bbox=dict(facecolor='white', edgecolor='red', boxstyle='square,pad=0.1', alpha=1.0), zorder=30)
                    continue

        # 7. Horizontal/Vertical
        if c_type in ["Horizontal", "Vertical"]:
            if len(targets) == 2:
                p1 = get_pos_coord(targets[0]); p2 = get_pos_coord(targets[1])
                draw_alignment_line(p1, p2, c_type)
            else:
                for t in targets:
                    tid = t["id"]
                    prim = prim_map.get(tid)
                    if prim["type"] == "Line":
                        s, e = prim["start"], prim["end"]
                        mx = s[0] + (e[0] - s[0]) * 0.2
                        my = s[1] + (e[1] - s[1]) * 0.2
                        nx, ny = get_outward_normal(s, e)
                        off = scale_ref * 0.4
                        fx, fy = mx + nx*off, my + ny*off
                        label = "H" if c_type == "Horizontal" else "V"
                        ax.text(fx, fy, label, color='red', fontsize=11, weight='bold', ha='center', va='center',
                                bbox=dict(facecolor='white', edgecolor='red', boxstyle='square,pad=0.1', alpha=1.0))
                    else:
                        mx, my = get_pos_coord({"id": tid, "pos": 0})
                        mx, my = get_label_offset(mx, my)
                        ax.text(mx, my, c_type[0], color='red', fontsize=12, **text_props,
                                bbox=dict(facecolor='white', edgecolor='red', boxstyle='circle,pad=0.1'))
            continue

        # 8. Coincident / Perpendicular
        if c_type in ["Coincident", "Perpendicular"]:
            shared_v = find_shared_vertex(targets)
            if shared_v:
                dx, dy = shared_v
            else:
                coords = [get_pos_coord(t) for t in targets]
                dx = np.mean([p[0] for p in coords])
                dy = np.mean([p[1] for p in coords])

            if c_type == "Perpendicular":
                vx, vy = center_x - dx, center_y - dy
                ln = math.sqrt(vx*vx + vy*vy)
                if ln > 0: vx, vy = vx/ln, vy/ln
                off = scale_ref * 0.8
                ax.text(dx + vx*off, dy + vy*off, "⊥", color='red', fontsize=20, weight='bold', ha='center', va='center')
            else:
                ax.scatter([dx], [dy], color='red', s=50, zorder=20, edgecolors='white', linewidth=1.0)
            continue

        # 9. Others
        for t in targets:
            tid = t["id"]
            mx, my = get_pos_coord({"id": tid, "pos": 0})
            mx, my = get_label_offset(mx, my)
            ax.text(mx, my, c_type[:3], color='red', fontsize=10, **text_props,
                    bbox=dict(facecolor='white', edgecolor='red', pad=0.1))

    ax.set_xlim(view_min_x, view_max_x)
    ax.set_ylim(view_min_y, view_max_y)
    ax.axis('off')

    # Add label in top-left corner if provided
    if title:
        ax.text(
            0.02,  # 2% from left edge in axes coordinates
            0.98,  # 98% from bottom (2% from top) in axes coordinates
            title,
            fontsize=title_fontsize,
            fontweight='bold',
            color='blue',
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.4', linewidth=2, alpha=1.0),
            zorder=1000
        )

    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()


if __name__ == "__main__":
    input_files = sorted(
        glob.glob("/home/ryotaro/github/vitruvion/scripts/prediction_fc_search/history_gt_ar_refined_*.py"),
        key=lambda x: int(x.split("_")[-1].split(".")[0]))
    INPUT_SCRIPT_PATH = input_files[0]
    OUTPUT_IMAGE_PATH = "vlm_judge_input.png"

    assert os.path.isfile(INPUT_SCRIPT_PATH)
    print(f"Checking {os.path.basename(INPUT_SCRIPT_PATH)}")

    # 1. 実行・抽出・クローズ
    data = load_and_extract_from_file(INPUT_SCRIPT_PATH)

    if data:
        # 2. JSONとして表示（デバッグ用）
        # print(json.dumps(data, indent=2))

        # 3. 可視化
        visualize_sketch_for_vlm(data, save_path=OUTPUT_IMAGE_PATH)