"""
Qwen3 VLM Constraint Judge Script

Uses Qwen3 VLM to judge whether GT or Predicted constraints are more natural/preferable for CAD sketches.
Bias avoidance: GT and Pred are randomly assigned as "Sketch A" or "Sketch B" for each sample.
"""

import base64
import glob
import io
import json
import os
import random
import re
import sys
import tempfile

import cairosvg
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler

from v20260116_draw_constraints import visualize_sketch_for_vlm

# Add FreeCAD Libs (conditionally)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part
import Sketcher
from FreeCAD import Vector


# ==========================================
# Model Configuration
# ==========================================
MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf"
MMPROJ_PATH = "/home/ryotaro/github/llama.cpp/models/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf"

# Data directory
DATA_DIR = "/home/ryotaro/github/vitruvion/scripts/prediction_fc_search"


# ==========================================
# Data Extraction Functions
# ==========================================

def parse_vector(vec_str):
    """Extracts numbers from 'Vector(x, y, z)'"""
    nums = re.findall(r"[-+]?\d*\.?\d+", vec_str)
    return [float(n) for n in nums[:2]]


def extract_primitives_from_script(filepath: str) -> list:
    """
    Parse FreeCAD script to extract primitives (addGeometry lines).
    """
    if not os.path.exists(filepath):
        return []

    primitives = []
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('fc_sketch.addGeometry('):
            is_construction = 'True' in line.split(',')[-1]

            if 'Part.LineSegment' in line:
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
                inner_match = re.search(r'Part\.Circle\s*\((.*)\)', line)
                if inner_match:
                    inner_content = inner_match.group(1)
                    vecs = re.findall(r'Vector\s*\([^)]+\)', inner_content)
                    if len(vecs) >= 1:
                        center = parse_vector(vecs[0])
                        v2_end_idx = inner_content.rfind('Vector')
                        if v2_end_idx != -1:
                            v2_close = inner_content.find(')', v2_end_idx)
                            if v2_close != -1:
                                remainder = inner_content[v2_close+1:]
                                radius_match = re.search(r"[-+]?\d*\.?\d+", remainder)
                                if radius_match:
                                    radius = float(radius_match.group(0))
                                    primitives.append({
                                        "type": "Circle",
                                        "center": center,
                                        "radius": radius,
                                        "is_construction": is_construction
                                    })
            elif 'Part.ArcOfCircle' in line:
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

            else:
                raise NotImplementedError(f"Unknown primitive type: {line}")

    return primitives


def extract_constraints_from_script(filepath: str) -> list:
    """
    Parse FreeCAD script to extract constraints (addConstraint lines).
    Returns list of constraint command strings.
    """
    if not os.path.exists(filepath):
        return []

    constraints = []
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('fc_sketch.addConstraint('):
            # Extract the constraint definition
            inner = re.search(r"Sketcher\.Constraint\((.*)\)", line)
            if inner:
                args_str = inner.group(1).rstrip(')')
                raw_args = [a.strip().strip("'").strip('"') for a in args_str.split(',')]
                if raw_args:
                    c_type = raw_args[0]
                    c_args = []
                    for a in raw_args[1:]:
                        try:
                            if '.' in a:
                                c_args.append(float(a))
                            else:
                                c_args.append(int(a))
                        except:
                            c_args.append(a)
                    constraints.append({"type": c_type, "args": c_args})

    return constraints


def format_constraint_to_json(ctype: str, args: list) -> dict:
    """Convert raw FreeCAD constraint arguments to structured JSON format.

    Based on FreeCAD constraint API signatures from convert_sketch_to_freecad.py.

    Args:
        ctype: Constraint type name (e.g., 'Coincident', 'Horizontal', 'Tangent')
        args: List of constraint arguments from parse_freecad_script

    Returns:
        dict with keys: type, entity1, subnode1, entity2, subnode2, entity3, subnode3, value
    """
    c_out = {
        "type": ctype,
        "entity1": None, "subnode1": None,
        "entity2": None, "subnode2": None,
        "entity3": None, "subnode3": None,
        "value": None
    }

    def get_arg(idx):
        return args[idx] if idx < len(args) else None

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
            c_out["value"] = get_arg(1)
        elif len(args) >= 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["value"] = get_arg(4)

    elif ctype in ["Diameter", "Radius"]:
        # Diameter(e1, value) - 2 args
        # Radius(e1, value) - 2 args
        c_out["entity1"] = get_arg(0)
        c_out["value"] = get_arg(1)

    elif ctype == "Angle":
        # Angle(e1, value) - 2 args (single line angle)
        # Angle(e1, e2, value) - 3 args (two lines)
        # Angle(e1, s1, e2, s2, value) - 5 args (two points)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
            c_out["value"] = get_arg(1)
        elif len(args) == 3:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
            c_out["value"] = get_arg(2)
        elif len(args) >= 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            c_out["value"] = get_arg(4)

    else:
        # Unknown constraint type - just store the raw args
        c_out["entity1"] = get_arg(0) if len(args) > 0 else None
        if len(args) > 1:
            c_out["entity2"] = get_arg(1)

    return c_out


# ==========================================
# Image Generation Functions
# ==========================================

from collections import defaultdict
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def execute_script_and_export_dict(filepath: str) -> dict | None:
    """
    Execute a FreeCAD script file and export the sketch data as a dictionary.
    Similar to load_and_extract_from_file but returns structured data for visualization.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    exec_globals = {
        "FreeCAD": FreeCAD,
        "Part": Part,
        "Sketcher": Sketcher,
        "Vector": FreeCAD.Vector
    }

    with open(filepath, 'r') as f:
        script_content = f.read()

    try:
        exec(script_content, exec_globals)
    except Exception as e:
        print(f"Error executing script: {e}")
        return None

    target_sketch = None
    for key, val in exec_globals.items():
        if hasattr(val, "TypeId") and val.TypeId == "Sketcher::SketchObject":
            target_sketch = val
            break

    if target_sketch is None and FreeCAD.ActiveDocument:
        for obj in FreeCAD.ActiveDocument.Objects:
            if obj.TypeId == "Sketcher::SketchObject":
                target_sketch = obj
                break

    if target_sketch is None:
        print("Error: No Sketch object found after script execution.")
        if FreeCAD.ActiveDocument:
            FreeCAD.closeDocument(FreeCAD.ActiveDocument.Name)
        return None

    # Export to dict
    drawing_data = export_sketch_to_dict(target_sketch)

    # Close document
    doc_name = target_sketch.Document.Name
    FreeCAD.closeDocument(doc_name)

    return drawing_data


def export_sketch_to_dict(sketch_obj) -> dict:
    """Export sketch geometry and constraints to dictionary format."""
    drawing_data = {
        "primitives": [],
        "constraints": []
    }

    # Primitives
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
        else:
            raise ValueError(f"Unknown geometry type: {type(geom)}")

        prim_data["construction"] = sketch_obj.getConstruction(i)
        drawing_data["primitives"].append(prim_data)

    # Constraints
    for const in sketch_obj.Constraints:
        c_data = {
            "type": const.Type,
            "target": []
        }

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


def create_primitives_only_data(full_data: dict) -> dict:
    """Create a copy of data with only primitives (no constraints)."""
    return {
        "primitives": full_data["primitives"],
        "constraints": []
    }


def generate_images_for_comparison(
    gt_filepath: str,
    pred_filepath: str,
    output_dir: str,
    sample_idx: int,
    is_gt_first: bool = True
):
    """
    Generate three images for comparison:
    1. Primitives only
    2. Sketch A with constraints (GT or Pred based on is_gt_first)
    3. Sketch B with constraints (the other)

    Args:
        gt_filepath: Path to GT script
        pred_filepath: Path to Pred script
        output_dir: Directory to save images
        sample_idx: Sample index for naming
        is_gt_first: If True, GT is Sketch A; if False, Pred is Sketch A

    Returns:
        Tuple of (prim_path, sketch_a_path, sketch_b_path, gt_data, pred_data)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Execute GT script and get data
    gt_data = execute_script_and_export_dict(gt_filepath)
    if gt_data is None:
        return None, None, None, None, None

    # Execute Pred script and get data
    pred_data = execute_script_and_export_dict(pred_filepath)
    if pred_data is None:
        return None, None, None, None, None

    # Generate image paths
    prim_only_path = os.path.join(output_dir, f"primitives_only_{sample_idx}.png")
    sketch_a_path = os.path.join(output_dir, f"sketch_a_{sample_idx}.png")
    sketch_b_path = os.path.join(output_dir, f"sketch_b_{sample_idx}.png")

    # Primitives only (use GT data but remove constraints)
    prim_only_data = create_primitives_only_data(gt_data)
    visualize_sketch_for_vlm(prim_only_data, prim_only_path, title="Primitives", title_fontsize=20)

    # Assign A and B based on is_gt_first
    if is_gt_first:
        sketch_a_data = gt_data
        sketch_b_data = pred_data
    else:
        sketch_a_data = pred_data
        sketch_b_data = gt_data

    # Generate Sketch A and B images with labels
    visualize_sketch_for_vlm(sketch_a_data, sketch_a_path, title="Sketch A", title_fontsize=20)
    visualize_sketch_for_vlm(sketch_b_data, sketch_b_path, title="Sketch B", title_fontsize=20)

    return prim_only_path, sketch_a_path, sketch_b_path, gt_data, pred_data


# ==========================================
# Qwen3 Judge Functions
# ==========================================

def prepare_judge_prompt(
    primitives_image_path: str,
    sketch_a_image_path: str,
    sketch_b_image_path: str,
    primitives: list,
    sketch_a_constraints: list,
    sketch_b_constraints: list
) -> list:
    """
    Prepare messages for Qwen3 to judge which sketch has better constraints.
    """
    # Encode images
    def encode_image(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    prim_img = encode_image(primitives_image_path)
    sketch_a_img = encode_image(sketch_a_image_path)
    sketch_b_img = encode_image(sketch_b_image_path)

    # Format constraints for display
    sketch_a_constraints_str = json.dumps(sketch_a_constraints, indent=2)
    sketch_b_constraints_str = json.dumps(sketch_b_constraints, indent=2)
    primitives_str = json.dumps(primitives, indent=2)

    system_prompt = """You are an expert CAD engineer specializing in parametric 2D sketching.

Your task is to evaluate two sets of geometric constraints applied to the same 2D CAD sketch primitives.
You will judge which set of constraints results in a more **natural** and **editable** CAD sketch delivering the design intent without overly distorting the original shape.

[Primitives Format]
The input consists of primitives with the following formats:
- Point(x, y, is_construction)
- LineSegment((start_x, start_y), (end_x, end_y), is_construction)
- Circle((center_x, center_y), is_clockwise, radius, is_construction)
- Arc((start_x, start_y), (mid_x, mid_y), (end_x, end_y), is_construction)
* Note: mid_x/y for Arc is a point *on the arc*, not the center or the mid-coord point.
The primitives are implicitly indexed starting from 0 based on their order.

[Constraint Format]
Each constraint follows this structure:
{
  "type": "ConstraintTypeName",
  "entity1": <int or null>,
  "subnode1": <int or null>,
  "entity2": <int or null>,
  "subnode2": <int or null>,
  "entity3": <int or null>,
  "subnode3": <int or null>,
  "value": <float or null>
}

- entityN: The 0-based index of the primitive
- subnodeN: Which part of the primitive is referenced:
    - 0: The Entity itself (e.g., the edge of a line or circle)
    - 1: StartPoint
    - 2: EndPoint
    - 3: CenterPoint (strictly the center of a Circle or Arc)
- value: For dimensional constraints (Distance, Radius, Diameter, Angle)

[Constraint Types and Signatures]
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
    * Fixes distance between two points or a point to a curve.
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

[Evaluation Criteria]
1. **Naturalness**: Do the constraints capture the intended geometric relationships that a human CAD designer would expect?
2. **Editability**: Would the constraints allow for easy modifications without breaking the design intent?
3. **Non-destructive**: Do the constraints preserve the original shape except for the global scale and shift?
4. **Completeness**: Are the constraints sufficient to fully define the geometry (ideally DoF=2, leaving only global translation)?
5. **Redundancy**: Are there unnecessary or conflicting constraints?
6. **Best Practices**: Do the constraints follow CAD best practices (e.g., avoid excessive dimension constraints)?

[Note]
- Do to the type and order of applying constraints, the absolute sketch sizes are different between the two sketches. This difference should not affect the evaluation, i.e., we don't care whether one edge length is 999.0 mm or 1000.0 mm.
- In some cases, constraints on the sketch are overlapped and a bit hard to recognize. Since this is an issue on the sketch drawing, don't rely too much on the sketch visual cues, and mainly focus on the constraint definition commands for evaluation.

[Response Format]
You MUST respond with ONLY a valid JSON object. No other text is allowed.
{
  "choice": "A" or "B",
  "reasoning": "Brief explanation of why you chose this sketch."
}
Your response must start with '{' and end with '}'. No other text is allowed."""

    user_content = [
        {"type": "text", "text": "I will show you three images and two sets of constraints.\n\n**Image 1**: The base primitives (geometry only, no constraints)\n**Image 2**: Sketch A (primitives + Constraint Set A)\n**Image 3**: Sketch B (primitives + Constraint Set B)\n\nPlease analyze which constraint set (A or B) is more natural and editable."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prim_img}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sketch_a_img}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sketch_b_img}"}},
        {"type": "text", "text": f"""
[Primitives (shared by both sketches)]
```json
{primitives_str}
```

[Constraint Set A]
```json
{sketch_a_constraints_str}
```

[Constraint Set B]
```json
{sketch_b_constraints_str}
```

Based on the images and constraint definitions, which sketch (A or B) has more natural, editable CAD constraints?
Respond with a JSON object containing "choice" (either "A" or "B") and "reasoning"."""}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return messages


def extract_json_from_thinking_output(text: str) -> str:
    """Extract JSON from Thinking model output."""
    # Remove </think> and everything before it
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Find JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group(0)

    return text


# ==========================================
# Main Execution
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3 VLM Constraint Judge")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--output_dir", type=str, default="qwen_judge_images", help="Directory for generated images")
    args = parser.parse_args()

    # Find all sample indices
    gt_files = sorted(glob.glob(os.path.join(DATA_DIR, "history_gt_ar_refined_*.py")))
    sample_indices = []
    for f in gt_files:
        match = re.search(r'history_gt_ar_refined_(\d+)\.py$', f)
        if match:
            sample_indices.append(int(match.group(1)))

    if args.num_samples:
        sample_indices = sample_indices[:args.num_samples]

    print(f"Processing {len(sample_indices)} samples...")

    # Initialize model
    print("Loading Qwen3 VLM model...")
    llm = Llama(
        model_path=MODEL_PATH,
        chat_handler=Qwen3VLChatHandler(
            clip_model_path=MMPROJ_PATH,
            force_reasoning=True,
            image_min_tokens=1024,
        ),
        n_gpu_layers=-1,
        n_ctx=65536,
        swa_full=True,
    )

    # Results storage
    results = []
    gt_wins = 0
    pred_wins = 0

    # Results path
    results_path = os.path.join(args.output_dir, "judge_results.jsonl")
    print(f"Results will be appended to: {results_path}")

    # Check for existing results to allow resuming
    if os.path.exists(results_path):
        print("Found existing results file. checking for processed samples...")
        with open(results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    idx = data.get("sample_idx")
                    if idx is not None:
                        # Add to results list for summary calculation
                        results.append(data)
                        if data.get("winner") == "GT":
                            gt_wins += 1
                        elif data.get("winner") == "Pred":
                            pred_wins += 1
                except json.JSONDecodeError:
                    continue

        existing_indices = set(r["sample_idx"] for r in results)
        print(f"Found {len(existing_indices)} already processed samples.")

        # Filter indices to process
        original_count = len(sample_indices)
        sample_indices = [idx for idx in sample_indices if idx not in existing_indices]
        print(f"Resuming: {len(sample_indices)} samples remaining (skipped {original_count - len(sample_indices)})")

    for idx in sample_indices:
        print(f"\n=== Processing sample {idx} ===")

        gt_file = os.path.join(DATA_DIR, f"history_gt_ar_refined_{idx}.py")
        pred_file = os.path.join(DATA_DIR, f"history_pred_with_solver_{idx}.py")

        if not os.path.exists(gt_file) or not os.path.exists(pred_file):
            print(f"Skipping sample {idx}: files not found")
            continue

        # Random A/B assignment for bias avoidance
        is_gt_first = random.choice([True, False])

        # Generate images with A/B labels
        prim_img, sketch_a_img, sketch_b_img, gt_data, pred_data = generate_images_for_comparison(
            gt_file, pred_file, args.output_dir, idx, is_gt_first=is_gt_first
        )

        if prim_img is None:
            print(f"Skipping sample {idx}: image generation failed")
            continue

        # Extract primitives and constraints
        primitives = extract_primitives_from_script(gt_file)
        gt_constraints_raw = extract_constraints_from_script(gt_file)
        pred_constraints_raw = extract_constraints_from_script(pred_file)

        # Convert constraints to structured JSON format
        gt_constraints = [format_constraint_to_json(c["type"], c["args"]) for c in gt_constraints_raw]
        pred_constraints = [format_constraint_to_json(c["type"], c["args"]) for c in pred_constraints_raw]

        # Assign constraints based on A/B order
        if is_gt_first:
            sketch_a_constraints, sketch_a_label = gt_constraints, "GT"
            sketch_b_constraints, sketch_b_label = pred_constraints, "Pred"
        else:
            sketch_a_constraints, sketch_a_label = pred_constraints, "Pred"
            sketch_b_constraints, sketch_b_label = gt_constraints, "GT"

        print(f"  A = {sketch_a_label}, B = {sketch_b_label}")

        # Prepare prompt
        messages = prepare_judge_prompt(
            prim_img, sketch_a_img, sketch_b_img,
            primitives, sketch_a_constraints, sketch_b_constraints
        )

        # Call Qwen3 with streaming
        llm.reset()
        output_text = ""
        token_count = 0

        result_entry = None

        try:
            print("  Calling Qwen3 (streaming)...")
            for chunk in llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=32768,
            ):
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    output_text += content
                    token_count += 1
                    # Stream thinking output
                    sys.stdout.write(content)
                    sys.stdout.flush()

            print()  # Newline after streaming
            print(f"  Total tokens: {token_count}")

            # Extract JSON
            json_text = extract_json_from_thinking_output(output_text)
            result = json.loads(json_text)

            choice = result.get("choice", "")
            reasoning = result.get("reasoning", "")

            # Determine winner
            if choice == "A":
                winner = sketch_a_label
            elif choice == "B":
                winner = sketch_b_label
            else:
                winner = "Unknown"

            if winner == "GT":
                gt_wins += 1
            elif winner == "Pred":
                pred_wins += 1

            result_entry = {
                "sample_idx": idx,
                "a_label": sketch_a_label,
                "b_label": sketch_b_label,
                "choice": choice,
                "winner": winner,
                "reasoning": reasoning
            }

            results.append(result_entry)

            print(f"  Choice: {choice} ({winner})")
            print(f"  Reasoning: {reasoning[:200]}..." if len(reasoning) > 200 else f"  Reasoning: {reasoning}")

        except Exception as e:
            print(f"  Error: {e}")
            result_entry = {
                "sample_idx": idx,
                "a_label": sketch_a_label,
                "b_label": sketch_b_label,
                "choice": "Error",
                "winner": "Error",
                "reasoning": str(e)
            }
            results.append(result_entry)

        # Incrementally write to JSONL
        if result_entry:
            with open(results_path, 'a') as f:
                f.write(json.dumps(result_entry) + '\n')

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"GT wins: {gt_wins} ({100*gt_wins/len(results):.1f}%)" if results else "GT wins: 0")
    print(f"Pred wins: {pred_wins} ({100*pred_wins/len(results):.1f}%)" if results else "Pred wins: 0")

    # Save summary
    summary_path = os.path.join(args.output_dir, "judge_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "summary": {
                "total": len(results),
                "gt_wins": gt_wins,
                "pred_wins": pred_wins
            }
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
