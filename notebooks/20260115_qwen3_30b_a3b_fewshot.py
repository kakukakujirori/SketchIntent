import base64
import gc
import glob
import json
import os
import re
from io import BytesIO

import cairosvg
import PIL.Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler


# --- Model Configuration ---
# Define the path to the main model file
#MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-30B-A3B-Thinking-Q4_K_M.gguf"
MODEL_PATH = "/home/ryotaro/github/llama.cpp/models/Qwen3VL-30B-A3B-Thinking-Q8_0.gguf"
# Define the path to the multi-modal projector file
MMPROJ_PATH = "/home/ryotaro/github/llama.cpp/models/mmproj-Qwen3VL-30B-A3B-Thinking-F16.gguf"

# --- Initialize the Llama Model ---
llm = Llama(
    model_path=MODEL_PATH,
    # Set up the chat handler for Qwen3-VL, specifying the projector path
    chat_handler=Qwen3VLChatHandler(
      clip_model_path=MMPROJ_PATH,
      force_reasoning=True,
      image_min_tokens=1024, # Note: Qwen-VL models require at minimum 1024 image tokens to function correctly on bbox grounding tasks
    ),
    n_gpu_layers=-1,  # Offload all layers to the GPU
    n_ctx=65536, #32768,      # Set the context window size
    swa_full=True,
)


def parse_vector(vec_str):
    # Extracts numbers from "Vector(x, y, z)"
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", vec_str)
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


def export_to_freecad_script(primitives: list, constraints_json: dict, output_path: str):
    """
    Convert parsed primitives and inferred constraints JSON to a FreeCAD Python script.

    Args:
        primitives: List of primitive dicts (from parse_freecad_script)
        constraints_json: Dict with 'constraints' key containing list of constraint items
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
            lines.append(f"fc_sketch.addGeometry(Part.ArcOfCircle(Part.Circle(Vector({sx}, {sy}, 0), Vector({mx}, {my}, 0), Vector({ex}, {ey}, 0)), ), {is_const})")
            # Note: ArcOfCircle construction may need adjustment based on actual FreeCAD API

    lines.append("")
    lines.append("# --- Add Constraints ---")

    constraints = constraints_json.get("constraints", [])
    for c in constraints:
        ctype = c.get("type", "")
        e1 = c.get("entity1")
        s1 = c.get("subnode1")
        e2 = c.get("entity2")
        s2 = c.get("subnode2")
        e3 = c.get("entity3")
        s3 = c.get("subnode3")

        # Build constraint arguments based on type
        if ctype in ["Coincident", "Tangent"]:
            # 4-arg: entity1, subnode1, entity2, subnode2
            if s1 is not None and s2 is not None:
                lines.append(f"fc_sketch.addConstraint(Sketcher.Constraint('{ctype}', {e1}, {s1}, {e2}, {s2}))")
            else:
                lines.append(f"# Skipped {ctype}: missing subnode info")
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
        elif ctype in ["Distance", "Angle"]:
            # Dimensional - value would be needed but we skip value inference
            lines.append(f"# Skipped {ctype}: dimensional constraint (value inference not supported yet)")
        elif ctype in ["Radius", "Diameter"]:
            lines.append(f"# Skipped {ctype}: dimensional constraint")
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
            # args[1] is the value, skip
        elif len(args) >= 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)
            # args[4] is the value, skip

    elif ctype in ["Diameter", "Radius"]:
        # Diameter(e1, value) - 2 args
        # Radius(e1, value) - 2 args
        c_out["entity1"] = get_arg(0)
        # args[1] is the value, skip

    elif ctype == "Angle":
        # Angle(e1, value) - 2 args (single line angle)
        # Angle(e1, e2, value) - 3 args (two lines)
        # Angle(e1, s1, e2, s2, value) - 5 args (two points)
        if len(args) == 2:
            c_out["entity1"] = get_arg(0)
        elif len(args) == 3:
            c_out["entity1"] = get_arg(0)
            c_out["entity2"] = get_arg(1)
        elif len(args) >= 5:
            c_out["entity1"] = get_arg(0)
            c_out["subnode1"] = get_arg(1)
            c_out["entity2"] = get_arg(2)
            c_out["subnode2"] = get_arg(3)

    else:
        raise NotImplementedError(f"Unknown constraint type: {ctype}")

    return c_out


def prepare_prompts(primitives_list: list, image_path_list: str | list[str], example_files: list[str] = []):
    # 1. Collect all images (Target images + Example images are handled differently in prompt construction)
    # Actually, based on the plan, we want to present Example Images first, then Target Image.
    # But image_path_list argument in this function signature was originally for the target.

    # Let's normalize the target image list
    if isinstance(image_path_list, str):
        target_image_path_list = [image_path_list]
    else:
        target_image_path_list = image_path_list

    # 2. Prepare Example Content
    all_encoded_images = []
    example_text_block = ""

    # Process examples
    for i, ex_file in enumerate(example_files):
        # Parse example
        ex_primitives, ex_constraints_gt = parse_freecad_script(ex_file)

        # Determine example image path
        # Assuming .py -> .svg convention
        # Check if the file name follows the pattern "history_gt_ar_refined_*.py" -> "gt_sketch_*.svg"
        if "history_gt_ar_refined_" in ex_file:
            ex_svg_path = ex_file.replace("history_gt_ar_refined_", "gt_sketch_").replace(".py", ".svg")
        else:
            # Fallback or strict assumption? Let's assume strict or simple replace if needed.
            # If naming is different, might need adjustment. For now assume same pattern as main.
            ex_svg_path = ex_file.replace(".py", ".svg")

        if not os.path.exists(ex_svg_path):
            print(f"Warning: Example image not found: {ex_svg_path}")
            continue

        # Encode example image
        png_bytes = cairosvg.svg2png(url=ex_svg_path, scale=1.0)
        encoded_data = base64.b64encode(png_bytes).decode("utf-8")
        all_encoded_images.append(encoded_data)

        # Format primitives
        ex_prim_text = json.dumps(ex_primitives, indent=2)

        # Format constraints to match the prompt schema
        formatted_constraints = [format_constraint_to_json(c["type"], c["args"]) for c in ex_constraints_gt]

        # Format constraints
        ex_cons_text = json.dumps({"constraints": formatted_constraints}, indent=2)

        # Append to example text block
        # Image index is i+1 (1-based)
        example_text_block += f"""
[Example {i+1}]
Reference Image: The {i+1}-th image provided.

[Primitives]
```
{ex_prim_text}
```
[Constraints]
```
{ex_cons_text}
```
"""

    # 3. Process Target Image(s)
    # We append target images AFTER example images
    for target_image_path in target_image_path_list:
        assert target_image_path.endswith(".svg"), f"{target_image_path=}"
        png_bytes = cairosvg.svg2png(url=target_image_path, scale=1.0)
        encoded_data = base64.b64encode(png_bytes).decode("utf-8")
        all_encoded_images.append(encoded_data)

    # Target image index starts after examples
    target_image_index = len(example_files) + 1 # 1-based index for the *first* target image (assuming 1 target usually)

    # 4. Primitives info for Target
    target_primitives_text = json.dumps(primitives_list, indent=2)

    # 5. Provide System Prompt
    base_system_prompt_text = """
You are an expert CAD engineer.
Analyze the provided 2D sketch image and the list of geometric primitives.

Your task is to infer the necessary **Geometric Constraints** to fully constrain this sketch.
More specifically, the sketch is fully constrained if its degree-of-freedom is 2, i.e., leaving only the global xy translation.
It is also highly desirable that the constraints are not malformed, conflicting, or redundant.

The grammer follows that of FreeCAD, i.e., the primitives are assigned reference numbers,
and they are used to specify the position on which the constraints are applied.

[Input Format]
The input consists of a list of geometric primitives.
The supported primitives and their arguments are:
- Point(x, y, is_construction)
- LineSegment((start_x, start_y), (end_x, end_y), is_construction)
- Circle((center_x, center_y), is_clockwise, radius, is_construction)
- Arc((start_x, start_y), (mid_x, mid_y), (end_x, end_y), is_construction)
* Note: mid_x/y for Arc is a point *on the arc* (roughly halfway), not the center or the mid-coord point.
The primitives are implicitly indexed starting from 0 (0-indexed) based on their order in the input list.

[Output Format]
Output a list of constraints inferred from the visual arrangement and geometric properties of the primitives.
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
Please strictly adhere to the following signatures for each constraint type. Arguments not listed for a type should be omitted or null.
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

[Rules]
1. Identify relationships strictly based on the visual appearance in the image and the provided IDs.
2. Prioritize geometric relationships (Coincident, Horizontal, Vertical, Tangent, Parallel, Perpendicular, etc.).
3. Only use dimensional constraints (Length, Radius, etc.) if the geometry cannot be fixed by relationships alone.
4. Use the exact IDs provided in the Primitives Information.
5. If you cannot name all the constraints to achieve DoF=2, output the constraints you have identified so far to achieve the lowest DoF.

[Response Format - CRITICAL]
You MUST respond with ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or code blocks.
The JSON must strictly follow this schema:
```
{
  "constraints": [
    {
      "type": "ConstraintTypeName",
      "entity1": 0,
      "subnode1": 0,
      "entity2": null,
      "subnode2": null,
      "entity3": null,
      "subnode3": null
    }
  ]
}
```
Field descriptions:
- "type": String, the constraint type (e.g., "Coincident", "Horizontal", "Parallel", "Tangent", etc.)
- "entity1", "entity2", "entity3": Integer (0-based primitive index) or null if not applicable
- "subnode1", "subnode2", "subnode3": Integer (0=Entity, 1=Start, 2=End, 3=Center) or null if not applicable
Example output for two constraints:
```
{
  "constraints":[
    {
      "type":"Coincident",
      "entity1":0,
      "subnode1":2,
      "entity2":1,
      "subnode2":1,
      "entity3":null,
      "subnode3":null
    },
    {
      "type":"Horizontal",
      "entity1":0,
      "subnode1":null,
      "entity2":null,
      "subnode2":null,
      "entity3":null,
      "subnode3":null
    }
  ]
}
```
Your response must start with '{' and end with '}'. No other text is allowed.
"""

    if example_text_block:
        system_prompt_text = base_system_prompt_text + "\n[Few-Shot Examples]\n" + example_text_block
    else:
        system_prompt_text = base_system_prompt_text

    # 6. Construct User Message Content
    user_messages = []

    # Add all images (Examples + Target)
    for encoded_data in all_encoded_images:
        encoded_text = f"data:image/png;base64,{encoded_data}"
        user_messages.append({"type": "image_url", "image_url": {"url": encoded_text}})

    # Add final text prompt
    user_prompt_text = f"""
Predict all the natural constraints to make the primitives fully constrained, i.e. to make the Degree of Freedom 2.
Use the **last** provided 2D sketch image (the {target_image_index}-th image) as the reference.

[Primitives Information]
```
{target_primitives_text}
```
"""
    user_messages.append({"type": "text", "text": user_prompt_text})

    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_messages}
    ]

    return messages


def extract_json_from_thinking_output(text: str) -> str:
        """Thinkingモデルの出力から思考部分を除去してJSONを抽出する"""
        # </think> タグの後の部分を取得
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        # JSONオブジェクトを正規表現で抽出（{...}の形式）
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            return json_match.group(0)

        return text


if __name__ == '__main__':
    import sys

    #load data
    target_file_list = sorted(glob.glob("/home/ryotaro/github/vitruvion/scripts/prediction_fc_search/history_gt_ar_refined_*.py"))

    # Example files (first 3)
    example_files = target_file_list[0:3]

    for target_file in target_file_list[3:]:

        sample_primitives, constraints_gt = parse_freecad_script(target_file)
        sample_svg_path = target_file.replace("history_gt_ar_refined_", "gt_sketch_").replace(".py", ".svg")
        number = int(target_file.split("_")[-1].replace(".py", ""))

        print(f"=== Processing sample {number} ===")
        print(f"File: {target_file}")
        print(f"Primitives: {len(sample_primitives)} items")
        print(f"Example files: {len(example_files)} items")
        print("=" * 50)

        # ストリーミングモードで実行
        messages = prepare_prompts(sample_primitives, sample_svg_path, example_files=example_files)
        llm.reset()

        output_text = ""
        token_count = 0

        try:
            print("\n--- Streaming Output (Thinking + Response) ---\n")
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
                    # リアルタイム出力
                    sys.stdout.write(content)
                    sys.stdout.flush()

        except RuntimeError as e:
            print(f"\n\n=== RuntimeError after {token_count} tokens ===")
            print(f"Error: {e}")
            print("\n--- Output so far ---")
            print(output_text[-2000:] if len(output_text) > 2000 else output_text)  # 最後の2000文字

            # 途中結果を保存
            with open(f"thinking_output_{number}.txt", "w") as f:
                f.write(output_text)
            print(f"\nFull output saved to: thinking_output_{number}.txt")

        else:
            print("\n\n=== Generation Complete ===")
            print(f"Total tokens: {token_count}")

            # 結果を抽出
            json_text = extract_json_from_thinking_output(output_text)
            print(f"\nExtracted JSON:\n{json_text}")

            try:
                constraints_json = json.loads(json_text)
                export_to_freecad_script(sample_primitives, constraints_json, f"qwen3_8b_output_{number}.py")
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
