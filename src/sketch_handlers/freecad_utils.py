from typing import TYPE_CHECKING, Any
import sys
import os


def ensure_freecad_imported(libpath: str = "~/github/FreeCAD/build/release/lib") -> tuple[Any, Any, Any]:
    """
    Safely imports and returns FreeCAD modules (FreeCAD, Part, Sketcher).
    Ensures the path is set up correctly before importing.

    Returns:
        tuple: (FreeCAD, Part, Sketcher) modules.

    Raises:
        ImportError: If FreeCAD modules cannot be found or imported.
    """
    # 1. Ensure FreeCAD path is in sys.path
    # This is typically handled in __init__.py, but safe to double-check here
    freecad_lib = os.path.expanduser(libpath)
    if freecad_lib not in sys.path:
        sys.path.insert(0, freecad_lib)

    try:
        import FreeCAD
        import Part
        import Sketcher
        return FreeCAD, Part, Sketcher
    except ImportError as e:
        raise ImportError(f"Failed to import FreeCAD modules: {e}. "
                          "Ensure you are running in an environment with FreeCAD available "
                          "or that the path is configured correctly.") from e


if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


def load_freecad_commands(commands_or_file: list[str] | str) -> tuple['FreeCAD.Document', 'Sketcher.SketchObject']:
    """
    Load freecad commands to create a sketch.
    """
    if isinstance(commands_or_file, str):
        if os.path.isfile(commands_or_file):
            with open(commands_or_file, 'r') as f:
                script_content = f.read()
        else:
            script_content = commands_or_file
    else:
        script_content = "\n".join(commands_or_file)

    FreeCAD, Part, Sketcher = ensure_freecad_imported()
    exec_globals = {
        "FreeCAD": FreeCAD,
        "Part": Part,
        "Sketcher": Sketcher,
        "Vector": FreeCAD.Vector,
    }
    fc_sketch = None
    initial_docs = set(FreeCAD.listDocuments().keys())

    try:
        exec(script_content, exec_globals)

        # 1. Find the target sketch from exec_globals
        for key, val in exec_globals.items():
            if hasattr(val, "TypeId") and val.TypeId == "Sketcher::SketchObject":
                fc_sketch = val
                break

        # 2. If not found in globals, search newly created documents
        current_docs = set(FreeCAD.listDocuments().keys())
        new_docs = list(current_docs - initial_docs)

        if fc_sketch is None and new_docs:
            for doc_name in new_docs:
                doc_obj = FreeCAD.getDocument(doc_name)
                sketches = doc_obj.findObjects(Type="Sketcher::SketchObject")
                if sketches:
                    fc_sketch = sketches[0]
                    break

        if fc_sketch is None:
            raise ValueError(f"No sketches found in {commands_or_file}")

        doc = fc_sketch.Document
        if not doc:
            raise ValueError("Sketch is found but not attached to any document")

        return doc, fc_sketch

    except Exception as e:
        # Safely clean up ONLY the documents created during this execution
        current_docs = set(FreeCAD.listDocuments().keys())
        for doc_name in (current_docs - initial_docs):
            FreeCAD.closeDocument(doc_name)
        raise e


def dump_freecad_commands(fc_sketch: 'Sketcher.SketchObject', doc_name: str = "MySketch") -> list[str]:
    """
    Dump the freecad commands to reconstruct the sketch.

    Args:
        fc_sketch: The FreeCAD sketch object.
        doc_name: Document name for the FreeCAD project.

    Returns:
        A list of strings representing the history commands.
    """
    commands = []
    commands.append("import FreeCAD")
    commands.append("import Part")
    commands.append("import Sketcher")
    commands.append(f"doc = FreeCAD.newDocument('{doc_name}')")
    commands.append("body = doc.addObject('PartDesign::Body', 'Body')")
    commands.append("fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')")
    commands.append("xy_plane = body.Origin.OriginFeatures[3]")
    commands.append("fc_sketch.AttachmentSupport = xy_plane")
    commands.append("fc_sketch.MapMode = 'FlatFace'")

    # 1. Dump Geometry
    for i, geo in enumerate(fc_sketch.Geometry):
        geo_type = type(geo).__name__
        is_const = fc_sketch.getConstruction(i)

        if geo_type == "LineSegment":
            p1 = geo.StartPoint
            p2 = geo.EndPoint
            cmd = f"fc_sketch.addGeometry(Part.LineSegment(FreeCAD.Vector({p1.x}, {p1.y}, 0), FreeCAD.Vector({p2.x}, {p2.y}, 0)), {is_const})"
            commands.append(cmd)

        elif geo_type == "Circle":
            c = geo.Center
            r = geo.Radius
            cmd = f"fc_sketch.addGeometry(Part.Circle(FreeCAD.Vector({c.x}, {c.y}, 0), FreeCAD.Vector(0, 0, 1), {r}), {is_const})"
            commands.append(cmd)

        elif geo_type == "ArcOfCircle":
            p_start = geo.StartPoint
            # FreeCAD ArcOfCircle uses 3 points or Center/Radius/Start/End, we'll use 3 points (Start, Mid, End)
            import math
            sp = geo.FirstParameter
            ep = geo.LastParameter
            # Handle modulus math carefully
            mid_param = (sp + ep) / 2.0
            p_mid = geo.value(mid_param)
            p_end = geo.EndPoint
            cmd = f"fc_sketch.addGeometry(Part.ArcOfCircle(FreeCAD.Vector({p_start.x}, {p_start.y}, 0), FreeCAD.Vector({p_mid.x}, {p_mid.y}, 0), FreeCAD.Vector({p_end.x}, {p_end.y}, 0)), {is_const})"
            commands.append(cmd)

        elif geo_type == "Point":
            cmd = f"fc_sketch.addGeometry(Part.Point(FreeCAD.Vector({geo.X}, {geo.Y}, 0)), {is_const})"
            commands.append(cmd)

        else:
            raise ValueError(f"Unsupported geometry type: {geo_type} for index {i}")

    # 2. Dump Constraints
    for i, c in enumerate(fc_sketch.Constraints):
        t = c.Type
        args = [f"'{t}'"]

        if t in ['Radius', 'Diameter']:
            args.extend([str(c.First), f"{c.Value}"])
        elif t in ['Horizontal', 'Vertical']:
            if c.FirstPos == 0 and c.Second <= 0: # single line
                args.extend([str(c.First)])
            else: # two points
                args.extend([str(c.First), str(c.FirstPos), str(c.Second), str(c.SecondPos)])
        elif t in ['Parallel', 'Perpendicular', 'Equal', 'Concentric']:
            args.extend([str(c.First), str(c.Second)])
        elif t == 'PointOnObject':
            args.extend([str(c.First), str(c.FirstPos), str(c.Second)])
        elif t == 'Tangent':
            if c.FirstPos == 0 and c.SecondPos == 0:
                args.extend([str(c.First), str(c.Second)])
            else:
                args.extend([str(c.First), str(c.FirstPos), str(c.Second), str(c.SecondPos)])
        elif t == 'Symmetric':
            args.extend([str(c.First), str(c.FirstPos), str(c.Second), str(c.SecondPos), str(c.Third), str(c.ThirdPos)])
        elif t in ['Distance', 'DistanceX', 'DistanceY', 'Angle']:
            if c.FirstPos == 0 and c.SecondPos == 0:
                if c.Second >= 0:
                    args.extend([str(c.First), str(c.Second), f"{c.Value}"])
                else:
                    args.extend([str(c.First), f"{c.Value}"])
            elif c.FirstPos != 0 and c.SecondPos == 0:
                args.extend([str(c.First), str(c.FirstPos), str(c.Second), f"{c.Value}"])
            else:
                args.extend([str(c.First), str(c.FirstPos), str(c.Second), str(c.SecondPos), f"{c.Value}"])
        elif t == 'Block':
            args.extend([str(c.First)])
        else:
            # Coincident / Fallback
            args.extend([str(c.First), str(c.FirstPos), str(c.Second), str(c.SecondPos)])

        cmd = f"fc_sketch.addConstraint(Sketcher.Constraint({', '.join(args)}))"
        commands.append(cmd)

    commands.append("doc.recompute()")
    return commands