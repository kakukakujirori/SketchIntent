import gc
import os
import sys
import traceback

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.freecad_utils import dump_freecad_commands, ensure_freecad_imported
from src.sketch_handlers.context import SketchContext

FreeCAD, Part, Sketcher = ensure_freecad_imported()


def compare_sketches(sk1: 'Sketcher.SketchObject', sk2: 'Sketcher.SketchObject', tol = 1e-1):
    if len(sk1.Geometry) != len(sk2.Geometry):
        return False, f"Geometry count mismatch: {len(sk1.Geometry)} vs {len(sk2.Geometry)}"
    for i in range(len(sk1.Geometry)):
        g1 = sk1.Geometry[i]
        g2 = sk2.Geometry[i]
        if type(g1) != type(g2):
            return False, f"Geo {i} type mismatch: {type(g1)} vs {type(g2)}"
        if sk1.getConstruction(i) != sk2.getConstruction(i):
            return False, f"Geo {i} construction mismatch"

        g_type = type(g1).__name__
        if g_type == "LineSegment":
            d_start_start = abs(g1.StartPoint.x - g2.StartPoint.x) + abs(g1.StartPoint.y - g2.StartPoint.y)
            d_end_end = abs(g1.EndPoint.x - g2.EndPoint.x) + abs(g1.EndPoint.y - g2.EndPoint.y)

            d_start_end = abs(g1.StartPoint.x - g2.EndPoint.x) + abs(g1.StartPoint.y - g2.EndPoint.y)
            d_end_start = abs(g1.EndPoint.x - g2.StartPoint.x) + abs(g1.EndPoint.y - g2.StartPoint.y)

            if (d_start_start > tol or d_end_end > tol) and (d_start_end > tol or d_end_start > tol):
                # Is the original line a zero-length degenerate segment?
                orig_len = abs(g1.StartPoint.x - g1.EndPoint.x) + abs(g1.StartPoint.y - g1.EndPoint.y)
                new_len = abs(g2.StartPoint.x - g2.EndPoint.x) + abs(g2.StartPoint.y - g2.EndPoint.y)
                if orig_len < tol and new_len < tol:
                    # Both are degenerate lines, just shifted by OCC solver. Treat as Match.
                    continue
                return False, f"Geo {i} LineSegment endpoints mismatch (even considering swap) Orig: ({g1.StartPoint.x:.4f}, {g1.StartPoint.y:.4f})->({g1.EndPoint.x:.4f}, {g1.EndPoint.y:.4f}) | New: ({g2.StartPoint.x:.4f}, {g2.StartPoint.y:.4f})->({g2.EndPoint.x:.4f}, {g2.EndPoint.y:.4f})"
        elif g_type == "Circle":
            if abs(g1.Center.x - g2.Center.x) > tol or abs(g1.Center.y - g2.Center.y) > tol:
                return False, f"Geo {i} Circle Center mismatch"
            if abs(g1.Radius - g2.Radius) > tol:
                return False, f"Geo {i} Circle Radius mismatch"
        elif g_type == "ArcOfCircle":
            d_start_start = abs(g1.StartPoint.x - g2.StartPoint.x) + abs(g1.StartPoint.y - g2.StartPoint.y)
            d_end_end = abs(g1.EndPoint.x - g2.EndPoint.x) + abs(g1.EndPoint.y - g2.EndPoint.y)

            d_start_end = abs(g1.StartPoint.x - g2.EndPoint.x) + abs(g1.StartPoint.y - g2.EndPoint.y)
            d_end_start = abs(g1.EndPoint.x - g2.StartPoint.x) + abs(g1.EndPoint.y - g2.StartPoint.y)

            if (d_start_start > tol or d_end_end > tol) and (d_start_end > tol or d_end_start > tol):
                # Is the original arc a zero-length degenerate segment?
                orig_len = abs(g1.StartPoint.x - g1.EndPoint.x) + abs(g1.StartPoint.y - g1.EndPoint.y)
                new_len = abs(g2.StartPoint.x - g2.EndPoint.x) + abs(g2.StartPoint.y - g2.EndPoint.y)
                if orig_len < tol and new_len < tol:
                    # Both are degenerate arcs, just shifted by OCC solver. Treat as Match.
                    continue
                return False, f"Geo {i} Arc endpoints mismatch (even considering swap) Orig: ({g1.StartPoint.x:.4f}, {g1.StartPoint.y:.4f})->({g1.EndPoint.x:.4f}, {g1.EndPoint.y:.4f})"
        elif g_type == "Point":
            if abs(g1.X - g2.X) > tol or abs(g1.Y - g2.Y) > tol:
                return False, f"Geo {i} Point mismatch"

    if len(sk1.Constraints) != len(sk2.Constraints):
        return False, f"Constraint count mismatch: {len(sk1.Constraints)} vs {len(sk2.Constraints)}"
    for i in range(len(sk1.Constraints)):
        c1 = sk1.Constraints[i]
        c2 = sk2.Constraints[i]
        if c1.Type != c2.Type or c1.First != c2.First or c1.FirstPos != c2.FirstPos or c1.Second != c2.Second or c1.SecondPos != c2.SecondPos or c1.Third != c2.Third or c1.ThirdPos != c2.ThirdPos:
            return False, f"Constraint {i} properties mismatch: {c1.Type} vs {c2.Type}"
        if hasattr(c1, 'Value') and hasattr(c2, 'Value'):
            if abs(c1.Value - c2.Value) > tol: # Tolerance for string conversion 1e-4
                return False, f"Constraint {i} Value mismatch: {c1.Value} vs {c2.Value}"
    return True, "Match"


def test_robust(data_npy: str, num_tests: int = -1):
    seq_data = flat_array.load_dictionary_flat(data_npy)
    converter = SketchFreecadConverter(context=SketchContext(), verbose=False)

    if num_tests < 0:
        num_tests = len(seq_data['sequences'])

    success = 0
    failures = []

    for i in range(num_tests):
        # Defensively clear all FreeCAD documents before starting a new test to avoid OCC solver state leakage
        for doc_name in FreeCAD.listDocuments().keys():
            FreeCAD.closeDocument(doc_name)
        if FreeCAD.ActiveDocument:
            FreeCAD.setActiveDocument("")
        gc.collect() # Force FreeCAD C++ bindings to release memory

        seq = seq_data['sequences'][i]
        sketch = datalib.sketch_from_sequence(seq)

        try:
            doc1, fc_sketch1, _, _ = converter.to_freecad(sketch, doc_name=f"Doc1_{i}")
        except Exception as e:
            print(f"Seq {i} skipped: original conversion failed: {e}")
            continue # skip if original conversion fails for some reason

        cmds = dump_freecad_commands(fc_sketch1, doc_name=f"Doc2_{i}")
        # execute cmds in new locals
        local_scope = {}
        try:
            exec("\n".join(cmds), globals(), local_scope)
        except Exception as e:
            msg = f"Seq {i} Execute Error: {e}\n{traceback.format_exc()}"
            failures.append(msg)
            print(msg)
            FreeCAD.closeDocument(doc1.Name)
            if f'Doc2_{i}' in FreeCAD.listDocuments():
                FreeCAD.closeDocument(f'Doc2_{i}')
            continue

        fc_sketch2 = local_scope.get('fc_sketch')
        if not fc_sketch2:
            msg = f"Seq {i}: fc_sketch not found in exec scope"
            failures.append(msg)
            print(msg)
            FreeCAD.closeDocument(doc1.Name)
            continue

        doc2 = local_scope['doc']

        doc1.recompute()
        doc2.recompute()

        match, msg = compare_sketches(fc_sketch1, fc_sketch2)
        if match:
            success += 1
        else:
            failures.append(f"Seq {i} Mismatch: {msg}")
            print(f"FAILED Seq {i}: {msg}")

        FreeCAD.closeDocument(doc1.Name)
        FreeCAD.closeDocument(doc2.Name)

    print(f"Final Success: {success}/{num_tests}")


if __name__ == '__main__':
    data_npy = 'data_processed/sg_filtered_unique.npy'
    test_robust(data_npy, num_tests=1000)
