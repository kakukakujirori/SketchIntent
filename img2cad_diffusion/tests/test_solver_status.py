"""Test script to explore FreeCAD solver status API.

This script:
1. Loads one sample from sg_filtered_unique.npy
2. Converts it to FreeCAD sketch
3. Queries solver status and related information
"""

import os
import sys

# Add FreeCAD Libs
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

import FreeCAD
import Part
import Sketcher

# Import SketchGraphs
from sketchgraphs.data import flat_array, sketch_from_sequence
import sketchgraphs.data as datalib

# Import our converter
from img2cad_diffusion.convert_sequence_to_freecad import convert_sequence_to_freecad, meter_to_millimeter, resolve_subnodes, resolve_redundant_coincident, resolve_tangent, convert_primitive, convert_constraint

def create_sketch_from_sequence(seq):
    """Create FreeCAD sketch from SketchGraphs sequence and return sketch object."""
    doc = FreeCAD.newDocument("TestDoc")
    body = doc.addObject('PartDesign::Body', 'Body')
    sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    xy_plane = body.Origin.OriginFeatures[3]
    sketch.AttachmentSupport = xy_plane
    sketch.MapMode = 'FlatFace'

    # Rescale
    seq = [meter_to_millimeter(op, False) for op in seq]

    # Pass 1: Map Building
    node_idx_map = resolve_subnodes(seq)

    # Pass 2: Constraint Refinement
    seq = resolve_redundant_coincident(seq, node_idx_map, False)
    seq = resolve_tangent(seq, node_idx_map)

    # Pass 3: Generation
    for i, op in enumerate(seq):
        if isinstance(op, datalib.NodeOp):
            convert_primitive(op, sketch)
        elif isinstance(op, datalib.EdgeOp):
            try:
                convert_constraint(op, sketch, node_idx_map)
            except Exception as e:
                print(f"  Constraint error: {e}")

    doc.recompute()
    return doc, sketch


def explore_solver_status(sketch):
    """Explore what solver information is available."""
    print("\n=== Solver Status Exploration ===")

    # Basic solve
    result = sketch.solve()
    print(f"sketch.solve() = {result}")

    # getSolverStatus() - may not exist in all versions
    if hasattr(sketch, 'getSolverStatus'):
        status = sketch.getSolverStatus()
        print(f"sketch.getSolverStatus() = {status}")
    else:
        print("sketch.getSolverStatus() not available")

    # Degrees of Freedom
    if hasattr(sketch, 'getDoF'):
        dof = sketch.getDoF()
        print(f"sketch.getDoF() = {dof}")
    else:
        print("sketch.getDoF() not available")

    # DoF - alternative method
    if hasattr(sketch, 'DoF'):
        print(f"sketch.DoF = {sketch.DoF}")

    # Constraint related methods
    print("\n--- Constraint Info ---")
    print(f"ConstraintCount = {sketch.ConstraintCount}")
    print(f"GeometryCount = {sketch.GeometryCount}")

    # Check for redundant/conflicting constraints
    if hasattr(sketch, 'redundant'):
        print(f"sketch.redundant = {sketch.redundant}")

    if hasattr(sketch, 'conflicting'):
        print(f"sketch.conflicting = {sketch.conflicting}")

    # MalformedConstraints
    if hasattr(sketch, 'MalformedConstraints'):
        print(f"sketch.MalformedConstraints = {sketch.MalformedConstraints}")

    # List all sketch attributes for exploration
    print("\n--- All sketch attributes ---")
    attrs = [a for a in dir(sketch) if not a.startswith('_')]
    solver_related = [a for a in attrs if 'solv' in a.lower() or 'constr' in a.lower() or 'dof' in a.lower() or 'redund' in a.lower() or 'conflict' in a.lower()]
    print(f"Solver-related attributes: {solver_related}")


if __name__ == '__main__':
    # Load one sample
    print("Loading data...")
    seq_data = flat_array.load_dictionary_flat('data_processed/sg_filtered_unique.npy')

    # Try first few samples to find one that works
    for sample_idx in [0, 1, 2, 5, 10]:
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx}")
        print('='*60)

        seq = seq_data['sequences'][sample_idx]

        try:
            doc, sketch = create_sketch_from_sequence(seq)
            explore_solver_status(sketch)

            # Close document
            FreeCAD.closeDocument(doc.Name)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
