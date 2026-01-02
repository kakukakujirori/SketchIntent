"""
Just search for fully constrained data in Vitruvion data
Supports multiprocessing and incremental result writing for resumability.
"""
import argparse
import os
import sys
from multiprocessing import Pool, Lock, Manager
from functools import partial
from tqdm import tqdm
import filelock

import matplotlib.pyplot as plt

# Global variables for worker processes
_freecad_initialized = False


def _init_worker():
    """Initialize FreeCAD in each worker process."""
    global _freecad_initialized
    if not _freecad_initialized:
        freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
        if freecad_lib not in sys.path:
            sys.path.insert(0, freecad_lib)
        import FreeCAD  # noqa: F401
        _freecad_initialized = True


def _get_freecad():
    """Get FreeCAD module, initializing if necessary."""
    _init_worker()
    import FreeCAD
    return FreeCAD


# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, NodeOp, EdgeOp

from img2cad_diffusion.convert_sequence_to_freecad import (
    convert_constraint,
    convert_primitive,
    meter_to_millimeter,
    resolve_redundant_coincident,
    resolve_subnodes,
    resolve_tangent
)

import math
from dataclasses import dataclass

# Geometry stability constants
M_TO_MM = 1000.0
GEOMETRY_TOLERANCE = 1.0  # mm - max allowed drift


@dataclass
class GeometryPoint:
    """A point in 2D space."""
    x: float
    y: float

    def distance_to(self, other: 'GeometryPoint') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def get_original_geometry_points(seq) -> list[tuple[str, list[GeometryPoint]]]:
    """Extract geometry points from SketchGraphs sequence (before FreeCAD conversion)."""
    sketch = datalib.sketch_from_sequence(seq)
    result = []

    for entity_id, entity in sketch.entities.items():
        if entity is None:
            continue

        entity_type = type(entity).__name__
        points = []

        if entity_type == 'Line':
            px, py = entity.pntX * M_TO_MM, entity.pntY * M_TO_MM
            dx, dy = entity.dirX, entity.dirY
            start_param = entity.startParam * M_TO_MM
            end_param = entity.endParam * M_TO_MM

            start_x = px + dx * start_param
            start_y = py + dy * start_param
            end_x = px + dx * end_param
            end_y = py + dy * end_param

            points = [GeometryPoint(start_x, start_y), GeometryPoint(end_x, end_y)]

        elif entity_type == 'Circle':
            cx = entity.xCenter * M_TO_MM
            cy = entity.yCenter * M_TO_MM
            points = [GeometryPoint(cx, cy)]

        elif entity_type == 'Arc':
            cx = entity.xCenter * M_TO_MM
            cy = entity.yCenter * M_TO_MM
            points = [GeometryPoint(cx, cy)]  # Center only

        elif entity_type == 'Point':
            x = entity.x * M_TO_MM
            y = entity.y * M_TO_MM
            points = [GeometryPoint(x, y)]

        if points:
            result.append((entity_type, points))

    return result


def get_freecad_geometry_points(fc_sketch) -> list[tuple[str, list[GeometryPoint]]]:
    """Extract geometry points from FreeCAD sketch."""
    FreeCAD = _get_freecad()
    import Part

    result = []

    for geo in fc_sketch.Geometry:
        geo_type = type(geo).__name__
        points = []

        if isinstance(geo, Part.LineSegment):
            start = geo.StartPoint
            end = geo.EndPoint
            points = [GeometryPoint(start.x, start.y), GeometryPoint(end.x, end.y)]

        elif isinstance(geo, Part.Circle):
            center = geo.Center
            points = [GeometryPoint(center.x, center.y)]

        elif isinstance(geo, Part.ArcOfCircle):
            center = geo.Center
            points = [GeometryPoint(center.x, center.y)]  # Center only

        elif isinstance(geo, Part.Point):
            points = [GeometryPoint(geo.X, geo.Y)]

        if points:
            result.append((geo_type, points))

    return result


def check_geometry_stability(
    original_points: list[tuple[str, list[GeometryPoint]]],
    fc_points: list[tuple[str, list[GeometryPoint]]],
    tolerance: float = GEOMETRY_TOLERANCE,
) -> bool:
    """Check if geometry is stable (no significant drift after constraints).

    Returns True if stable, False if collapsed.
    """
    type_mapping = {'Line': 'LineSegment', 'Arc': 'ArcOfCircle'}

    def group_by_type(points_list):
        groups = {}
        for geo_type, pts in points_list:
            normalized_type = type_mapping.get(geo_type, geo_type)
            if normalized_type not in groups:
                groups[normalized_type] = []
            groups[normalized_type].append(pts)
        return groups

    orig_groups = group_by_type(original_points)
    fc_groups = group_by_type(fc_points)

    # Check counts match
    for geo_type in orig_groups:
        if len(orig_groups.get(geo_type, [])) != len(fc_groups.get(geo_type, [])):
            return False

    # Check geometry stability with nearest-neighbor matching
    for geo_type in orig_groups:
        orig_geos = orig_groups[geo_type]
        fc_geos = fc_groups.get(geo_type, [])

        fc_used = [False] * len(fc_geos)

        for orig_pts in orig_geos:
            best_fc_idx = -1
            best_distance = float('inf')

            for fc_idx, fc_pts in enumerate(fc_geos):
                if fc_used[fc_idx] or len(orig_pts) != len(fc_pts):
                    continue

                total_dist = sum(o.distance_to(f) for o, f in zip(orig_pts, fc_pts))

                if len(orig_pts) == 2:  # Line: try reversed
                    reversed_dist = (orig_pts[0].distance_to(fc_pts[1]) +
                                    orig_pts[1].distance_to(fc_pts[0]))
                    total_dist = min(total_dist, reversed_dist)

                if total_dist < best_distance:
                    best_distance = total_dist
                    best_fc_idx = fc_idx

            if best_fc_idx >= 0:
                fc_used[best_fc_idx] = True
                fc_pts = fc_geos[best_fc_idx]

                # Calculate max drift
                normal_dists = [orig_pts[i].distance_to(fc_pts[i]) for i in range(len(orig_pts))]
                min_max_drift = max(normal_dists)

                if len(orig_pts) == 2:
                    reversed_dists = [orig_pts[0].distance_to(fc_pts[1]),
                                     orig_pts[1].distance_to(fc_pts[0])]
                    if max(reversed_dists) < min_max_drift:
                        min_max_drift = max(reversed_dists)

                if min_max_drift > tolerance:
                    return False

    return True


def process_sequence(
    seq: list[NodeOp | EdgeOp],
    output_py_path: str | None,
    debug: bool = False,
    return_sketch: bool = False,
):
    """Process sequence and optionally return sketch for geometry stability check.

    Args:
        seq: Sequence of operations
        output_py_path: Path to save Python history
        debug: Enable debug output
        return_sketch: If True, return (status, sketch, doc) instead of just status.
                      Caller is responsible for closing doc.

    Returns:
        If return_sketch=False: solver_status dict
        If return_sketch=True: (solver_status, sketch, doc) - caller must close doc
    """
    FreeCAD = _get_freecad()

    doc = FreeCAD.newDocument("MyDoc")
    body = doc.addObject('PartDesign::Body', 'Body')
    sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    xy_plane = body.Origin.OriginFeatures[3]
    sketch.AttachmentSupport = xy_plane
    sketch.MapMode = 'FlatFace'

    history_commands = []
    history_commands.append("import FreeCAD")
    history_commands.append("import Part")
    history_commands.append("import Sketcher")
    history_commands.append("doc = FreeCAD.newDocument('MyDoc')")
    history_commands.append("body = doc.addObject('PartDesign::Body', 'Body')")
    history_commands.append("sketch = body.newObject('Sketcher::SketchObject', 'Sketch')")
    history_commands.append("# ... attachment setup skipped for brevity ...")

    # rescale
    seq = [meter_to_millimeter(op, debug) for op in seq]
    if debug:
        geom_cnt = 0
        for i, op in enumerate(seq):
            if isinstance(op, NodeOp):
                print(f"{i:02d} {geom_cnt:2d}", op)
                geom_cnt += 1
            else:
                print(f"{i:02d}    {op}")

    # Pass 1: Map Building (node_idx_map = {node_id: (primitive_id, 0~3), ...} where node_id includes both primitives and subnodes)
    node_idx_map = resolve_subnodes(seq)
    if debug: print(f"{node_idx_map=}")

    # Pass 2: Constraint Refinement
    seq = resolve_redundant_coincident(seq, node_idx_map, debug)
    seq = resolve_tangent(seq, node_idx_map)

    # Pass 3: Generation
    for i, op in enumerate(seq):

        if isinstance(op, NodeOp):
            # Primitive Creation
            mapping, cmd = convert_primitive(op, sketch)
            if cmd:
                cmd = cmd.replace("Vector (", "FreeCAD.Vector(")
                history_commands.append(cmd)

        elif isinstance(op, EdgeOp):
            # Constraint Application
            cmd = convert_constraint(op, sketch, node_idx_map)
            if cmd:
                history_commands.append(cmd)

    doc.recompute()

    # Get solver status BEFORE closing the document
    solver_status = sketch.getSolverStatus()

    # Dump History to File
    if output_py_path:
        with open(output_py_path, "w") as f:
            history_commands.append("doc.recompute()")
            f.write("\n".join(history_commands))
        if debug: print(f"History saved to {output_py_path}")

    if return_sketch:
        # Return sketch and doc for caller to do geometry check and close
        return solver_status, sketch, doc
    else:
        # Close the document to free memory
        FreeCAD.closeDocument(doc.Name)
        return solver_status


def write_to_file_atomic(file_path: str, idx: int):
    """Write an index to a file with file lock for process safety."""
    lock_path = file_path + ".lock"
    with filelock.FileLock(lock_path):
        with open(file_path, "a") as f:
            f.write(f"{idx}\n")


def load_indices_from_file(file_path: str) -> set[int]:
    """Load indices from a file (one per line)."""
    if not os.path.exists(file_path):
        return set()

    indices = set()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    indices.add(int(line))
                except ValueError:
                    continue
    return indices


def process_single_sequence(args: tuple) -> tuple[int, bool, str | None]:
    """
    Process a single sequence and return the result.
    Writes FC results and errors immediately to respective files.

    FC criteria:
    1. DoF == 2
    2. No redundant constraints
    3. No conflicting constraints
    4. Geometry stability (no drift > 1mm)

    Args:
        args: Tuple of (index, sequence, fc_file_path, error_file_path, unstable_file_path)

    Returns:
        Tuple of (index, is_fc, error_message)
        - is_fc: True if all FC criteria are met including geometry stability
        - error_message: None if successful, otherwise the error string
    """
    FreeCAD = _get_freecad()
    idx, seq, fc_file_path, error_file_path, unstable_file_path = args

    try:
        # Process with return_sketch=True to do geometry stability check
        status, fc_sketch, doc = process_sequence(seq, None, return_sketch=True)

        # FC criteria: DoF == 2 AND no redundant AND no conflicting constraints
        is_fc = (
            status["DoF"] == 2 and
            len(status.get("RedundantConstraints", [])) == 0 and
            len(status.get("ConflictingConstraints", [])) == 0
        )

        if is_fc:
            # Additional check: geometry stability
            orig_points = get_original_geometry_points(seq)
            fc_points = get_freecad_geometry_points(fc_sketch)
            is_stable = check_geometry_stability(orig_points, fc_points, GEOMETRY_TOLERANCE)

            if is_stable:
                # Passed all checks - write to FC file
                write_to_file_atomic(fc_file_path, idx)
            else:
                # FC but geometry unstable - write to unstable file
                if unstable_file_path:
                    write_to_file_atomic(unstable_file_path, idx)
                is_fc = False

        # Close document
        FreeCAD.closeDocument(doc.Name)

        return (idx, is_fc, None)
    except Exception as e:
        # Write error index to error file
        import traceback
        print(f"Error processing index {idx}:")
        traceback.print_exc()
        write_to_file_atomic(error_file_path, idx)
        return (idx, False, str(e))


def get_max_index_from_file(file_path: str) -> int:
    """Get the maximum index from a file. Returns -1 if file doesn't exist or is empty."""
    if not os.path.exists(file_path):
        return -1

    max_idx = -1
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    max_idx = max(max_idx, int(line))
                except ValueError:
                    continue
    return max_idx


def process_multiprocess(
    sequences: list,
    outdir: str = "sketchgraphs_fc_search",
    num_processes: int = 4
):
    """
    Process sequences using multiprocessing with immediate result writing.
    Supports resumption from previous runs.

    Files used:
    - sketchgraphs_fc.txt: Contains indices where DoF == 2 AND geometry stable
    - sketchgraphs_error.txt: Contains indices that caused errors
    - sketchgraphs_unstable.txt: Contains indices that are FC but geometry unstable

    On resume, processing starts from max(recorded indices) + 1.
    This may cause slight duplicate processing at the boundary, but is very efficient.
    """
    fc_file_path = os.path.join(outdir, "sketchgraphs_fc.txt")
    error_file_path = os.path.join(outdir, "sketchgraphs_error.txt")
    unstable_file_path = os.path.join(outdir, "sketchgraphs_unstable.txt")

    # Find the maximum index already processed
    max_fc_idx = get_max_index_from_file(fc_file_path)
    max_error_idx = get_max_index_from_file(error_file_path)
    max_unstable_idx = get_max_index_from_file(unstable_file_path)
    start_idx = max(max_fc_idx, max_error_idx, max_unstable_idx) + 1

    # Count existing results for statistics
    existing_fc = load_indices_from_file(fc_file_path)
    existing_error = load_indices_from_file(error_file_path)
    existing_unstable = load_indices_from_file(unstable_file_path)

    print(f"Found {len(existing_fc)} FC sequences from previous runs")
    print(f"Found {len(existing_error)} error sequences from previous runs")
    print(f"Found {len(existing_unstable)} geometry-unstable sequences from previous runs")
    print(f"Resuming from index {start_idx}")

    total_sequences = len(sequences)
    remaining = total_sequences - start_idx

    if remaining <= 0:
        print("All sequences already processed!")
        return existing_fc

    # Create tasks starting from start_idx
    tasks = [
        (i, sequences[i], fc_file_path, error_file_path, unstable_file_path)
        for i in range(start_idx, total_sequences)
    ]

    print(f"Processing {len(tasks)} remaining sequences with {num_processes} processes")

    fc_count = len(existing_fc)
    error_count = len(existing_error)

    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_processes, initializer=_init_worker) as pool:
        results = pool.imap_unordered(process_single_sequence, tasks)

        for idx, is_fc, error in tqdm(results, total=len(tasks), desc="Processing"):
            if error:
                error_count += 1
                if error_count <= 10:  # Only print first 10 errors
                    print(f"\nError at index {idx}: {error}")
            elif is_fc:
                fc_count += 1

    print(f"\nProcessing complete!")
    print(f"Total FC sequences found: {fc_count}")
    print(f"Total errors: {error_count}")

    # Return all FC indices
    return load_indices_from_file(fc_file_path)


def process(sequences: list, outdir: str = "sketchgraphs_fc_search"):
    """Original single-process implementation for backward compatibility."""
    FC = []
    output_file_path = os.path.join(outdir, "sketchgraphs_fc.txt")

    for i, seq in enumerate(tqdm(sequences)):
        try:
            status = process_sequence(seq, None)
            if status["DoF"] == 2:
                FC.append(i)
                # Write immediately
                with open(output_file_path, "a") as f:
                    f.write(f"{i}\n")
        except Exception as e:
            print(f"Error occured: {e}")
            pass

    return set(FC)


def render_fc_sketches(sequences: list, fc_indices: set[int], outdir: str):
    """Render sketches for all FC sequences."""
    print(f"\nRendering {len(fc_indices)} FC sketches...")

    for i in tqdm(sorted(fc_indices), desc="Rendering"):
        output_path = os.path.join(outdir, f"{i}_gt.svg")
        if os.path.exists(output_path):
            continue  # Skip already rendered

        try:
            seq = sequences[i]
            sketch = datalib.sketch_from_sequence(seq)
            fig = datalib.render_sketch(sketch)
            fig.savefig(output_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error rendering sketch {i}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data_processed/sg_filtered_unique.npy")
    parser.add_argument("--outdir", type=str, default="sketchgraphs_fc_search")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--skip_render", action="store_true", help="Skip rendering FC sketches")
    args = parser.parse_args()

    seq_data = flat_array.load_dictionary_flat(args.data_path)
    sequences = seq_data['sequences']

    os.makedirs(args.outdir, exist_ok=True)

    if args.num_processes > 1:
        fc_indices = process_multiprocess(sequences, args.outdir, args.num_processes)
    else:
        fc_indices = process(sequences, args.outdir)

    if not args.skip_render:
        render_fc_sketches(sequences, fc_indices, args.outdir)