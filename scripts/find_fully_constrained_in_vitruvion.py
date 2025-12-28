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

from img2cad_diffusion.convert_to_freecad import (
    convert_constraint,
    convert_primitive,
    meter_to_millimeter,
    resolve_redundant_coincident,
    resolve_subnodes,
    resolve_tangent
)


def process_sequence(
    seq: list[NodeOp | EdgeOp],
    output_py_path: str | None,
    debug: bool = False,
):
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

    Args:
        args: Tuple of (index, sequence, fc_file_path, error_file_path)

    Returns:
        Tuple of (index, is_fc, error_message)
        - is_fc: True if DoF == 2
        - error_message: None if successful, otherwise the error string
    """
    idx, seq, fc_file_path, error_file_path = args

    try:
        status = process_sequence(seq, None)
        is_fc = status["DoF"] == 2

        # If FC, write immediately to file with file lock
        if is_fc:
            write_to_file_atomic(fc_file_path, idx)

        return (idx, is_fc, None)
    except Exception as e:
        # Write error index to error file
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
    - sketchgraphs_fc.txt: Contains indices where DoF == 2
    - sketchgraphs_error.txt: Contains indices that caused errors

    On resume, processing starts from max(recorded indices) + 1.
    This may cause slight duplicate processing at the boundary, but is very efficient.
    """
    fc_file_path = os.path.join(outdir, "sketchgraphs_fc.txt")
    error_file_path = os.path.join(outdir, "sketchgraphs_error.txt")

    # Find the maximum index already processed
    max_fc_idx = get_max_index_from_file(fc_file_path)
    max_error_idx = get_max_index_from_file(error_file_path)
    start_idx = max(max_fc_idx, max_error_idx) + 1

    # Count existing results for statistics
    existing_fc = load_indices_from_file(fc_file_path)
    existing_error = load_indices_from_file(error_file_path)

    print(f"Found {len(existing_fc)} FC sequences from previous runs")
    print(f"Found {len(existing_error)} error sequences from previous runs")
    print(f"Resuming from index {start_idx}")

    total_sequences = len(sequences)
    remaining = total_sequences - start_idx

    if remaining <= 0:
        print("All sequences already processed!")
        return existing_fc

    # Create tasks starting from start_idx
    tasks = [
        (i, sequences[i], fc_file_path, error_file_path)
        for i in range(start_idx, total_sequences)
    ]

    print(f"Processing {len(tasks)} remaining sequences with {num_processes} processes")

    fc_count = len(existing_fc)
    error_count = len(existing_error)

    with Pool(processes=num_processes, initializer=_init_worker) as pool:
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