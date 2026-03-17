import argparse
import copy
import math
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data._constraint import LocalReferenceParameter, QuantityParameter
from src.sketch_handlers.sketch_utils import complete_constraint_values


def test_complete_constraint_values(sketch_ori: datalib.Sketch, eps: float = 0.1, verbose: bool = False):

    dprint = lambda msg : print(msg) if verbose else None

    for pid, prim in sketch_ori.entities.items():
        dprint(f"{pid}: {prim}")

    # Recover dimensions
    sketch_new = complete_constraint_values(copy.deepcopy(sketch_ori))

    # list to store before/after
    before_after: list[tuple[float, float, str]] = []

    # Accumulate comparisons
    for cid, constr in sketch_ori.constraints.items():
        dprint(f"\n{constr}")
        for pid, param in enumerate(constr.parameters):
            if isinstance(param, LocalReferenceParameter):
                dprint(param.value)

            if isinstance(param, QuantityParameter):
                orig_expr = param.expression
                orig_paramId = param.parameterId
                # orig_val = param.value  <== Always 0 or None
                try:
                    orig_val, orig_unit = orig_expr.split(" ", 1)
                    orig_val = float(orig_val)
                    if 'deg' in orig_unit:
                        orig_val = math.radians(orig_val)
                except ValueError as e:
                    continue

                new_param = sketch_new.constraints[cid].parameters[pid]
                assert isinstance(new_param, QuantityParameter)
                new_val = new_param.value

                before_after.append((orig_val, new_val, orig_paramId))

                dprint(f"{param}: {orig_expr=}, {orig_val=} ==> {new_val=}")

    # Verify the computed values against original values
    length_ratios = []
    angle_diff = []
    for before, after, param_type in before_after:
        if param_type == 'length':
            length_ratios.append(abs(after) / before)  # complete_constraint_values allows negative values
        elif param_type == 'angle':
            diff = abs(after - before) % (2 * math.pi)
            angle_diff.append(diff)

    dprint(f"\n{length_ratios=}\n{angle_diff=}")

    if length_ratios and eps < max(length_ratios) - min(length_ratios):
        dprint(f"Length ratios are not within tolerance: {length_ratios}")
        return False
    if angle_diff and eps < max(angle_diff):
        dprint(f"Angle differences are not within tolerance: {angle_diff}")
        return False
    dprint("Success!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, default="/home/ryotaro/my_works/My-Vitruvion/data_processed/sg_filtered_unique_valid.npy")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--process_num", type=int, default=1, help="Number of sketches to process")
    parser.add_argument("--eps", type=float, default=0.1, help="Tolerance for comparison")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if not os.path.exists(args.sequence_path):
        print(f"Data file not found: {args.sequence_path}")
        exit(0)

    sequence_array = flat_array.load_dictionary_flat(args.sequence_path)
    sequences = sequence_array['sequences']

    failed_indices = []
    for i in range(args.start_idx, args.start_idx + args.process_num):
        seq = sequences[i]
        sketch = datalib.sketch_from_sequence(seq)
        ret = test_complete_constraint_values(sketch, eps=args.eps, verbose=args.verbose)
        if not ret:
            failed_indices.append(i)

    if len(failed_indices) == 0:
        print(f"All tests passed from {args.start_idx} to {args.start_idx + args.process_num}")
    else:
        print(f"Test Failed: {failed_indices}")