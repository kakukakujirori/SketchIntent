import argparse
import glob
import math
import os
import shutil
from tqdm import tqdm

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from scripts.qwen3_refine_data_with_solver import parse_freecad_script, verify_constraints


def extract_primitives(file_py: str) -> list[str]:
    with open(file_py, "r") as f:
        lines = f.read().splitlines()

    primitives = []
    for l in lines:
        if l.startswith("fc_sketch.addGeometry("):
            primitives.append(l)

    return primitives


def get_max_deviation(prim_a_str: str, prim_b_str: str) -> float:
    prim_a_list, _ = parse_freecad_script([prim_a_str])
    prim_b_list, _ = parse_freecad_script([prim_b_str])

    if not prim_a_list or not prim_b_list:
        return float('inf')

    prim_a = prim_a_list[0]
    prim_b = prim_b_list[0]

    if prim_a["type"] != prim_b["type"]:
        return float('inf')

    devs = []

    for key in ["start", "end", "mid", "center"]:
        if key in prim_a and key in prim_b:
            p1 = prim_a[key]
            p2 = prim_b[key]
            dev = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            devs.append(dev)

    if "x" in prim_a and "y" in prim_a and "x" in prim_b and "y" in prim_b:
        dev = math.sqrt((prim_a["x"] - prim_b["x"])**2 + (prim_a["y"] - prim_b["y"])**2)
        devs.append(dev)

    if "radius" in prim_a and "radius" in prim_b:
        dev = abs(prim_a["radius"] - prim_b["radius"])
        devs.append(dev)

    return max(devs) if devs else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_qwen3_refined_with_solver/commands")
    parser.add_argument("--out_dir", type=str, default="data_qwen3_refined_with_solver/commands_refined")
    parser.add_argument("--geo_tolerance", type=float, default=1e-3)
    parser.add_argument("--fc_only", action='store_true')
    parser.add_argument("--fc_dof", type=int, default=2)
    args = parser.parse_args()

    if args.fc_only:
        args.out_dir += "_fc"

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    valid_count = 0

    for refined_py in tqdm(glob.glob(os.path.join(args.data_dir, "*_refined.py"))):
        original_py = refined_py.replace("_refined.py", "_original.py")
        if not os.path.isfile(original_py):
            continue

        refined_primitives = extract_primitives(refined_py)
        original_primitives = extract_primitives(original_py)
        if len(refined_primitives) == 0:
            continue  # NOTE: Entire file is commented out
        if len(refined_primitives) != len(original_primitives):
            raise ValueError(f"Number of primitives are different:\n{refined_primitives=}\n{original_primitives=}")

        try:
            # check the geometric stability
            deviating = False
            for refined_p, original_p in zip(refined_primitives, original_primitives):
                if args.geo_tolerance < get_max_deviation(refined_p, original_p):
                    deviating = True
                    break

            if deviating:
                continue

            # check the solver DoF
            if args.fc_only:
                prims, consts = parse_freecad_script(refined_py)
                res = verify_constraints(prims, consts, geo_threshold_mm=args.geo_tolerance)
                if res.status != "Converged" or res.dof > args.fc_dof:
                    continue

        except Exception as e:
            print(f"Error processing {refined_py}: {e}")
            continue

        # save the file
        shutil.copy(refined_py, args.out_dir)
        valid_count += 1

    print(f"Valid ratio: {valid_count} / {len(glob.glob(os.path.join(args.data_dir, '*_refined.py')))}")
