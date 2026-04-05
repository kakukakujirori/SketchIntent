import argparse
import csv
import math
import os
import random
import shutil
from collections import defaultdict


def main(args):
    csv_path = os.path.join(args.data_dir, "_sketch_dof_info.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        exit(1)

    # 1. Load data and group by DoF
    dof_groups = defaultdict(list)
    total_count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dof_groups[row['dof']].append(row['filename'])
            total_count += 1

    if total_count == 0:
        print("No sketches found in CSV.")
        return

    # 2. Determine number of annotators
    num_annotators = math.ceil(total_count / args.num_sketches_per_annotator)
    print(f"Total sketches: {total_count}")
    print(f"Target sketches per annotator: {args.num_sketches_per_annotator}")
    print(f"Number of annotators: {num_annotators}")

    # 3. Stratified distribution
    annotator_files = [[] for _ in range(num_annotators)]

    # Shuffle each group and distribute round-robin to maintain distribution
    sorted_dofs = sorted(dof_groups.keys(), key=lambda x: float(x))
    current_annotator = 0
    for dof in sorted_dofs:
        filenames = dof_groups[dof]
        random.shuffle(filenames)
        for fname in filenames:
            annotator_files[current_annotator].append(fname)
            current_annotator = (current_annotator + 1) % num_annotators

    # 4. Copy files to numbered subdirectories
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, files in enumerate(annotator_files):
        if not files:
            continue

        anno_dir = os.path.join(args.out_dir, str(i))
        os.makedirs(anno_dir, exist_ok=True)

        print(f"Copying {len(files)} sketches to {anno_dir}...")
        for fname in files:
            # Copy .FCStd
            fcstd_src = os.path.join(args.data_dir, fname + ".FCStd")
            if os.path.exists(fcstd_src):
                shutil.copy2(fcstd_src, anno_dir)

            # Copy .py
            py_src = os.path.join(args.data_dir, fname + ".py")
            if os.path.exists(py_src):
                shutil.copy2(py_src, anno_dir)

        # Zip the directory
        print(f"Zipping {anno_dir}...")
        shutil.make_archive(anno_dir, 'zip', anno_dir)

    print(f"Successfully split and zipped into {num_annotators} packages in {args.out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split sketches into multiple folders for annotators while maintaining DoF distribution.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .FCStd files and _sketch_dof_info.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--num_sketches_per_annotator", type=int, default=50, help="Target number of sketches to outsource per annotator")
    args = parser.parse_args()

    main(args)
