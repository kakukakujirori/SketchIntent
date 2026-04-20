import argparse
import csv
import math
import os
import random
import shutil
from collections import defaultdict


def price_model(dof: int) -> float:
    return round(0.35 * (dof - 2)**0.5, 6)


def main(args):
    random.seed(args.seed)
    csv_path = os.path.join(args.data_dir, "_sketch_dof_info.csv")
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        exit(1)

    # 1. Load data and group by DoF
    filename_info = {}
    dof_groups = defaultdict(list)
    total_count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            dof = int(float(row['dof']))
            prim_count = row.get('prim_count', 'Unknown')

            filename_info[fname] = {
                'dof': dof,
                'prim_count': prim_count
            }
            dof_groups[dof].append(fname)
            total_count += 1

    if total_count == 0:
        print("No sketches found in CSV.")
        return

    # 2. Determine number of annotators
    num_annotators = math.ceil(total_count / args.num_sketches_per_annotator)
    print(f"Total sketches: {total_count}")
    print(f"Target sketches per annotator: {args.num_sketches_per_annotator}")
    print(f"Number of annotators: {num_annotators}")

    # 3. Stratified distribution (Uniform Complexity)
    annotator_files = [[] for _ in range(num_annotators)]

    # Flatten all filenames into a single list sorted by DoF
    all_filenames_sorted = []
    for dof in sorted(dof_groups.keys()):
        # We already shuffled inside each DoF group if we wanted,
        # but let's just use the order we have.
        filenames = dof_groups[dof]
        random.shuffle(filenames)
        all_filenames_sorted.extend(filenames)

    # Distribute files in chunks of num_annotators, shuffling the destination for each chunk
    for i in range(0, len(all_filenames_sorted), num_annotators):
        chunk = all_filenames_sorted[i : i + num_annotators]

        # Shuffle the target annotator indices for this specific chunk
        dest_indices = list(range(num_annotators))
        random.shuffle(dest_indices)

        for j, fname in enumerate(chunk):
            annotator_files[dest_indices[j]].append(fname)

    # 3.5 Re-order folders by maximum DoF (ascending)
    # This ensures that folders with smaller "worst-case" complexity come first.
    annotator_files.sort(key=lambda files: max(filename_info[f]['dof'] for f in files) if files else 0)

    # 4. Copy files and generate info
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_data = []

    for i, files in enumerate(annotator_files):
        if not files:
            continue

        # Sort files within the folder by DoF for better readability
        files.sort(key=lambda f: filename_info[f]['dof'])

        folder_name = str(i)
        anno_dir = os.path.join(args.out_dir, folder_name)
        os.makedirs(anno_dir, exist_ok=True)

        print(f"Processing {len(files)} sketches for package {folder_name}...")

        local_info = []
        dofs = []
        total_price = 0.0

        for fname in files:
            # Copy .FCStd
            fcstd_src = os.path.join(args.data_dir, fname + ".FCStd")
            if os.path.exists(fcstd_src):
                shutil.copy2(fcstd_src, anno_dir)

            # Copy .py
            py_src = os.path.join(args.data_dir, fname + ".py")
            if os.path.exists(py_src):
                shutil.copy2(py_src, anno_dir)

            # Prepare local CSV info
            info = filename_info[fname]
            dof = info['dof']
            price = price_model(dof)

            local_info.append({
                'filename': fname,
                'dof': dof,
                'prim_count': info['prim_count'],
                'price': price
            })
            dofs.append(dof)
            total_price += price

        # Write local _sketch_dof_info.csv
        local_csv_path = os.path.join(anno_dir, "_sketch_dof_info.csv")
        with open(local_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'dof', 'prim_count', 'price'])
            writer.writeheader()
            writer.writerows(local_info)

        # Zip the directory such that it contains the folder itself
        print(f"Zipping {anno_dir}...")
        # root_dir is out_dir, base_dir is folder_name
        shutil.make_archive(anno_dir, 'zip', args.out_dir, folder_name)

        # Collect summary stats
        count = len(files)
        mean_dof = sum(dofs) / count if count > 0 else 0
        var_dof = sum((x - mean_dof)**2 for x in dofs) / count if count > 0 else 0
        max_dof = max(dofs) if dofs else 0

        summary_data.append({
            'folder': folder_name,
            'num_files': count,
            'dof_mean': round(mean_dof, 4),
            'dof_variance': round(var_dof, 4),
            'max_dof': max_dof,
            'total_price': round(total_price, 4)
        })

    # 5. Write global summary CSV
    summary_csv_path = os.path.join(args.out_dir, "_sketch_dof_info_summary.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['folder', 'num_files', 'dof_mean', 'dof_variance', 'max_dof', 'total_price'])
        writer.writeheader()
        writer.writerows(summary_data)

    print(f"Successfully split and zipped into {num_annotators} packages in {args.out_dir}")
    print(f"Summary saved to {summary_csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split sketches into multiple folders for annotators while maintaining DoF distribution.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing .FCStd files and _sketch_dof_info.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--num_sketches_per_annotator", type=int, default=50, help="Target number of sketches to outsource per annotator")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    main(args)
