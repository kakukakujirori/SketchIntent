import argparse
import csv
import glob
import os
import re
import shutil


def main(args):
    if args.log_path is None:
        # By default, the log is in the parent of the commands dir
        args.log_path = os.path.join(os.path.dirname(args.data_dir), "freecad_conversion_log.csv")

    if not os.path.isfile(args.log_path):
        print(f"Log file not found: {args.log_path}")
        return

    # Identify sketches to remove (DoF=2 means global translation only)
    to_remove = set()
    dof_map = {}
    with open(args.log_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # DoF is often stored as string representation of an integer
            dof_val = row['dof']
            dof_map[row['name']] = dof_val
            if int(float(dof_val)) == 2:
                to_remove.add(row['name'])

    if not to_remove:
        print("No sketches with dof=2 found in log.")

    # Handle output directory
    target_dir = args.data_dir
    if args.out_dir:
        if os.path.isdir(args.out_dir):
            shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Copying files from {args.data_dir} to {args.out_dir}...")
        for f in glob.glob(os.path.join(args.data_dir, "*")):
            if os.path.isfile(f):
                shutil.copy2(f, args.out_dir)
        target_dir = args.out_dir

    print(f"Finding and removing sketches from {target_dir}...")

    # Iterate through all .py files in target_dir and check if they match the to_remove list
    all_py_files = sorted(glob.glob(os.path.join(target_dir, "*.py")))
    count = 0
    remaining_info = []

    for py_file in all_py_files:
        filename = os.path.basename(py_file)
        if "_sketch" not in filename:
            continue

        # NOTE: CSV's name column is not equal to the filename! Thus we need to check the SketchName in the py file.
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # The script contains: fc_sketch = body.newObject('Sketcher::SketchObject', 'SketchName')
            # Extract the internal sketch name to reconstruct the full name used in the log
            match = re.search(r"newObject\('Sketcher::SketchObject',\s*'(.*?)'\)", content)
            if match:
                full_name = match.group(1)

                # Count primitives (those that add geometry)
                prim_count = len(re.findall(r"fc_sketch\.addGeometry\(Part\.", content))

                is_outside_range = not (args.prim_num_range[0] <= prim_count <= args.prim_num_range[1])

                if full_name in to_remove or is_outside_range:
                    # Remove the .py file and corresponding .FCStd file
                    os.remove(py_file)
                    fcstd_file = py_file.replace(".py", ".FCStd")
                    if os.path.exists(fcstd_file):
                        os.remove(fcstd_file)
                    count += 1
                else:
                    # Keep track of files that are NOT removed
                    remaining_info.append({
                        'filename': os.path.splitext(filename)[0],
                        'dof': dof_map.get(full_name, "Unknown"),
                        'prim_count': prim_count
                    })
        except Exception as e:
            print(f"Error processing file {py_file}: {e}")

    # Write remaining files info to CSV
    if remaining_info:
        csv_out_path = os.path.join(target_dir, "_sketch_dof_info.csv")
        with open(csv_out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'dof', 'prim_count'])
            writer.writeheader()
            writer.writerows(remaining_info)
        print(f"Saved remaining sketches info to {csv_out_path}")

    print(f"Summary: Removed {count} sketch pairs (.py and .FCStd) from {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove sketches with DoF=2 (fully constrained except for global translation) from the output directory.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing converted .py and .FCStd files")
    parser.add_argument("--out_dir", type=str, default=None, help="If provided, results will be saved here instead of modifying data_dir in place.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to the freecad_conversion_log.csv. Defaults to [data_dir]/../freecad_conversion_log.csv if not provided.")
    parser.add_argument("--prim_num_range", type=int, nargs=2, default=(6, 16), help="Only convert sketches with number of primitives in this range")
    args = parser.parse_args()
    main(args)
