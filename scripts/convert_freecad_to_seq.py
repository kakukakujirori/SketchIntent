import argparse
import os
import sys
import traceback
from tqdm import tqdm

import rootutils
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter

FreeCAD, Part, Sketcher = ensure_freecad_imported()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a batch of FreeCAD python command data into a single npy file.")
    parser.add_argument("--data_dir", type=str, help="Path to the folder containing FreeCAD python command data.")
    parser.add_argument("--out_path", type=str, help="Path to output the npy file.")
    parser.add_argument("--num_quant_bins", type=int, default=64, help="FreeCAD conversion should work under this quantization lelvel.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"{args.data_dir} is not a folder.")

    if not args.out_path.endswith(".npy"):
        raise ValueError(f"{args.out_path} must end with .npy")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    context = SketchContext()  # NOTE: It's configuration doesn't affect the freecad <--> sketchgraphs conversion
    converter = SketchFreecadConverter(context, raise_on_error=True)

    # Process
    files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".py")])
    sequences = []
    sketch_ids = []
    error_list = []

    for i, f in enumerate(tqdm(files)):
        try:
            sketch = converter.from_freecad_commands(f)
        except Exception:
            print(f"Error processing {f}:")
            traceback.print_exc()
            error_list.append((f, traceback.format_exc()))
            continue

        try:
            seq = datalib.sequence.sketch_to_sequence(sketch_or_error)
            sequences.append(seq)
            # NOTE: (doc_id, part_id, sketch_id), where part_id and sketch_id are dummy
            # Ref: sketchgraphs/pipeline/make_sequence_dataset.py
            doc_id = os.path.basename(f)[:24].encode('utf-8')
            _part_id = 0
            _sketch_id = 0
            sketch_ids.append((doc_id, _part_id, _sketch_id))
        except Exception:
            print(f"Error converting to sequence {f}:")
            traceback.print_exc()
            error_list.append((f, traceback.format_exc()))

    sequence_lengths = np.array([len(seq) for seq in sequences], dtype=np.int64)
    sketch_ids_arr = np.array(
        sketch_ids, dtype=[('document_id', 'S24'), ('part_idx', '<i4'), ('sketch_idx', '<i4')])

    offsets, sequence_data = flat_array.raw_list_flat(sequences)
    flat_data = flat_array.pack_list_flat(offsets, sequence_data)

    out_npy = flat_array.pack_dictionary_flat({
        'sequences': flat_data,
        'sequence_lengths': sequence_lengths,
        'sketch_ids': sketch_ids_arr,
    })

    # Save
    np.save(args.out_path, out_npy, allow_pickle=False)
    print(f"Output saved at {args.out_path}.")
    print(f"Files with errors: {len(error_list)}/{len(files)}")
    if error_list:
        with open(args.out_path + ".errors", "w") as f:
            for f_path, e_msg in error_list:
                f.write(f"{f_path}:\n{e_msg}\n\n")
