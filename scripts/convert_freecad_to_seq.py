import argparse
import os
import sys
import traceback

import rootutils
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, sequence
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter

FreeCAD, Part, Sketcher = ensure_freecad_imported()


def exec_freecad(freecad_python_file: str) -> Sketcher.Sketch:
    try:
        initial_docs = set(FreeCAD.listDocuments().keys())

        exec_globals = {
            "FreeCAD": FreeCAD,
            "Part": Part,
            "Sketcher": Sketcher,
            "Vector": FreeCAD.Vector,
        }
        with open(freecad_python_file, 'r') as f:
            script_content = f.read()
        exec(script_content, exec_globals)

        current_docs = set(FreeCAD.listDocuments().keys())
        new_docs = list(current_docs - initial_docs)

        if not new_docs:
            if FreeCAD.ActiveDocument:
                doc = FreeCAD.ActiveDocument
            else:
                raise ValueError(f"No active document found for {freecad_python_file}")
        else:
            doc = FreeCAD.getDocument(new_docs[0])

        sketches = doc.findObjects(Type="Sketcher::SketchObject")
        if not sketches:
            raise ValueError(f"No sketches found in {freecad_python_file}")

        fc_sketch = sketches[0]
        return fc_sketch
    finally:
        if 'doc_orig' in locals() and doc_orig:
            FreeCAD.closeDocument(doc_orig.Name)


def process(freecad_python_file: str) -> datalib.Sketch | str:
    context = SketchContext()  # NOTE: It's configuration doesn't affect the freecad <--> sketchgraphs conversion
    converter = SketchFreecadConverter(context)

    try:
        fc_sketch = exec_freecad(freecad_python_file)
        sg_sketch = converter.from_freecad(fc_sketch)
        return sg_sketch
    except Exception:
        return traceback.format_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a batch of FreeCAD python command data into a single npy file.")
    parser.add_argument("data_folder", type=str, help="Path to the folder containing FreeCAD python command data.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output npy file. If None, `data_folder` name is use.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_folder):
        raise FileNotFoundError(f"{args.data_folder} is not a folder.")

    if args.output_file is None:
        args.output_file = os.path.normpath(args.data_folder) + ".npy"
    elif not args.output_file.endswith(".npy"):
        raise ValueError(f"{args.output_file} must end with the .npy extension.")

    # Process
    files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder) if f.endswith(".py")]
    sequences = []
    sketch_ids = []
    error_list = []

    for f in files:
        sketch_or_error = process(f)
        if isinstance(sketch_or_error, str):
            print(f"Error processing {f}:")
            print(sketch_or_error)
            error_list.append((f, sketch_or_error))
        else:
            try:
                seq = sequence.sketch_to_sequence(sketch_or_error)
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
    np.save(args.output_file, out_npy, allow_pickle=False)
    print(f"Output saved at {args.output_file}.")
    print(f"Files with errors: {len(error_list)}/{len(files)}")
    if error_list:
        with open(args.output_file + ".errors", "w") as f:
            for f_path, e_msg in error_list:
                f.write(f"{f_path}:\n{e_msg}\n\n")