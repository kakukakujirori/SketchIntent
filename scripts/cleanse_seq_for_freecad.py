"""
This script takes SketchGraphs sequence data, removes structurally duplicate
sketches (uniqueify), and then removes those that fail FreeCAD conversion.
"""
import argparse
import os
from tqdm import tqdm

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.freecad_utils import ensure_freecad_imported
from src.sketch_handlers.sketch_freecad_converter import SketchFreecadConverter
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_primitives, cleanse_sketch_constraints,
    complete_constraint_values, normalize_sketch
)

FreeCAD, Part, Sketcher = ensure_freecad_imported()



def cleanse_sketch(sketch: datalib.Sketch) -> datalib.Sketch:
    _ = normalize_sketch(sketch)
    sketch = cleanse_sketch_primitives(sketch)
    sketch = cleanse_sketch_constraints(sketch)
    sketch, _ = complete_constraint_values(sketch)
    return sketch


def compute_unique_indexes(data: np.ndarray, offsets: np.ndarray) -> list[int]:
    """Computes indices of unique sequences in the data."""
    lengths = np.diff(offsets)
    unique_lengths = np.unique(lengths)
    unique_idxs = []

    for length in unique_lengths:
        idx = np.flatnonzero(lengths == length)
        idx_expand = np.expand_dims(offsets[idx], -1) + np.arange(length)

        sequences = data[idx_expand]
        _, idx_unique = np.unique(sequences, axis=0, return_index=True)
        unique_idxs.append(idx[idx_unique])

    unique_idxs = np.concatenate(unique_idxs)
    unique_idxs.sort()

    return unique_idxs.tolist()


def check_sketch_primitives(sketch: datalib.Sketch, converter: SketchFreecadConverter) -> bool:
    doc = None
    try:
        doc, fc_sketch, primitive_freecad_mapping = converter.to_freecad_primitives(sketch)
        return doc and fc_sketch
    except Part.OCCError as e:
        print(str(e))
        return False
    finally:
        FreeCAD.closeDocument(doc.Name) if doc else None


def _offsets_from_counts(counts: np.ndarray):
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def subsample_sequences(sequence_data, idx: list[int]) -> np.ndarray:
    sketch_ids = sequence_data['sketch_ids'][idx]
    sequence_lengths = sequence_data['sequence_lengths'][idx]

    sequences: flat_array.FlatSerializedArray = sequence_data['sequences']
    idx_sequences_data = [sequences.get_raw_bytes(i) for i in idx]
    counts = np.array([len(x) for x in idx_sequences_data], dtype=np.int64)

    sequences = flat_array.pack_list_flat(_offsets_from_counts(counts), np.concatenate(idx_sequences_data))

    return flat_array.pack_dictionary_flat({
        'sequences': sequences,
        'sketch_ids': sketch_ids,
        'sequence_lengths': sequence_lengths
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Uniqueify and remove sketches that cannot be converted to FreeCAD.")
    parser.add_argument("sequence_file", type=str, help="Input sequence numpy data (sg_filtered.npy).")
    parser.add_argument("--num_quantization_bins", type=int, default=64, help="Number of quantization bins for uniqueify.")
    args = parser.parse_args()

    context = SketchContext(num_quantization_bins=args.num_quantization_bins, max_token_length=None)
    tokenizer = SketchTokenizer(context)
    converter = SketchFreecadConverter(context, raise_on_error=False)

    sequence_data = flat_array.load_dictionary_flat(args.sequence_file)

    # --- Step 1: Uniqueify ---
    print("=== Step 1: Uniqueifying Sequences ===")
    token_seqs = []
    valid_step1_indices = []
    for i, seq in enumerate(tqdm(sequence_data['sequences'], desc="Tokenizing")):
        sketch = datalib.sketch_from_sequence(seq)
        try:
            sketch = cleanse_sketch(sketch)
        except Exception as e:
            print(f"Error cleansing sketch {i}: {e}")
            continue
        tokens, _ = tokenizer.tokenize_primitives(sketch)
        token_seqs.append(tokens['p_val'])
        valid_step1_indices.append(i)

    counts = np.array([len(t) for t in token_seqs], dtype=np.int64)
    flat_tokens = np.concatenate(token_seqs)
    offsets = _offsets_from_counts(counts)

    unique_idx_relative = compute_unique_indexes(flat_tokens, offsets)
    unique_idx = [valid_step1_indices[i] for i in unique_idx_relative]
    print(f'{len(counts)} / {len(sequence_data["sequences"])} sketches are valid after cleansing.')
    print(f'Kept {len(unique_idx)} / {len(counts)} sequences ({len(unique_idx) / len(counts):.2%}) after uniqueify.')

    # --- Step 2: FreeCAD validation ---
    # Iterate over unique_idx directly into the original sequence_data (still dict-like).
    # subsample_sequences is called only once at the very end with the final valid indices.
    print("=== Step 2: FreeCAD Validation ===")
    valid_orig_indices = []
    for orig_idx in tqdm(unique_idx, desc="Validating"):
        seq = sequence_data['sequences'][orig_idx]
        sketch = datalib.sketch_from_sequence(seq)
        sketch = cleanse_sketch(sketch)

        if check_sketch_primitives(sketch, converter):
            valid_orig_indices.append(orig_idx)

    sequence_data = subsample_sequences(sequence_data, valid_orig_indices)

    # output
    output_path = args.sequence_file.replace(".npy", "_unique_valid.npy")
    print(f'Kept {len(valid_orig_indices)} / {len(unique_idx)} sequences ({len(valid_orig_indices) / len(unique_idx):.2%}) after FreeCAD validation.')
    print(f'Saving at {output_path}')
    np.save(output_path, sequence_data, allow_pickle=False)
