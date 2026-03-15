"""
This script converts SketchGraphs sequence data to FreeCAD format, and
removes those that failed the conversion because of Part.OCCError.

TODO: Integrate this script to convert_sketchgraphs_to_seq.py
"""
import argparse
import os
from functools import partial
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
from src.sketch_handlers.sketch_utils import normalize_sketch

FreeCAD, Part, Sketcher = ensure_freecad_imported()


_ArcPatched = type('Arc', (datalib.Arc,), {
    'start_point': None,
    'mid_point': None,
    'end_point': None
})


def normalize_and_quantize(sketch: datalib.Sketch, n_bins: int, min_val: float = -0.5, max_val: float = 0.5) -> datalib.Sketch:

    # Normalize to [-0.5, 0.5]^2
    normalize_sketch(sketch)

    quantizer = partial(SketchTokenizer.quantize_params, n_bins=n_bins, min_val=min_val, max_val=max_val)
    for e_id, entity in sketch.entities.items():
        # NOTE: Don't use primitive handlers, because we need to put them back to entities.
        if isinstance(entity, datalib.Point):
            params = np.array([entity.x, entity.y]).clip(min=min_val, max=max_val)
            quant_params = quantizer(params)
            entity.x, entity.y = quant_params[0], quant_params[1]
        elif isinstance(entity, datalib.Line):
            params = np.concatenate([entity.start_point, entity.end_point]).clip(min=min_val, max=max_val)
            quant_params = quantizer(params)
            entity = entity.from_info({'id': e_id, 'startPoint': quant_params[:2], 'endPoint': quant_params[2:]})
        elif isinstance(entity, datalib.Circle):
            params = np.append(entity.center_point, entity.radius).clip(min=min_val, max=max_val)
            quant_params = quantizer(params)
            entity = entity.from_info({'id': e_id, 'center': quant_params[:2], 'radius': quant_params[2], 'clockwise': entity.clockwise})
        elif isinstance(entity, datalib.Arc):
            params = np.concatenate([entity.start_point, entity.mid_point, entity.end_point]).clip(min=min_val, max=max_val)
            quant_params = quantizer(params)
            # NOTE: [start/mid/end]_point are complex to calculate from their actual parameters.
            #       So we create a temporal subclass that does have a real [start/mid/end]_point attributes.
            entity.__class__ = _ArcPatched
            entity.start_point = quant_params[0:2]
            entity.mid_point = quant_params[2:4]
            entity.end_point = quant_params[4:6]
        else:
            raise ValueError(f"Unknown entity type: {type(entity)}")

        sketch.entities[e_id] = entity

    return sketch


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
    counts = [len(x) for x in idx_sequences_data]

    sequences = flat_array.pack_list_flat(_offsets_from_counts(counts), np.concatenate(idx_sequences_data))

    return flat_array.pack_dictionary_flat({
        'sequences': sequences,
        'sketch_ids': sketch_ids,
        'sequence_lengths': sequence_lengths
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove sketches that cannot be converted to a FreeCAD format.")
    parser.add_argument("sequence_file", type=str, help="Input sequence numpy data.")
    parser.add_argument("--num_quantization_bins", type=int, default=64, help="FreeCAD conversion should work under this quantization lelvel.")
    args = parser.parse_args()

    context = SketchContext(num_quantization_bins=args.num_quantization_bins)
    converter = SketchFreecadConverter(context)

    # check data
    sequence_data = flat_array.load_dictionary_flat(args.sequence_file)
    valid_seq_indices = []
    for idx, seq in enumerate(tqdm(sequence_data['sequences'])):
        sketch = datalib.sketch_from_sequence(seq)
        sketch = normalize_and_quantize(sketch, n_bins=context.num_quantization_bins)
        valid = check_sketch_primitives(sketch, converter)
        if valid:
            valid_seq_indices.append(idx)

    # sieve valid sketches
    sequence_data = subsample_sequences(sequence_data, valid_seq_indices)
    output_path = os.path.join(os.path.dirname(args.sequence_file), "sg_filtered_unique_valid.npy")
    print(f'Extracted sketches: {len(valid_seq_indices)}/{len(sequence_data["sequences"])}. Saving valid sequences at {output_path}')
    np.save(output_path, sequence_data, allow_pickle=False)
