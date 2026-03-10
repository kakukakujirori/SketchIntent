"""Filter and tokenize sequences from sketchgraphs source shards.

This script combines the functionality of filtering raw sketchgraphs json shards
and subsequently tokenizing/uniqueifying them into a final sequence dataset ready for training.
"""

import dataclasses
import enum
import functools
import logging
import os
from typing import Dict, Sequence

import hydra
import rootutils
import numpy as np
import omegaconf
from hydra.core.config_store import ConfigStore

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.pipeline import make_sequence_dataset
from sketchgraphs.pipeline.make_sequence_dataset import FilterReason

from . import dataset
from src.sketch_handlers.sketch_utils import normalize_sketch


@dataclasses.dataclass
class FilterConfiguration:
    min_entities: int = 6
    max_entities: int = 16


@dataclasses.dataclass
class TokenizeSequenceConfig:
    num_position_bins: int = 64


@dataclasses.dataclass
class ConvertSketchgraphsRunConfiguration:
    filter: FilterConfiguration = dataclasses.field(default_factory=FilterConfiguration)
    tokenize: TokenizeSequenceConfig = dataclasses.field(default_factory=TokenizeSequenceConfig)
    input_folder: str = omegaconf.MISSING
    output_folder: str = omegaconf.MISSING
    total_sketches: int = 16261381


class ExtendedFilterReason(enum.Enum):
    CannotNormalize = 8
    ZeroSizedEntity = 9


def _filter_sketch_render_valid(sketch: datalib.Sketch) -> FilterReason:
    """Evaluates whether the sketch is valid from a rendering perspective."""
    try:
        scale = normalize_sketch(sketch)
    except Exception:
        # Also discard sequences where normalize_sketch fails.
        return ExtendedFilterReason.CannotNormalize

    if scale == -1:
        return ExtendedFilterReason.CannotNormalize

    # Discard some pathological sketches which have zero-sized entities
    for ent in sketch.entities.values():
        if hasattr(ent, 'radius'):
            if getattr(ent, 'radius') == 0:
                return ExtendedFilterReason.ZeroSizedEntity
        # Discard zero-length lines/arcs
        if isinstance(ent, datalib.Line):
            if np.allclose(ent.start_point, ent.end_point):
                return ExtendedFilterReason.ZeroSizedEntity
        if isinstance(ent, datalib.Arc):
            if np.allclose(ent.start_point, ent.mid_point):
                return ExtendedFilterReason.ZeroSizedEntity

    return FilterReason.Accepted


def filter_sketch(sketch: datalib.Sketch, config: FilterConfiguration) -> FilterReason:
    config_s = make_sequence_dataset.make_default_filter_config(config.min_entities, config.max_entities)
    reason = make_sequence_dataset.filter_sketch(sketch, config_s)

    if reason != FilterReason.Accepted:
        return reason

    return _filter_sketch_render_valid(sketch)


def _list_directory(directory: str):
    files = os.listdir(directory)
    return [os.path.join(directory, f) for f in files if f.endswith('.tar.zst')]


def _offsets_from_counts(counts: np.ndarray):
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def compute_unique_indexes(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
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

    return unique_idxs


def subsample_sequences(sequence_data, idx: Sequence[int]) -> np.ndarray:
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


@hydra.main(config_name='conf')
def main(config: ConvertSketchgraphsRunConfiguration):
    logger = logging.getLogger(__name__)

    # --- Step 1: Filter Sequences ---
    logger.info("=== Step 1: Filtering Sequences ===")
    input_folder = hydra.utils.to_absolute_path(config.input_folder)
    output_folder = hydra.utils.to_absolute_path(config.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    shards = _list_directory(input_folder)
    filter_function = functools.partial(filter_sketch, config=config.filter)

    num_threads = len(os.sched_getaffinity(0))

    filtered_result = make_sequence_dataset.process(
        shards, num_threads, filter_function, total_sketches=config.total_sketches
    )

    filtered_sequence_path = os.path.join(output_folder, 'sg_filtered.npy')
    logger.info(f'Saving filtered intermediate result at {filtered_sequence_path}')
    np.save(filtered_sequence_path, filtered_result, allow_pickle=False)

    # --- Step 2: Uniqueify Sequences ---
    logger.info("=== Step 2: Uniqueifying Sequences ===")

    # We use PrimitiveDataset temporarily just to convert the sequences into comparable
    # discrete tokens to figure out which ones are visually/structurally unique.
    # TODO: Refactor the code so that dataset.PrimitiveDataset is not used.
    primitive_dataset = dataset.PrimitiveDataset(filtered_sequence_path, config.tokenize.num_position_bins)

    data = dataset.process_sequence_data(primitive_dataset)
    data = {k: np.asarray(v) for k, v in data.items()}

    counts = np.diff(data['offsets'])

    logger.info('Uniqueifying token sequences.')
    unique_idx = compute_unique_indexes(data['val'], data['offsets'])
    logger.info(f'Kept {len(unique_idx)} / {len(counts)} sequences ({len(unique_idx) / len(counts):.2%}).')

    logger.info('Subsampling actual sequence data.')
    sequence_output_path = os.path.join(output_folder, 'sg_filtered_unique.npy')
    sequence_data = flat_array.load_dictionary_flat(filtered_sequence_path)
    sequence_data = subsample_sequences(sequence_data, unique_idx)
    logger.info(f'Saving unique sequences at {sequence_output_path}')
    np.save(sequence_output_path, sequence_data, allow_pickle=False)

    logger.info('=== Conversion Complete ===')


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=ConvertSketchgraphsRunConfiguration)
    main()
