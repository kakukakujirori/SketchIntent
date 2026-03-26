"""Filter sequences from sketchgraphs source shards.

This script filters raw sketchgraphs json shards and saves the result
as an intermediate sequence dataset (sg_filtered.npy) for further processing.
"""

import dataclasses
import enum
import functools
import os

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

from src.sketch_handlers.sketch_utils import normalize_sketch


@dataclasses.dataclass
class FilterConfiguration:
    min_entities: int = 6
    max_entities: int = 16


@dataclasses.dataclass
class ConvertSketchgraphsRunConfiguration:
    filter: FilterConfiguration = dataclasses.field(default_factory=FilterConfiguration)
    input_folder: str = "data_shards"
    output_folder: str = "data_processed"
    total_sketches: int = 16261381


class ExtendedFilterReason(enum.Enum):
    CannotNormalize = 8
    ZeroSizedEntity = 9


def _filter_sketch_render_valid(sketch: datalib.Sketch) -> FilterReason | ExtendedFilterReason:
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


def filter_sketch(sketch: datalib.Sketch, config: FilterConfiguration) -> FilterReason | ExtendedFilterReason:
    config_s = make_sequence_dataset.make_default_filter_config(config.min_entities, config.max_entities)
    reason = make_sequence_dataset.filter_sketch(sketch, config_s)

    if reason != FilterReason.Accepted:
        return reason

    return _filter_sketch_render_valid(sketch)


@hydra.main(config_name='conf')
def main(config: ConvertSketchgraphsRunConfiguration):
    input_folder = hydra.utils.to_absolute_path(config.input_folder)
    output_folder = hydra.utils.to_absolute_path(config.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    shards = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tar.zst')]
    filter_function = functools.partial(filter_sketch, config=config.filter)

    num_threads = len(os.sched_getaffinity(0))

    filtered_result = make_sequence_dataset.process(
        shards, num_threads, filter_function, total_sketches=config.total_sketches
    )

    filtered_sequence_path = os.path.join(output_folder, 'sg_filtered.npy')
    print(f'Saving result at {filtered_sequence_path}')
    np.save(filtered_sequence_path, filtered_result, allow_pickle=False)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=ConvertSketchgraphsRunConfiguration)
    main()
