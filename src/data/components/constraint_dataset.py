from typing import Literal
import random
import torch

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, Arc, Circle, Line, Point

from src.data.components.primitive_noise import PrimitiveNoise
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import cleanse_sketch_primitives, cleanse_sketch_constraints, complete_constraint_values, normalize_sketch


class ConstraintDataset(torch.utils.data.Dataset):
    """Constraint generation dataset."""

    def __init__(
        self,
        sequence_file: str,
        context: SketchContext,
        primitive_noise: PrimitiveNoise,
    ):
        """Create a new constraint dataset.

        Parameters
        ----------
        sequence_file : str
            Path to the sequence file to load.
        context : SketchContext
            Context configuration for the dataset.
        primitive_noise : PrimitiveNoise
            Configuration for the primitive noise to apply.
        """
        self.context = context
        self.primitive_noise = primitive_noise
        self.tokenizer = SketchTokenizer(self.context)

        sequence_array = flat_array.load_dictionary_flat(sequence_file)
        self.sequences = sequence_array['sequences']

    def __getitem__(self, index: int) -> dict[str, int | datalib.Sketch | torch.Tensor]:
        seq = self.sequences[index]

        sketch = datalib.sketch_from_sequence(seq)

        # NOTE: Keep the order: complete_constraint_values is based on the correct cleanse_sketch_primitives output.
        _ = normalize_sketch(sketch)
        sketch = cleanse_sketch_primitives(sketch)
        sketch = cleanse_sketch_constraints(sketch)
        sketch, _ = complete_constraint_values(sketch)

        # Apply noise and renormalize so that noisy values stay within [-0.5, 0.5] range
        sketch_noisy = self.primitive_noise.apply_primitive_noise(sketch)
        _ = normalize_sketch(sketch_noisy)

        result_tokens = self.tokenizer.tokenize(sketch_noisy)

        result: dict[str, int | datalib.Sketch | torch.Tensor] = {"sketch_index": index, "sketch": sketch}
        for key, val in result_tokens.items():
            result[key] = torch.from_numpy(val)

        return result

    def __len__(self) -> int:
        return len(self.sequences)