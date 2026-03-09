from typing import Literal
import random
import torch

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, Arc, Circle, Line, Point

from src.data.components.primitive_noise import PrimitiveNoise
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import normalize_sketch, cleanse_sketch


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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[index]

        sketch = datalib.sketch_from_sequence(seq)

        # Apply noise first (if enabled), then normalize
        # This ensures noisy values stay within [-0.5, 0.5] range
        sketch = self.primitive_noise.apply_primitive_noise(sketch)

        normalize_sketch(sketch)

        # Remove degenerate constraints before tokenization
        cleanse_sketch(sketch)

        result = self.tokenizer.tokenize(sketch)

        return {key: torch.from_numpy(val) for key, val in result.items()}

    def __len__(self) -> int:
        return len(self.sequences)