from typing import Literal
import random
import torch

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, Arc, Circle, Line, Point

from src.data.components.primitive_noise import PrimitiveNoise
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import normalize_sketch


class ConstraintDataset(torch.utils.data.Dataset):
    """Constraint generation dataset."""

    def __init__(
        self,
        sequence_file: str,
        max_token_length: int,
        num_quantization_bins: int,
        primitive_noise: PrimitiveNoise = PrimitiveNoise(std=0, max_diff=0),
        align_constraint_token_length_to: int | None = None,
        use_stop_padding: bool = False,
        constraint_order: Literal["sort_by_ref_id", "random_shuffle", "original"] = "sort_by_ref_id",
    ):
        """Create a new constraint dataset.

        Parameters
        ----------
        sequence_file : str
            Path to the sequence file to load.
        max_token_length : int
            Maximum length of the tokenized sequence.
        num_quantization_bins : int
            Number of bins for positional quantization.
        primitive_noise : PrimitiveNoise
            Configuration for the primitive noise to apply.
        align_constraint_token_length : bool
            If True, each constraint uses exactly 4 tokens with InnerPad padding.
        use_stop_padding : bool
            If True, pad constraints with STOP tokens instead of PAD.
        constraint_order : str
            Constraint ordering: "original", "sort_by_ref_id", or "random_shuffle".
        """
        self.primitive_noise = primitive_noise

        self.context = SketchContext(
            max_token_length=max_token_length,
            num_quantization_bins=num_quantization_bins,
            align_constraint_token_length_to=align_constraint_token_length_to,
            use_stop_padding=use_stop_padding,
            constraint_order=constraint_order,
        )
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

        result = self.tokenizer.tokenize(sketch)

        return {key: torch.from_numpy(val) for key, val in result.items()}

    def __len__(self) -> int:
        return len(self.sequences)