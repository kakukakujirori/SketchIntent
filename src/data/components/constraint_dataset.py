import random
import torch

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array, Arc, Circle, Line, Point

from src.data.components.primitive_noise import PrimitiveNoise
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import (
    cleanse_sketch_primitives,
    cleanse_sketch_constraints,
    complete_constraint_values,
    normalize_sketch,
    remove_dimensional_constraints,
)


class ConstraintDataset(torch.utils.data.Dataset):
    """Constraint generation dataset."""

    @staticmethod
    def _metadata_to_str(value) -> str:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").rstrip("\x00")
        return str(value)

    @classmethod
    def _format_sketch_id(cls, sketch_id) -> str:
        try:
            document_id = cls._metadata_to_str(sketch_id["document_id"])
            part_idx = int(sketch_id["part_idx"])
            sketch_idx = int(sketch_id["sketch_idx"])
            return f"{document_id}:{part_idx}:{sketch_idx}"
        except Exception:
            return cls._metadata_to_str(sketch_id)

    def __init__(
        self,
        sequence_file: str,
        context: SketchContext,
        primitive_noise: PrimitiveNoise,
        relational_only: bool = False,
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
        self.relational_only = relational_only

        sequence_array = flat_array.load_dictionary_flat(sequence_file)
        self.sequences = sequence_array['sequences']
        self.source_files = sequence_array.get('source_files')
        self.sketch_ids = sequence_array.get('sketch_ids')

    def __getitem__(self, index: int) -> dict[str, int | str | datalib.Sketch | torch.Tensor]:
        seq = self.sequences[index]

        sketch = datalib.sketch_from_sequence(seq)
        if self.relational_only:
            sketch = remove_dimensional_constraints(sketch)

        # NOTE: Keep the order: complete_constraint_values is based on the correct cleanse_sketch_primitives output.
        _ = normalize_sketch(sketch)
        sketch = cleanse_sketch_primitives(sketch)
        sketch = cleanse_sketch_constraints(sketch)
        sketch, _ = complete_constraint_values(sketch)

        # Apply noise and renormalize so that noisy values stay within [-0.5, 0.5] range
        sketch_noisy = self.primitive_noise.apply_primitive_noise(sketch)
        _ = normalize_sketch(sketch_noisy)

        result_tokens = self.tokenizer.tokenize(sketch_noisy)

        result: dict[str, int | str | datalib.Sketch | torch.Tensor] = {"sketch_index": index, "sketch": sketch}
        if self.source_files is not None:
            result["source_file"] = self._metadata_to_str(self.source_files[index])
        if self.sketch_ids is not None:
            result["sketch_id"] = self._format_sketch_id(self.sketch_ids[index])
        for key, val in result_tokens.items():
            result[key] = torch.from_numpy(val)

        return result

    def __len__(self) -> int:
        return len(self.sequences)
