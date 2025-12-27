"""This module handles constraint dataset functionality including tokenization.
"""

import copy
import random
import dataclasses
from typing import Sequence

import numpy as np
import lightning as L
import torch
import torch.utils.data
from typing import TypedDict
from scipy.stats import truncnorm

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data import Arc, Circle, Line, Point

from img2cad_diffusion import sketch_utils, primitive_data
from img2cad_diffusion.primitive_data import _pad_or_truncate_to_length

# Import token definitions from centralized module
from img2cad_diffusion.tokens import (
    ConstraintToken,
    PrimitiveParamToken,
    ConstraintParamToken,
)
from .tokenization import GatherMap, tokenize_primitives, tokenize_sketch


class ConstraintTokenInput(TypedDict):
    val: torch.Tensor
    coord: torch.Tensor
    pos: torch.Tensor
    c_val: torch.Tensor
    # c_coord: torch.Tensor
    # c_pos: torch.Tensor


@dataclasses.dataclass
class PrimitiveNoiseConfig:
    """Configuration for primitive noise model.
    The primitive noise is implemented as a truncated normal noise.

    Attributes
    ----------
    enabled : bool
        Whether to use primitive noise model
    std : float
        Standard deviation of noise to add
    max_difference : float
        Maximum difference between noisy coordinate and original coordinate.
    """
    enabled: bool = True
    std: float = 0.15
    max_difference: float = 0.15


# =============================================================================
# Sketch Noise Functions (moved from noise_models.py)
# =============================================================================

def _trunc_normal_entity_noise(entity: datalib.Entity, std: float = 0.2, max_diff: float = 0.1):
    """Applies truncated normal noise to entity parameters.

    Parameters
    ----------
    entity : Entity
        The entity to apply noise to.
    std : float
        Standard deviation of the truncated normal noise distribution.
    max_diff : float
        Maximum allowed magnitude of change in a parameter value.

    Returns
    -------
    Entity
        New entity with noisy parameters.
    """
    assert std > 0.0 and max_diff > 0.0

    params, _ = sketch_utils.parameterize_entity(entity)
    trial = 0
    while True:
        noise = truncnorm.rvs(a=-max_diff, b=max_diff,
                              scale=std / (2 ** trial), size=params.size)
        new_params = params + noise

        # Preserve clockwise attribute for Arc entities
        clockwise = entity.clockwise if isinstance(entity, Arc) else None
        new_ent = sketch_utils.entity_from_params(new_params, entity_id=entity.entityId, clockwise=clockwise)

        if isinstance(entity, Arc):
            trial += 1
            if new_ent is None:
                continue
            actual_new_params, _ = sketch_utils.parameterize_entity(new_ent)
            if (np.abs(actual_new_params - params) <= max_diff).all():
                if (np.abs(entity.center_point - new_ent.center_point) <= max_diff).all():
                    break
        else:
            break
    return new_ent


def _noisify_sketch_ents(sketch: datalib.Sketch, std: float = 0.2, max_diff: float = 0.1):
    """Inject noise into entity parameters of the given sketch (in-place)."""
    for ent_key, ent in sketch.entities.items():
        new_ent = _trunc_normal_entity_noise(ent, std, max_diff)
        new_ent.isConstruction = ent.isConstruction
        sketch.entities[ent_key] = new_ent
    return sketch


def apply_primitive_noise(sketch: datalib.Sketch, std: float = 0.15, max_difference: float = 0.15) -> datalib.Sketch:
    """Apply noise to sketch entities, returning a new noisy copy."""
    noise_sketch = copy.deepcopy(sketch)
    try:
        _noisify_sketch_ents(noise_sketch, std=std, max_diff=max_difference)
    except:
        noise_sketch = sketch
    return noise_sketch


# =============================================================================
# Dataset and DataModule
# =============================================================================

class ConstraintDataset(torch.utils.data.Dataset):
    """Constraint generation dataset."""

    def __init__(self, sequence_file: str, num_bins: int, max_length: int | None = None,
                 primitive_noise_config: PrimitiveNoiseConfig | None = None):
        """Create a new constraint dataset.

        Parameters
        ----------
        sequence_file : str
            Path to the sequence file to load.
        num_bins : int
            Number of bins for positional quantization.
        max_length : int, optional
            Length to which to pad or truncate sequence tokens.
        primitive_noise_config : PrimitiveNoiseConfig, optional
            If not `None`, configuration for the primitive noise to apply.
        """

        if primitive_noise_config is None:
            primitive_noise_config = PrimitiveNoiseConfig(enabled=False)

        self.primitive_noise_config = primitive_noise_config
        self.num_bins = num_bins
        self.max_length = max_length

        sequence_array = flat_array.load_dictionary_flat(sequence_file)
        if 'sequences' in sequence_array:
            self.sequences = sequence_array['sequences']  # = [cpt_0, cpt_1, ...]
            self.is_purturbed = False
        elif 'perturbations' in sequence_array:
            self.sequences = sequence_array['perturbations']  # = [[cpt_0, cpt_1, ..., cpt_49], ...]
            self.is_purturbed = True
        else:
            raise KeyError("Neither 'sequences' or 'perturbations' found in the key")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[index]
        if self.is_purturbed:
            if len(seq) == 0:
                print(f"Empty sequence at index {index}. Using the previous sequence.")
                seq = self.sequences[max(0, index - 1)]
            seq = random.choice(seq)

        sketch = datalib.sketch_from_sequence(seq)

        # Apply noise first (if enabled), then normalize
        # This ensures noisy values stay within [-0.5, 0.5] range
        if self.primitive_noise_config.enabled:
            sketch = apply_primitive_noise(sketch, self.primitive_noise_config.std, self.primitive_noise_config.max_difference)

        sketch_utils.normalize_sketch(sketch)

        result = tokenize_sketch(
            sketch,
            self.num_bins,
            self.max_length,
            normalize=False,  # NOTE: manually normalized above by `sketch_utils.normalize_sketch`
            uniquify_primitive_val_tokens=True,   # NOTE: THIS IS MUST!!! PrimitiveModel implicitly assumes the len(PrimitiveToken)-offset when doing torch.gather.
            uniquify_constraint_val_tokens=True,  # NOTE: THIS IS MUST!!! ConstraintMMDModel implicitly assumes the len(ConstraintToken)-offset when doing torch.gather.
        )

        return {
            'val': torch.from_numpy(result.prim_tokens['val']),
            'coord': torch.from_numpy(result.prim_tokens['coord']),
            'pos': torch.from_numpy(result.prim_tokens['pos']),
            'c_val': torch.from_numpy(result.constr_tokens['val']),
            # 'c_coord': torch.from_numpy(result.constr_tokens['coord']),
            # 'c_pos': torch.from_numpy(result.constr_tokens['pos']),
        }

    def __len__(self) -> int:
        if self.is_purturbed:
            return len(self.sequences) // len(self.sequences[0])
        else:
            return len(self.sequences)


@dataclasses.dataclass
class ConstraintDataConfig(primitive_data.PrimitiveDataConfig):
    max_token_length: int = 130
    primitive_noise: PrimitiveNoiseConfig = dataclasses.field(default_factory=lambda: PrimitiveNoiseConfig(enabled=True))


class ConstraintDataModule(L.LightningDataModule):
    def __init__(self, config: ConstraintDataConfig, batch_size: int, num_workers: int=8):
        super().__init__()

        self.batch_size = batch_size
        self.config = config
        self.num_workers = num_workers

        self._dataset = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        self._dataset = ConstraintDataset(
            self.config.sequence_path,
            self.config.num_position_bins,
            self.config.max_token_length,
            self.config.primitive_noise)

        self._train_dataset, self._valid_dataset, self._test_dataset = primitive_data.split_dataset(
            self._dataset, self.config.validation_fraction, self.config.test_fraction)

    def _make_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader[ConstraintTokenInput]:
        return torch.utils.data.DataLoader(
            dataset, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[ConstraintTokenInput]:
        assert self._train_dataset is not None
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader[ConstraintTokenInput]:
        assert self._valid_dataset is not None
        return self._make_dataloader(self._valid_dataset, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader[ConstraintTokenInput]:
        assert self._test_dataset is not None
        return self._make_dataloader(self._test_dataset, shuffle=False)

    @property
    def train_dataset_size(self) -> int:
        assert self._train_dataset is not None
        return len(self._train_dataset)
