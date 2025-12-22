"""This module handles constraint dataset functionality including tokenization.
"""

import copy
import random
import dataclasses
from typing import Dict, Optional, Sequence

import numpy as np
import lightning as L
import torch
import torch.utils.data
from typing import TypedDict
from scipy.stats import truncnorm

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data._entity import Arc

from img2cad_diffusion import sketch_utils, primitive_data
from img2cad_diffusion.primitive_data import _pad_or_truncate_to_length

# Import token definitions from centralized module
from img2cad_diffusion.tokens import (
    ConstraintToken as Token,
    NON_COORD_TOKEN,
    CONSTRAINT_COORD_TOKENS,
)


class TokenInput(TypedDict):
    val: torch.Tensor
    coord: torch.Tensor
    pos: torch.Tensor
    c_val: torch.Tensor
    c_coord: torch.Tensor
    c_pos: torch.Tensor


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

    params = sketch_utils.parameterize_entity(entity)
    trial = 0
    while True:
        noise = truncnorm.rvs(a=-max_diff, b=max_diff,
                              scale=std / (2 ** trial), size=params.size)
        new_params = params + noise
        new_ent = sketch_utils.entity_from_params(new_params, entity_id=entity.entityId)

        if isinstance(entity, Arc):
            trial += 1
            if new_ent is None:
                continue
            actual_new_params = sketch_utils.parameterize_entity(new_ent)
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
# Constraint Tokenization
# =============================================================================

def tokenize_constraints(seq: datalib.sequence, gather_idxs: Sequence[int], max_length: Optional[int] = None):
    """Tokenizes the constraints in a sketch construction sequence.

    Parameters
    ----------
    seq : datalib.sequence
        The sketch construction sequence to tokenize.
    gather_idxs : Sequence[int]
        Indices produced by `primitive_data.tokenize_sketch` used to track entity tokens.
    max_length : Optional[int]
        If not `None`, truncates the sequence to this length.
    """
    val_tokens = [Token.Start]
    coord_tokens = [NON_COORD_TOKEN]
    pos_idx = 1  # 0 is reserved for padding
    pos_tokens = [pos_idx]

    # Iterate through edge ops
    for op in seq:
        # Ensure op is applicable edge op
        if not isinstance(op, datalib.EdgeOp):
            continue
        if not op.label.name in Token.__members__:
            continue
        refs = op.references
        if 0 in refs:  # skip external constraints
            continue

        # Add constraint type tokens
        val_tokens.append(Token[op.label.name])
        coord_tokens.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos_tokens.append(pos_idx)

        # Add reference parameters
        val_tokens.extend([gather_idxs[ref] + len(Token) for ref in sorted(refs)])
        coord_tokens.extend(CONSTRAINT_COORD_TOKENS[:len(refs)])
        pos_tokens.extend([pos_idx] * len(refs))

    val_tokens.append(Token.Stop)
    coord_tokens.append(NON_COORD_TOKEN)
    pos_tokens.append(pos_idx + 1)

    sample = {
        'val': _pad_or_truncate_to_length(np.array(val_tokens, dtype=np.int64), max_length),
        'coord': _pad_or_truncate_to_length(np.array(coord_tokens, dtype=np.int64), max_length),
        'pos': _pad_or_truncate_to_length(np.array(pos_tokens, dtype=np.int64), max_length)
    }

    return sample


# =============================================================================
# Dataset and DataModule
# =============================================================================

class ConstraintDataset(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    """Constraint generation dataset."""

    def __init__(self, sequence_file: str, num_bins: int, max_length: Optional[int]=None,
                 primitive_noise_config: PrimitiveNoiseConfig=None, tokenize: bool=True):
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
        tokenize : bool
            If `True` (default), indicates that the sketches should be tokenized into
            primitive and constraint token sequences. Otherwise, simply returns the sketch
            (with potential primitive noise applied).
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
        self.tokenize = tokenize

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        if self.is_purturbed:
            if len(seq) == 0:
                print(f"Empty sequence at index {idx}. Using the previous sequence.")
                seq = self.sequences[max(0, idx - 1)]
            seq = random.choice(seq)

        sketch = datalib.sketch_from_sequence(seq)
        sketch_utils.normalize_sketch(sketch)

        if self.primitive_noise_config.enabled:
            sketch = apply_primitive_noise(sketch, self.primitive_noise_config.std, self.primitive_noise_config.max_difference)

        if not self.tokenize:
            return sketch

        sample, gather_idx = primitive_data.tokenize_sketch(sketch, self.num_bins, self.max_length)
        c_sample = tokenize_constraints(seq, gather_idx, self.max_length)
        c_sample = {f'c_{k}': v for k, v in c_sample.items()}

        sample = {
            **sample,
            **c_sample
        }

        sample = {k: torch.from_numpy(v) for k, v in sample.items()}
        return sample

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

    def setup(self, stage: Optional[str]=None) -> None:
        self._dataset = ConstraintDataset(
            self.config.sequence_path,
            self.config.num_position_bins,
            self.config.max_token_length,
            self.config.primitive_noise)

        self._train_dataset, self._valid_dataset, self._test_dataset = primitive_data.split_dataset(
            self._dataset, self.config.validation_fraction, self.config.test_fraction)

    def _make_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader[TokenInput]:
        return torch.utils.data.DataLoader(
            dataset, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[TokenInput]:
        assert self._train_dataset is not None
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader[TokenInput]]:
        assert self._valid_dataset is not None
        return self._make_dataloader(self._valid_dataset, shuffle=False)

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader[TokenInput]]:
        assert self._test_dataset is not None
        return self._make_dataloader(self._test_dataset, shuffle=False)

    @property
    def train_dataset_size(self) -> int:
        assert self._train_dataset is not None
        return len(self._train_dataset)

