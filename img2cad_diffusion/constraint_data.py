"""This module handles constraint dataset functionality including tokenization.
"""

import copy
import json
import random
import dataclasses
from pathlib import Path
from typing import Sequence

import numpy as np
import lightning as L
import torch
import torch.utils.data
from typing import TypedDict, Required, NotRequired
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
    c_cnt: NotRequired[torch.Tensor]
    data_index: NotRequired[torch.Tensor]


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

    def __init__(
        self,
        sequence_file: str,
        num_bins: int,
        max_length: int | None = None,
        primitive_noise_config: PrimitiveNoiseConfig | None = None,
        align_constraint_token_length: bool = False,
        use_stop_padding: bool = False,
        constraint_order: str = "original",
    ):
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
        align_constraint_token_length : bool
            If True, each constraint uses exactly 4 tokens with InnerPad padding.
        use_stop_padding : bool
            If True, pad constraints with STOP tokens instead of PAD.
        constraint_order : str
            Constraint ordering: "original", "sort_by_ref_id", or "random_shuffle".
        """

        if primitive_noise_config is None:
            primitive_noise_config = PrimitiveNoiseConfig(enabled=False)

        self.primitive_noise_config = primitive_noise_config
        self.num_bins = num_bins
        self.max_length = max_length
        self.align_constraint_token_length = align_constraint_token_length
        self.use_stop_padding = use_stop_padding
        self.constraint_order = constraint_order

        sequence_array = flat_array.load_dictionary_flat(sequence_file)
        if 'sequences' in sequence_array:
            self.sequences = sequence_array['sequences']  # = [cpt_0, cpt_1, ...]
            self.is_purturbed = False
        elif 'perturbations' in sequence_array:
            self.sequences = sequence_array['perturbations']  # = [[cpt_0, cpt_1, ..., cpt_49], ...]
            self.is_purturbed = True
        else:
            raise KeyError("Neither 'sequences' or 'perturbations' found in the key")

    def __getitem__(self, index: int) -> ConstraintTokenInput:
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
            align_constraint_token_length=self.align_constraint_token_length,
            use_stop_padding=self.use_stop_padding,
            constraint_order=self.constraint_order,
        )

        return ConstraintTokenInput(
            val=torch.from_numpy(result.prim_tokens['val']),
            coord=torch.from_numpy(result.prim_tokens['coord']),
            pos=torch.from_numpy(result.prim_tokens['pos']),
            c_val=torch.from_numpy(result.constr_tokens['val']),
            # c_coord=torch.from_numpy(result.constr_tokens['coord']),
            # c_pos=torch.from_numpy(result.constr_tokens['pos']),
            c_cnt=torch.tensor(result.constr_tokens['cnt']),
            # data_index=torch.tensor([index]),                         # NOTE: Uncomment during debug
        )

    def __len__(self) -> int:
        if self.is_purturbed:
            return len(self.sequences) // len(self.sequences[0])
        else:
            return len(self.sequences)


@dataclasses.dataclass
class ConstraintDataConfig(primitive_data.PrimitiveDataConfig):
    max_token_length: int = 130
    primitive_noise: PrimitiveNoiseConfig = dataclasses.field(default_factory=lambda: PrimitiveNoiseConfig(enabled=True))
    # Path to FC clusters JSON file (contains selected_for_validation indices)
    fc_clusters_path: str | None = None
    # Fixed-length tokenization: each constraint uses exactly 4 tokens
    align_constraint_token_length: bool = False
    # Use STOP tokens for tail padding instead of PAD (required for Hungarian matching)
    use_stop_padding: bool = False
    # Constraint ordering: "original", "sort_by_ref_id", or "random_shuffle"
    constraint_order: str = "original"
    # dataloader-related configs
    batch_size: int = 256
    num_data_workers: int = 8


class ConstraintDataModule(L.LightningDataModule):
    def __init__(self, config: ConstraintDataConfig):
        super().__init__()

        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_data_workers

        self._dataset = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None
        self._fc_valid_dataset = None  # FC-only validation dataset

    def _load_fc_val_test_indices(self) -> tuple[list[int], list[int]]:
        """Load FC validation indices from cluster JSON file."""
        if self.config.fc_clusters_path is None:
            print("FC clusters path not specified. Skipping FC validation.")
            return [], []

        fc_path = Path(self.config.fc_clusters_path)
        if not fc_path.exists():
            raise FileNotFoundError(f"FC clusters file not found: {fc_path}")

        with open(fc_path, 'r') as f:
            data = json.load(f)
            print(f"Loaded FC clusters from {fc_path}")

        return data.get('selected_for_validation', []), data.get('selected_for_test', [])

    def setup(self, stage: str | None = None) -> None:
        self._dataset = ConstraintDataset(
            self.config.sequence_path,
            self.config.num_position_bins,
            self.config.max_token_length,
            self.config.primitive_noise,
            self.config.align_constraint_token_length,
            self.config.use_stop_padding,
            self.config.constraint_order,
        )

        # Load FC validation indices
        fc_val_indices, fc_test_indices = self._load_fc_val_test_indices()

        # Create FC validation dataset from specific indices
        self._fc_val_dataset = torch.utils.data.Subset(self._dataset, fc_val_indices)
        self._fc_test_dataset = torch.utils.data.Subset(self._dataset, fc_test_indices)

        # Exclude FC indices from main dataset before splitting
        fc_set = set(fc_val_indices) | set(fc_test_indices)
        remaining_indices = [i for i in range(len(self._dataset)) if i not in fc_set]
        remaining_dataset = torch.utils.data.Subset(self._dataset, remaining_indices)

        # Split remaining data into train/val/test
        self._train_dataset, self._valid_dataset, self._test_dataset = primitive_data.split_dataset(
            remaining_dataset, self.config.validation_fraction, self.config.test_fraction)

    def _make_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader[ConstraintTokenInput]:
        return torch.utils.data.DataLoader(
            dataset, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[ConstraintTokenInput]:
        assert self._train_dataset is not None
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> list[torch.utils.data.DataLoader[ConstraintTokenInput]] | torch.utils.data.DataLoader[ConstraintTokenInput]:
        """Returns validation dataloaders.

        If FC validation is enabled, returns a list:
          - dataloader_idx=0: Normal validation set
          - dataloader_idx=1: FC-only validation set
        Otherwise, returns a single dataloader.
        """
        assert self._valid_dataset is not None
        normal_loader = self._make_dataloader(self._valid_dataset, shuffle=False)

        if self._fc_val_dataset is not None and len(self._fc_val_dataset) > 0:
            fc_loader = self._make_dataloader(self._fc_val_dataset, shuffle=False)
            print(f"Returning 2 validation dataloaders: normal={len(self._valid_dataset)}, fc={len(self._fc_val_dataset)}")
            return [normal_loader, fc_loader]

        return normal_loader

    def test_dataloader(self) -> list[torch.utils.data.DataLoader[ConstraintTokenInput]] | torch.utils.data.DataLoader[ConstraintTokenInput]:
        assert self._test_dataset is not None
        normal_loader = self._make_dataloader(self._test_dataset, shuffle=False)

        if self._fc_test_dataset is not None and len(self._fc_test_dataset) > 0:
            fc_loader = self._make_dataloader(self._fc_test_dataset, shuffle=False)
            print(f"Returning 2 test dataloaders: normal={len(self._test_dataset)}, fc={len(self._fc_test_dataset)}")
            return [normal_loader, fc_loader]

        return normal_loader

    @property
    def train_dataset_size(self) -> int:
        assert self._train_dataset is not None
        return len(self._train_dataset)

    @property
    def has_fc_validation(self) -> bool:
        """Whether FC validation dataloader is available."""
        return self._fc_val_dataset is not None and len(self._fc_val_dataset) > 0
