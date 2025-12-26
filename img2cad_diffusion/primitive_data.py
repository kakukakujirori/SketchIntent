"""This module handles primitive dataset functionality including tokenization.

This module contains:
- Tokenization functions for sketches (tokenize_sketch, sketch_from_tokens)
- Dataset classes (PrimitiveDataset, CachedPrimitiveDataset)
- DataModule for Lightning (PrimitiveDataModule)
- Configuration dataclasses (PrimitiveDataConfig)
"""

import copy
import dataclasses
import logging
import math
import os
from collections import OrderedDict
from random import shuffle
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import lightning as L
import torch
import torch.nn.functional
import torch.utils.data
import tqdm

import sketchgraphs.data as datalib
from sketchgraphs.data import flat_array
from sketchgraphs.data import Arc, Circle, Line, Point

from . import sketch_utils, tokenization
from .tokens import (
    PrimitiveToken,
    PrimitiveParamToken,
    NUM_PARAMS,
)


# =============================================================================
# Type Definitions
# =============================================================================

class SketchTokenSet(TypedDict):
    """Token set representing a sketch."""
    val: np.ndarray
    coord: np.ndarray
    pos: np.ndarray


class PrimitiveTokenInput(TypedDict):
    """Batch input type for primitive models."""
    val: torch.Tensor
    coord: torch.Tensor
    pos: torch.Tensor


@dataclasses.dataclass
class EntityTokenMapping:
    """Mapping from an entity to its token positions in the primitive sequence.

    This replaces the implicit gather_idxs list with an explicit structure.
    External entities are not included in the mapping (they are identified
    by their absence, not by assuming index 0).

    Attributes
    ----------
    entity_idx : int
        Node index in the SketchGraphs sequence (1-based for entities, 0 is External).
    entity_type : type
        The entity type (Arc, Circle, Line, Point).
    entity_token_pos : int
        Position of the entity type token in the val sequence.
    subnode_positions : Dict[str, int]
        Mapping from subnode name to token position.
        e.g., {"entity": 5, "center": 6, "start": 8, "end": 10}
    """
    entity_idx: int
    entity_type: type
    entity_token_pos: int
    subnode_positions: Dict[str, int]


# Type alias for the gather mapping
GatherMap = Dict[int, EntityTokenMapping]


# =============================================================================
# Configuration
# =============================================================================

@dataclasses.dataclass
class PrimitiveDataConfig:
    """Configuration for creating a new `PrimitiveDataset`.

    Attributes
    ----------
    num_position_bins
        Number of bins used for quantization of position.
    sequence_path
        Path to file containing sequence data.
    max_token_length
        Maximum length for token sequences representing primitives.
    validation_fraction : float
        Fraction of data to use as validation set.
    test_fraction : float
        Fraction of data to use as test set.
    permute_entities : bool
        If `True`, indicates that entity order should be permuted in the primitive dataset.
    """
    num_position_bins: int = 64
    sequence_path: str = 'data_processed/sg_filtered_unique.npy'
    max_token_length: int = 130
    dataset_size: Optional[int] = None
    validation_fraction: Optional[float] = 0.025
    test_fraction: Optional[float] = 0.05
    permute_entities: bool = False


# =============================================================================
# Tokenization Functions
# =============================================================================

INCLUDE_CONSTRUCTION = True
INCLUDE_CLOCKWISE = True  # clockwise tokens for Arc entities

def _pad_or_truncate_to_length(arr: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
    """Ensures that the array has the given length by truncating or padding.

    If `target_length` is `None`, returns `arr` unchanged.
    """
    if target_length is None:
        return arr

    if len(arr) > target_length:
        return arr[:target_length]

    if isinstance(arr, np.ndarray):
        return np.pad(arr, (0, target_length - len(arr)), constant_values=PrimitiveToken.Pad)
    elif isinstance(arr, torch.Tensor):
        return torch.nn.functional.pad(arr, (0, target_length - len(arr)), value=PrimitiveToken.Pad)
    else:
        raise ValueError('arr must be either numpy array or torch Tensor')


def param_seq_from_tokens(tokens, num_bins):
    """Convert value tokens to sequence of quantized entity parameters.

    Parameters
    ----------
    tokens : np.array
        The sequence of value tokens
    num_bins : int
        The number of quantization bins

    Returns
    -------
    all_params : list
        A list of tuples (params, clockwise, isConstruction) where:
        - params: quantized entity parameters
        - clockwise: True/False for Arc, None for other entities
        - isConstruction: True/False for all entities
    """
    reverse_construction_toks = {
        len(PrimitiveToken) + num_bins: True,
        len(PrimitiveToken) + num_bins + 1: False
    }

    reverse_clockwise_toks = {
        len(PrimitiveToken) + num_bins + 2: True,
        len(PrimitiveToken) + num_bins + 3: False
    }

    all_params = []
    curr_params = []
    clockwise = None  # Only set for Arc entities
    isConstruction = False
    for token in tokens:
        if token == PrimitiveToken.Start:
            continue
        if token < len(PrimitiveToken):
            if curr_params:
                all_params.append((curr_params, clockwise, isConstruction))
                curr_params = []
                clockwise = None  # Reset for next entity
                isConstruction = False
        if token in [PrimitiveToken.Stop, PrimitiveToken.Pad]:
            break
        if token >= len(PrimitiveToken):
            if token <= len(PrimitiveToken) + (num_bins - 1):
                # Numerical coordinate
                curr_params.append(token - len(PrimitiveToken))
            elif token in reverse_clockwise_toks:
                # clockwise attribute (Arc only)
                clockwise = reverse_clockwise_toks[token]
            elif token in reverse_construction_toks:
                # isConstruction attribute
                isConstruction = reverse_construction_toks[token]

    if curr_params:
        # Append possibly leftover entity parameters
        all_params.append((curr_params, clockwise, isConstruction))
    return all_params


def sketch_from_tokens(tokens, num_bins):
    """Convert value tokens to Sketch instance.

    Parameters
    ----------
    tokens : np.array
        The sequence of value tokens
    num_bins : int
        The number of quantization bins

    Returns
    -------
    Sketch
        A Sketch instance corresponding to the given value tokens
    """
    all_params = param_seq_from_tokens(tokens, num_bins)
    sketch = datalib.Sketch()
    for idx, (ent_params, clockwise, isConstruction) in enumerate(all_params):
        ent_params = sketch_utils.dequantize_params(ent_params, num_bins)
        ent_params = ent_params.tolist()
        ent = sketch_utils.entity_from_params(ent_params, clockwise=clockwise)
        if ent is not None:
            ent.entityId = str(idx + 1)
            ent.isConstruction = isConstruction
        sketch.entities[str(idx)] = ent
    return sketch


# =============================================================================
# Dataset Classes
# =============================================================================

class PrimitiveDataset(torch.utils.data.Dataset[SketchTokenSet]):
    """This dataset encapsulates a sequence of sketch as a dataset of tokenized primitives."""

    def __init__(self, sequence_file: str, num_bins: int, max_length: Optional[int] = None, permute: bool = False):
        """Create a new primitive dataset from the given sequence.

        Parameters
        ----------
        sequence_file : str
            Path to the sequence file containing the sketch data.
        num_bins : int
            Number of bins to used for quantizing positional features.
        max_length : int, optional
            If not `None`, the length to which the sequences are padded or truncated.
            Otherwise, no padding / truncation is performed.
        permute : bool, optional
            If True, the primitives are randomly permuted.
            Note, this breaks constraint references. (default False)
        """
        super().__init__()
        self.sequences = flat_array.load_dictionary_flat(sequence_file)['sequences']
        self.max_length = max_length
        self.num_bins = num_bins
        self.permute = permute

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        sketch = datalib.sketch_from_sequence(seq)
        sketch_utils.normalize_sketch(sketch)

        sample, _ = tokenization.tokenize_primitives(sketch, self.num_bins, self.max_length)
        sample = {k: torch.from_numpy(v) for k, v in sample.items()}
        return sample

    def __len__(self) -> int:
        return len(self.sequences)


class CachedPrimitiveDataset(torch.utils.data.Dataset[SketchTokenSet]):
    """This dataset encapsulates a set of pre-tokenized sequences.

    This dataset presents functionality similar to `PrimitiveDataset`, but instead
    of tokenizing sequences on the fly, it relies on a pre-computed sequence file.
    """
    _offsets: torch.Tensor
    _val_tokens: torch.Tensor
    _pos_tokens: torch.Tensor
    _coord_tokens: torch.Tensor

    def __init__(self, cached_sequence_file: str, num_bins: int, max_length: Optional[int] = None):
        super().__init__()

        data = np.load(cached_sequence_file)

        if data['num_bins'] != num_bins:
            raise ValueError('Specified number of bins does not match number of bins in cached sequence file.')

        self._offsets = torch.from_numpy(data['offsets'])
        self._val_tokens = torch.from_numpy(data['val'])
        self._pos_tokens = torch.from_numpy(data['pos'])
        self._coord_tokens = torch.from_numpy(data['coord'])

        self.num_bins = num_bins
        self.max_length = max_length

    def __getitem__(self, idx: int) -> SketchTokenSet:
        idx_start = self._offsets[idx]
        idx_stop = self._offsets[idx + 1]
        segment = slice(idx_start, idx_stop)

        result = {
            'val': self._val_tokens[segment],
            'pos': self._pos_tokens[segment],
            'coord': self._coord_tokens[segment]
        }

        # Pytorch much prefers 64-bit integer indexing for Embeddings.
        return {
            k: _pad_or_truncate_to_length(v.to(dtype=torch.int64), self.max_length)
            for k, v in result.items()
        }

    def __len__(self) -> int:
        return len(self._offsets) - 1


# =============================================================================
# Utilities
# =============================================================================

def _counts_to_offset(x):
    x = np.asarray(x)
    offsets = np.zeros(len(x) + 1, dtype=np.int32)
    np.cumsum(x, out=offsets[1:])
    return offsets


def concatenate_tokens(tokens: Sequence[SketchTokenSet]) -> Dict[str, torch.Tensor]:
    """Concatenates a sequence of token sequences.

    Parameters
    ----------
    tokens : Sequence[SketchTokenSet]
        A sequence of tokens to concatenate.
    """
    counts = np.concatenate([
        np.diff(np.asarray(x['offsets'])) if 'offsets' in x else np.array([len(x['val'])])
        for x in tokens
    ])

    tokens_cat = {
        k: torch.cat([torch.as_tensor(x[k]) for x in tokens])
        for k in tokens[0].keys()
    }

    offsets = torch.from_numpy(_counts_to_offset(counts))

    return {
        **tokens_cat,
        'offsets': offsets
    }


def process_sequence_data(dataset: PrimitiveDataset, num_workers: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Tokenizes all the data from the given `PrimitiveDataset`.

    Parameters
    ----------
    dataset : PrimitiveDataset
        The dataset from which to tokenize the data.
    num_workers : optional, int
        The number of workers to use when processing the data.
        If `None`, uses all available CPUs on the current machine.
    """
    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))

    dataloader = torch.utils.data.dataloader.DataLoader(
        dataset, 512,
        num_workers=num_workers,
        collate_fn=concatenate_tokens
    )

    results = []

    for batch in tqdm.tqdm(dataloader, smoothing=0.01):
        batch = {k: v.clone() for k, v in batch.items()}
        for k in ('val', 'coord', 'pos'):
            batch[k] = batch[k].to(dtype=torch.int16)
        results.append(batch)

    result = concatenate_tokens(results)
    result['num_bins'] = dataset.num_bins

    return result


def split_dataset(
    dataset: torch.utils.data.Dataset,
    validation_fraction: Optional[float] = None,
    test_fraction: Optional[float] = None
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Splits the given dataset into training, validation, and test datasets.

    The split is performed randomly (but the same every time).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split.
    validation_fraction : float, optional
        If not `None`, the fraction of samples to reserve for the validation set.
    test_fraction : float, optional
        If not `None`, the fraction of samples to reserve for the test set.

    Returns
    -------
    train_dataset : torch.utils.data.Subset
        The training dataset split.
    valid_dataset : torch.utils.data.Subset
        The validation dataset split.
    test_dataset : torch.utils.data.Subset
        The test dataset split.
    """
    generator = torch.Generator().manual_seed(4242424242)
    perm = torch.randperm(len(dataset), generator=generator)

    if validation_fraction is None:
        validation_fraction = 0.0

    if test_fraction is None:
        test_fraction = 0.0

    num_train_samples = int(math.ceil(len(dataset) * (1 - validation_fraction - test_fraction)))
    num_test_samples = int(math.ceil(len(dataset) * test_fraction))

    if num_train_samples + num_test_samples > len(dataset):
        num_test_samples = len(dataset) - num_train_samples

    num_valid_samples = len(dataset) - num_train_samples - num_test_samples

    train_idx, valid_idx, test_idx = torch.split(perm, [num_train_samples, num_valid_samples, num_test_samples])

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, valid_dataset, test_dataset


def _get_cache_data_path(sequence_path: os.PathLike, num_position_bins: int) -> str:
    path, _ = os.path.splitext(sequence_path)
    return path + f'_b{num_position_bins}.cache.npz'


def cache_data_if_needed(sequence_path: os.PathLike, num_position_bins: int) -> bool:
    """Generates cached sequence data if does not exist.

    Parameters
    ----------
    sequence_path : os.PathLike
        Path to the file containing raw sequences.
    num_position_bins : int
        Number of bins to use for quantizing positional parameters.
    """
    cache_data_path = _get_cache_data_path(sequence_path, num_position_bins)

    log = logging.getLogger(__name__)
    if os.path.exists(cache_data_path):
        log.info('Found cached data at location {}'.format(cache_data_path))
        return False

    log.info('Cached data not found. Generating cache.')
    primitive_dataset = PrimitiveDataset(sequence_path, num_position_bins)

    data = process_sequence_data(primitive_dataset)
    data = {k: np.asarray(v) for k, v in data.items()}

    log.info('Done processing data, saving data to {}'.format(cache_data_path))
    np.savez(cache_data_path, **data)
    return True


# =============================================================================
# DataModule
# =============================================================================

class PrimitiveDataModule(L.LightningDataModule):
    """This class provides a wrapper over `PrimitiveDataset` to handle data loading."""

    def __init__(self, config: PrimitiveDataConfig, batch_size: int = 128, num_workers: int = 8):
        super().__init__()

        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._dataset = None
        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._valid_dataset_no_noise = None
        self._test_dataset_no_noise = None

    def prepare_data(self) -> None:
        cache_data_if_needed(self.config.sequence_path, self.config.num_position_bins)

    def setup(self, stage: Optional[str] = None):
        dataset_args = {
            'sequence_file': self.config.sequence_path,
            'num_bins': self.config.num_position_bins,
            'max_length': self.config.max_token_length
        }

        if getattr(self.config, 'permute_entities', False):
            # Use getattr to preserve compatibility with some saved models
            logging.info('Loading permuted primitive dataset.')
            self._dataset = PrimitiveDataset(**dataset_args, permute=True)
            non_noise_dataset = PrimitiveDataset(**dataset_args, permute=False)

            _, self._valid_dataset_no_noise, self._test_dataset_no_noise = split_dataset(
                non_noise_dataset, self.config.validation_fraction, self.config.test_fraction
            )
        else:
            logging.info('Loading cached primitive dataset.')
            self._dataset = CachedPrimitiveDataset(
                self.cache_data_file_path,
                self.config.num_position_bins,
                self.config.max_token_length
            )

        self._train_dataset, self._valid_dataset, self._test_dataset = split_dataset(
            self._dataset, self.config.validation_fraction, self.config.test_fraction
        )

    def _make_dataloader(self, dataset, shuffle: bool) -> torch.utils.data.DataLoader[PrimitiveTokenInput]:
        return torch.utils.data.DataLoader(
            dataset, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader[PrimitiveTokenInput]:
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self):
        dl1 = self._make_dataloader(self._valid_dataset, shuffle=False)

        if self._valid_dataset_no_noise is not None:
            return [self._make_dataloader(self._valid_dataset_no_noise, shuffle=False), dl1]
        else:
            return dl1

    def test_dataloader(self):
        dl1 = self._make_dataloader(self._test_dataset, shuffle=False)
        if self._test_dataset_no_noise is not None:
            return [self._make_dataloader(self._test_dataset_no_noise, shuffle=False), dl1]
        else:
            return dl1

    @property
    def cache_data_file_path(self) -> str:
        """Path to the cached token data file."""
        return _get_cache_data_path(self.config.sequence_path, self.config.num_position_bins)

    @property
    def train_dataset_size(self) -> int:
        return len(self._train_dataset)
