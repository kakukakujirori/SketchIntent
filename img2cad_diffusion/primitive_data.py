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

from . import sketch_utils as i2c_utils
from .tokens import (
    PrimitiveToken as Token,
    NON_COORD_TOKEN,
    PRIMITIVE_COORD_TOKEN_MAP as COORD_TOKEN_MAP,
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
        return np.pad(arr, (0, target_length - len(arr)), constant_values=Token.Pad)
    elif isinstance(arr, torch.Tensor):
        return torch.nn.functional.pad(arr, (0, target_length - len(arr)), value=Token.Pad)
    else:
        raise ValueError('arr must be either numpy array or torch Tensor')


def tokenize_sketch(
    sketch: datalib.Sketch,
    num_bins: int,
    max_length: Optional[int] = None,
    permute: bool = False,
    include_stop: bool = True
) -> Tuple[SketchTokenSet, List[int]]:
    """Tokenizes the given sketch.

    Parameters
    ----------
    sketch : datalib.Sketch
        Sketch from which to generate the tokens.
    num_bins : int
        Number of bins used to quantize sketch positions.
    max_length : int, optional
        Length to which the token sequence is padded or truncated.
    permute : bool, optional
        If True, the primitives are randomly permuted. (default False)
    include_stop : bool, optional
        If `True`, indicates that the stop token is included at the end of the
        sequence. Otherwise, no stop token is included at the end of the sequence.

    Returns
    -------
    data : SketchTokenSet
        A dictionary containing the token sequence corresponding to the given sketch.
        The token sequence is padded or truncated to the given length if specified.
    gather_idx : List[int]
        A list of integers representing indices to track for the constraint model.
    """
    val_tokens = [Token.Start]
    coord_tokens = [NON_COORD_TOKEN]
    pos_idx = 1  # 0 is reserved for padding
    pos_tokens = [pos_idx]

    # Index tracking for constraint model's gather operation
    # Maps entity subnodes to token positions in the primitive token sequence.
    # gather_idxs[0] = None is a placeholder for External (node_idx=0).
    # EdgeOp.references use 0-based node indices:
    #   ref=0 -> External (skipped in tokenize_constraints)
    #   ref=1 -> first real entity -> gather_idxs[1]
    # This allows direct access: gather_idxs[ref] where ref is the node index.
    gather_map = {
        Arc: [0, 1, 3, 5],   # [entity_token, center(x/y), start(x/y), end(x/y)]
        Circle: [0, 1],      # [entity_token, center(x/y)]
        Line: [0, 1, 3],     # [entity_token, start(x/y), end(x/y)]
        Point: [0]           # [entity_token]
    }
    gather_idxs = [None]  # Index 0 is placeholder for External (ref=0)

    # isConstruction tokens
    construction_tok_dict = {
        True: len(Token) + num_bins,
        False: len(Token) + num_bins + 1
    }

    # clockwise tokens (Arc only)
    clockwise_tok_dict = {
        True: len(Token) + num_bins + 2,
        False: len(Token) + num_bins + 3
    }

    if permute:
        # Randomly permute primitive ordering
        entities = list(sketch.entities.items())
        shuffle(entities)
        sketch.entities = OrderedDict(entities)

    for ent in sketch.entities.values():
        gather_idxs.extend([len(val_tokens) + gidx for gidx in gather_map[type(ent)]])

        val_tokens.append(Token[ent.type.name])
        coord_tokens.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos_tokens.append(pos_idx)

        params = i2c_utils.parameterize_entity(ent)
        try:
            param_bins = i2c_utils.quantize_params(params, num_bins)
        except:
            # Zero-length line segment issue
            params[params > 0.5] = 0.5
            params[params < -0.5] = -0.5
            param_bins = i2c_utils.quantize_params(params, num_bins)

        val_tokens.extend(param_bins + len(Token))
        coord_tokens.extend(COORD_TOKEN_MAP[type(ent)])
        pos_tokens.extend([pos_idx] * param_bins.size)

        # Add clockwise attribute for Arc entities
        if isinstance(ent, Arc):
            val_tokens.append(clockwise_tok_dict[ent.clockwise])
            coord_tokens.append(NON_COORD_TOKEN)
            pos_tokens.append(pos_idx)

        # Add isConstruction attribute
        if INCLUDE_CONSTRUCTION:
            val_tokens.append(construction_tok_dict[ent.isConstruction])
            coord_tokens.append(NON_COORD_TOKEN)
            pos_tokens.append(pos_idx)

    if include_stop:
        val_tokens.append(Token.Stop)
        coord_tokens.append(NON_COORD_TOKEN)
        pos_tokens.append(pos_idx + 1)

    sample = {
        'val': _pad_or_truncate_to_length(np.array(val_tokens, dtype=np.int64), max_length),
        'coord': _pad_or_truncate_to_length(np.array(coord_tokens, dtype=np.int64), max_length),
        'pos': _pad_or_truncate_to_length(np.array(pos_tokens, dtype=np.int64), max_length)
    }

    return sample, gather_idxs


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
        len(Token) + num_bins: True,
        len(Token) + num_bins + 1: False
    }

    reverse_clockwise_toks = {
        len(Token) + num_bins + 2: True,
        len(Token) + num_bins + 3: False
    }

    all_params = []
    curr_params = []
    clockwise = None  # Only set for Arc entities
    isConstruction = False
    for token in tokens:
        if token == Token.Start:
            continue
        if token < len(Token):
            if curr_params:
                all_params.append((curr_params, clockwise, isConstruction))
                curr_params = []
                clockwise = None  # Reset for next entity
                isConstruction = False
        if token in [Token.Stop, Token.Pad]:
            break
        if token >= len(Token):
            if token <= len(Token) + (num_bins - 1):
                # Numerical coordinate
                curr_params.append(token - len(Token))
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
        ent_params = i2c_utils.dequantize_params(ent_params, num_bins)
        ent_params = ent_params.tolist()
        ent = i2c_utils.entity_from_params(ent_params, clockwise=clockwise)
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        sketch = datalib.sketch_from_sequence(seq)
        i2c_utils.normalize_sketch(sketch)

        sample, _ = tokenize_sketch(sketch, self.num_bins, self.max_length, permute=self.permute)
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
        if dataset is None:
            return None

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
