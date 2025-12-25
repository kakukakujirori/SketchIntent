"""Unified tokenization interface for img2cad_diffusion.

This module provides a unified API for tokenizing sketches and constraints,
consolidating logic from primitive_data.py and constraint_data.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import sketchgraphs.data as datalib

from . import primitive_data
from . import constraint_data
from . import sketch_utils
from .tokens import (
    TokenVocabulary,
    PrimitiveToken,
    ConstraintToken,
    ENTITY_TOKEN_OFFSETS,
)
from .primitive_data import (
    SketchTokenSet,
    EntityTokenMapping,
    GatherMap,
)


# =============================================================================
# Tokenized Data Structures
# =============================================================================

@dataclass
class TokenizedPrimitives:
    """Tokenization result for primitives only.

    Attributes
    ----------
    tokens : SketchTokenSet
        Dictionary with 'val', 'coord', 'pos' arrays.
    gather_map : GatherMap
        Mapping from entity node_idx to EntityTokenMapping.
    sketch : datalib.Sketch
        The normalized sketch (for visualization/debugging).
    """
    tokens: SketchTokenSet
    gather_map: GatherMap
    sketch: datalib.Sketch


@dataclass
class ConstraintTokenSet:
    """Token set for constraints."""
    val: np.ndarray
    coord: np.ndarray
    pos: np.ndarray


@dataclass
class TokenizedSample:
    """Complete tokenization for training sample (primitives + constraints).

    Attributes
    ----------
    primitives : TokenizedPrimitives
        Tokenized primitive data with gather_map.
    constraint_tokens : ConstraintTokenSet
        Tokenized constraint data.
    sequence : datalib.sequence
        Original sequence (for debugging).
    """
    primitives: TokenizedPrimitives
    constraint_tokens: ConstraintTokenSet
    sequence: datalib.sequence  # Original for reference


# =============================================================================
# Unified Tokenization Functions
# =============================================================================

def tokenize_primitives(
    sketch: datalib.Sketch,
    vocab: TokenVocabulary,
    max_length: Optional[int] = None,
    permute: bool = False,
    normalize: bool = True,
) -> TokenizedPrimitives:
    """Tokenize a sketch's primitives.

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch to tokenize.
    vocab : TokenVocabulary
        Token vocabulary for index computation.
    max_length : Optional[int]
        Maximum token sequence length.
    permute : bool
        Whether to randomly permute entity order.
    normalize : bool
        Whether to normalize the sketch before tokenization.

    Returns
    -------
    TokenizedPrimitives
        Tokenized primitive data with gather_map and sketch.
    """
    # Normalize if needed
    if normalize:
        sketch = datalib.Sketch(entities=sketch.entities.copy(), constraints=sketch.constraints.copy())
        sketch_utils.normalize_sketch(sketch)

    # Tokenize
    tokens, gather_map = primitive_data.tokenize_sketch(
        sketch,
        num_bins=vocab.num_bins,
        max_length=max_length,
        permute=permute
    )

    return TokenizedPrimitives(
        tokens=tokens,
        gather_map=gather_map,
        sketch=sketch
    )


def tokenize_sample(
    seq: datalib.sequence,
    vocab: TokenVocabulary,
    max_length: Optional[int] = None,
    permute: bool = False,
) -> TokenizedSample:
    """Tokenize a complete sample (primitives + constraints).

    This is the main entry point for tokenizing training data.

    Parameters
    ----------
    seq : datalib.sequence
        The SketchGraphs sequence to tokenize.
    vocab : TokenVocabulary
        Token vocabulary for index computation.
    max_length : Optional[int]
        Maximum token sequence length.
    permute : bool
        Whether to randomly permute entity order.

    Returns
    -------
    TokenizedSample
        Complete tokenized sample with primitives and constraints.
    """
    # Convert sequence to sketch
    sketch = datalib.sketch_from_sequence(seq)
    sketch_utils.normalize_sketch(sketch)

    # Tokenize primitives
    prim_tokens, gather_map = primitive_data.tokenize_sketch(
        sketch,
        num_bins=vocab.num_bins,
        max_length=max_length,
        permute=permute
    )

    # Tokenize constraints
    c_tokens = constraint_data.tokenize_constraints(
        seq,
        gather_map,
        max_length
    )

    return TokenizedSample(
        primitives=TokenizedPrimitives(
            tokens=prim_tokens,
            gather_map=gather_map,
            sketch=sketch
        ),
        constraint_tokens=ConstraintTokenSet(
            val=c_tokens['val'],
            coord=c_tokens['coord'],
            pos=c_tokens['pos']
        ),
        sequence=seq
    )


def sample_to_tensors(sample: TokenizedSample) -> Dict[str, torch.Tensor]:
    """Convert a TokenizedSample to tensor dictionary for training.

    This matches the format expected by ConstraintMDDModel.

    Parameters
    ----------
    sample : TokenizedSample
        Tokenized sample to convert.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'val', 'coord', 'pos', 'c_val', 'c_coord', 'c_pos' tensors.
    """
    return {
        'val': torch.from_numpy(sample.primitives.tokens['val']),
        'coord': torch.from_numpy(sample.primitives.tokens['coord']),
        'pos': torch.from_numpy(sample.primitives.tokens['pos']),
        'c_val': torch.from_numpy(sample.constraint_tokens.val),
        'c_coord': torch.from_numpy(sample.constraint_tokens.coord),
        'c_pos': torch.from_numpy(sample.constraint_tokens.pos),
    }
