"""Detokenization functions for converting tokens back to Sketch objects.

This module provides the inverse operations of tokenization.py:
- reconstruct_primitives: primitive tokens -> Sketch (entities only)
- reconstruct_constraints: constraint tokens -> constraint list
- reconstruct_sketch: full reconstruction with primitives and constraints

For tokenization (Sketch -> tokens), see tokenization.py.
"""
import numpy as np

import sketchgraphs.data as datalib
from sketchgraphs.data import Arc, Circle, Line, Point
from sketchgraphs.data._constraint import Constraint, LocalReferenceParameter

from . import sketch_utils
from .tokenization import GatherMap, tokenize_primitives
from .tokens import (
    PrimitiveToken,
    ConstraintToken,
)

# =============================================================================
# Primitive Detokenization
# =============================================================================

def param_seq_from_tokens(tokens, num_bins: int):
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
    # Attribute token offsets (must match tokenization.py)
    attribute_offset = len(PrimitiveToken) + num_bins  # 72

    # clockwise: attribute_offset + 0/1 (72=False, 73=True)
    # isConstruction: attribute_offset + 2/3 (74=False, 75=True)
    clockwise_base = attribute_offset
    construction_base = attribute_offset + 2

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
            elif token == clockwise_base or token == clockwise_base + 1:
                # clockwise attribute (Arc only): 72=False, 73=True
                clockwise = bool(token - clockwise_base)
            elif token == construction_base or token == construction_base + 1:
                # isConstruction attribute: 74=False, 75=True
                isConstruction = bool(token - construction_base)

    if curr_params:
        # Append possibly leftover entity parameters
        all_params.append((curr_params, clockwise, isConstruction))
    return all_params


def reconstruct_primitives(tokens, num_bins: int) -> datalib.Sketch:
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
# Constraint Detokenization
# =============================================================================

def reconstruct_constraints(c_val_tokens: np.ndarray) -> list[tuple[ConstraintToken, list[int]]]:
    """Convert constraint tokens back to (constraint_type, gather_indices) pairs.

    This function is the inverse of tokenize_constraints for the value tokens.
    It parses the token sequence and extracts constraint types with their
    associated entity references (as gather indices).

    Parameters
    ----------
    c_val_tokens : np.ndarray
        The constraint value token sequence (c_val from tokenization).
        Should contain [Start, <constraints...>, Stop, <padding>].

    Returns
    -------
    list[tuple[ConstraintToken, list[int]]]
        List of (constraint_type, [gather_idx1, gather_idx2, ...]) tuples.
        Each gather_idx is an index into the primitive token positions.
    """
    constraints = []
    current_constraint = None
    current_refs = []

    for token in c_val_tokens:
        token = int(token)

        # Skip padding, mask, and inner padding tokens
        if (token == ConstraintToken.Pad or \
            token == ConstraintToken.Mask or \
            token == ConstraintToken.InnerPad):
            continue

        # Control tokens
        if token == ConstraintToken.Start:
            continue
        if token == ConstraintToken.Stop:
            # Finalize last constraint if any
            if current_constraint is not None:
                constraints.append((current_constraint, current_refs))
            break

        # Check if this is a constraint type token (4-16)
        if token >= ConstraintToken.Coincident and token <= ConstraintToken.Vertical:
            # Save previous constraint if exists
            if current_constraint is not None:
                constraints.append((current_constraint, current_refs))
            # Start new constraint
            current_constraint = ConstraintToken(token)
            current_refs = []
        elif token >= len(ConstraintToken):
            # This is a reference token: gather_idx = token - len(ConstraintToken)
            gather_idx = token - len(ConstraintToken)
            current_refs.append(gather_idx)

    return constraints


def build_gather_idx_mapping(
    gather_map: GatherMap,
) -> dict[int, tuple[int, int]]:
    """Build a mapping from primitive token position to (entity_idx, subnode_idx).

    Parameters
    ----------
    gather_map : GatherMap
        Mapping from node_idx to EntityTokenMapping.

    Returns
    -------
    dict[int, tuple[int, int]]
        Mapping from primitive token position to (entity_idx, subnode_idx).
        entity_idx is 1-based (0 is reserved for external).
        subnode_idx is derived from param_tokens:
        - 0 = entity type token
        - 1+ = param token positions (CenterX=1, CenterY=2, etc.)
    """

    mapping = {}
    for entity_idx, entity_mapping in gather_map.items():
        entity_pos = entity_mapping.entity_token_pos

        # Entity type token position (offset 0)
        mapping[entity_pos] = (entity_idx, 0)

        # Parameter token positions (offset 1+)
        for i, param_token in enumerate(entity_mapping.param_tokens):
            token_pos = entity_pos + 1 + i  # +1 for entity type token
            # subnode_idx based on semantic meaning
            # Use param_token value as subnode_idx for now (can be refined)
            mapping[token_pos] = (entity_idx, i + 1)

    return mapping


# =============================================================================
# Full Reconstruction
# =============================================================================

def reconstruct_sketch(
    prim_val_tokens: np.ndarray,
    c_val_tokens: np.ndarray,
    num_bins: int
) -> datalib.Sketch:
    """Reconstruct a Sketch with entities and constraints from tokens.

    This function takes primitive tokens and constraint tokens, and produces
    a Sketch object with both entities and constraints populated.

    Parameters
    ----------
    prim_val_tokens : np.ndarray
        The primitive value token sequence (from tokenize_primitives).
    c_val_tokens : np.ndarray
        The constraint value token sequence (from tokenize_constraints).
    num_bins : int
        Number of bins used for primitive quantization.

    Returns
    -------
    datalib.Sketch
        A Sketch object with entities and constraints.
    """
    # Reconstruct primitives
    sketch = reconstruct_primitives(prim_val_tokens, num_bins)

    # Normalize before re-tokenizing to avoid out-of-range warnings
    # This operation doesn't affect the following gather_map
    # since it only depends on structure (entity types/order), not parameter values
    sketch_utils.normalize_sketch(sketch)

    # Get gather_map for constraint reference resolution
    _, gather_map = tokenize_primitives(sketch, num_bins, max_length=None)
    gather_to_entity = build_gather_idx_mapping(gather_map)

    # Reconstruct constraints
    constraint_list = reconstruct_constraints(c_val_tokens)

    # Subnode slot to subnode name mapping
    subnode_slot_to_name = {
        Arc: {1: 'center', 2: 'start', 3: 'end'},
        Circle: {1: 'center'},
        Line: {1: 'start', 2: 'end'},
        Point: {}
    }

    # Add constraints to sketch
    entity_list = list(sketch.entities.values())
    for constraint_idx, (constraint_type, gather_refs) in enumerate(constraint_list):
        # Convert gather refs to entity reference strings
        params = []
        for ref_idx, prim_tok_pos in enumerate(gather_refs):
            if prim_tok_pos in gather_to_entity:
                entity_id, subnode_slot = gather_to_entity[prim_tok_pos]

                # Build reference string (e.g., "1" or "1.start")
                if entity_id >= 1 and entity_id <= len(entity_list):
                    ent = entity_list[entity_id - 1]
                    if ent is not None:
                        ent_type = type(ent)
                        subnode_name = subnode_slot_to_name.get(ent_type, {}).get(subnode_slot)
                        if subnode_name:
                            ref_str = f"{entity_id}.{subnode_name}"
                        else:
                            ref_str = str(entity_id)
                    else:
                        ref_str = str(entity_id)
                else:
                    ref_str = str(prim_tok_pos)
            else:
                ref_str = str(prim_tok_pos)

            param_id = f"local{ref_idx}"
            params.append(LocalReferenceParameter(param_id, ref_str))

        # Map ConstraintToken to ConstraintType
        constraint_type_name = constraint_type.name
        if constraint_type_name in datalib.ConstraintType.__members__:
            ct = datalib.ConstraintType[constraint_type_name]
        else:
            continue

        # Create Constraint and add to sketch
        constraint_id = f"c{constraint_idx}"
        constraint = Constraint(constraint_id, ct, params)
        sketch.constraints[constraint_id] = constraint

    return sketch
