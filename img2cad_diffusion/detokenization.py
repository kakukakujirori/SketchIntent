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
    CONSTRAINT_ACCEPTABLE_REF_COUNTS,
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
        A list of tuples (params, isConstruction) where:
        - params: quantized entity parameters
        - isConstruction: True/False for all entities
    """
    # Attribute token offsets (must match tokenization.py)
    attribute_offset = len(PrimitiveToken) + num_bins

    # isConstruction: attribute_offset + 0/1
    construction_base = attribute_offset

    all_params = []
    curr_params = []
    isConstruction = False
    for token in tokens:
        # 1. Handle Control Tokens
        if token == PrimitiveToken.Start:
            continue
        elif token == PrimitiveToken.Pad:
            continue
        elif token == PrimitiveToken.Stop:
            if curr_params:
                all_params.append((curr_params, isConstruction))
            break

        # 2. Handle Entity Type Tokens (Arc, Line, etc.)
        # Only treat as new entity if it's a valid primitive type (>= Arc)
        elif token < len(PrimitiveToken):
            if curr_params:
                # Store previous entity
                all_params.append((curr_params, isConstruction))
                curr_params = []
            # Reset for new entity
            isConstruction = False
            continue

        # 3. Handle Parameters and Attributes
        elif token >= len(PrimitiveToken):
            if token < len(PrimitiveToken) + num_bins:
                # Numerical coordinate
                curr_params.append(token - len(PrimitiveToken))
            elif token == construction_base or token == construction_base + 1:
                # isConstruction attribute: False, True
                isConstruction = bool(token - construction_base)

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
    for idx, (ent_params, isConstruction) in enumerate(all_params):
        ent_params = sketch_utils.dequantize_params(ent_params, num_bins)
        ent_params = ent_params.tolist()
        ent = sketch_utils.entity_from_params(ent_params)
        entity_id = str(idx + 1)  # 1-based entity ID
        if ent is not None:
            ent.entityId = entity_id
            ent.isConstruction = isConstruction
        sketch.entities[entity_id] = ent  # Key matches entityId (1-based)
    return sketch


# =============================================================================
# Constraint Detokenization
# =============================================================================

def reconstruct_constraints(c_val_tokens: np.ndarray, token_len_per_constraint: int | None = None) -> tuple[
        list[tuple[ConstraintToken, list[int], tuple[int, int]]],
        list[tuple[ConstraintToken, list[int], tuple[int, int]]],
    ]:
    """Convert constraint tokens back to (constraint_type, reference_indices, token_[start,end]_pos) pairs.

    This function is the inverse of tokenize_constraints for the value tokens.
    It parses the token sequence and extracts constraint types with their
    associated entity references (as gather indices).

    Parameters
    ----------
    c_val_tokens : np.ndarray
        The constraint value token sequence (c_val from tokenization).
        Should contain [Start, <constraints...>, Stop, <padding>].
    token_len_per_constraint : int | None
        The number of tokens per constraint. If None, the number of tokens
        is variable per constraint, and is inferred from the token sequence.

    Returns
    -------
    valid_constraint_list: list[tuple[ConstraintToken, list[int], tuple[int, int]]]
        List of valid constraint tuples, including token ranges [start_pos, end_pos).
    invalid_constraint_list: list[tuple[ConstraintToken, list[int], tuple[int, int]]]
        List of invalid constraint tuples, including token ranges [start_pos, end_pos).
    """
    valid_constraints = []
    invalid_constraints = []
    seen_valid_constraints = set()

    current_constraint = None
    current_refs = []
    current_start_pos = -1

    def validate_constraints(constraint_token: int, references: list[int]) -> bool:
        """Check if constraint has valid number of references."""
        # Is the constraint token valid?
        if constraint_token < 0 or len(ConstraintToken) <= constraint_token:
            return False
        # Are the references in a valid range?
        if any(ref < 0 for ref in references):
            return False
        # Are the reference num acceptable?
        if len(references) not in CONSTRAINT_ACCEPTABLE_REF_COUNTS.get(ConstraintToken(constraint_token), set()):
            return False
        # Is the constraint already seen?
        if (constraint_token, tuple(sorted(references))) in seen_valid_constraints:
            return False
        # Pass all checks
        return True

    def may_add_constraint(
        current_constraint: int | None,
        current_refs: list[int],
        start_pos: int,
        end_pos: int,
    ):
        # Save previous constraint if exists
        if current_constraint is None:
            return
        if not ConstraintToken.has_value(current_constraint):
            invalid_constraints.append((current_constraint, current_refs, (start_pos, end_pos)))
            return

        item = (ConstraintToken(current_constraint), current_refs, (start_pos, end_pos))
        is_valid = validate_constraints(current_constraint, current_refs)
        if is_valid:
            valid_constraints.append(item)
            seen_valid_constraints.add((current_constraint, tuple(sorted(current_refs))))
        else:
            invalid_constraints.append(item)
        return

    if token_len_per_constraint is None:
        # Variable length per constraint
        for idx, token in enumerate(c_val_tokens):
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
                may_add_constraint(current_constraint, current_refs, current_start_pos, idx)
                break

            # Check if this is a constraint type token
            if token < len(ConstraintToken):
                # Finalize previous constraint
                may_add_constraint(current_constraint, current_refs, current_start_pos, idx)
                # Start new constraint
                current_constraint = ConstraintToken(token)
                current_refs = []
                current_start_pos = idx
            else:
                # This is a reference token: gather_idx = token - len(ConstraintToken)
                gather_idx = token - len(ConstraintToken)
                current_refs.append(gather_idx)

    else:
        if token_len_per_constraint <= 0:
            raise ValueError("token_len_per_constraint must be positive")

        start_idx = 1 if c_val_tokens[0] == ConstraintToken.Start else 0

        for idx in range(start_idx, len(c_val_tokens), token_len_per_constraint):
            current_constraint = int(c_val_tokens[idx])
            current_start_pos = idx
            current_refs = [
                int(ref_token) - len(ConstraintToken)
                for ref_token in c_val_tokens[idx + 1 : idx + token_len_per_constraint]
                if ref_token != ConstraintToken.InnerPad]

            # Skip padding and mask tokens
            if (current_constraint == ConstraintToken.Pad or \
                current_constraint == ConstraintToken.Mask):
                continue

            # Control tokens
            if current_constraint == ConstraintToken.Start:
                continue
            if current_constraint == ConstraintToken.Stop:
                break

            # may be a constraint block
            may_add_constraint(current_constraint, current_refs, current_start_pos, current_start_pos + token_len_per_constraint)

    return valid_constraints, invalid_constraints


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
    num_bins: int,
    token_len_per_constraint: int | None,
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
    token_len_per_constraint : int | None
        Number of tokens per constraint. If None, variable length per constraint.

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
    constraint_list, _ = reconstruct_constraints(c_val_tokens, token_len_per_constraint)

    # Subnode slot to subnode name mapping
    # Slot corresponds to param token offset from entity type token.
    # Constraint refs only use X coordinate slots (tokenize_constraints uses StartX, EndX, CenterX).
    # Y coordinate slots should NOT appear - if they do, it's a bug.
    # For Line: params = [StartX, StartY, EndX, EndY] → slot 1=start, slot 3=end
    # For Arc: params = [StartX, StartY, MidX, MidY, EndX, EndY, CenterX, CenterY]
    #   → slot 1=start, 3=mid, 5=end, 7=center
    # For Circle: params = [CenterX, CenterY, Radius] → slot 1=center
    subnode_slot_to_name = {
        Arc: {1: 'start', 3: 'mid', 5: 'end', 7: 'center'},
        Circle: {1: 'center'},
        Line: {1: 'start', 3: 'end'},
        Point: {}
    }

    # Add constraints to sketch
    entity_list = list(sketch.entities.values())
    for constraint_idx, (constraint_type, gather_refs, _) in enumerate(constraint_list):
        # Convert gather refs to entity reference strings
        params = []
        for ref_idx, prim_tok_pos in enumerate(gather_refs):
            if prim_tok_pos in gather_to_entity:
                entity_id, subnode_slot = gather_to_entity[prim_tok_pos]

                # Build reference string (e.g., "1" or "1.start")
                # entity_id is 1-based (from gather_map which uses sketch.entities keys)
                try:
                    entity_id_int = int(entity_id)
                except (ValueError, TypeError) as e:
                    raise e

                if entity_id_int >= 1 and entity_id_int <= len(entity_list):
                    ent = entity_list[entity_id_int - 1]  # Convert 1-based to 0-based index
                    if ent is not None:
                        ent_type = type(ent)
                        subnode_name = subnode_slot_to_name.get(ent_type, {}).get(subnode_slot)
                        # entity_id is already 1-based, use directly
                        if subnode_slot == 0:
                            # Slot 0 = entity type token → edge reference
                            ref_str = str(entity_id)
                        elif subnode_name:
                            # Valid X-coordinate slot → point reference
                            ref_str = f"{entity_id}.{subnode_name}"
                        else:
                            # Unexpected slot (e.g., Y-coordinate) - this is a bug
                            raise ValueError(
                                f"Unexpected subnode_slot {subnode_slot} for entity type {ent_type.__name__}. "
                                f"Only X-coordinate slots should be referenced: {subnode_slot_to_name.get(ent_type, {})}"
                            )
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
