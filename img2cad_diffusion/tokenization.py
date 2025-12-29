"""Unified tokenization interface for img2cad_diffusion.

This module provides the main API for tokenizing sketches:
- tokenize_sketch(sketch, ...) -> TokenizedSketch
- tokenize_sequence(seq, ...) -> TokenizedSketch (convenience wrapper)
"""

from dataclasses import dataclass

import numpy as np
import torch

import sketchgraphs.data as datalib

from . import sketch_utils
from .tokens import (
    SpecialToken,
    PrimitiveToken,
    ConstraintToken,
    PrimitiveParamToken,
    ConstraintParamToken,
)


# =============================================================================
# Type Aliases
# =============================================================================

SketchTokenSet = dict[str, np.ndarray]  # {'val': ..., 'coord': ..., 'pos': ...}


@dataclass
class EntityTokenMapping:
    """Mapping info for a single entity in the token sequence."""
    entity_idx: int
    entity_type: type
    entity_token_pos: int
    param_tokens: list[PrimitiveParamToken]


GatherMap = dict[int, EntityTokenMapping]


# =============================================================================
# Tokenized Data Structures
# =============================================================================

@dataclass
class TokenizedSketch:
    """Complete tokenization result for a sketch.

    Attributes
    ----------
    prim_tokens : SketchTokenSet
        Primitive token arrays (val, coord, pos).
    constr_tokens : SketchTokenSet
        Constraint token arrays (val, coord, pos).
    gather_map : GatherMap
        Entity index -> token position mapping.
    """
    prim_tokens: SketchTokenSet
    constr_tokens: SketchTokenSet
    gather_map: GatherMap


# =============================================================================
# Primitive Tokenization
# =============================================================================

# Coordinate token maps for each entity type
from sketchgraphs.data import Arc, Circle, Line, Point


def _pad_or_truncate(arr: np.ndarray, length: int | None = None, pad_value: int = SpecialToken.Pad) -> np.ndarray:
    """Pad or truncate array to fixed length."""
    if length is None:
        return arr
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), constant_values=pad_value)


def tokenize_primitives(
    sketch: datalib.Sketch,
    num_bins: int,
    max_length: int | None = None,
    uniquify_val_tokens: bool = True,
) -> tuple[SketchTokenSet, GatherMap]:
    """Tokenize sketch primitives (entities).

    Parameters
    ----------
    sketch : Sketch
        Normalized sketch to tokenize.
    num_bins : int
        Number of quantization bins.
    max_length : int | None
        Max sequence length (for padding/truncation).

    Returns
    -------
    tokens : SketchTokenSet
        Token arrays (val, coord, pos).
    gather_map : GatherMap
        Entity index -> token position mapping.
    """

    val_tokens = [PrimitiveToken.Start]
    coord_tokens = [PrimitiveParamToken.Type]  # Start token is entity type
    pos_idx = len(SpecialToken)
    pos_tokens = [pos_idx]

    gather_map: GatherMap = {}

    # offsets to possibly uniquify val tokens
    if uniquify_val_tokens:
        parameter_offset = len(PrimitiveToken)
        attribute_offset = len(PrimitiveToken) + num_bins
    else:
        parameter_offset = len(SpecialToken)
        attribute_offset = len(SpecialToken)

    for entity_id, ent in sketch.entities.items():
        entity_idx = int(entity_id)  # Convert '1', '4', '6' to int
        entity_start_pos = len(val_tokens)

        # Entity type token
        val_tokens.append(PrimitiveToken[ent.type.name])
        coord_tokens.append(PrimitiveParamToken.Type)
        pos_idx += 1
        pos_tokens.append(pos_idx)

        # Parameter tokens
        params, param_tokens = sketch_utils.parameterize_entity(ent)

        # Clip parameters to valid range with warning if significantly outside
        # Use small tolerance to avoid false positives from floating point precision
        eps = 1e-6
        if params.min() < -0.5 - eps or params.max() > 0.5 + eps:
            import warnings
            warnings.warn(
                f"Entity {entity_id} has params outside [-0.5, 0.5]: "
                f"[{params.min():.4f}, {params.max():.4f}]. Clipping.",
                RuntimeWarning
            )
        # Always clip to ensure values are in valid range
        params = np.clip(params, -0.5, 0.5)

        param_bins = sketch_utils.quantize_params(params, num_bins)

        val_tokens.extend(parameter_offset + param_bins)
        coord_tokens.extend(param_tokens)
        pos_tokens.extend([pos_idx] * len(param_bins))

        # Arc: clockwise attribute
        if isinstance(ent, Arc):
            val_tokens.append(attribute_offset + int(ent.clockwise))
            coord_tokens.append(PrimitiveParamToken.Clockwise)
            pos_tokens.append(pos_idx)

        # Construction attribute
        clockwise_offset = 2 if uniquify_val_tokens else 0
        val_tokens.append(attribute_offset + clockwise_offset + int(ent.isConstruction))
        coord_tokens.append(PrimitiveParamToken.Construction)
        pos_tokens.append(pos_idx)

        # Build gather_map entry using sketch entity ID
        gather_map[entity_idx] = EntityTokenMapping(
            entity_idx=entity_idx,
            entity_type=type(ent),
            entity_token_pos=entity_start_pos,
            param_tokens=param_tokens,
        )

    # Stop token
    val_tokens.append(PrimitiveToken.Stop)
    coord_tokens.append(PrimitiveParamToken.Type)  # Stop is like a type token
    pos_tokens.append(pos_idx + 1)

    tokens = {
        'val': _pad_or_truncate(np.array(val_tokens, dtype=np.int64), max_length),
        'coord': _pad_or_truncate(np.array(coord_tokens, dtype=np.int64), max_length),
        'pos': _pad_or_truncate(np.array(pos_tokens, dtype=np.int64), max_length),
    }

    return tokens, gather_map


# =============================================================================
# Constraint Tokenization (using sketch.constraints)
# =============================================================================

# Fixed token length for aligned constraint tokenization
CONSTRAINT_TOKEN_LEN = 4  # [Type, Ref1, Ref2orInnerPad, Ref3orInnerPad]


def tokenize_constraints(
    sketch: datalib.Sketch,
    gather_map: GatherMap,
    max_length: int | None = None,
    uniquify_val_tokens: bool = True,
    sort_by_ref_id: bool = False,
    align_token_length: bool = False,
    use_stop_padding: bool = False,
) -> SketchTokenSet:
    """Tokenize sketch constraints.

    Uses sketch.constraints (not sequence EdgeOps) for cleaner access.
    Constraints are sorted by minimum reference token position for canonical ordering.

    Parameters
    ----------
    sketch : Sketch
        The sketch containing constraints.
    gather_map : GatherMap
        Entity -> token position mapping from tokenize_primitives.
    max_length : int | None
        Max sequence length.
    align_token_length : bool
        If True, each constraint emits exactly CONSTRAINT_TOKEN_LEN tokens (4),
        padding with InnerPad for unused reference slots.
    use_stop_padding : bool
        If True, pad with STOP tokens instead of PAD tokens. Required for Hungarian matching.

    Returns
    -------
    tokens : SketchTokenSet
        Constraint token arrays (val, coord, pos).
    """
    val_token_index_offset = len(ConstraintToken) if uniquify_val_tokens else 0

    # Helper to compute token_pos for a reference (including subnode offset)
    def get_token_pos(entity_idx: int, subnode_name: str | None) -> int:
        mapping = gather_map[entity_idx]
        if subnode_name is not None:
            subnode_to_param = {
                'start': PrimitiveParamToken.StartX,
                'end': PrimitiveParamToken.EndX,
                'center': PrimitiveParamToken.CenterX
            }
            target_param = subnode_to_param.get(subnode_name)
            if target_param is None or target_param not in mapping.param_tokens:
                return mapping.entity_token_pos  # Fallback
            offset = list(mapping.param_tokens).index(target_param) + 1
            return mapping.entity_token_pos + offset
        return mapping.entity_token_pos

    # ==========================================================================
    # Phase 1: Collect all valid constraints with parsed refs
    # ==========================================================================
    # Each entry: (constraint_type: str, sorted_refs: list[tuple[int, str | None]], min_token_pos: int)
    parsed_constraints: list[tuple[str, list[tuple[int, str | None]], int]] = []

    for constraint in sketch.constraints.values():
        # Get constraint type
        constraint_type = constraint.constraint_type.name
        if constraint_type not in ConstraintToken.__members__:
            continue

        # Get entity references with subnode info
        # param.value can be '10' (entity) or '10.start' (subnode)
        refs: list[tuple[int, str | None]] = []
        for param in constraint.parameters:
            if not hasattr(param, 'value'):
                continue
            value = param.value
            if not isinstance(value, str):
                continue
            if '.' in value:
                parts = value.split('.')
                assert len(parts) == 2, f"Invalid subnode ref format: {value}"
                entity_idx = int(parts[0])
                subnode_name = parts[1]  # 'start', 'end', 'center'
                assert subnode_name in ('start', 'end', 'center'), f"Unknown subnode name: {subnode_name}"
                refs.append((entity_idx, subnode_name))
            else:
                entity_idx = int(value)
                refs.append((entity_idx, None))

        # Skip External constraints (ref not in gather_map)
        if any(entity_idx not in gather_map for entity_idx, _ in refs):
            continue

        # Deduplicate consecutive refs
        if len(refs) >= 2:
            unique_refs = [refs[0]]
            for r in refs[1:]:
                if r != unique_refs[-1]:  # Compare full (entity_idx, subnode_name) tuple
                    unique_refs.append(r)
            refs = unique_refs

        # Skip if any reference's token_pos (including subnode offset) exceeds max_length
        if max_length is not None:
            if any(get_token_pos(entity_idx, subnode) >= max_length for entity_idx, subnode in refs):
                continue

        # Sort refs within this constraint by token position
        # Midpoint: ensure midpoint (point/subnode) is always first for consistency
        if constraint_type == 'Midpoint':
            if len(refs) == 2:
                # 2-ref: swap if first ref is a curve (Line/Arc/Circle), not a point/subnode
                first_entity_idx, first_subnode = refs[0]
                first_entity_str = str(first_entity_idx)
                if not isinstance(sketch.entities.get(first_entity_str, None), Point):
                    sorted_refs = list(reversed(refs))
                else:
                    sorted_refs = refs
            else:
                # 3+ refs: first is midpoint (keep), sort rest by token position
                sorted_refs = [refs[0]] + sorted(refs[1:], key=lambda r: get_token_pos(r[0], r[1]))
        else:
            sorted_refs = sorted(refs, key=lambda r: get_token_pos(r[0], r[1]))

        # Compute minimum token position for sorting constraints
        min_token_pos = min(get_token_pos(entity_idx, subnode) for entity_idx, subnode in sorted_refs)

        parsed_constraints.append((constraint_type, sorted_refs, min_token_pos))

    # ==========================================================================
    # Phase 2: Sort constraints by min_token_pos and emit tokens
    # ==========================================================================
    if sort_by_ref_id:
        parsed_constraints.sort(key=lambda x: x[2])  # Sort by min_token_pos

    val_tokens = [ConstraintToken.Start]
    coord_tokens = [ConstraintParamToken.Type]  # Start token
    pos_idx = len(SpecialToken)
    pos_tokens = [pos_idx]

    for constraint_type, sorted_refs, _ in parsed_constraints:
        # Constraint type token
        val_tokens.append(ConstraintToken[constraint_type])
        coord_tokens.append(ConstraintParamToken.Type)
        pos_idx += 1
        pos_tokens.append(pos_idx)

        # Reference tokens (sorted refs already, now append)
        num_refs = len(sorted_refs)

        if align_token_length:
            # Fixed 4-token format: [Type, Ref1, Ref2orInnerPad, Ref3orInnerPad]
            # Type token already added, need exactly 3 ref slots
            max_refs = CONSTRAINT_TOKEN_LEN - 1  # 3 reference slots
            if num_refs > max_refs:
                raise ValueError(
                    f"Constraint '{constraint_type}' has {num_refs} refs, "
                    f"exceeds max {max_refs} for fixed {CONSTRAINT_TOKEN_LEN}-token format"
                )

            # Add actual references
            for entity_idx, subnode_name in sorted_refs:
                token_pos = get_token_pos(entity_idx, subnode_name)
                val_tokens.append(token_pos + val_token_index_offset)
                coord_tokens.append(ConstraintParamToken.Position)
                pos_tokens.append(pos_idx)

            # Pad with InnerPad to reach fixed length
            for _ in range(max_refs - num_refs):
                val_tokens.append(ConstraintToken.InnerPad)
                coord_tokens.append(ConstraintParamToken.InnerPad)
                pos_tokens.append(pos_idx)
        else:
            # Original variable-length format
            for entity_idx, subnode_name in sorted_refs:
                token_pos = get_token_pos(entity_idx, subnode_name)
                val_tokens.append(token_pos + val_token_index_offset)
                coord_tokens.append(ConstraintParamToken.Position)
                pos_tokens.append(pos_idx)

    # Stop token
    val_tokens.append(ConstraintToken.Stop)
    coord_tokens.append(ConstraintParamToken.Type)
    pos_tokens.append(pos_idx + 1)

    # Determine tail padding token
    tail_pad_value = ConstraintToken.Stop if use_stop_padding else ConstraintToken.Pad

    return {
        'val': _pad_or_truncate(np.array(val_tokens, dtype=np.int64), max_length, tail_pad_value),
        'coord': _pad_or_truncate(np.array(coord_tokens, dtype=np.int64), max_length, tail_pad_value),
        'pos': _pad_or_truncate(np.array(pos_tokens, dtype=np.int64), max_length, tail_pad_value),
    }


# =============================================================================
# Main API
# =============================================================================

def tokenize_sketch(
    sketch: datalib.Sketch,
    num_bins: int,
    max_length: int | None = None,
    normalize: bool = True,
    uniquify_primitive_val_tokens: bool = True,
    uniquify_constraint_val_tokens: bool = True,
    align_constraint_token_length: bool = False,
    use_stop_padding: bool = False,
) -> TokenizedSketch:
    """Tokenize a complete sketch (primitives + constraints).

    This is the main entry point for tokenization.

    Parameters
    ----------
    sketch : Sketch
        The sketch to tokenize.
    num_bins : int
        Number of quantization bins for coordinates.
    max_length : int | None
        Max token sequence length.
    normalize : bool
        Whether to normalize the sketch first.
    align_constraint_token_length : bool
        If True, each constraint uses exactly 4 tokens (with InnerPad padding).
    use_stop_padding : bool
        If True, pad constraints with STOP tokens instead of PAD.

    Returns
    -------
    TokenizedSketch
        Complete tokenization with primitives and constraints.
    """
    if normalize:
        sketch = datalib.Sketch(
            entities=sketch.entities.copy(),
            constraints=sketch.constraints.copy()
        )
        sketch_utils.normalize_sketch(sketch)

    prim_tokens, gather_map = tokenize_primitives(sketch, num_bins, max_length, uniquify_primitive_val_tokens)
    constr_tokens = tokenize_constraints(
        sketch, gather_map, max_length, uniquify_constraint_val_tokens,
        align_token_length=align_constraint_token_length,
        use_stop_padding=use_stop_padding
    )

    return TokenizedSketch(
        prim_tokens=prim_tokens,
        constr_tokens=constr_tokens,
        gather_map=gather_map
    )


def tokenize_sequence(
    seq: datalib.sequence,
    num_bins: int,
    max_length: int | None = None,
) -> TokenizedSketch:
    """Tokenize a construction sequence.

    Convenience wrapper that converts sequence to sketch first.

    Parameters
    ----------
    seq : sequence
        SketchGraphs construction sequence.
    num_bins : int
        Number of quantization bins.
    max_length : int | None
        Max token sequence length.

    Returns
    -------
    TokenizedSketch
        Complete tokenization.
    """
    sketch = datalib.sketch_from_sequence(seq)
    return tokenize_sketch(sketch, num_bins, max_length, normalize=True)
