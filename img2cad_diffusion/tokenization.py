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
    param_tokens: tuple[PrimitiveParamToken, ...]


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
        params, param_tokens = sketch_utils.parameterize_entity(ent)  # type: ignore[no-redef]

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


ConstraintOrder = str  # Literal["original", "sort_by_ref_id", "random_shuffle"]


def _refine_constraint_type(
    constraint_type: str,
    refs: list[tuple[int, str | None]],
    parameters: list,
) -> str | None:
    """Refine constraint type based on references and parameters.

    Returns the refined constraint type string (e.g. 'DistanceX') or None if the
    constraint should be dropped (e.g. invalid Mirror/Offset).
    """
    # 1. Mirror Logic
    if constraint_type == 'Mirror':
        # 2-ref Mirror usually refers to external entities, so drop it
        return constraint_type if len(refs) == 3 else None

    # 2. Offset Logic
    if constraint_type == 'Offset':
        # hard to treat offset constraint
        return None

    # 3. Distance Logic (Direction Mapping)
    if constraint_type == 'Distance':
        # Identify direction from parameters
        direction = None
        for p in parameters:
            p_id = getattr(p, 'parameterId', '')
            if p_id == 'direction':
                direction = getattr(p, 'value', None)
                break

        if direction == 'HORIZONTAL':
            return 'DistanceX'
        elif direction == 'VERTICAL':
            return 'DistanceY'
        else:
            return 'Distance'

    # 4. Other Logics
    return constraint_type


def tokenize_constraints(
    sketch: datalib.Sketch,
    gather_map: GatherMap,
    max_length: int | None = None,
    uniquify_val_tokens: bool = True,
    constraint_order: ConstraintOrder = "original",
    align_token_length: bool = False,
    use_stop_padding: bool = False,
) -> SketchTokenSet:
    """Tokenize sketch constraints.

    Uses sketch.constraints (not sequence EdgeOps) for cleaner access.
    Constraints can be reordered based on constraint_order.

    Parameters
    ----------
    sketch : Sketch
        The sketch containing constraints.
    gather_map : GatherMap
        Entity -> token position mapping from tokenize_primitives.
    max_length : int | None
        Max sequence length.
    uniquify_val_tokens : bool
        If True, reference primitive indices are offset by len(ConstraintToken) to differentiate from constraint type tokens.
    constraint_order : str
        Constraint ordering: "original" (sequence order), "sort_by_ref_id" (canonical),
        or "random_shuffle" (random each call).
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
        # Only LocalReferenceParameter contains entity references
        # param.value can be '10' (entity) or '10.start' (subnode)
        refs: list[tuple[int, str | None]] = []
        for param in constraint.parameters:
            # Only process LocalReferenceParameter - skip QuantityParameter and EnumParameter
            if type(param).__name__ != 'LocalReferenceParameter':
                continue

            value = getattr(param, 'value', None)
            if not isinstance(value, str):
                raise ValueError(f"LocalReferenceParameter.value should be str, got {type(value)}: {value}")

            if '.' in value:
                parts = value.split('.')
                if not parts[0].isdigit():
                    raise ValueError(f"Unexpected non-numeric reference prefix: {value}")
                if len(parts) != 2:
                    raise ValueError(f"Unexpected reference format with {len(parts)} parts: {value}")
                entity_idx, subnode_name = int(parts[0]), parts[1]
                if subnode_name not in ('start', 'end', 'center'):
                    raise ValueError(f"Unknown subnode name: {subnode_name} in {value}")
            else:
                if not value.isdigit():
                    raise ValueError(f"LocalReferenceParameter.value should be numeric, got: {value}")
                entity_idx, subnode_name = int(value), None

            # --- FILTERING: External Entity check ---
            if entity_idx not in gather_map:
                # Reference to external entity (not in primitive map) -> DROP
                refs = []
                break # To outer loop continue

            refs.append((entity_idx, subnode_name))

        # Check references count and validity
        if not refs:
            continue

        # --- SPECIFIC CONSTRAINT MAPPING & FILTERING ---
        # --- SPECIFIC CONSTRAINT MAPPING & FILTERING ---
        constraint_type = _refine_constraint_type(constraint_type, refs, constraint.parameters)
        if constraint_type is None:
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
        # Midpoint: ensure midpoint (center point) is always first for consistency
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
                # 3+ refs: determine midpoint geometrically
                # For Midpoint(mid, p1, p2), mid.coords = (p1.coords + p2.coords) / 2
                # We test all permutations and find the best one
                def get_point_coords(entity_idx: int, subnode_name: str | None):
                    """Get (x, y) coordinates for a reference."""
                    ent = sketch.entities.get(str(entity_idx))
                    if ent is None:
                        return None
                    if isinstance(ent, Point):
                        return (ent.x, ent.y)
                    elif isinstance(ent, Line):
                        if subnode_name == 'start':
                            sp = ent.start_point
                            return (sp[0], sp[1])
                        elif subnode_name == 'end':
                            ep = ent.end_point
                            return (ep[0], ep[1])
                        else:
                            # Edge reference - use midpoint of line
                            sp = ent.start_point
                            ep = ent.end_point
                            return ((sp[0] + ep[0]) / 2, (sp[1] + ep[1]) / 2)
                    elif isinstance(ent, (Circle, Arc)):
                        if subnode_name in ('start', 'end'):
                            # For arc start/end, would need more complex calc
                            return (ent.xCenter, ent.yCenter)
                        return (ent.xCenter, ent.yCenter)
                    return None

                # Find which ref is the geometric midpoint
                best_mid_idx = 0
                best_err = float('inf')

                for mid_idx in range(len(refs)):
                    mid_entity_idx, mid_subnode = refs[mid_idx]
                    mid_coords = get_point_coords(mid_entity_idx, mid_subnode)
                    if mid_coords is None:
                        continue

                    # Get other refs
                    other_refs = [refs[i] for i in range(len(refs)) if i != mid_idx]
                    if len(other_refs) < 2:
                        continue

                    # Check if mid is midpoint of first two others
                    p1_coords = get_point_coords(other_refs[0][0], other_refs[0][1])
                    p2_coords = get_point_coords(other_refs[1][0], other_refs[1][1])
                    if p1_coords is None or p2_coords is None:
                        continue

                    calc_mid_x = (p1_coords[0] + p2_coords[0]) / 2
                    calc_mid_y = (p1_coords[1] + p2_coords[1]) / 2
                    err = ((mid_coords[0] - calc_mid_x)**2 + (mid_coords[1] - calc_mid_y)**2)**0.5

                    if err < best_err:
                        best_err = err
                        best_mid_idx = mid_idx

                # Reorder: midpoint first, then others sorted by token position
                mid_ref = refs[best_mid_idx]
                other_refs = [refs[i] for i in range(len(refs)) if i != best_mid_idx]
                sorted_refs = [mid_ref] + sorted(other_refs, key=lambda r: get_token_pos(r[0], r[1]))
        else:
            sorted_refs = sorted(refs, key=lambda r: get_token_pos(r[0], r[1]))

        # Compute minimum token position for sorting constraints
        min_token_pos = min(get_token_pos(entity_idx, subnode) for entity_idx, subnode in sorted_refs)

        parsed_constraints.append((constraint_type, sorted_refs, min_token_pos))

    # ==========================================================================
    # Phase 2: Order constraints and emit tokens
    # ==========================================================================
    if constraint_order == "sort_by_ref_id":
        parsed_constraints.sort(key=lambda x: x[2])  # Sort by min_token_pos
    elif constraint_order == "random_shuffle":
        import random
        random.shuffle(parsed_constraints)
    elif constraint_order == "original":
        pass
    else:
        raise NotImplementedError(f"Unknown constraint_order: {constraint_order}")

    val_tokens = [ConstraintToken.Start]
    coord_tokens = [ConstraintParamToken.Type]  # Start token
    pos_idx = pos_idx_start = len(SpecialToken)
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

    ret =  {
        'val': _pad_or_truncate(np.array(val_tokens, dtype=np.int64), max_length, tail_pad_value),
        'coord': _pad_or_truncate(np.array(coord_tokens, dtype=np.int64), max_length, pad_value=ConstraintParamToken.Pad),
        'pos': _pad_or_truncate(np.array(pos_tokens, dtype=np.int64), max_length, pad_value=0),
    }

    # count constraints
    ret['cnt'] = ret['pos'].max() - pos_idx_start - 1

    return ret


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
    constraint_order: ConstraintOrder = "original",
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
    constraint_order : str
        Constraint ordering: "original", "sort_by_ref_id", or "random_shuffle".

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
        constraint_order=constraint_order,
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
