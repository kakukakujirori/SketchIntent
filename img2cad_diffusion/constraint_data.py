"""This module handles constraint dataset functionality including tokenization.
"""

import copy
import random
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

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

        # Preserve clockwise attribute for Arc entities
        clockwise = entity.clockwise if isinstance(entity, Arc) else None
        new_ent = sketch_utils.entity_from_params(new_params, entity_id=entity.entityId, clockwise=clockwise)

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

def deduplicate_refs(op: datalib.EdgeOp) -> datalib.EdgeOp:
    """Remove duplicate references from EdgeOp.

    Some Coincident EdgeOps in SketchGraphs data have duplicate references like (5, 5, 3, 3).
    This function deduplicates them. Only applies to 4+ refs cases.

    For refs with <= 3 elements (normal constraints including Midpoint),
    the original EdgeOp is returned unchanged to preserve ordering.

    Note: Sorting is NOT performed here - that responsibility belongs to the caller
    (e.g., tokenize_constraints sorts refs based on gather_idxs).

    Parameters
    ----------
    op : EdgeOp
        The EdgeOp to process.

    Returns
    -------
    EdgeOp
        EdgeOp with deduplicated references for 4+ refs cases, original otherwise.
    """
    refs = op.references
    if len(refs) >= 4:
        # This should only happen for duplicated Coincident constraints
        assert op.label == datalib.ConstraintType.Coincident, \
            f"4+ refs bug should only occur for Coincident, got {op.label}"
        deduped = tuple(set(refs))
        assert len(deduped) == 2, f"Expected 2 unique refs after dedup, got {deduped}"
        return datalib.EdgeOp(op.label, deduped, op.parameters)
    return op


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
        if 0 in op.references:  # skip external constraints
            continue

        # Deduplicate refs (handles cases like (5, 5, 3, 3) -> (3, 5))
        op = deduplicate_refs(op)
        refs = op.references

        # Skip constraints with references that would be out of bounds after truncation
        # gather_idxs[0]=None (External placeholder, ref=0)
        # gather_idxs[ref] gives the first token position for entity with node_idx=ref
        if max_length is not None:
            if any(gather_idxs[ref] >= max_length for ref in refs):
                continue

        # Add constraint type tokens
        val_tokens.append(Token[op.label.name])
        coord_tokens.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos_tokens.append(pos_idx)

        # Add reference parameters
        # For Midpoint constraints, the first reference is the midpoint and should stay fixed
        # Only sort the remaining references (symmetric points)
        if op.label == datalib.ConstraintType.Midpoint and len(refs) == 3:
            # Midpoint(A, B, C) means A is the midpoint of B and C
            # Keep A fixed, sort B and C
            sorted_refs = [refs[0]] + sorted(refs[1:], key=lambda r: gather_idxs[r])
        else:
            sorted_refs = sorted(refs, key=lambda r: gather_idxs[r])

        val_tokens.extend([gather_idxs[ref] + len(Token) for ref in sorted_refs])
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
# Constraint Detokenization
# =============================================================================

def detokenize_constraints(c_val_tokens: np.ndarray) -> List[Tuple[Token, List[int]]]:
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
    List[Tuple[Token, List[int]]]
        List of (constraint_type, [gather_idx1, gather_idx2, ...]) tuples.
        Each gather_idx is an index into the gather_idxs list from primitive tokenization.
    """
    constraints = []
    current_constraint = None
    current_refs = []

    for token in c_val_tokens:
        token = int(token)

        # Skip padding and mask tokens
        if token == Token.Pad or token == Token.Mask:
            continue

        # Control tokens
        if token == Token.Start:
            continue
        if token == Token.Stop:
            # Finalize last constraint if any
            if current_constraint is not None:
                constraints.append((current_constraint, current_refs))
            break

        # Check if this is a constraint type token (4-16)
        if token >= Token.Coincident and token <= Token.Vertical:
            # Save previous constraint if exists
            if current_constraint is not None:
                constraints.append((current_constraint, current_refs))
            # Start new constraint
            current_constraint = Token(token)
            current_refs = []
        elif token >= len(Token):
            # This is a reference token: gather_idx = token - len(Token)
            gather_idx = token - len(Token)
            current_refs.append(gather_idx)

    return constraints


def build_gather_idx_mapping(sketch: datalib.Sketch, num_bins: int) -> Dict[int, Tuple[int, int]]:
    """Build a mapping from primitive token position to (entity_id, subnode_idx).

    The gather_idxs created during primitive tokenization map entity sub-references
    (like start/end points) to positions in the primitive token sequence.
    This function creates a reverse mapping to recover which entity and which
    sub-part of that entity a primitive token position refers to.

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch containing the entities.
    num_bins : int
        Number of bins used for primitive tokenization.

    Returns
    -------
    Dict[int, Tuple[int, int]]
        Mapping from primitive token position to (entity_idx, subnode_idx).
        entity_idx is 1-based (0 is reserved for external).
        subnode_idx meanings depend on entity type:
        - Arc: 0=entity, 1=start, 2=mid, 3=end
        - Circle: 0=entity, 1=center
        - Line: 0=entity, 1=start, 2=end
        - Point: 0=entity
    """
    from sketchgraphs.data import Arc, Circle, Line, Point

    # Re-tokenize to get the actual gather_idxs (positions in primitive token sequence)
    _, gather_idxs = primitive_data.tokenize_sketch(sketch, num_bins, max_length=None, include_stop=False)

    # gather_map: for each entity type, which subnodes are tracked
    gather_map = {
        Arc: [0, 1, 2, 3],      # entity, start, mid, end
        Circle: [0, 1],          # entity, center
        Line: [0, 1, 2],         # entity, start, end
        Point: [0]               # entity only
    }


    # Build mapping: primitive_token_position -> (entity_idx, subnode_idx)
    # gather_idxs[0]=None (External placeholder), gather_idxs[1]+ are real entities
    # Note: entity_idx here is 1-based (first entity = 1)
    mapping = {}

    gather_list_idx = 1  # Skip index 0 (External placeholder) and start at 1 (first real entity)
    entity_idx = 1  # 1-based entity index (matching SketchGraphs refs)

    for ent in sketch.entities.values():
        if ent is None:
            entity_idx += 1
            continue

        ent_type = type(ent)
        if ent_type not in gather_map:
            entity_idx += 1
            continue

        subnode_types = gather_map[ent_type]
        for subnode_idx in subnode_types:
            if gather_list_idx < len(gather_idxs):
                prim_token_pos = gather_idxs[gather_list_idx]
                if prim_token_pos is not None:  # Skip None placeholders
                    mapping[prim_token_pos] = (entity_idx, subnode_idx)
                gather_list_idx += 1

        entity_idx += 1

    return mapping


def reconstruct_sketch_with_constraints(
    sketch: datalib.Sketch,
    c_val_tokens: np.ndarray,
    num_bins: int
) -> datalib.sequence:
    """Reconstruct a full construction sequence with constraints from tokens.

    This function takes a sketch (reconstructed from primitive tokens) and
    constraint tokens, and produces a complete construction sequence that
    includes both NodeOp (for entities) and EdgeOp (for constraints).

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch containing entities, typically from primitive_data.sketch_from_tokens().
    c_val_tokens : np.ndarray
        The constraint value token sequence.
    num_bins : int
        Number of bins used for primitive quantization (for re-tokenizing to get gather_idxs).

    Returns
    -------
    datalib.sequence
        A complete construction sequence with NodeOp and EdgeOp.
    """
    # Re-tokenize to get gather_idxs mapping
    _, gather_idxs = primitive_data.tokenize_sketch(sketch, num_bins, max_length=None, include_stop=False)

    # Build mapping from primitive token position to (entity_idx, subnode_idx)
    gather_to_entity = build_gather_idx_mapping(sketch, num_bins)

    # Detokenize constraints
    constraint_list = detokenize_constraints(c_val_tokens)

    # Build the sequence with proper subnode structure
    sequence = []
    node_ops = []  # Track NodeOps for subnode reference
    subnode_map = {}  # Maps (entity_idx, subnode_type) -> node_idx
    entity_node_map = {}  # Maps entity_idx -> node_idx of the entity NodeOp

    # Subnode types for each entity - ORDER MUST MATCH ORIGINAL SKETCHGRAPHS SEQUENCE
    # Arc subnodes appear as: Center, Start, End in the original sequence
    from sketchgraphs.data import Arc, Circle, Line, Point

    subnode_types = {
        Arc: [datalib.SubnodeType.SN_Center, datalib.SubnodeType.SN_Start, datalib.SubnodeType.SN_End],
        Circle: [datalib.SubnodeType.SN_Center],
        Line: [datalib.SubnodeType.SN_Start, datalib.SubnodeType.SN_End],
        Point: []  # Points have no subnodes
    }

    # Helper to convert entity to parameters dict
    def entity_to_parameters(ent) -> dict:
        """Convert a sketch entity to a parameters dictionary for NodeOp."""
        params = {'isConstruction': getattr(ent, 'isConstruction', False)}

        if isinstance(ent, Arc):
            params.update({
                'xCenter': ent.xCenter,
                'yCenter': ent.yCenter,
                'radius': ent.radius,
                'xDir': getattr(ent, 'xDir', 1.0),
                'yDir': getattr(ent, 'yDir', 0.0),
                'startParam': getattr(ent, 'startParam', 0.0),
                'endParam': getattr(ent, 'endParam', 3.14159),
                'clockwise': getattr(ent, 'clockwise', False),
            })
        elif isinstance(ent, Circle):
            params.update({
                'xCenter': ent.xCenter,
                'yCenter': ent.yCenter,
                'radius': ent.radius,
            })
        elif isinstance(ent, Line):
            start = ent.start_point
            end = ent.end_point
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = (dx**2 + dy**2)**0.5
            if length > 1e-10:
                dirX, dirY = dx/length, dy/length
            else:
                dirX, dirY = 1.0, 0.0
            params.update({
                'pntX': start[0],
                'pntY': start[1],
                'dirX': dirX,
                'dirY': dirY,
                'startParam': 0.0,
                'endParam': length,
            })
        elif isinstance(ent, Point):
            params.update({
                'x': ent.x,
                'y': ent.y,
            })

        return params

    # Add NodeOps for entities and their subnodes
    entity_idx = 1  # 1-based entity indexing (External=0 is skipped)
    for ent_key, ent in sketch.entities.items():
        if ent is None:
            entity_idx += 1
            continue

        # Add entity NodeOp
        entity_node_idx = len(node_ops)
        entity_node_map[entity_idx] = entity_node_idx
        node_op = datalib.NodeOp(label=ent.type, parameters=entity_to_parameters(ent))
        node_ops.append(node_op)
        sequence.append(node_op)

        # Add subnode NodeOps for this entity
        ent_type = type(ent)
        for sn_type in subnode_types.get(ent_type, []):
            subnode_node_idx = len(node_ops)
            subnode_op = datalib.NodeOp(label=sn_type, parameters={})
            node_ops.append(subnode_op)
            sequence.append(subnode_op)
            subnode_map[(entity_idx, sn_type)] = subnode_node_idx

            # Add Subnode EdgeOp to link subnode to parent entity
            subnode_edge = datalib.EdgeOp(
                label=datalib.ConstraintType.Subnode,
                references=(subnode_node_idx, entity_node_idx)
            )
            sequence.append(subnode_edge)

        entity_idx += 1

    # Map: subnode index in gather_idxs -> subnode type
    # For Arc: gather slots are [entity(0), center(1), start(2), end(3)]
    # For Line: gather slots are [entity(0), start(1), end(2)]
    subnode_idx_to_type = {
        Arc: {0: None, 1: datalib.SubnodeType.SN_Center, 2: datalib.SubnodeType.SN_Start, 3: datalib.SubnodeType.SN_End},
        Circle: {0: None, 1: datalib.SubnodeType.SN_Center},
        Line: {0: None, 1: datalib.SubnodeType.SN_Start, 2: datalib.SubnodeType.SN_End},
        Point: {0: None}
    }

    # 2. Add EdgeOps for constraints
    for constraint_type, gather_refs in constraint_list:
        # Convert gather refs (primitive token positions) to node indices
        node_refs = []
        for prim_tok_pos in gather_refs:
            if prim_tok_pos in gather_to_entity:
                entity_id, subnode_slot = gather_to_entity[prim_tok_pos]

                # Get entity type to determine what subnode type this slot represents
                ent_list = list(sketch.entities.values())
                if entity_id >= 1 and entity_id <= len(ent_list):
                    ent = ent_list[entity_id - 1]
                    if ent is not None:
                        ent_type = type(ent)
                        sn_type = subnode_idx_to_type.get(ent_type, {}).get(subnode_slot)

                        if sn_type is not None and (entity_id, sn_type) in subnode_map:
                            # Reference the subnode's node index
                            node_refs.append(subnode_map[(entity_id, sn_type)])
                        elif entity_id in entity_node_map:
                            # Reference the entity's node index (slot 0 = entity itself)
                            node_refs.append(entity_node_map[entity_id])
                        else:
                            node_refs.append(0)
                    else:
                        node_refs.append(0)
                else:
                    node_refs.append(prim_tok_pos)
            else:
                node_refs.append(prim_tok_pos)

        # Map Token to ConstraintType
        constraint_type_name = constraint_type.name
        if constraint_type_name in datalib.ConstraintType.__members__:
            constraint_label = datalib.ConstraintType[constraint_type_name]
        else:
            continue  # Skip unknown constraint types

        # Create EdgeOp
        edge_op = datalib.EdgeOp(
            label=constraint_label,
            references=tuple(node_refs)
        )
        sequence.append(edge_op)

    return sequence


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
