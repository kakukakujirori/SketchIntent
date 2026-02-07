"""This module contains the implementation for the "sequence" projection of SketchGraphs.

The sequence projection (consisting of `NodeOp` representing entities, and `EdgeOp` representing constraints),
is a streamlined representation of sketches, adapted for machine learning applications.

"""

import collections
import typing
import json

from ._entity import Entity, EntityType, SubnodeType, ENTITY_TYPE_TO_CLASS
from ._constraint import Constraint, ConstraintType, ConstraintParameterType, LocalReferenceParameter
from ._constraint import NUMERIC_IDS, ENUM_IDS, BOOL_IDS, QuantityParameter, EnumParameter, BooleanParameter, BooleanValue
from .sketch import Sketch


class NodeOp(typing.NamedTuple):
    """This class represents a node (or entity) operation in a construction sequence.

    An entity is specified by a label, along with a dictionary of parameters.
    """
    label: EntityType
    parameters: dict = {}


class EdgeOp(typing.NamedTuple):
    """This class represents an edge (or constraint) operation in a construction sequence.

    An edge is specified by a label, a variable number of targets, and a dictionary of parameters.
    Targets for an edge are nodes, which are referred to using their index in the construction sequence.
    """
    label: ConstraintType
    references: typing.Tuple[int, ...]
    parameters: dict = {}


def _get_entity_parameters(entity: Entity):
    parameter_ids = type(entity).bool_ids + type(entity).float_ids
    return {param_id: getattr(entity, param_id) for param_id in parameter_ids}


def _get_constraint_parameters(constraint: Constraint):
    parameters = {}
    if not constraint.type.has_parameters:
        return parameters

    schema = [param.parameterId for param in constraint.parameters]
    ref_schema = constraint.type.normalize(schema)

    if ref_schema is False:
        # print("[EdgeOp] _get_constraint_parameters() returns False")
        return None  # return None for unsupported schema

    for param in constraint.parameters:
        param_id = param.parameterId
        # Normalize the parameter ID (e.g. localOffset -> local1)
        normalized_id = constraint.type.normalize_param_id(param_id)

        if normalized_id in ref_schema:
            if param.type == ConstraintParameterType.Quantity:
                parameters[normalized_id] = param.expression
            elif param.type in (ConstraintParameterType.Enum, ConstraintParameterType.Boolean):
                parameters[normalized_id] = param.value

    return parameters


def sketch_to_sequence(sketch: Sketch, ignore_invalid_constraints=True) -> typing.List[typing.Union[NodeOp, EdgeOp]]:
    """Creates a sequence representation in terms of `NodeOp` and `EdgeOp` from the given sketch.

    All the entities in the sketch are converted, along with the respective constraints.
    Constraints are expressed in terms of entity indices (instead of original string identifiers).
    The sequence is ordered such that the nodes are in original insertion order, and the edges
    are placed such that they directly follow the last node they reference.

    Parameters
    ----------
    sketch : Sketch
        The sketch object to convert to sequence.
    ignore_invalid_constraints : bool
        If True, indicates that invalid constraints should be ignored. Otherwise, raises a ValueError
        on encountering an invalid constraint.

    Returns
    -------
    list
        A list of construction operations

    Raises
    ------
    ValueError
        If ``ignore_invalid_constraints`` is set to False, a ValueError will be raised
        upon encountering invalid constraints (e.g. constraints which refer to non-existing nodes,
        or whose parameters cannot be processed according to the specified schema).
    """
    node_sequence = []
    edge_sequence = []
    node_idx_map = {}

    node_sequence.append(NodeOp(EntityType.External))
    node_idx_map['External'] = 0

    for node_id, node in sketch.entities.items():
        if node.type == EntityType.Unknown:
            continue

        current_node_index = len(node_sequence)
        node_idx_map[node_id] = current_node_index
        node_sequence.append(NodeOp(node.type, _get_entity_parameters(node)))

        for subnode_type, subnode_id in zip(node.get_subnode_types(), node.get_subnode_ids()):
            current_subnode_index = len(node_sequence)
            node_idx_map[subnode_id] = current_subnode_index
            node_sequence.append(NodeOp(subnode_type))
            edge_sequence.append(EdgeOp(ConstraintType.Subnode, (current_subnode_index, current_node_index)))

    node_sequence.append(NodeOp(EntityType.Stop))

    for edge in sketch.constraints.values():
        if edge.type == ConstraintType.Unknown:
            continue

        references = []
        ignore_reference = False
        for r in edge.get_references():
            r_idx = node_idx_map.get(r)

            if r_idx is not None:
                references.append(r_idx)
                continue

            if ignore_invalid_constraints:
                ignore_reference = True
                break
            else:
                raise ValueError('Invalid constraint reference {0}'.format(r))

        if ignore_reference:
            continue

        references.sort(reverse=True)
        references = tuple(references)
        parameters = _get_constraint_parameters(edge)

        if edge.type.has_parameters and parameters is None:
            if ignore_invalid_constraints:
                # The constraint is expected to have parameters but schema is invalid.
                continue
            raise ValueError('Invalid constraint: schema not supported.')

        edge_sequence.append(EdgeOp(edge.type, references, parameters))

    edge_sequence.sort(key=lambda x: x.references[0])

    sequence = []

    j = 0

    for i, node in enumerate(node_sequence):
        sequence.append(node)

        while j < len(edge_sequence) and edge_sequence[j].references[0] == i:
            sequence.append(edge_sequence[j])
            j += 1

    # The above should exhaust all edges as no edges refer to the last (stop) node.
    assert j == len(edge_sequence)

    return sequence


def sketch_from_sequence(seq, ignore_external_constraints=True) -> Sketch:
    """Builds a Sketch object from the given sequence.

    Parameters
    ----------
    seq : List[Union[NodeOp, EdgeOp]]
        A construction sequence representing a sketch.
    ignore_external_constraints : bool, optional
        If True, ignores external constraints (referencing the external node).

    Returns
    -------
    Sketch
        A sketch object representing the given construction sequence.
    """
    entities = collections.OrderedDict()
    constraints = collections.OrderedDict()

    # Map from node index in sequence to (main_node_index, subnode_label)
    # Used for translating subnode references to entity_id.subnode_suffix
    main_node_map = {}

    # Map from SubnodeType to string suffix
    subnode_label_to_string = {
        SubnodeType.SN_Start: 'start',
        SubnodeType.SN_End: 'end',
        SubnodeType.SN_Center: 'center'
    }

    node_idx = -1
    constraint_idx = 1

    def create_entity_id(node_idx: int) -> str:
        """Helper function for creating entity id - manages sub-entities"""
        if node_idx not in main_node_map:
            return str(node_idx)
        else:
            main_idx, subnode_label = main_node_map[node_idx]
            return '%i.%s' % (main_idx, subnode_label_to_string[subnode_label])

    for op in seq:
        if isinstance(op, NodeOp):
            node_idx += 1
            # Check if this is a main (non-sub) node
            if isinstance(op.label, EntityType):
                last_main_idx = node_idx
                entityId = str(node_idx)
                # Don't create External or Stop entity
                if op.label == EntityType.External:
                    if not ignore_external_constraints:
                        # Create a dummy Point for External geometry to allow constraints to reference it
                        entities[entityId] = ENTITY_TYPE_TO_CLASS[EntityType.Point](entityId, isConstruction=True)
                elif op.label != EntityType.Stop:
                    entities[entityId] = ENTITY_TYPE_TO_CLASS[op.label](entityId)
                    # Extract parameters for numerical entities
                    for param_id, val in op.parameters.items():
                        setattr(entities[entityId], param_id, val)
            else:
                # Mark main reference of this subnode
                main_node_map[node_idx] = (last_main_idx, op.label)

        else:

            if op.label == ConstraintType.Subnode:
                # Verify EdgeOp Subnode matches NodeOp-inferred parent relationship
                if len(op.references) >= 2:
                    child_idx = op.references[0]
                    parent_idx = op.references[1]
                    if child_idx in main_node_map:
                        inferred_parent, subnode_label = main_node_map[child_idx]
                        assert inferred_parent == parent_idx, \
                            f"EdgeOp Subnode mismatch: EdgeOp says child {child_idx} -> parent {parent_idx}, " \
                            f"but NodeOp ordering inferred parent {inferred_parent}"
                # Don't include subnode edges in constraints
                continue

            constraint_ent_id = 'c_%i' % constraint_idx
            constraint_idx += 1
            constraint_type = op.label

            # Check if all referenced entities exist (handling filtered External node)
            # We need to check if ANY reference points to a missing entity (due to filtering or Stop node)
            references_valid = True
            for ref_idx in op.references:
                # Get main node index if it's a subnode
                # We must use the same logic as create_entity_id to find the main node
                if ref_idx in main_node_map:
                    check_idx, _ = main_node_map[ref_idx]
                else:
                    check_idx = ref_idx

                # Convert to string ID for main node
                check_id = str(check_idx)

                # Check if the main entity exists in our constructed entities dict
                # This handles both External (if skipped) and Stop nodes (always skipped)
                if check_id not in entities:
                    references_valid = False
                    break

            if not references_valid:
                continue

            # Build reference parameters
            params = []

            if constraint_type == ConstraintType.Midpoint:
                # Midpoint: preserve original Sequence order (midpoint first, then curve/points)
                for i, ref_idx in enumerate(op.references):
                    param_id = f'local{i}'
                    params.append(LocalReferenceParameter(param_id, create_entity_id(ref_idx)))
            else:
                # Other constraints: symmetric, order doesn't matter semantically
                # Use localFirst/localSecond naming for compatibility
                for i, ref_idx in enumerate(op.references):
                    if i == 0:
                        param_id = 'localFirst'
                    elif i == 1:
                        param_id = 'localSecond'
                    else:
                        param_id = f'local{i}'
                    params.append(LocalReferenceParameter(param_id, create_entity_id(ref_idx)))

            # Extract parameters for numerical constraints
            params_are_ok = True
            for param_id, val in op.parameters.items():
                if param_id in NUMERIC_IDS:
                    if val == 'OTHER':  # skip unsupported values
                        params_are_ok = False
                    val = val.replace('METER', 'm')
                    val = val.replace('DEGREE', 'deg')
                    params.append(QuantityParameter(param_id, '0.0', val))  # val is really expression here
                elif param_id in ENUM_IDS:
                    params.append(EnumParameter(param_id, val))
                elif param_id in BOOL_IDS:
                    val = bool(BooleanValue[val])
                    params.append(BooleanParameter(param_id, val))
                else:
                    raise ValueError(f"Unknown parameter id: {param_id}")

            if params_are_ok:
                constraints[constraint_ent_id] = Constraint(constraint_ent_id, constraint_type, params)

    return Sketch(entities=entities, constraints=constraints)


def pgvgraph_from_sequence(seq):
    """Builds a pgv.AGraph from the sequence.

    Hyperedges are not supported in the resulting graph.

    Parameters
    ----------
    seq : List[Union[NodeOp, EdgeOp]]
        A construction sequence representing a sketch.

    Returns
    -------
    pgv.AGraph
        A pygraphviz AGraph corresponding to the given sequence.
    """
    import pygraphviz as pgv

    graph = pgv.AGraph(strict=False)
    idx = 0
    for op in seq:
        if isinstance(op, NodeOp):
            if op.label != EntityType.Stop:
                graph.add_node(idx, label=op.label.name, parameters=json.dumps(op.parameters), index=idx)
                idx += 1
    idx = 0
    for op in seq:
        if isinstance(op, EdgeOp):
            if len(op.references) > 2:
                continue  # hyperedges not supported
            node_a = op.references[0]
            if len(op.references) == 1:
                node_b = node_a
            elif len(op.references) == 2:
                node_b = op.references[1]
            else:
                raise ValueError("Invalid number of references in EdgeOp.")
            graph.add_edge(node_a, node_b, label=op.label.name, parameters=json.dumps(op.parameters), index=idx)
            idx += 1
    return graph