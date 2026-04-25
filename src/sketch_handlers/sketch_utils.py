"""This module contains some utilities for processing the sketch data for
img2cad.
"""
import math
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

import sketchgraphs.data as datalib
from sketchgraphs.data import EntityType, BooleanParameter, ConstraintParameter, EnumParameter, LocalReferenceParameter, QuantityParameter
from sketchgraphs.data import Arc, Circle, Line, Point


# Positional parameters
POS_PARAMS = {
    EntityType.Arc: ('xCenter', 'yCenter'),
    EntityType.Circle: ('xCenter', 'yCenter'),
    EntityType.Line: ('pntX', 'pntY'),
    EntityType.Point: ('x', 'y')
}


# Scale parameters
SCALE_PARAMS = {
    EntityType.Arc: ('radius',),
    EntityType.Circle: ('radius',),
    EntityType.Line: ('startParam', 'endParam'),
    EntityType.Point: ()
}


from src.sketch_handlers.registry import PRIMITIVE_HANDLERS

def _get_entity_bbox(ent):
    """Compute the bounding box for the given entity.

    Parameters
    ----------
    ent : Entity
        The entity of interest for bbox computation

    Returns
    -------
    np.array
        Bounding box of the form `[[x0, y0], [x1, y1]]`
    """
    handler_cls = PRIMITIVE_HANDLERS.get(type(ent).__name__)
    if not handler_cls:
        return None

    # Instantiate handler (lightweight) and compute bbox
    return handler_cls().get_bbox(ent)


def get_sketch_bbox(sketch):
    """Compute the bounding box for the sketch.

    The bounding box is only computed based on entities currently supported
    in `_get_entity_bbox` (`Arc`, `Circle`, `Line`, `Point`).
    TODO: raise error if sketch has unsupported entities

    Parameters
    ----------
    sketch : Sketch
        The sketch of interest for bbox computation

    Returns
    -------
    np.array
        Bounding box of the form `[[x0, y0], [x1, y1]]`
    """
    # Get entity bboxes
    bboxes = [_get_entity_bbox(ent) for ent in sketch.entities.values()]
    # Remove any None bboxes
    bboxes = np.array([bbox for bbox in bboxes if bbox is not None])
    if bboxes.size == 0:
        # Return origin as bounding box for empty sketch
        return np.array([[0., 0.], [0., 0.]])
    # Compute overall bbox
    x0, y0 = np.min(bboxes[:,0,:], axis=0)
    x1, y1 = np.max(bboxes[:,1,:], axis=0)
    return np.array([[x0, y0], [x1, y1]])


def center_sketch(sketch):
    """Center the sketch's bounding box over the origin.

    The entity parameters of the given sketch are modified in place.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be centered

    Returns
    -------
    None
    """
    # Get bounding box
    (x0, y0), (x1, y1) = get_sketch_bbox(sketch)
    # Compute current offset from origin
    x_offset = np.mean([x0, x1])
    y_offset = np.mean([y0, y1])
    # Modify the sketch's entities
    for ent in sketch.entities.values():
        pos_params = POS_PARAMS.get(ent.type)
        if pos_params is None:
            # Skip unsupported entities
            continue
        for param_id in pos_params:
            this_offset = x_offset if 'x' in param_id.lower() else y_offset
            curr_val = getattr(ent, param_id)
            setattr(ent, param_id, curr_val - this_offset)


def rescale_sketch(sketch: datalib.Sketch, dst_length: float = 1.0) -> float:
    """Rescale the sketch such that the long axis of the bounding box is dst_length.

    The entity parameters of the given sketch are modified in place. This
    function should only be called on sketches that are centered; an error is
    raised it the given sketch is not.

    Parameters
    ----------
    sketch : Sketch
        The sketch to be rescaled
    dst_length: float
        The target length of the long axis of the bounding box.

    Returns
    -------
    float
        The normalizing scale factor. If the sketch is zero-dim,
        -1 is returned instead.
    """

    # Get bounding box
    (x0, y0), (x1, y1) = get_sketch_bbox(sketch)
    # Ensure sketch is already centered
    if not np.isclose(x0, -x1) or not np.isclose(y0, -y1):
        raise ValueError("sketch must be centered before rescaling")
    # Calculate scale factor
    w = x1 - x0
    h = y1 - y0
    factor = max(w, h) / dst_length
    if factor == 0:
        return -1
    # Modify the sketch's entities
    for ent in sketch.entities.values():
        pos_params = POS_PARAMS.get(ent.type)
        scale_params = SCALE_PARAMS.get(ent.type)
        if pos_params is None or scale_params is None:
            # Skip unsupported entities
            continue
        params = pos_params + scale_params  # rescale both types of params
        for param_id in params:
            curr_val = getattr(ent, param_id)
            setattr(ent, param_id, curr_val/factor)
    return factor


def normalize_sketch(sketch: datalib.Sketch, dst_length: float = 1.0) -> float:
    """Helper function that both centers and rescales the given sketch in place.
    """
    center_sketch(sketch)
    scale_factor = rescale_sketch(sketch, dst_length)
    return scale_factor


################################################################


def get_refs(constr: datalib.Constraint) -> list[tuple[str, str | None]]:
    refs = [p.value.split(".") for p in constr.parameters if isinstance(p, LocalReferenceParameter)]  # e.g., 1.start
    refs = [(x[0], x[1] if (len(x) == 2 and len(x[1]) > 0) else None) for x in refs]
    if not all(y in (None, 'start', 'end', 'center', 'mid') for _, y in refs):
        raise ValueError(f"Invalid ref subnodes: {refs}")
    return sorted(refs, key=lambda x: (x[0], {'start': 1, 'end': 2, 'center': 3, 'mid': 4}.get(x[1], 0)))


def _resolve_tangent_and_perpendicular(sketch: datalib.Sketch) -> datalib.Sketch:
    """Resolve
    - Tangent(edge, edge) + Coincident(point, point) pairs to Tangent(edge, point, edge, point).
    - Perpendicular(edge, edge) + Coincident(point, point) pairs to Perpendicular(edge, point, edge, point).

    FreeCAD judges the former as redundant.
    """
    # Store all Point-Point Coincidents and Edge-Edge Tangents
    point_point_coincident_map = defaultdict(list)
    edge_edge_tangent_map = {}
    edge_edge_perpendicular_map = {}
    for cid, c in sketch.constraints.items():
        if c.constraint_type.name == 'Coincident':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                if pt1 and pt2:  # Point-Point
                    point_point_coincident_map[(geo1, geo2)].append((cid, refs))
        if c.constraint_type.name == 'Tangent':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                if pt1 is None and pt2 is None:  # Edge-Edge
                    edge_edge_tangent_map[(geo1, geo2)] = (cid, refs)
        if c.constraint_type.name == 'Perpendicular':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                if pt1 is None and pt2 is None:  # Edge-Edge
                    edge_edge_perpendicular_map[(geo1, geo2)] = (cid, refs)

    # Overwrite Tangent(edge, edge) with Tangent(edge, point, edge, point) and drop Coincident(point, point)
    common_tangent_keys = point_point_coincident_map.keys() & edge_edge_tangent_map.keys()
    for key in common_tangent_keys:
        tangent_cid, ((geo1, _), (geo2, _)) = edge_edge_tangent_map[key]
        for coincident_cid, ((_geo1, pt1), (_geo2, pt2)) in point_point_coincident_map[key]:
            assert geo1 == _geo1 and geo2 == _geo2
            # new params
            new_param = []
            for p in sketch.constraints[tangent_cid].parameters:
                if isinstance(p, LocalReferenceParameter) and (p.value == str(geo1) or p.value.startswith(f"{geo1}.")):
                    p.value = f"{geo1}.{pt1}"
                elif isinstance(p, LocalReferenceParameter) and (p.value == str(geo2) or p.value.startswith(f"{geo2}.")):
                    p.value = f"{geo2}.{pt2}"
                new_param.append(p)
            sketch.constraints[tangent_cid] = sketch.constraints[tangent_cid]._replace(parameters=new_param)
            sketch.constraints.pop(coincident_cid)

    # Overwrite Perpendicular(edge, edge) with Perpendicular(edge, point, edge, point) and drop Coincident(point, point)
    common_perpendicular_keys = point_point_coincident_map.keys() & edge_edge_perpendicular_map.keys()
    for key in common_perpendicular_keys:
        perpendicular_cid, ((geo1, _), (geo2, _)) = edge_edge_perpendicular_map[key]
        for coincident_cid, ((_geo1, pt1), (_geo2, pt2)) in point_point_coincident_map[key]:
            assert geo1 == _geo1 and geo2 == _geo2
            # new params
            new_param = []
            for p in sketch.constraints[perpendicular_cid].parameters:
                if isinstance(p, LocalReferenceParameter) and (p.value == str(geo1) or p.value.startswith(f"{geo1}.")):
                    p.value = f"{geo1}.{pt1}"
                elif isinstance(p, LocalReferenceParameter) and (p.value == str(geo2) or p.value.startswith(f"{geo2}.")):
                    p.value = f"{geo2}.{pt2}"
                new_param.append(p)
            sketch.constraints[perpendicular_cid] = sketch.constraints[perpendicular_cid]._replace(parameters=new_param)
            sketch.constraints.pop(coincident_cid)

    return sketch


def _resolve_redundant_coincident(sketch: datalib.Sketch) -> datalib.Sketch:
    """When same (geo1, geo2) pair has both types of Coincident:
        - Point-on-Object (one ref has pt=None, the other has pt='start'/'end'/'center')
        - Point-Point (both refs have pt='start'/'end'/'center')

    The Point-on-Object is redundant and should be skipped.
    """
    # Parse all Coincident constraints
    coincident_map = defaultdict(list)
    for cid, c in sketch.constraints.items():
        if c.constraint_type.name == 'Coincident':
            refs = get_refs(c)
            if len(refs) == 2:
                (geo1, pt1), (geo2, pt2) = refs
                coincident_map[(geo1, geo2)].append((cid, (pt1, pt2)))

    # Remove redundant Point-on-Object (also represented as Coincident in SketchGraphs)
    redundant_point_on_object = []
    for geo_pair, cid_pts_list in coincident_map.items():
        firsts = {pt1 for (cid, (pt1, pt2)) in cid_pts_list if pt1 is not None and pt2 is not None}
        seconds = {pt2 for (cid, (pt1, pt2)) in cid_pts_list if pt1 is not None and pt2 is not None}

        for cid, (pt1, pt2) in cid_pts_list:
            if pt1 is not None and pt2 is not None:
                continue
            if pt1 in firsts or pt2 in seconds:
                redundant_point_on_object.append(cid)
            # NOTE: Since get_refs returns sorted references,
            # (pt2 in firsts or pt1 in seconds) check is unnecessary.

    for cid in redundant_point_on_object:
        del sketch.constraints[cid]

    return sketch


def _resolve_concentric_of_arcs_forming_circle(sketch: datalib.Sketch) -> datalib.Sketch:
    """Fix the over-constrained pattern that arises when two arcs together form
    a closed circle (or any configuration where they share *both* endpoints and
    are concentric).

    **Why this is over-constrained**

    Two arcs whose start/end endpoints are pairwise Coincident already share the
    same chord.  The perpendicular bisector of that chord passes through both
    circle centres.  Adding a Coincident (or Concentric) on the *centres* further
    fixes the centre to a single point – that is 2 DoF removed for a constraint
    that is geometrically only 1 DoF free (the centre can still slide along the
    bisector).  FreeCAD therefore flags the centre constraint as redundant.

    **Fix strategy**

    1. Remove the centre–centre Coincident constraint.
    2. Replace one of the endpoint Coincident(pt_a, pt_b) constraints with
       Tangent(arc_a, pt_a, arc_b, pt_b), which encodes the same contact point
       information while also enforcing tangency (arcs of equal radius forming a
       circle are always tangent at the junction).  This is the representation
       that FreeCAD itself generates for such geometry.
    """
    # ------------------------------------------------------------------ #
    # 1. Collect point-point Coincident constraints for Arc pairs          #
    # ------------------------------------------------------------------ #
    # Step 1: Identify all arcs and their endpoint Coincidents
    arc_endpoint_coin: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    # Step 2: Identify all center-to-center Coincidents
    center_coin: dict[tuple[str, str], str] = {}

    for cid, c in sketch.constraints.items():
        if c.constraint_type.name not in ('Coincident', 'Concentric'):
            continue
        refs = get_refs(c)
        if len(refs) != 2:
            continue
        (g1, pt1), (g2, pt2) = refs

        ent1 = sketch.entities.get(str(g1))
        ent2 = sketch.entities.get(str(g2))
        if not (isinstance(ent1, Arc) and isinstance(ent2, Arc)):
            continue

        # Centre–centre: both refs point to 'center' (or bare ID for Concentric)
        if (pt1 in ('center', None)) and (pt2 in ('center', None)):
            key = (min(g1, g2), max(g1, g2))
            if key not in center_coin:  # keep first occurrence
                center_coin[key] = cid
            continue

        # Endpoint–endpoint
        if pt1 in ('start', 'end') and pt2 in ('start', 'end'):
            key = (min(g1, g2), max(g1, g2))
            arc_endpoint_coin[key].append((cid, f"{g1}.{pt1}", f"{g2}.{pt2}"))

    # ------------------------------------------------------------------ #
    # 2. For each arc pair that has both ≥2 endpoint Coincidents AND a    #
    #    centre Coincident, apply the fix.                                #
    # ------------------------------------------------------------------ #
    for key, ep_list in arc_endpoint_coin.items():
        if key not in center_coin:
            continue
        if len(ep_list) < 2:
            # Only one shared endpoint – not the closed-circle pattern
            continue

        centre_cid = center_coin[key]
        # Pick the first endpoint Coincident to convert to Tangent
        coin_cid_to_convert, ref_a, ref_b = ep_list[0]
        ga, pt_a = ref_a.split('.')
        gb, pt_b = ref_b.split('.')

        # Remove the centre–centre Coincident
        sketch.constraints.pop(centre_cid, None)

        # Replace the chosen endpoint Coincident with Tangent(arc_a.pt_a, arc_b.pt_b)
        existing_constr = sketch.constraints.get(coin_cid_to_convert)
        if existing_constr is not None:
            new_params = [
                LocalReferenceParameter(parameterId='localFirst',  value=f"{ga}.{pt_a}"),
                LocalReferenceParameter(parameterId='localSecond', value=f"{gb}.{pt_b}"),
            ]
            sketch.constraints[coin_cid_to_convert] = existing_constr._replace(
                constraint_type=datalib.ConstraintType.Tangent,
                parameters=new_params,
            )

    return sketch


def _resolve_midpoint_mirror_ref_order(sketch: datalib.Sketch, midpoint_tol: float = 1e-6) -> tuple[datalib.Sketch, dict[str, str]]:
    """Midpoint and Mirror have the following reference ordering: [EntityA, EntityB, SymmetyPoint/Axis]
    However, SketchGraphs format doesn't guarantee this ordering, so we recover it by checking the actual geometry.

    Parameters
    ----------
    midpoint_tol : float
        Absolute tolerance for the midpoint collinearity check (np.allclose atol).
        The default (1e-6) is appropriate for raw sketch data.  After coordinate
        transformations such as align_sketch the alignment error can be O(1e-2),
        so a larger value (e.g. 0.05) may be required.
    """

    def sort_triplet_refs(
        refs: tuple[tuple[str, str | None], tuple[str, str | None], tuple[str, str | None]],
        entities: dict[str, datalib.Entity],
    ) -> tuple[tuple[str, str | None], tuple[str, str | None], tuple[str, str | None]]:

        (geo1, pt1), (geo2, pt2), (geo3, pt3) = refs
        coord1 = np.array(get_entity_point(entities[str(geo1)], pt1) or [])
        coord2 = np.array(get_entity_point(entities[str(geo2)], pt2) or [])
        coord3 = np.array(get_entity_point(entities[str(geo3)], pt3) or [])

        if (coord1.size == 2) and (coord2.size == 2) and (coord3.size == 2):
            coords = [coord1, coord2, coord3]
            for i in range(3):
                a, b = coords[(i+1) % 3], coords[(i+2) % 3]
                if np.allclose(2*coords[i], a+b, atol=midpoint_tol):
                    return (refs[(i+1) % 3], refs[(i+2) % 3], refs[i])
                # else:
                #     print(f"{i}: {coords[i]=}, {a=}, {b=}")
            raise ValueError(f"Three points are not equidistant collinear: {coord1=}, {coord2=}, {coord3=}")
        elif coord1.size == 0:
            if coord2.size != 2 or coord3.size != 2:
                raise ValueError(f"Remaining references must be points: {coord2.size=}, {coord3.size=}")
            return (refs[1], refs[2], refs[0])
        elif coord2.size == 0:
            if coord1.size != 2 or coord3.size != 2:
                raise ValueError(f"Remaining references must be points: {coord1.size=}, {coord3.size=}")
            return (refs[0], refs[2], refs[1])
        elif coord3.size == 0:
            if coord1.size != 2 or coord2.size != 2:
                raise ValueError(f"Remaining references must be points: {coord1.size=}, {coord2.size=}")
            return (refs[0], refs[1], refs[2])
        else:
            raise ValueError(f"At least two references must be points: got {refs=}")

    def sort_line_point_refs(
        refs: tuple[tuple[str, str | None], tuple[str, str | None]],
        entities: dict[str, datalib.Entity],
    ) -> tuple[tuple[str, str | None], tuple[str, str | None]]:

        (geo1, pt1), (geo2, pt2) = refs
        if isinstance(entities[str(geo1)], datalib.Line) and pt1 is None:
            if not (pt2 or isinstance(entities[str(geo2)], datalib.Point)):
                raise ValueError(f"Second reference must be a Point for Line midpoint: got {refs=}")
            return (refs[0], refs[1])
        elif isinstance(entities[str(geo2)], datalib.Line) and pt2 is None:
            if not (pt1 or isinstance(entities[str(geo1)], datalib.Point)):
                raise ValueError(f"First reference must be a Point for Line midpoint: got {refs=}")
            return (refs[1], refs[0])
        elif isinstance(entities[str(geo1)], datalib.Arc) and pt1 is None:
            if not (pt2 or isinstance(entities[str(geo2)], datalib.Point)):
                raise ValueError(f"Second reference must be a Point for Arc midpoint: got {refs=}")
            return (refs[0], refs[1])
        elif isinstance(entities[str(geo2)], datalib.Arc) and pt2 is None:
            if not (pt1 or isinstance(entities[str(geo1)], datalib.Point)):
                raise ValueError(f"First reference must be a Point for Arc midpoint: got {refs=}")
            return (refs[1], refs[0])
        else:
            raise ValueError(f"Refs must consist of a Line and a Point: got {refs=}")


    invalid_cids = {}
    for cid, constr in sketch.constraints.items():
        if constr.constraint_type.name not in ('Midpoint', 'Mirror'):
            continue

        # gather ref params
        local_refs = [p.value.split(".") for p in constr.parameters if isinstance(p, LocalReferenceParameter)]
        local_refs = tuple((x[0], str(x[1]) if (len(x) == 2 and x[1] != "") else None) for x in local_refs)

        # sort the refs
        try:
            if len(local_refs) == 3:
                # [PointA, PointB, PointSym]
                local_refs_sorted = sort_triplet_refs(local_refs, sketch.entities)
            elif len(local_refs) == 2:
                # [Line, Point]
                local_refs_sorted = sort_line_point_refs(local_refs, sketch.entities)
            else:
                continue
        except (ValueError, KeyError) as e:
            invalid_cids[cid] = traceback.format_exc()
            continue

        local_refs_sorted = [f"{ref[0]}.{ref[1]}" if ref[1] is not None else str(ref[0]) for ref in local_refs_sorted]

        # set the new order
        ref_params = []
        other_params = []
        for p in constr.parameters:
            if isinstance(p, LocalReferenceParameter):
                ref_params.append(p)
            else:
                other_params.append(p)
        new_params = sorted(ref_params, key=lambda p: local_refs_sorted.index(p.value)) + other_params
        sketch.constraints[cid] = constr._replace(parameters=new_params)

    return sketch, invalid_cids


def _convert_concentric_to_coincident(sketch: datalib.Sketch) -> datalib.Sketch:
    for cid, constr in list(sketch.constraints.items()):
        if constr.type == datalib.ConstraintType.Concentric:
            new_params = []
            for p in constr.parameters:
                if isinstance(p, LocalReferenceParameter):
                    # Concentric(circle/arc/point1, circle/arc/point2) -> Coincident(point1, point2)
                    val = p.value
                    if "." not in val:
                        ent = sketch.entities.get(val)
                        if isinstance(ent, (datalib.Circle, datalib.Arc)):
                            val = f"{val}.center"
                    new_params.append(LocalReferenceParameter(parameterId=p.parameterId, value=val))
                else:
                    new_params.append(p)
            sketch.constraints[cid] = constr._replace(
                constraint_type=datalib.ConstraintType.Coincident,
                parameters=new_params
            )
    return sketch


def _convert_line_length_to_distance(sketch: datalib.Sketch) -> tuple[datalib.Sketch, dict[str, str]]:
    invalid_cids = {}
    for cid, constr in list(sketch.constraints.items()):
        if constr.type == datalib.ConstraintType.Length:
            # check the entity
            refs = get_refs(constr)
            refs = list(set(refs))  # deduplication of a corner case: refs = [(x, None), (x, None)]
            if len(refs) != 1:
                invalid_cids[cid] = f"Length constraint must have one reference: got {refs=}"
                continue

            ent_id, _ = refs[0]
            ent = sketch.entities.get(str(ent_id))
            if isinstance(ent, datalib.Line):
                # Length(line) -> Distance(line.start, line.end)
                direction_val = 'MINIMUM'
                for p in constr.parameters:
                    if isinstance(p, EnumParameter) and p.parameterId == 'direction':
                        direction_val = p.value
                        break
                if direction_val != 'MINIMUM':
                    invalid_cids[cid] = f"Length constraint direction must be 'MINIMUM': got {direction_val=}"
                    continue

                new_params: list[datalib.ConstraintParameter] = [
                    LocalReferenceParameter(parameterId='localFirst', value=f"{ent_id}.start"),
                    LocalReferenceParameter(parameterId='localSecond', value=f"{ent_id}.end"),
                    EnumParameter(parameterId='direction', value=direction_val),
                ]
                # preserve other parameters
                for p in constr.parameters:
                    if isinstance(p, EnumParameter) and p.parameterId == 'direction':
                        continue
                    if not isinstance(p, LocalReferenceParameter):
                        new_params.append(p)

                sketch.constraints[cid] = constr._replace(
                    constraint_type=datalib.ConstraintType.Distance,
                    parameters=new_params
                )

    return sketch, invalid_cids


def _convert_diameter_to_radius(sketch: datalib.Sketch) -> datalib.Sketch:
    for cid, constr in list(sketch.constraints.items()):
        if constr.type == datalib.ConstraintType.Diameter:
            new_params = []
            for p in constr.parameters:
                if isinstance(p, QuantityParameter):
                    # Diameter -> Radius (halve the value)
                    try:
                        new_val = float(p.value) / 2.0
                    except ValueError:
                        new_val = p.value + " / 2"
                    new_params.append(QuantityParameter(parameterId='length', value=str(new_val), expression=p.expression + " / 2"))
                else:
                    new_params.append(p)

            sketch.constraints[cid] = constr._replace(
                constraint_type=datalib.ConstraintType.Radius,
                parameters=new_params
            )
    return sketch


def _convert_horizontal_vertical_on_points_to_line(sketch: datalib.Sketch) -> datalib.Sketch:
    """Canonicalize Horizontal/Vertical constraints:
    Horizontal(Line.start, Line.end) -> Horizontal(Line)
    Vertical(Line.start, Line.end) -> Vertical(Line)
    """
    for cid, constr in list(sketch.constraints.items()):
        if constr.type in (datalib.ConstraintType.Horizontal, datalib.ConstraintType.Vertical):
            refs = get_refs(constr)
            if len(refs) == 2:
                (g1, pt1), (g2, pt2) = refs
                # Check if it's the same line and both endpoints
                if g1 == g2 and (pt1, pt2) in [('start', 'end'), ('end', 'start')]:
                    ent = sketch.entities.get(str(g1))
                    if isinstance(ent, datalib.Line):
                        # Convert to line-based constraint
                        new_params = [LocalReferenceParameter(parameterId='localFirst', value=str(g1))] + \
                                     [p for p in constr.parameters if not isinstance(p, LocalReferenceParameter)]
                        sketch.constraints[cid] = constr._replace(parameters=new_params)
    return sketch


def cleanse_sketch_constraints(
    sketch: datalib.Sketch,
    resolve_redundancy: bool = True,
    resolve_synonym: bool = True,
    delete_invalid: bool = False,
    midpoint_tol: float = 1e-6,
) -> datalib.Sketch:
    """Remove or fix degenerate constraints from the sketch in-place.

    Currently handles:
    - Duplicate references: deduplicate LocalReferenceParameters that share
      the same value (e.g., Coincident(A, A, B, B) → Coincident(A, B)).
    - Self-referential: remove constraints where all LocalReferenceParameters
      point to the same entity/subnode after dedup (e.g., Coincident(A, A), Tangent(A, 1, A, 1) → removed).
    - Sort Midpoint and Mirror references so that the last entity is the symmetry point or axis.

    If resolve_redundancy is True, it will also handles:
    - Merge Tangent(edge, ednge) and Coincident(point, point)
    - Merge PointOnObject(point, edge) and Coincident(point, point)

    If resolve_synonym is True, it will also handle:
    - Concentric(circle/arc1, circle/arc2) is translated into Coincident(circle/arc1, 3, circle/arc2, 3)
    - Length(line) is translated into Distance(line, 1, line, 2)
    - Diameter is translated into Radius

    Parameters
    ----------
    sketch : datalib.Sketch
        The sketch to cleanse. Modified in-place.

    Returns
    -------
    sketch : datalib.Sketch
        The sketch with degenerate constraints removed.
    """
    keys_to_remove = []

    for cid, constr in sketch.constraints.items():
        # Deduplicate LocalReferenceParameters by value
        local_refs = []
        seen_refs = set()
        deduped_params = []
        for p in constr.parameters:
            if isinstance(p, LocalReferenceParameter):
                local_refs.append(p.value)
                if p.value in seen_refs:
                    continue  # skip duplicate
                seen_refs.add(p.value)
            deduped_params.append(p)

        # Skip constraints with 0 or 1 local ref (nothing to check)
        if len(local_refs) <= 1:
            continue

        # After dedup: if only 1 unique local ref remains, it's self-referential
        if len(seen_refs) <= 1:
            keys_to_remove.append(cid)
            continue

        # If dedup changed anything, update the constraint
        if len(deduped_params) < len(constr.parameters):
            sketch.constraints[cid] = constr._replace(parameters=deduped_params)

    for cid in keys_to_remove:
        del sketch.constraints[cid]

    ################################################################

    def maybe_delete_invalid_constraints(sketch: datalib.Sketch, invalid_cids: dict[str, str]) -> datalib.Sketch:
        if invalid_cids:
            if delete_invalid:
                for cid in invalid_cids.keys():
                    del sketch.constraints[cid]
            else:
                raise ValueError(f"Invalid constraints found: {invalid_cids}")
        return sketch

    # resolve midpoint/mirror reference ordering
    sketch, invalid_cids = _resolve_midpoint_mirror_ref_order(sketch, midpoint_tol=midpoint_tol)
    sketch = maybe_delete_invalid_constraints(sketch, invalid_cids)

    if resolve_redundancy:
        sketch = _resolve_concentric_of_arcs_forming_circle(sketch)
        sketch = _resolve_tangent_and_perpendicular(sketch)
        sketch = _resolve_redundant_coincident(sketch)

    if resolve_synonym:
        sketch = _convert_concentric_to_coincident(sketch)
        sketch, invalid_cids = _convert_line_length_to_distance(sketch)
        sketch = maybe_delete_invalid_constraints(sketch, invalid_cids)
        sketch = _convert_diameter_to_radius(sketch)
        sketch = _convert_horizontal_vertical_on_points_to_line(sketch)

    return sketch


################################################################


@dataclass
class SketchProperties:
    direction: str = ""
    aligned: bool = True


def get_properties(constr: datalib.Constraint) -> SketchProperties:
    props = SketchProperties()

    for param in constr.parameters:
        # direction
        if isinstance(param, EnumParameter) and param.parameterId == 'direction':
            props.direction = param.value
        # aligned
        elif isinstance(param, BooleanParameter) and param.parameterId == 'aligned':
            props.aligned = param.value

    if props.direction not in ("", "MINIMUM", "HORIZONTAL", "VERTICAL"):
        raise ValueError(f"ERROR: Invalid direction for Distance constraint {constr}")

    return props


def get_entity_point(entity: datalib.Entity, subnode: str | None = None) -> tuple[float, float] | None:
    """Get the point coordinates for an entity and optional subnode."""
    if isinstance(entity, datalib.Point):
        return (entity.x, entity.y)
    elif isinstance(entity, datalib.Line):
        if subnode == 'start':
            return entity.start_point.tolist()
        elif subnode == 'end':
            return entity.end_point.tolist()
    elif isinstance(entity, datalib.Circle):
        if subnode == 'center':
            return entity.center_point.tolist()
    elif isinstance(entity, datalib.Arc):
        if subnode == 'start':
            return entity.start_point.tolist()
        elif subnode == 'end':
            return entity.end_point.tolist()
        elif subnode == 'center':
            return entity.center_point.tolist()
        elif subnode == 'mid':
            return entity.mid_point.tolist()
    return None


def complete_constraint_values(sketch: datalib.Sketch, expr_watermark: str = "VALUE_ASSIGNED:", skip_invalid_constraints: bool = False) -> tuple[datalib.Sketch, list[str]]:
    """
    Fill in the void dimensional constraint values by the primitive position.

    Dimensional constraints in the original SketchGraphs dataset contain contraint values in param.expression.
    Unfortunately, this is a plain text and param.value is left empty.
    This function assigns a value to param.value based on the primitive positions.

    If skip_invalid_constraints = True, the second returned list contains the ids of the constraints whose values could not be assigned.
    However, the invalid constraints remain in the sketch.

    NOTE: After the value is assigned, param.expression is updated in the format of "<expr_watermark>: <value>".
    NOTE: The value match between the calculated param.value and existing param.expression is not verified in this function,
          which is instead checked by `tests/test_compute_constraint_values.py`.
    """

    def compute_dist_points(p1: tuple[float, float], p2: tuple[float, float], direction: str) -> float:
        """Calculate Euclidean distance between two points."""
        if direction == "HORIZONTAL":
            return p2[0] - p1[0]
        elif direction == "VERTICAL":
            return p2[1] - p1[1]
        elif direction == "MINIMUM":
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        else:
            raise ValueError(f"ERROR: Invalid direction for Distance constraint {direction}")

    def compute_dist_point_line(p: tuple[float, float], line: datalib.Line) -> float:
        dx = p[0] - line.pntX
        dy = p[1] - line.pntY
        area = abs(dx * line.dirY - dy * line.dirX)
        norm = math.sqrt(line.dirX**2 + line.dirY**2)
        return area / norm if norm > 0 else 0.0

    def compute_dist_point_circle(p: tuple[float, float], circle: datalib.Circle | datalib.Arc) -> float:
        d_centers = math.sqrt((p[0] - circle.xCenter)**2 + (p[1] - circle.yCenter)**2)
        return abs(d_centers - circle.radius)  # For Point-Circle, FreeCAD expects non-negative values.

    def compute_dist_line_circle(line: datalib.Line, circle: datalib.Circle | datalib.Arc) -> float:
        d_center_to_line = compute_dist_point_line((circle.xCenter, circle.yCenter), line)
        return d_center_to_line - circle.radius

    def compute_dist_circle_circle(c1: datalib.Circle | datalib.Arc, c2: datalib.Circle | datalib.Arc) -> float:
        d_centers = math.sqrt((c1.xCenter - c2.xCenter)**2 + (c1.yCenter - c2.yCenter)**2)
        if d_centers < abs(c1.radius - c2.radius):
            # whole circle is included by the other (positive)
            return abs(c1.radius - c2.radius) - d_centers
        else:
            # circles are disjoint (positive) or partially overlapping (negative)
            return d_centers - (c1.radius + c2.radius)

    def compute_dist_line_line(l1: datalib.Line, l2: datalib.Line) -> float:
        # Check parallelism: |v1 x v2| should be close to 0
        cross_prod = abs(l1.dirX * l2.dirY - l1.dirY * l2.dirX)
        norm1 = math.sqrt(l1.dirX**2 + l1.dirY**2)
        norm2 = math.sqrt(l2.dirX**2 + l2.dirY**2)
        if cross_prod / (norm1 * norm2) > 1e-3:
            raise ValueError(f"Lines are not parallel: {l1}, {l2}")

        # Calculate distance from l1.start to l2
        p1_start = get_entity_point(l1, 'start')
        assert p1_start is not None
        return compute_dist_point_line(p1_start, l2)

    def compute_line_length(line: datalib.Line, direction: str) -> float:
        """Calculate the length of a line entity."""
        start_point = get_entity_point(line, 'start')
        end_point = get_entity_point(line, 'end')
        assert start_point is not None and end_point is not None
        return compute_dist_points(start_point, end_point, direction)

    def compute_angle_between_lines(line1: datalib.Line, line2: datalib.Line) -> float:
        """Calculate the directed angle from line1 to line2 based on parameters."""
        a1 = math.atan2(line1.dirY, line1.dirX)
        a2 = math.atan2(line2.dirY, line2.dirX)
        diff = a2 - a1
        return diff

    def compute_angle_arc(arc: datalib.Arc) -> float:
        """Compute the angle of an arc in radians."""
        sx, sy = arc.start_point
        tx, ty = arc.end_point
        a1 = math.atan2(sy - arc.yCenter, sx - arc.xCenter)
        a2 = math.atan2(ty - arc.yCenter, tx - arc.xCenter)
        diff = a2 - a1
        return diff

    def assign_value(
        constr: datalib.Constraint,
        value: float,
        param_id: Literal['length', 'angle'],
        expr_watermark: str,
    ) -> datalib.Constraint:
        new_params = []
        value_assigned_times = 0
        for param in constr.parameters:
            if isinstance(param, QuantityParameter):
                assert param_id == param.parameterId, f"ERROR: parameterId mismatch {param_id=}, {param.parameterId=}"
                new_params.append(QuantityParameter(parameterId=param.parameterId, value=value, expression=expr_watermark+str(value)))
                value_assigned_times += 1
            else:
                new_params.append(param)

        if value_assigned_times == 0:
            # No value assigned in the original constraint, which sometimes happens
            new_params.append(QuantityParameter(parameterId=param_id, value=value, expression=expr_watermark+str(value)))
            value_assigned_times += 1

        if value_assigned_times != 1:
            raise ValueError(f"ERROR: value assigned {value_assigned_times} times for constraint {constr}")

        return constr._replace(parameters=new_params)

    def assign_boolean(
        constr: datalib.Constraint,
        flag: bool,
        name: Literal['aligned', 'clockwise'],
    ) -> datalib.Constraint:
        new_params = []
        boolean_assigned_times = 0
        for param in constr.parameters:
            if isinstance(param, BooleanParameter) and param.parameterId == name:
                new_params.append(BooleanParameter(parameterId=name, value=flag))
                boolean_assigned_times += 1
            else:
                new_params.append(param)

        if boolean_assigned_times == 0:
            # No value assigned in the original constraint, which sometimes happens
            new_params.append(BooleanParameter(parameterId=name, value=flag))
            boolean_assigned_times += 1

        if boolean_assigned_times != 1:
            raise ValueError(f"ERROR: value assigned {boolean_assigned_times} times for constraint {constr}")

        return constr._replace(parameters=new_params)


    invalid_constraints = []
    for constr_id, constr in sketch.constraints.items():
        # Extract entity references
        entity_refs = []
        for param in constr.parameters:
            if isinstance(param, LocalReferenceParameter):
                entity_refs.append(param.value)  # (e.g.) 5, 3.center, ...

        # properties
        props = get_properties(constr)

        # Process each constraint type
        try:
            if constr.type == datalib.ConstraintType.Angle:
                # NOTE: Since the original SketchGraphs angle data is unclear in how it was computed, we instead use angles calculated on the fly.
                #       Call src.sketch_handlers.sketch_utils.complete_constraint_values() first to make that change.
                #       - Value is already set in radians (used as-is by FreeCAD)
                #       - Aligned is always True
                #       - Clockwise is neglected
                if len(entity_refs) == 1:
                    ent1_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                    ent1 = sketch.entities.get(ent1_id)

                    if isinstance(ent1, datalib.Line):
                        x_axis = datalib.Line(entityId=-1, pntX=0, pntY=0, dirX=1, dirY=0, startParam=-0.5, endParam=0.5)
                        angle = compute_angle_between_lines(ent1, x_axis)
                        new_constr = assign_value(constr, angle, 'angle', expr_watermark)
                        new_constr = assign_boolean(new_constr, True, 'aligned')
                        sketch.constraints[constr_id] = new_constr
                    elif isinstance(ent1, datalib.Arc):
                        angle = compute_angle_arc(ent1)
                        new_constr = assign_value(constr, angle, 'angle', expr_watermark)
                        new_constr = assign_boolean(new_constr, True, 'aligned')
                        sketch.constraints[constr_id] = new_constr
                    else:
                        raise ValueError(f"ERROR: Invalid references for Angle constraint {entity_refs}")

                elif len(entity_refs) == 2:
                    ent1_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                    ent2_id = entity_refs[1].split('.')[0] if '.' in entity_refs[1] else entity_refs[1]

                    ent1 = sketch.entities.get(ent1_id)
                    ent2 = sketch.entities.get(ent2_id)

                    if isinstance(ent1, datalib.Line) and isinstance(ent2, datalib.Line):
                        angle = compute_angle_between_lines(ent1, ent2)
                        new_constr = assign_value(constr, angle, 'angle', expr_watermark)
                        new_constr = assign_boolean(new_constr, True, 'aligned')
                        sketch.constraints[constr_id] = new_constr
                    else:
                        raise ValueError(f"ERROR: Invalid references for Angle constraint {entity_refs}")

                else:
                    raise ValueError(f"ERROR: Invalid references for Angle constraint {entity_refs}")

            elif constr.type == datalib.ConstraintType.Diameter:
                assert len(entity_refs) == 1, f"ERROR: Invalid references for Diameter constraint {entity_refs}"
                ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                entity = sketch.entities.get(ent_id)

                if isinstance(entity, (datalib.Circle, datalib.Arc)):
                    diameter = entity.radius * 2
                    sketch.constraints[constr_id] = assign_value(constr, diameter, 'length', expr_watermark)
                else:
                    raise ValueError(f"ERROR: Invalid references for Diameter constraint {entity_refs}")

            elif constr.type == datalib.ConstraintType.Distance:

                if len(entity_refs) == 1:
                    # Distance constraint on a single entity (line length)
                    ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                    entity = sketch.entities.get(ent_id)

                    if isinstance(entity, datalib.Line):
                        length = compute_line_length(entity, props.direction)
                        sketch.constraints[constr_id] = assign_value(constr, length, 'length', expr_watermark)
                    else:
                        raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")

                elif len(entity_refs) == 2:
                    # Distance between two entities
                    ent1_ref = entity_refs[0]
                    ent2_ref = entity_refs[1]

                    # Parse entity IDs and subnodes
                    if '.' in ent1_ref:
                        ent1_id, sub1 = ent1_ref.split('.', 1)
                    else:
                        ent1_id, sub1 = ent1_ref, None

                    if '.' in ent2_ref:
                        ent2_id, sub2 = ent2_ref.split('.', 1)
                    else:
                        ent2_id, sub2 = ent2_ref, None

                    ent1 = sketch.entities.get(ent1_id)
                    ent2 = sketch.entities.get(ent2_id)
                    assert ent1 and ent2, f"ERROR: Invalid references for Distance constraint {entity_refs}"

                    p1 = get_entity_point(ent1, sub1)
                    p2 = get_entity_point(ent2, sub2)
                    if p1 and p2:
                        distance = compute_dist_points(p1, p2, props.direction)
                    elif p1:
                        assert props.direction == 'MINIMUM'
                        if isinstance(ent2, datalib.Line):
                            distance = compute_dist_point_line(p1, ent2)
                        elif isinstance(ent2, (datalib.Circle, datalib.Arc)):
                            distance = compute_dist_point_circle(p1, ent2)
                        else:
                            raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")
                    elif p2:
                        assert props.direction == 'MINIMUM'
                        if isinstance(ent1, datalib.Line):
                            distance = compute_dist_point_line(p2, ent1)
                        elif isinstance(ent1, (datalib.Circle, datalib.Arc)):
                            distance = compute_dist_point_circle(p2, ent1)
                        else:
                            raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")
                    else:
                        assert props.direction == 'MINIMUM'
                        if isinstance(ent1, datalib.Line) and isinstance(ent2, (datalib.Circle, datalib.Arc)):
                            distance = compute_dist_line_circle(ent1, ent2)
                        elif isinstance(ent1, (datalib.Circle, datalib.Arc)) and isinstance(ent2, datalib.Line):
                            distance = compute_dist_line_circle(ent2, ent1)
                        elif isinstance(ent1, (datalib.Circle, datalib.Arc)) and isinstance(ent2, (datalib.Circle, datalib.Arc)):
                            distance = compute_dist_circle_circle(ent1, ent2)
                        elif isinstance(ent1, datalib.Line) and isinstance(ent2, datalib.Line):
                            distance = compute_dist_line_line(ent1, ent2)
                        else:
                            raise ValueError(f"ERROR: Invalid references for Distance constraint {ent1=}, {ent2=}")

                    sketch.constraints[constr_id] = assign_value(constr, distance, 'length', expr_watermark)

                else:
                    raise ValueError(f"ERROR: Invalid references for Distance constraint {entity_refs}")

            elif constr.type == datalib.ConstraintType.Length:
                assert len(entity_refs) == 1, f"ERROR: Invalid references for Length constraint {entity_refs}"
                ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                entity = sketch.entities.get(ent_id)

                if isinstance(entity, datalib.Line):
                    length = compute_line_length(entity, props.direction)
                    sketch.constraints[constr_id] = assign_value(constr, length, 'length', expr_watermark)
                elif isinstance(entity, datalib.Arc):
                    # Arc length = radius * angle (in radians)
                    angle_diff = abs(entity.endParam - entity.startParam)
                    # Handle full circle case
                    if angle_diff > math.pi * 2:
                        angle_diff = angle_diff % (math.pi * 2)
                    length = entity.radius * angle_diff
                    sketch.constraints[constr_id] = assign_value(constr, length, 'length', expr_watermark)
                else:
                    raise ValueError(f"ERROR: Invalid references for Length constraint {entity_refs}")

            elif constr.type == datalib.ConstraintType.Radius:
                assert len(entity_refs) >= 1, f"ERROR: Invalid references for Radius constraint {entity_refs}"
                ent_id = entity_refs[0].split('.')[0] if '.' in entity_refs[0] else entity_refs[0]
                entity = sketch.entities.get(ent_id)

                if isinstance(entity, (datalib.Circle, datalib.Arc)):
                    radius = entity.radius
                    sketch.constraints[constr_id] = assign_value(constr, radius, 'length', expr_watermark)
                else:
                    raise ValueError(f"ERROR: Invalid references for Radius constraint: {entity=}")

        except Exception as e:
            if skip_invalid_constraints:
                # Add watermark for get_constraint_props() to work in src/sketch_handlers/constraints/base.py
                new_params = []
                for param in constr.parameters:
                    if isinstance(param, QuantityParameter):
                        new_params.append(QuantityParameter(parameterId=param.parameterId, value=0.0, expression=expr_watermark + "INVALID"))
                    else:
                        new_params.append(param)
                sketch.constraints[constr_id] = constr._replace(parameters=new_params)
                invalid_constraints.append(constr_id)
                print(f"WARNING: Invalid references for constraint {constr_id}: {e}")
            else:
                raise e

    return sketch, invalid_constraints

################

def cleanse_sketch_primitives(sketch: datalib.Sketch) -> datalib.Sketch:
    """Force startParam < endParam for datalib.Line, which makes the direction vector definition aligned with FreeCAD.
    This is beneficial for Angle constraints.
    """
    for pid, p in sketch.entities.items():
        if isinstance(p, datalib.Line):
            if p.startParam > p.endParam:
                p.dirX, p.dirY = -p.dirX, -p.dirY
                p.startParam, p.endParam = -p.startParam, -p.endParam
            sketch.entities[pid] = p
    return sketch