"""Solver feedback utilities for constraint evaluation.

This module provides functionality to:
1. Convert model output tokens to FreeCAD sketch
2. Run the geometry solver
3. Return solver status (DoF, conflicts, redundancies)
"""
from __future__ import annotations
import os
import sys
from tqdm import tqdm
from dataclasses import dataclass, field

import numpy as np
import torch

# Add FreeCAD Libs (conditionally)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

try:
    import FreeCAD
    import Part
    import Sketcher
    HAS_FREECAD = True
except ImportError:
    HAS_FREECAD = False
    print("Warning: FreeCAD not available. Solver feedback disabled.")

from .detokenization import reconstruct_sketch
from .convert_sketch_to_freecad import convert_sketch_to_freecad, FreeCADConversionError


@dataclass
class SolverStatus:
    """Solver status returned by FreeCAD."""
    DoF: int = 0
    Status: str = "Unknown"
    MalformedConstraints: list[int] = field(default_factory=list)
    ConflictingConstraints: list[int] = field(default_factory=list)
    RedundantConstraints: list[int] = field(default_factory=list)
    ConflictGroups: list[list[int]] = field(default_factory=list)
    MappedMalformedIndices: list[int] = field(default_factory=list)
    MappedConflictingIndices: list[int] = field(default_factory=list)
    MappedRedundantIndices: list[int] = field(default_factory=list)
    MappedConflictGroups: list[list[int]] = field(default_factory=list)


def tokens_to_freecad_sketch(
    prim_tokens: np.ndarray | torch.Tensor,
    cons_tokens: np.ndarray | torch.Tensor,
    num_bins: int,
    token_len_per_constraint: int | None,
) -> tuple[FreeCAD.Document, "Sketcher.SketchObject", list[str], dict[int, str]]:
    """Convert primitive and constraint tokens to FreeCAD sketch.

    Parameters
    ----------
    prim_tokens : array-like
        Primitive value tokens [L_prim]
    cons_tokens : array-like
        Constraint value tokens [L_cons]
    num_bins : int
        Number of quantization bins for primitives
    token_len_per_constraint : int | None
        Number of tokens per constraint. If None, variable length per constraint.

    Returns
    -------
    tuple[FreeCAD.Document, Sketcher.SketchObject, list[str], dict[int, str]]
        FreeCAD document, sketch object, history commands, and constraint mapping
    """
    # Convert to numpy if needed
    if isinstance(prim_tokens, torch.Tensor):
        prim_tokens = prim_tokens.cpu().numpy()
    if isinstance(cons_tokens, torch.Tensor):
        cons_tokens = cons_tokens.cpu().numpy()

    # Step 1: Tokens → Sketch using standard reconstruction
    sketch = reconstruct_sketch(prim_tokens, cons_tokens, num_bins, token_len_per_constraint)

    # Step 2: Sketch → FreeCAD (direct conversion)
    doc, fc_sketch, history_commands, entity_to_geoid, mapping = convert_sketch_to_freecad(sketch, doc_name="SolverEval")

    return doc, fc_sketch, history_commands, mapping


@dataclass
class GeometryValidationResult:
    """Result of geometry preservation validation."""
    geo_id: int
    geo_type: str
    before_points: list[tuple[float, float]]  # Before constraints
    after_points: list[tuple[float, float]]   # After constraints
    max_deviation: float  # Maximum deviation in mm
    is_preserved: bool    # True if max_deviation < threshold


def _extract_geometry_coords(fc_sketch: "Sketcher.SketchObject") -> dict[int, list[tuple[float, float]]]:
    """Extract coordinates from all geometries in a FreeCAD sketch.

    Returns dict mapping geo_id -> list of (x, y) points.
    """
    import Part

    coords = {}
    for geo_id, geo in enumerate(fc_sketch.Geometry):
        geo_type = type(geo).__name__
        points = []

        if geo_type == 'Point':
            points.append((geo.X, geo.Y))
        elif geo_type == 'LineSegment':
            sp = geo.StartPoint
            ep = geo.EndPoint
            points.append((sp.x, sp.y))
            points.append((ep.x, ep.y))
        elif geo_type in ('Circle', 'ArcOfCircle'):
            c = geo.Center
            points.append((c.x, c.y))
            if hasattr(geo, 'StartPoint'):
                sp = geo.StartPoint
                points.append((sp.x, sp.y))
            if hasattr(geo, 'EndPoint'):
                ep = geo.EndPoint
                points.append((ep.x, ep.y))

        coords[geo_id] = points

    return coords


def validate_constraint_geometry_impact(
    sketch: "datalib.Sketch",
    threshold_mm: float = 1.0,
    verbose: bool = True,
) -> tuple[list[GeometryValidationResult], "Sketcher.SketchObject", dict[int, int], dict[int, str]]:
    """Validate geometry impact of constraints by comparing before/after.

    This is the CORRECT validation approach:
    1. Add primitives only to FreeCAD → capture coordinates
    2. Add constraints → capture coordinates
    3. Compare the two to see how much constraints move geometry

    Parameters
    ----------
    sketch : datalib.Sketch
        Sketch with entities and constraints
    threshold_mm : float
        Maximum allowed deviation in mm
    verbose : bool
        If True, print detailed validation results

    Returns
    -------
    tuple[list[GeometryValidationResult], Sketcher.SketchObject, dict, dict]
        Validation results, final FreeCAD sketch, entity_to_geoid, constraint_mapping
    """
    import FreeCAD
    import Part
    import Sketcher
    from .convert_sketch_to_freecad import (
        _add_primitive, _add_constraint, _resolve_tangent,
        _resolve_redundant_coincident, _meter_to_millimeter
    )

    # Create FreeCAD document
    doc = FreeCAD.newDocument("ValidationDoc")
    body = doc.addObject('PartDesign::Body', 'Body')
    fc_sketch = body.newObject('Sketcher::SketchObject', 'Sketch')
    xy_plane = body.Origin.OriginFeatures[3]
    fc_sketch.AttachmentSupport = xy_plane
    fc_sketch.MapMode = 'FlatFace'

    entity_to_geoid: dict[int, int] = {}

    # Step 1: Add primitives only
    for entity_id, entity in sketch.entities.items():
        entity_idx = int(entity_id)
        _add_primitive(entity_idx, entity, fc_sketch, entity_to_geoid)

    doc.recompute()

    # Capture geometry BEFORE constraints
    coords_before = _extract_geometry_coords(fc_sketch)
    geo_types = {i: type(geo).__name__ for i, geo in enumerate(fc_sketch.Geometry)}

    # Step 2: Add constraints
    skip_coincident_ids, resolved_tangent_refs = _resolve_tangent(
        sketch.constraints, entity_to_geoid, sketch.entities
    )
    skip_redundant_ids = _resolve_redundant_coincident(
        sketch.constraints, entity_to_geoid, sketch.entities
    )
    skip_constraint_ids = skip_coincident_ids | skip_redundant_ids

    constraint_mapping: dict[int, str] = {}

    for constraint_id, constraint in sketch.constraints.items():
        if constraint_id in skip_constraint_ids:
            continue

        count_before = int(fc_sketch.ConstraintCount)
        try:
            _add_constraint(
                constraint, constraint_id, fc_sketch, entity_to_geoid,
                sketch.entities, resolved_tangent_refs
            )
        except Exception:
            continue

        count_after = int(fc_sketch.ConstraintCount)
        for i in range(count_before, count_after):
            constraint_mapping[i] = constraint_id

    doc.recompute()

    # Capture geometry AFTER constraints
    coords_after = _extract_geometry_coords(fc_sketch)

    # Step 3: Compare before and after
    results = []

    for geo_id in coords_before:
        before = coords_before[geo_id]
        after = coords_after.get(geo_id, [])

        if len(before) != len(after):
            continue

        max_deviation = 0.0
        for (bx, by), (ax, ay) in zip(before, after):
            deviation = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
            max_deviation = max(max_deviation, deviation)

        is_preserved = max_deviation < threshold_mm

        result = GeometryValidationResult(
            geo_id=geo_id,
            geo_type=geo_types.get(geo_id, "?"),
            before_points=before,
            after_points=after,
            max_deviation=max_deviation,
            is_preserved=is_preserved,
        )
        results.append(result)

        if verbose and not is_preserved:
            print(f"  [DEVIATION] geo {geo_id} ({geo_types.get(geo_id, '?')}): {max_deviation:.2f}mm")

    if verbose:
        failed_count = sum(1 for r in results if not r.is_preserved)
        max_dev = max((r.max_deviation for r in results), default=0)
        print(f"  Validation: {len(results) - failed_count}/{len(results)} preserved (max_dev={max_dev:.2f}mm)")

    return results, fc_sketch, entity_to_geoid, constraint_mapping


def get_solver_status(
    sketch: "Sketcher.SketchObject",
    constraint_mapping: dict[int, str] | None = None
) -> SolverStatus:
    """Get solver status from FreeCAD sketch.

    Note: FreeCAD solver indices in getSolverStatus() are 1-based.
    Our constraint_mapping keys are 0-based (internal FreeCAD indices).
    """
    sketch.solve()
    status = sketch.getSolverStatus()

    # Extract raw status (ensure 1-based indices are handled casually if they are just integers)
    # The user states getSolverStatus returns 1-based indices.
    # constraint_mapping keys are 0-based.
    # So we must subtract 1 from solver indices before looking up mapping.

    raw_malformed: list[int] = status['MalformedConstraints']
    raw_conflicts: list[int] = status['ConflictingConstraints']
    raw_redundant: list[int] = status['RedundantConstraints']
    raw_groups: list[list[int]] = status['ConflictGroups']

    # Helper to map a single index
    def map_idx(fc_idx_1based: int) -> int | None:
        fc_idx_0based = fc_idx_1based - 1
        if constraint_mapping and fc_idx_0based in constraint_mapping:
            cid = constraint_mapping[fc_idx_0based]
            assert cid.startswith('c') and cid[1:].isdigit(), f"[map_idx] Invalid cid: {cid}"
            return int(cid[1:])
        return None

    mapped_malformed = []
    for idx in raw_malformed:
        mid = map_idx(idx)
        if mid is not None:
            mapped_malformed.append(mid)

    mapped_conflicts = []
    for idx in raw_conflicts:
        mid = map_idx(idx)
        if mid is not None:
            mapped_conflicts.append(mid)

    mapped_redundant = []
    for idx in raw_redundant:
        mid = map_idx(idx)
        if mid is not None:
            mapped_redundant.append(mid)

    mapped_groups = []
    for group in raw_groups:
        mapped_group = []
        for idx in group:
            mid = map_idx(idx)
            if mid is not None:
                mapped_group.append(mid)
        if mapped_group:
            mapped_groups.append(mapped_group)

    return SolverStatus(
        DoF=status['DoF'],
        Status=status['Status'],
        MalformedConstraints=raw_malformed,
        ConflictingConstraints=raw_conflicts,
        RedundantConstraints=raw_redundant,
        ConflictGroups=raw_groups,
        MappedMalformedIndices=sorted(list(set(mapped_malformed))),
        MappedConflictingIndices=sorted(list(set(mapped_conflicts))),
        MappedRedundantIndices=sorted(list(set(mapped_redundant))),
        MappedConflictGroups=mapped_groups
    )


def evaluate_constraints_batch(
    prim_tokens_batch: torch.Tensor,
    cons_tokens_batch: torch.Tensor,
    num_bins: int,
    token_len_per_constraint: int | None,
    fully_constrained_dof: int = 0,
) -> dict[str, float]:
    """Evaluate constraint status for a batch of samples.

    Parameters
    ----------
    prim_tokens_batch : torch.Tensor
        Primitive tokens [B, L_prim]
    cons_tokens_batch : torch.Tensor
        Constraint tokens [B, L_cons]
    num_bins : int
        Number of quantization bins
    token_len_per_constraint : int | None
        Number of tokens per constraint. If None, variable length per constraint.
    fully_constrained_dof : int
        Number of degrees of freedom to be regarded as fully constrained

    Returns
    -------
    dict[str, float]
        Dictionary with rates: fc_rate, under_rate, over_rate, redundant_rate
    """
    assert prim_tokens_batch.size(0) == cons_tokens_batch.size(0), f"Batch size mismatch: {prim_tokens_batch.shape=} != {cons_tokens_batch.shape=}"
    B = prim_tokens_batch.size(0)

    counts = {
        'fully_constrained': 0,
        'under_constrained': 0,
        'over_constrained': 0,
        'redundant': 0,
        'malformed': 0,
        'conversion_failed': 0,
    }

    for primitives, constraints in tqdm(zip(prim_tokens_batch, cons_tokens_batch), desc="Solver Testing"):
        try:
            doc, sketch, history_commands, mapping = tokens_to_freecad_sketch(
                primitives,
                constraints,
                num_bins,
                token_len_per_constraint,
            )
        except FreeCADConversionError:
            counts['conversion_failed'] += 1
            continue

        status = get_solver_status(sketch, mapping)
        FreeCAD.closeDocument(doc.Name)

        if status.MalformedConstraints:
            counts['malformed'] += 1
        elif status.ConflictingConstraints:
            counts['over_constrained'] += 1
        elif status.RedundantConstraints:
            counts['redundant'] += 1
        elif status.DoF <= fully_constrained_dof:
            counts['fully_constrained'] += 1
        else:
            counts['under_constrained'] += 1

    # Compute rates
    total_valid = B - counts['conversion_failed']
    if total_valid == 0:
        return {}

    return {
        'fc_rate': counts['fully_constrained'] / total_valid,
        'under_rate': counts['under_constrained'] / total_valid,
        'over_rate': counts['over_constrained'] / total_valid,
        'redundant_rate': counts['redundant'] / total_valid,
        'malformed_rate': counts['malformed'] / total_valid,
        'conversion_fail_rate': counts['conversion_failed'] / B,
    }
