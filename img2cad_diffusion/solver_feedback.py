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
    doc, fc_sketch, history_commands, mapping = convert_sketch_to_freecad(sketch, doc_name="SolverEval")

    return doc, fc_sketch, history_commands, mapping


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
