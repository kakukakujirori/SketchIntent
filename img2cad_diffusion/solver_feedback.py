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
from typing import TypedDict

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

import sketchgraphs.data as datalib

from .detokenization import reconstruct_sketch
from .convert_sketch_to_freecad import convert_sketch_to_freecad


class SolverStatus(TypedDict):
    """Solver status dictionary returned by FreeCAD."""
    DoF: int
    Status: str  # 'Converged', 'Failed', etc.
    ConflictingConstraints: list[int]
    RedundantConstraints: list[int]
    ConflictGroups: list


def tokens_to_freecad_sketch(
    prim_tokens: np.ndarray | torch.Tensor,
    cons_tokens: np.ndarray | torch.Tensor,
    num_bins: int,
) -> tuple[FreeCAD.Document | None, Sketcher.SketchObject | None]:
    """Convert primitive and constraint tokens to FreeCAD sketch.

    Parameters
    ----------
    prim_tokens : array-like
        Primitive value tokens [L_prim]
    cons_tokens : array-like
        Constraint value tokens [L_cons]
    num_bins : int
        Number of quantization bins for primitives

    Returns
    -------
    tuple[FreeCAD.Document, Sketcher.SketchObject] | None
        FreeCAD document and sketch object, or None if conversion fails
    """
    # Convert to numpy if needed
    if isinstance(prim_tokens, torch.Tensor):
        prim_tokens = prim_tokens.cpu().numpy()
    if isinstance(cons_tokens, torch.Tensor):
        cons_tokens = cons_tokens.cpu().numpy()

    # Step 1: Tokens → Sketch (SketchGraphs format)
    sketch = reconstruct_sketch(prim_tokens, cons_tokens, num_bins)

    # Step 2: Sketch → FreeCAD (direct conversion)
    # May raise ValueError for degenerate primitives caused by quantization
    try:
        doc, fc_sketch, history_commands = convert_sketch_to_freecad(sketch, doc_name="SolverEval")
    except Exception as e:
        # Log the error for debugging, return None to indicate conversion failure
        print(f"[tokens_to_freecad_sketch] Conversion failed: {e}")
        return None, None

    return doc, fc_sketch


def get_solver_status(sketch: "Sketcher.SketchObject") -> SolverStatus:
    """Get solver status from FreeCAD sketch.

    Parameters
    ----------
    sketch : Sketcher.SketchObject
        FreeCAD sketch object

    Returns
    -------
    SolverStatus
        Dictionary with solver status information
    """
    sketch.solve()
    return sketch.getSolverStatus()


def evaluate_constraints_batch(
    prim_tokens_batch: torch.Tensor,
    cons_tokens_batch: torch.Tensor,
    num_bins: int,
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
        'conversion_failed': 0,
    }

    for primitives, constraints in tqdm(zip(prim_tokens_batch, cons_tokens_batch), desc="Solver Testing"):
        doc, sketch = tokens_to_freecad_sketch(
            primitives,
            constraints,
            num_bins
        )

        if doc is None:
            counts['conversion_failed'] += 1
            continue

        try:
            status = get_solver_status(sketch)

            if status['ConflictingConstraints']:
                counts['over_constrained'] += 1
            elif status['RedundantConstraints']:
                counts['redundant'] += 1
            elif status['DoF'] <= fully_constrained_dof:
                counts['fully_constrained'] += 1
            else:
                counts['under_constrained'] += 1

        except Exception as e:
            print(f"Warning: solver evaluation failed: {e}")
            counts['conversion_failed'] += 1
        finally:
            FreeCAD.closeDocument(doc.Name)

    # Compute rates
    total_valid = B - counts['conversion_failed']
    if total_valid == 0:
        return {}

    return {
        'fc_rate': counts['fully_constrained'] / total_valid,
        'under_rate': counts['under_constrained'] / total_valid,
        'over_rate': counts['over_constrained'] / total_valid,
        'redundant_rate': counts['redundant'] / total_valid,
        'conversion_fail_rate': counts['conversion_failed'] / B,
    }
