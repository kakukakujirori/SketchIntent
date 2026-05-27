"""Isolated worker module for ProcessPoolExecutor.

IMPORTANT: This module must NOT import torch, lightning, or any CUDA-related
library at the top level. It is loaded inside spawned worker processes, and
importing CUDA libraries from 32+ concurrent spawned processes causes the
CUDA driver's global lock to deadlock.

Only numpy, stdlib, and FreeCAD-related modules are allowed here.
"""
from __future__ import annotations

import numpy as np
import sketchgraphs.data as datalib

from src.models.components.solver_feedback import SolverFeedback
from src.sketch_handlers.context import SketchContext

_worker_solver_feedback: SolverFeedback | None = None


def init_worker(
    context: SketchContext,
    distortion_threshold: float,
    check_rigid_dimensional: bool = True,
) -> None:
    """Initialize the per-worker SolverFeedback instance.

    Called once per worker process by ProcessPoolExecutor(initializer=...).
    Must not trigger any torch/CUDA import.
    """
    global _worker_solver_feedback
    _worker_solver_feedback = SolverFeedback(
        context,
        distortion_threshold,
        check_rigid_dimensional=check_rigid_dimensional,
    )


def evaluate_single(args: tuple[np.ndarray, np.ndarray, datalib.Sketch | None]) -> object:
    """Evaluate a single (prim, cons, sketch) triple.

    Returns the result of SolverFeedback.evaluate_constraints, or None on
    unexpected failure (FreeCAD occasionally raises outside Python's
    exception hierarchy).
    """
    prim_np, cons_np, sketch_original = args
    assert _worker_solver_feedback is not None
    return _worker_solver_feedback.evaluate_constraints(
        prim_np, cons_np, sketch_original, check_one_by_one=False
    )
