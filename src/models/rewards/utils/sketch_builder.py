"""Shared helper for building a predicted sketch from token tensors."""
from __future__ import annotations

import numpy as np

import sketchgraphs.data as datalib

from src.models.components.solver_feedback import SolverFeedback
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer
from src.sketch_handlers.sketch_utils import complete_constraint_values


def build_pred_sketch(
    tokenizer: SketchTokenizer,
    cons_tokens_np: np.ndarray,
    prim_tokens_np: np.ndarray,
    sketch_original: datalib.Sketch | None,
) -> datalib.Sketch:
    """Decode token arrays into a Sketch with original primitive geometry.

    Passing ``reference_sketch=sketch_original`` makes the decoded entities
    adopt GT's entity keys by position, so constraint references in Pred and
    GT resolve to the same physical entity (required for downstream
    response-metric computations).
    """
    pred_sketch, _, _ = tokenizer.detokenize(
        {"p_val": prim_tokens_np, "c_val": cons_tokens_np},
        allow_invalid_constraints=True,
        reference_sketch=sketch_original,
    )
    if sketch_original is not None:
        pred_sketch = SolverFeedback.overwrite_primitives_with_original(
            pred_sketch, sketch_original
        )
    pred_sketch, _ = complete_constraint_values(
        pred_sketch, skip_invalid_constraints=True
    )
    return pred_sketch
