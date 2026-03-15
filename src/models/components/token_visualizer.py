import numpy as np
import torch

from src.sketch_handlers.constants import ConstraintToken


def visualize_constraint_sequence(tokens: torch.Tensor | np.ndarray, max_show: int | None = None):
    """Format constraint tokens for display."""
    if len(tokens.shape) > 1:
        return [visualize_constraint_sequence(t, max_show) for t in tokens]

    result = []
    for tok in tokens:
        tok_val = int(tok)
        if tok_val == ConstraintToken.Mask:  # NOTE: Unused
            result.append("[MASK]")
        elif tok_val == ConstraintToken.Pad:
            break
        elif tok_val == ConstraintToken.Start:
            result.append("[START]")
        elif tok_val == ConstraintToken.Stop:
            result.append("[STOP]")
            break
        elif tok_val == ConstraintToken.InnerPad:
            result.append("[PAD]")
        elif tok_val < len(ConstraintToken):
            result.append(ConstraintToken(tok_val).name)
        else:
            result.append(f"R{tok_val - len(ConstraintToken)}")
    if max_show and len(result) > max_show:
        return result[:max_show] + [f"... (+{len(result) - max_show} more)"]
    return result
