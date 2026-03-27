import torch
import numpy as np
import torchmetrics

from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


class ConstraintIoU(torchmetrics.Metric):
    """Accumulates constraint set IoU statistics across batches.

    Tracks:
    - Total intersection count
    - Total prediction count
    - Total target count
    - Total union count
    - Number of samples (for averaging precision/recall per sample)
    """

    def __init__(self, context: SketchContext, prefix_len: int = 1):
        super().__init__()
        self.context = context
        self.prefix_len = prefix_len
        self.tokenizer = SketchTokenizer(context)
        self.add_state("total_intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_union", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred_tokens: [B, L] predicted constraint tokens
            target_tokens: [B, L] ground truth constraint tokens
        """
        B = pred_tokens.shape[0]

        for b in range(B):
            pred_set = self._tokens_to_constraint_set(pred_tokens[b])
            target_set = self._tokens_to_constraint_set(target_tokens[b])

            intersection = pred_set & target_set
            union = pred_set | target_set

            self.total_intersection += len(intersection)
            self.total_pred += len(pred_set)
            self.total_target += len(target_set)
            self.total_union += len(union)
            self.num_samples += 1

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute IoU, precision, and recall.

        Returns:
            Dict with 'iou', 'precision', 'recall' tensors
        """
        # Avoid division by zero
        iou = self.total_intersection / self.total_union.clamp(min=1)
        precision = self.total_intersection / self.total_pred.clamp(min=1)
        recall = self.total_intersection / self.total_target.clamp(min=1)
        return {"iou": iou, "precision": precision, "recall": recall}

    def _tokens_to_constraint_set(self, tokens: torch.Tensor) -> set:
        """Convert token sequence to set of constraint tuples.

        Delegates to :meth:`SketchTokenizer.parse_constraint_tokens` for
        parsing, then wraps the result as a set for IoU computation.

        Args:
            tokens: [L] token sequence (single example)

        Returns:
            Set of (constraint_type, (sorted_refs...)) tuples
        """
        if len(tokens.shape) != 1:
            raise ValueError(f"Expected 1D tensor, got {tokens.shape}")

        val_tokens = tokens[self.prefix_len:].cpu().numpy()
        parsed_valid, parsed_malformed = self.tokenizer.parse_constraint_tokens(val_tokens)
        type_refs = [(typ, refs) for typ, refs, block, start_idx in parsed_valid]
        return set(type_refs)
