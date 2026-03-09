import torch
import torchmetrics

from src.sketch_handlers.constants import ConstraintToken
from src.sketch_handlers.context import SketchContext


class ConstraintIoU(torchmetrics.Metric):
    """Accumulates constraint set IoU statistics across batches.

    Tracks:
    - Total intersection count
    - Total prediction count
    - Total target count
    - Total union count
    - Number of samples (for averaging precision/recall per sample)
    """

    def __init__(self, context: SketchContext, prefix_len: int = 0):
        super().__init__()
        self.context = context
        self.block_size = context.align_constraint_token_length_to
        self.prefix_len = prefix_len
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

        If self.block_size = None, detects constraint boundaries by looking for constraint type tokens
        (Coincident through Vertical), rather than assuming fixed block sizes.

        Args:
            tokens: [L] token sequence (single example)

        Returns:
            Set of (constraint_type, (sorted_refs...)) tuples
        """
        if len(tokens.shape) != 1:
            raise ValueError(f"Expected 1D tensor, got {tokens.shape}")
        tokens = tokens.cpu().numpy().tolist()
        constraints = set()

        if self.block_size is None:
            current_type = None
            current_refs = []

            for tok in tokens:
                tok = int(tok)

                if tok == ConstraintToken.Pad:
                    continue
                if tok == ConstraintToken.Mask:
                    raise ValueError(f"Mask token found in the evaluation token.")
                if tok == ConstraintToken.Stop:
                    break
                if tok == ConstraintToken.InnerPad:
                    continue

                if tok < len(ConstraintToken):
                    assert tok != ConstraintToken.InnerPad
                    # Save previous constraint if exists
                    if current_type is not None:
                        constraint_tuple = (current_type, tuple(sorted(current_refs)))
                        constraints.add(constraint_tuple)
                    # Start new constraint
                    current_type = tok
                    current_refs = []
                else:
                    # This is a reference token
                    ref_idx = tok - len(ConstraintToken)
                    current_refs.append(ref_idx)

            # Don't forget the last constraint
            if current_type is not None:
                constraint_tuple = (current_type, tuple(sorted(current_refs)))
                constraints.add(constraint_tuple)

        else:
            assert self.block_size > 0

            idx = self.prefix_len
            while idx < len(tokens):
                block = tokens[idx:idx+self.block_size]
                idx += self.block_size

                # skip invalid constraints
                if len(block) < self.block_size:
                    break
                if block[0] in [ConstraintToken.Pad, ConstraintToken.Stop, ConstraintToken.Mask, ConstraintToken.InnerPad]:
                    continue
                if block[0] >= len(ConstraintToken):
                    continue
                while block[-1] in [ConstraintToken.InnerPad]:
                    block = block[:-1]
                if not all(b >= len(ConstraintToken) for b in block[1:]):
                    continue
                # if len(block[1:]) not in CONSTRAINT_ACCEPTABLE_REF_COUNTS[ConstraintToken(block[0])]:
                #     continue
                block = block[:1] + sorted(block[1:])
                constraints.add(tuple(block))

        return constraints
