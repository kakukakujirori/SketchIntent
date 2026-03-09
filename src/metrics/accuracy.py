import torch
import torchmetrics

from src.sketch_handlers.constants import SpecialToken


class TokenAccuracy(torchmetrics.Metric):
    def __init__(self, ignore_index: int = SpecialToken.Pad):
        super().__init__()
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            preds:  (batch, max_len)
            targets: (batch, max_len)
        """
        assert len(preds.shape) == 2, f"{preds.shape=}"
        assert len(targets.shape) == 2, f"{targets.shape=}"
        assert preds.shape == targets.shape, f"{preds.shape=}, {targets.shape=}"

        mask  = targets != self.ignore_index   # (batch, max_len)
        self.correct += (preds == targets)[mask].sum()
        self.total   += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total.float()