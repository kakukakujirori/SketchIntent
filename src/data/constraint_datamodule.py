from typing import Literal

import torch
import lightning as L

from src.data.components.constraint_dataset import ConstraintDataset
from src.data.components.primitive_noise import PrimitiveNoise
from src.sketch_handlers.context import SketchContext


class ConstraintDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_sequence_file: str,
        val_sequence_file: str,
        test_sequence_file: str,
        context: SketchContext,
        primitive_noise: PrimitiveNoise,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        self._train_dataset = ConstraintDataset(
            self.hparams.train_sequence_file,
            self.hparams.context,
            self.hparams.primitive_noise,
        )

        self._val_dataset = ConstraintDataset(
            self.hparams.val_sequence_file,
            self.hparams.context,
            self.hparams.primitive_noise,
        )

        self._test_dataset = ConstraintDataset(
            self.hparams.test_sequence_file,
            self.hparams.context,
            self.hparams.primitive_noise,
        )

    def _make_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        assert self._train_dataset is not None
        return self._make_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        assert self._val_dataset is not None
        return self._make_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        assert self._test_dataset is not None
        return self._make_dataloader(self._test_dataset, shuffle=False)

    @property
    def train_dataset_size(self) -> int:
        assert self._train_dataset is not None
        return len(self._train_dataset)
