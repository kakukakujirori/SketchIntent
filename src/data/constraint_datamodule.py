import math
from typing import Any

import torch
import lightning as L
from torch.utils.data.dataloader import default_collate

import sketchgraphs.data as datalib
from src.data.components.constraint_dataset import ConstraintDataset
from src.data.components.primitive_noise import PrimitiveNoise
from src.sketch_handlers.context import SketchContext


def sketch_collate_fn(batch: list[Any]):
    if isinstance(batch[0], datalib.Sketch):
        return batch

    if isinstance(batch[0], dict):
        return {key: sketch_collate_fn([item[key] for item in batch]) for key in batch[0]}
    if isinstance(batch[0], (tuple, list)):
        return type(batch[0])(sketch_collate_fn(samples) for samples in zip(*batch))
    return default_collate(batch)


def split_dataset(
    dataset: torch.utils.data.Dataset,
    validation_fraction: float | None = None,
    test_fraction: float | None = None,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Splits the given dataset into a training dataset and a validation dataset.
    The split is performed randomly (but the same every time).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to split.
    validation_fraction : float, optional
        If not `None`, the fraction of samples to reserve for the validation set.

    Returns
    -------
    train_dataset : torch.utils.data.Dataset
        The training dataset split
    valid_dataset : Optional[torch.utils.data.Dataset]
        If ``validation_fraction`` is not `None` and greater than 0,
        the validation dataset split.
    """

    generator = torch.Generator().manual_seed(4242424242)
    perm = torch.randperm(len(dataset), generator=generator)

    if validation_fraction is None:
        validation_fraction = 0.0

    if test_fraction is None:
        test_fraction = 0.0

    num_train_samples = int(math.ceil(len(dataset) * (1 - validation_fraction - test_fraction)))
    num_test_samples = int(math.ceil(len(dataset) * test_fraction))

    if num_train_samples + num_test_samples > len(dataset):
        num_test_samples = len(dataset) - num_train_samples

    num_valid_samples = len(dataset) - num_train_samples - num_test_samples

    train_idx, valid_idx, test_idx = torch.split(perm, [num_train_samples, num_valid_samples, num_test_samples])

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, valid_dataset, test_dataset


class ConstraintDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_sequence_file: str,
        val_sequence_file: str,
        test_sequence_file: str,
        context: SketchContext,
        primitive_noise: PrimitiveNoise,
        relational_only: bool = False,
        # validation_fraction: float = 0.025,
        # test_fraction: float = 0.05,
        batch_size: int = 32,
        num_workers: int = 0,
        multiprocessing_context: str | None = None,
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
            self.hparams.relational_only,
        )

        self._val_dataset = ConstraintDataset(
            self.hparams.val_sequence_file,
            self.hparams.context,
            self.hparams.primitive_noise,
            self.hparams.relational_only,
        )

        self._test_dataset = (
            ConstraintDataset(
                self.hparams.test_sequence_file,
                self.hparams.context,
                self.hparams.primitive_noise,
                self.hparams.relational_only,
            )
            if self.hparams.test_sequence_file
            else None
        )

        # self._train_dataset, self._val_dataset, self._test_dataset = split_dataset(
        #     self._train_dataset, self.hparams.validation_fraction, self.hparams.test_fraction)

    def _make_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        # PyTorch rejects multiprocessing_context when num_workers == 0.
        worker_context = (
            self.hparams.multiprocessing_context
            if self.hparams.num_workers > 0
            else None
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            multiprocessing_context=worker_context,
            collate_fn=sketch_collate_fn,
        )

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
