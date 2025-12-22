"""Utilities to interact with pytorch-lightning.
"""

import dataclasses
import logging
import os
from typing import Any, Dict, List, Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import ClusterEnvironment
from lightning.pytorch.strategies import DDPStrategy

import torch
import torch.distributed


log = logging.getLogger(__name__)


def _get_hydra_output_dir() -> str:
    """Get Hydra output directory, with fallback to current directory."""
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except (ImportError, ValueError):
        # Hydra not installed or not initialized (e.g., running outside @hydra.main)
        return '.'


@dataclasses.dataclass
class TrainingConfiguration:
    """Base configuration for training.

    Attributes
    ----------
    max_epochs : int
        Maximum number of epochs to train for.
    num_gpus : int
        Number of GPUs to use.
    mixed_precision : bool
        If `True`, uses 16-bit training.
    ligthning
        Additional arguments to pass when creating the pytorch-lightning trainer.
    """
    max_epochs: int = 30
    num_gpus: int = 1
    mixed_precision: bool = False
    lightning: Dict[str, Any] = dataclasses.field(default_factory=dict)


def make_trainer(config: TrainingConfiguration, **kwargs) -> pl.Trainer:
    """Creates a new `lightning.pytorch.Trainer` according to the given configuration.

    Parameters
    ----------
    config : TrainingConfiguration
        Configuration for the trainer to create.
    **kwargs
        Additional keyword arguments which are passed to the `lightning.pytorch.Trainer` constructor.
    """
    # Explicitly use TensorBoardLogger with Hydra-aware output directory
    log_dir = _get_hydra_output_dir()
    logger = TensorBoardLogger(save_dir=log_dir, name='lightning_logs')

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=-1,
            filename='epoch_{epoch}',
            auto_insert_metric_name=False),
        pl.callbacks.ModelCheckpoint(
            monitor='validation/loss',
            filename='epoch_{epoch}_vloss_{validation/loss:.3f}',
            auto_insert_metric_name=False,
            save_top_k=1),
        pl.callbacks.LearningRateMonitor(),
    ]

    if config.num_gpus > 0:
        callbacks.append(pl.callbacks.DeviceStatsMonitor())

    trainer_kwargs = {
        **config.lightning
    }


    trainer_kwargs['max_epochs'] = config.max_epochs

    if config.num_gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = config.num_gpus
    else:
        trainer_kwargs['accelerator'] = 'cpu'

    if config.num_gpus > 1:
        store_path = os.path.abspath('./torch_distributed_init.store')
        log.info(f'Using file-based distributed initialization at {store_path}')
        trainer_kwargs['strategy'] = CustomDDPStrategy(init_method='file://' + store_path, find_unused_parameters=False)

    if hasattr(config, 'optim') and hasattr(config.optim, 'gradient_clip_norm'):
        if config.optim.gradient_clip_norm is not None:
            trainer_kwargs['gradient_clip_val'] = config.optim.gradient_clip_norm

    if config.mixed_precision:
        if config.num_gpus == 0:
            logging.getLogger(__name__).warning('Requested 16-bit precision but no GPUs. 16-bit precision training is not available on CPU, it has been disabled for now.')
        else:
            trainer_kwargs['precision'] = 16

    trainer_kwargs.update(kwargs)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs)

    return trainer


class CustomDDPStrategy(DDPStrategy):
    """Custom DDP strategy which allows for the specification of the torch distributed initialization method.
    """
    def __init__(
        self,
        init_method: str = "env://",
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parallel_devices=parallel_devices, cluster_environment=cluster_environment, **kwargs)
        self.init_method = init_method

    def init_ddp_connection(self, global_rank: Optional[int]=None, world_size: Optional[int]=None) -> None:
        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        if not torch.distributed.is_initialized():
            log.info(f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}")
            torch.distributed.init_process_group(
                self.process_group_backend,
                init_method=self.init_method,
                rank=global_rank, world_size=world_size)
