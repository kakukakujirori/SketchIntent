"""Training script for constraint models."""
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from img2cad_diffusion import constraint_models, constraint_data


log = logging.getLogger(__name__)


def _get_hydra_output_dir() -> str:
    """Get Hydra output directory, with fallback to current directory."""
    try:
        return HydraConfig.get().runtime.output_dir
    except ValueError:
        return '.'


def make_trainer(config: constraint_models.ConstraintModelTrainingConfig) -> pl.Trainer:
    """Creates a Trainer from the given configuration."""
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

    trainer_kwargs = {
        'max_epochs': config.max_epochs,
    }

    if config.num_gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = config.num_gpus
        callbacks.append(pl.callbacks.DeviceStatsMonitor())
    else:
        trainer_kwargs['accelerator'] = 'cpu'

    if config.num_gpus > 1:
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)

    if config.optim.gradient_clip_norm is not None:
        trainer_kwargs['gradient_clip_val'] = config.optim.gradient_clip_norm

    if config.mixed_precision:
        if config.num_gpus == 0:
            log.warning('Requested mixed precision but no GPUs. Disabled.')
        else:
            trainer_kwargs['precision'] = "bf16-mixed"

    return pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)


@hydra.main(config_name='constraints', config_path='conf', version_base=None)
def main(config: constraint_models.ConstraintModelTrainingConfig):
    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)

    data = constraint_data.ConstraintDataModule(
        config.data,
        config.batch_size // config.num_gpus,
        config.num_data_workers // config.num_gpus
    )

    data.prepare_data()
    data.setup()

    config.data.dataset_size = data.train_dataset_size
    model = constraint_models.ConstraintMDDModule(config)
    trainer = make_trainer(config)

    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='constraints', node=constraint_models.ConstraintModelTrainingConfig)
    main()
