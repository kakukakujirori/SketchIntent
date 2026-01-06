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
            monitor='epoch',
            mode="max",
            filename='epoch_{epoch}',
            auto_insert_metric_name=False,
            save_top_k=1),
        pl.callbacks.ModelCheckpoint(
            monitor='validation/loss',
            filename='epoch_{epoch}_vloss_{validation/loss:.3f}',
            auto_insert_metric_name=False,
            save_top_k=1),
        pl.callbacks.LearningRateMonitor(),
    ]

    trainer_kwargs = {
        'max_epochs': config.max_epochs,
        # 'limit_train_batches': 10,
        # 'limit_val_batches': 10,
    }

    if config.num_gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = config.num_gpus
        # callbacks.append(pl.callbacks.DeviceStatsMonitor())
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

    if config.test_only:
        trainer_kwargs['devices'] = 1
        trainer_kwargs['num_nodes'] = 1

    return pl.Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)


@hydra.main(config_name='constraints', config_path='conf', version_base=None)
def main(config: constraint_models.ConstraintModelTrainingConfig):
    # Set global seed for reproducibility
    if hasattr(config, 'training_seed') and config.training_seed is not None:
        pl.seed_everything(config.training_seed, workers=True)
        log.info(f"Set training seed to {config.training_seed}")

    config.data.sequence_path = hydra.utils.to_absolute_path(config.data.sequence_path)

    # Convert fc_clusters_path to absolute path if specified
    if config.data.fc_clusters_path is not None:
        config.data.fc_clusters_path = hydra.utils.to_absolute_path(config.data.fc_clusters_path)

    # inverse-scale batch size and workers
    config.data.batch_size = max(1, config.data.batch_size // max(1, config.num_gpus))
    config.data.num_data_workers = max(1, config.data.num_data_workers // max(1, config.num_gpus))

    data = constraint_data.ConstraintDataModule(config.data)
    data.setup()

    if data.has_fc_validation:
        log.info(f"FC validation enabled with {len(data._fc_val_dataset)} samples")

    config.data.dataset_size = data.train_dataset_size

    # Select model based on config
    if config.model.net.model_type == "ar":
        from img2cad_diffusion import constraint_models_ar
        model = constraint_models_ar.ConstraintARModule(config)
    elif config.model.net.model_type == "mdd":
        model = constraint_models.ConstraintMDDModule(config)
    else:
        raise ValueError(f"Unknown model_type: {config.model.net.model_type}. Expected 'ar' or 'mdd'.")

    trainer = make_trainer(config)

    # Resolve checkpoint path if provided
    ckpt_path = None
    if config.checkpoint_path is not None:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)

    if config.test_only:
        if ckpt_path is None:
            raise ValueError("checkpoint_path is required when test_only=true")
        log.info(f"Test-only mode: loading checkpoint from {ckpt_path}")
        trainer.test(model, datamodule=data, ckpt_path=ckpt_path, weights_only=False)
    else:
        if ckpt_path is not None:
            log.info(f"Resuming training from checkpoint: {ckpt_path}")
        trainer.fit(model, datamodule=data, ckpt_path=ckpt_path, weights_only=False)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store(name='constraints', node=constraint_models.ConstraintModelTrainingConfig)
    main()
