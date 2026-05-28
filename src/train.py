import re
import sys
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _override_key(override: str) -> str | None:
    """Return the left-hand side key of a Hydra override."""
    if override.startswith("-"):
        return None
    item = override.lstrip("+")
    if item.startswith("~"):
        item = item[1:]
    for sep in ("=", ":"):
        if sep in item:
            return item.split(sep, 1)[0]
    return item or None


def _pop_resume_from(argv: list[str]) -> tuple[list[str], str | None]:
    """Accept a friendly ``--resume_from DIR`` flag before Hydra parses argv."""
    cleaned = [argv[0]]
    resume_from: str | None = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--resume_from":
            if i + 1 >= len(argv):
                raise SystemExit("--resume_from requires a run directory")
            resume_from = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--resume_from="):
            resume_from = arg.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned, resume_from


def _prepare_resume_argv(argv: list[str]) -> list[str]:
    """Replay saved run overrides and force Hydra to reuse the run directory."""
    cleaned, resume_from = _pop_resume_from(argv)
    if resume_from is None:
        return argv

    run_dir = Path(resume_from).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"--resume_from directory does not exist: {run_dir}")

    current_args = cleaned[1:]
    first_hydra_flag = next(
        (i for i, arg in enumerate(current_args) if arg.startswith("--")),
        len(current_args),
    )
    override_args = current_args[:first_hydra_flag]
    hydra_args = current_args[first_hydra_flag:]
    current_keys = {
        key for arg in override_args
        if (key := _override_key(arg)) is not None
    }

    saved_overrides: list[str] = []
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    if overrides_path.exists():
        loaded = OmegaConf.load(overrides_path)
        saved_overrides = [str(x) for x in loaded]

    resume_internal_keys = {"resume_from", "hydra.run.dir", "resume_ckpt_path"}
    replayed = [
        item for item in saved_overrides
        if (
            (key := _override_key(item)) not in current_keys
            and key not in resume_internal_keys
        )
    ]

    return [
        cleaned[0],
        *replayed,
        *override_args,
        f"resume_from={run_dir}",
        f"hydra.run.dir={run_dir}",
        *hydra_args,
    ]


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    ckpts = sorted(
        ckpt_dir.glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    return ckpts[-1] if ckpts else None


def _wandb_run_id(run_dir: Path) -> str | None:
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    candidates = [
        path for path in wandb_dir.iterdir()
        if path.is_dir() and path.name.startswith("run-")
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    match = re.match(r"run-\d{8}_\d{6}-(?P<id>.+)$", latest.name)
    return match.group("id") if match else None


def _configure_resume_from(cfg: DictConfig) -> None:
    resume_from = cfg.get("resume_from")
    if not resume_from:
        return

    run_dir = Path(str(resume_from)).expanduser().resolve()
    ckpt_path = _latest_checkpoint(run_dir)
    wandb_id = _wandb_run_id(run_dir)

    with open_dict(cfg):
        cfg.resume_from = str(run_dir)
        if cfg.get("resume_ckpt_path") is None and ckpt_path is not None:
            cfg.resume_ckpt_path = str(ckpt_path)
        if cfg.get("resume_ckpt_path") is not None:
            cfg.pretrained_ckpt_path = None
        if cfg.get("trainer") is not None:
            cfg.trainer.default_root_dir = str(run_dir)
        if cfg.get("logger") is not None and cfg.logger.get("wandb") is not None:
            cfg.logger.wandb.save_dir = str(run_dir)
            if wandb_id is not None:
                cfg.logger.wandb.id = wandb_id
                cfg.logger.wandb.resume = "allow"

    if ckpt_path is None:
        log.warning(f"No checkpoint found under {run_dir}/checkpoints; starting without trainer resume.")
    else:
        log.info(f"Resuming trainer checkpoint from {ckpt_path}")
    if wandb_id is None:
        log.warning(f"No wandb run id found under {run_dir}/wandb; starting a new wandb run.")
    else:
        log.info(f"Resuming wandb run id {wandb_id}")


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    _configure_resume_from(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    pretrained_ckpt_path = cfg.get("pretrained_ckpt_path")
    if pretrained_ckpt_path:
        log.info(f"Loading pretrained weights from {pretrained_ckpt_path}")
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            log.warning(f"Missing keys in model state_dict: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint state_dict: {unexpected_keys}")

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("resume_ckpt_path"), weights_only=False)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    sys.argv = _prepare_resume_argv(sys.argv)
    main()
