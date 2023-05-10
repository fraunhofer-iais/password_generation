import argparse
import logging
import os
from datetime import datetime as dt
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from password_generation import project_path
from password_generation.utils.callbacks import (
    LogReconstruction,
    SamplePasswords,
    WandbRemoveRun,
)
from password_generation.utils.helper import create_instance, timestamp
from password_generation.utils.yamls import expand_params, load_yaml_file

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=os.path.join(project_path, "configs/test_config.yaml"))
    parser.add_argument("-nc", "--no-cuda", action="store_true")
    args = parser.parse_args()
    return args


def run_config(config: Dict, args: argparse.Namespace) -> None:
    run_name = config["run"]
    config["run_time"] = timestamp()

    dataloader = create_instance(config["dataloader"])
    model = create_instance(
        config["model"],
        vocab_dim=dataloader.vocab_size,
        max_sequence_length=dataloader.max_sequence_length,
        sos_index=dataloader.sos_index,
        eos_index=dataloader.eos_index,
        pad_index=dataloader.pad_index,
        unk_index=dataloader.unk_index,
    )

    callbacks = init_callbacks(config, run_name, dataloader)

    logging_dir = config["logging"]["logging_dir"]
    os.makedirs(logging_dir, exist_ok=True)
    if config["logging"].get("use_wandb", True):
        train_logger = WandbLogger(project="password_generation", group=run_name, save_dir=logging_dir)
        train_logger.experiment.config.update(config)
    else:
        train_logger = TensorBoardLogger(name=run_name, save_dir=logging_dir)

    trainer = pl.Trainer(
        logger=train_logger,
        callbacks=callbacks,
        log_every_n_steps=100,
        gradient_clip_val=0.1,
        gradient_clip_algorithm="value",
        detect_anomaly=True,
        # strategy="dp2",
        gpus=-1 if (torch.cuda.is_available() and not args.no_cuda) else 0,
        **config["trainer"].get("args", {}),
    )

    trainer.fit(model, train_dataloaders=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    cleanup(trainer)


def cleanup(trainer: pl.Trainer) -> None:
    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


def main():
    args = parse_args()
    config = load_yaml_file(args.config)
    gs_configs: List[Dict] = expand_params(config, adjust_run_name=False)

    for index, config in enumerate(gs_configs):
        logger.info(
            f"\n"
            f"\t = = = = = = = = = = = = = = = = = = \n"
            f"\t RUNNING CONFIG NO. {index + 1} of {len(gs_configs)}.\n"
        )
        run_config(config, args)


def init_callbacks(config, run_name, dataloader) -> List[pl.Callback]:
    callbacks = []

    log_reconstruction = LogReconstruction(dataloader.tokenizer)
    callbacks.append(log_reconstruction)

    sample_passwords = SamplePasswords(dataloader.tokenizer, **config["sample_passwords"])
    callbacks.append(sample_passwords)

    save_checkpoints = config["model_checkpoints"].get("save_checkpoints", False)
    config["model_checkpoints"]["checkpoints_dir"] = os.path.join(
        config["model_checkpoints"]["checkpoints_dir"], run_name, config["run_time"]
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=config["model_checkpoints"]["checkpoints_dir"],
        monitor=config["model_checkpoints"].get("best_model_metric", "valid/loss"),
        save_last=save_checkpoints,  # only save last if any checkpoints are saved
        verbose=True,
        save_top_k=1 if save_checkpoints else 0,  # top_k = 0 -> no checkpointing,
    )
    callbacks.append(model_checkpoint)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    model_summary = ModelSummary(max_depth=3)
    callbacks.append(model_summary)

    return callbacks


if __name__ == "__main__":
    main()
