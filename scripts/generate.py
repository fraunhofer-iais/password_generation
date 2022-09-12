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
from password_generation.data.dataloaders import TokenizedTextDataLoader
from password_generation.models.base_model import Model
from password_generation.utils.callbacks import (
    LogReconstruction,
    SamplePasswords,
    WandbRemoveRun,
)
from password_generation.utils.generation import generate_passwords
from password_generation.utils.helper import create_instance, timestamp
from password_generation.utils.yamls import expand_params, load_yaml_file

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model.pth checkpoint file.")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output.txt file for storing generated passwords."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        help="Path to model config.yaml. Must match model model checkpoint.pth file. If not given, will look for config.yaml file in model checkpoint directory.",
    )
    parser.add_argument("-n", "--num-passwords", type=int, default=1_000_000, help="Number of passwords to generate.")
    parser.add_argument("-b", "--batch-size", type=int, default=100, help="Batch size for generation.")
    parser.add_argument("-nc", "--no-cuda", action="store_true")
    args = parser.parse_args()
    return args


def generate_from_config(config: Dict, args: argparse.Namespace) -> None:

    # dataloader: TokenizedTextDataLoader = create_instance(config["dataloader"])
    tokenizer = create_instance(config["dataloader"]["args"]["tokenizer_config"])
    model: Model = create_instance(
        config["model"],
        vocab_dim=len(tokenizer.vocab),
        max_sequence_length=config["dataloader"]["args"]["max_sequence_length"],
        sos_index=tokenizer.sos_index,
        eos_index=tokenizer.eos_index,
        pad_index=tokenizer.pad_index,
        unk_index=None,  # tokenizer.unk_index,
    )

    checkpoint = torch.load(args.model, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"])

    with open(args.output, "w") as f:
        for batch in generate_passwords(model, tokenizer, n=args.num_passwords, batch_size=args.batch_size):
            f.write("\n".join(batch) + "\n")


def cleanup(trainer: pl.Trainer) -> None:
    if isinstance(trainer.logger, WandbLogger):
        wandb.finish()


def find_config_path(model_path: os.PathLike) -> os.PathLike:
    parent_dir = os.path.dirname(os.path.realpath(model_path))
    config_path = os.path.join(parent_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File {config_path} does not exist.")
    return config_path


def main():
    args = parse_args()

    assert os.path.exists(args.model)
    if args.config is None:
        config_path = find_config_path(args.model)
    else:
        config_path = args.config
    config = load_yaml_file(config_path)

    generate_from_config(config, args)


if __name__ == "__main__":
    main()
