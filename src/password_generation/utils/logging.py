import logging
from functools import partial

import markdown
import pytorch_lightning as pl
import wandb
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


# TODO: Colored logs
logger = logging.getLogger(__name__)


def log_text(key: str, log_string: str, trainer: pl.Trainer) -> None:
    if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
        log_text_tensorboard(key, log_string, trainer)
    elif isinstance(trainer.logger, pl.loggers.WandbLogger):
        log_text_wandb(key, log_string)
    else:
        raise NotImplementedError


def log_text_tensorboard(key: str, log_string: str, trainer: pl.Trainer):
    trainer.logger.experiment.add_text(key, log_string, global_step=trainer.current_epoch)


def log_text_wandb(key: str, log_string: str):
    try:
        wandb.log({key: wandb.Html(markdown_to_html(log_string))})
    except wandb.errors.Error as e:
        logger.warning(
            f"Unable to log string with wandb. "
            f"If this happens in the validation sanity check, you can ignore this message. "
            f'Error: "{str(e)}"'
        )
        return


def markdown_to_html(markdown_string: str) -> str:
    return markdown.markdown(markdown_string)
