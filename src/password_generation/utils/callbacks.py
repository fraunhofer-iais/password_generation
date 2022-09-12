import logging
import sys
from typing import Any, List, Optional, Sequence, Set, Tuple

import humanize
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from password_generation.data.datasets import TokenizedTextDataset
from password_generation.models import Model
from password_generation.utils.logging import log_text, tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LogReconstruction(pl.Callback):
    """
    # log reconstruction
    # decode samples and check how many are in validation set
    """

    def __init__(self, tokenizer, num_log_reconstructions: int = 20, max_sequence_length: int = 15):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_log_reconstructions = num_log_reconstructions
        self.max_sequence_length = max_sequence_length

        self.train_targets = []
        self.train_preds = []
        self.valid_targets = []
        self.valid_preds = []
        self.test_targets = []
        self.test_preds = []

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Model,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if len(self.train_targets) < self.num_log_reconstructions:
            self.train_targets.extend(outputs["target"].tolist())
            self.train_preds.extend(outputs["pred"].tolist())

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Model,
        outputs: Optional[pl.utilities.types.STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if len(self.valid_targets) < self.num_log_reconstructions:
            self.valid_targets.extend(outputs["target"].tolist())
            self.valid_preds.extend(outputs["pred"].tolist())

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: Model,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if len(self.test_targets) < self.num_log_reconstructions:
            self.test_targets.extend(outputs["target"].tolist())
            self.test_preds.extend(outputs["pred"].tolist())

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Model) -> None:
        self.log_reconstruction(
            targets=self.train_targets,
            preds=self.train_preds,
            logging_prefix="train/reconstruction",
            trainer=trainer,
            pl_module=pl_module,
        )
        self.train_targets = []
        self.train_preds = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Model) -> None:
        self.log_reconstruction(
            targets=self.valid_targets,
            preds=self.valid_preds,
            logging_prefix="valid/reconstruction",
            trainer=trainer,
            pl_module=pl_module,
        )
        self.valid_targets = []
        self.valid_preds = []

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: Model) -> None:
        self.log_reconstruction(
            targets=self.test_targets,
            preds=self.test_preds,
            logging_prefix="test/reconstruction",
            trainer=trainer,
            pl_module=pl_module,
        )
        self.test_targets = []
        self.test_preds = []

    def log_reconstruction(
        self,
        targets: List[torch.Tensor],
        preds: List[torch.Tensor],
        logging_prefix: str,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        preds = preds[: self.num_log_reconstructions]
        preds_decoded = [self.tokenizer.decode(pred) for pred in preds]
        targets = targets[: self.num_log_reconstructions]
        targets_decoded = [self.tokenizer.decode(target) for target in targets]

        log_string = self.get_log_string(preds, targets, preds_decoded, targets_decoded)
        log_text(logging_prefix, log_string, trainer)

    def get_log_string(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
        preds_decoded: List[str],
        targets_decoded: List[str],
    ) -> str:
        log_string = f"\t{self.tokenizer.pad_token}: {self.tokenizer.pad_index} - "
        log_string += f"{self.tokenizer.unk_token}: {self.tokenizer.unk_index} - "
        log_string += f"{self.tokenizer.sos_token}: {self.tokenizer.sos_index} - "
        log_string += f"{self.tokenizer.eos_token}: {self.tokenizer.eos_index}\n\n"

        max_password_length = max([len(pwd) for pwd in preds_decoded + targets_decoded])
        max_index_length = len(str(len(self.tokenizer.vocab)))

        for pred_tensor, target_tensor, pred_string, target_string in zip(
            preds, targets, preds_decoded, targets_decoded
        ):
            pred_index_string = self.tensor_to_index_string(pred_tensor, max_index_length)
            target_index_string = self.tensor_to_index_string(target_tensor, max_index_length)

            log_string += f"\tTARGET: {target_string:<{max_password_length}} | {target_index_string}\n"
            log_string += f"\tRECON:  {pred_string:<{max_password_length}} | {pred_index_string}\n\n"
        return log_string

    def tensor_to_index_string(self, indices: torch.Tensor, max_index_length: int = 2) -> str:
        output = [f"{str(int(index)):>{max_index_length}}" for index in indices]
        return " ".join(output)


class SamplePasswords(pl.Callback):
    def __init__(
        self,
        tokenizer,
        num_samples: int,
        num_log_samples: int = 50,
        min_epoch: int = 1,
        sample_every: int = 1,
        batch_size: int = 128,
        compare_with: Optional[Sequence[str]] = ("valid", "test"),
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.num_log_samples = num_log_samples
        self.min_epoch = min_epoch
        self.sample_every = sample_every
        self.batch_size = batch_size

        self.compare_with = compare_with or []

        self.special_indices = [tokenizer.sos_index, tokenizer.eos_index, tokenizer.pad_index]

        self.samples: Optional[Set[Tuple]] = None
        self.valid_set: Optional[Set[Tuple]] = None
        self.test_set: Optional[Set[Tuple]] = None

    def password_tuple_set_from_dataset(self, dataset: TokenizedTextDataset, split: str = None) -> Set[Tuple]:
        desc_message = __name__ + f"Reading dataset {split} to tuples for password generation evaluation."
        tuples = [
            tuple([i for i in item.tolist() if i not in self.special_indices])
            for item, length in tqdm(dataset, desc=desc_message)
        ]
        set_tuples = set(tuples)
        logger.info(f"Size {len(set_tuples):,} unique passwords: {humanize.naturalsize(sys.getsizeof(set_tuples))}")
        return set_tuples

    def log_samples(
        self, samples: List[torch.Tensor], logging_prefix: str, trainer: pl.Trainer, pl_module: Model
    ) -> None:
        samples = samples[: self.num_log_samples]
        samples_decoded = [self.tokenizer.decode(sample) for sample in samples]

        log_string = self.get_log_string(samples, samples_decoded)
        log_text(logging_prefix, log_string, trainer)

    def get_log_string(self, samples: List[torch.Tensor], samples_decoded: List[str]) -> str:
        log_string = f"\t{self.tokenizer.pad_token}: {self.tokenizer.pad_index} - "
        log_string += f"{self.tokenizer.unk_token}: {self.tokenizer.unk_index} - "
        log_string += f"{self.tokenizer.sos_token}: {self.tokenizer.sos_index} - "
        log_string += f"{self.tokenizer.eos_token}: {self.tokenizer.eos_index}\n\n"

        max_password_length = max([len(pwd) for pwd in samples_decoded])
        max_index_length = len(str(len(self.tokenizer.vocab)))

        for sample_tensor, sample_string in zip(samples, samples_decoded):
            sample_index_string = self.tensor_to_index_string(sample_tensor, max_index_length)

            log_string += f"\t{sample_string:<{max_password_length}} | {sample_index_string}\n"
        return log_string

    def tensor_to_index_string(self, indices: torch.Tensor, max_index_length: int = 2) -> str:
        output = [f"{str(int(index)):>{max_index_length}}" for index in indices]
        return " ".join(output)

    def log_metrics(
        self,
        dataset_passwords: Set[Tuple],
        samples: Set[Tuple],
        logging_prefix: str,
        pl_module: Model,
        trainer: pl.Trainer,
    ) -> None:

        num_total = len(dataset_passwords)
        found = dataset_passwords.intersection(samples)
        num_found = len(found)
        ratio_found = num_found / num_total
        pl_module.log(logging_prefix + "/num_found", num_found)
        pl_module.log(logging_prefix + "/ratio_found", ratio_found)
        self.log_samples(
            [torch.tensor(sample) for sample in list(found)[: self.num_log_samples]],
            logging_prefix="samples_found",
            trainer=trainer,
            pl_module=pl_module,
        )

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Model) -> None:
        if trainer.current_epoch < self.min_epoch:
            return

        if trainer.current_epoch % self.sample_every != 0:
            self.samples = None
            return

        samples: List[Tuple] = []
        for _ in tqdm(
            range(self.num_samples // self.batch_size),
            desc=__name__ + f": Sampling passwords in batches of {self.batch_size}",
        ):
            batch_samples, batch_lens = pl_module.generate(self.batch_size)
            batch_samples = [
                tuple([index for index in batch_samples[i].tolist() if index not in self.special_indices])
                for i in range(len(batch_samples))
            ]
            samples.extend(batch_samples)
        logger.info(f"Sampled {len(samples):,} passwords after training epoch.")

        self.log_samples(
            [torch.tensor(sample) for sample in samples[: self.num_log_samples]],
            logging_prefix="samples",
            trainer=trainer,
            pl_module=pl_module,
        )

        self.samples = set(
            samples
        )  # this overrides the self.samples = None from init or the samples from the previous epoch
        unique_sample_ratio = len(self.samples) / len(samples)
        logger.info(f"Of {len(samples):,} samples, {len(self.samples):,} where unique ({unique_sample_ratio:.1%}).")
        pl_module.log("unique_samples", unique_sample_ratio)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Model) -> None:
        if "valid" not in self.compare_with:
            return

        if self.valid_set is None:
            self.valid_set = self.password_tuple_set_from_dataset(
                trainer.val_dataloaders[0].dataset.dataset, split="valid"
            )

        if self.samples is None:
            return
        self.log_metrics(self.valid_set, self.samples, logging_prefix="valid", pl_module=pl_module, trainer=trainer)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if "test" not in self.compare_with:
            return

        if self.test_set is None:
            self.test_set = self.password_tuple_set_from_dataset(
                trainer.test_dataloaders[0].dataset.dataset, split="test"
            )

        if self.samples is None:
            return
        self.log_metrics(self.test_set, self.samples, logging_prefix="test", pl_module=pl_module, trainer=trainer)


class WandbRemoveRun(pl.Callback):
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        if isinstance(exception, KeyboardInterrupt):
            print(
                """
             = = = = = = = = = = = = = = = = =
             THIS IS A KEYBOARDINTERRUPT

             WE CAN REMOVE THE WANDB RUN NOW
            = = = = = = = = = = = = = = = = =
            """
            )
