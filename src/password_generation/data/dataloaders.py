import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch.nn.utils.rnn
from torch.utils.data import DataLoader

from password_generation import project_path, test_data_file
from password_generation.data.datasets import TokenizedTextDataset
from password_generation.utils.helper import create_instance


class TokenizedTextDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer_config: Dict,
        batch_size: int,
        max_sequence_length: int,
        filter_config: Optional[Dict] = None,
        train_size: float = 0.9,
        valid_size: float = 0.1,
        test_size: float = 0.0,
        use_cache: bool = True,
        in_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        max_size: int = None,
        num_workers: Optional[int] = -1,
    ):
        super().__init__()
        data_path = os.path.join(
            project_path, data_path
        )  # absolute paths remain, relative paths are from project_path directory

        self.tokenizer = create_instance(tokenizer_config)
        self.filter = create_instance(filter_config) if filter_config is not None else None
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.num_workers = num_workers

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.train_ds, self.valid_ds, self.test_ds = TokenizedTextDataset.split(
            data_path=data_path,
            tokenizer=self.tokenizer,
            filter=self.filter,
            max_sequence_length=max_sequence_length,
            use_cache=use_cache,
            in_memory=in_memory,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            max_size=max_size,
        )

    @property
    def vocab_size(self):
        return self.train_ds.dataset.vocab_size

    @property
    def pad_index(self):
        return self.train_ds.dataset.pad_index

    @property
    def sos_index(self):
        return self.train_ds.dataset.sos_index

    @property
    def eos_index(self):
        return self.train_ds.dataset.eos_index

    @property
    def unk_index(self):
        return self.train_ds.dataset.unk_index

    def init_dataloader(self, ds: TokenizedTextDataset, shuffle: bool = True):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=partial(self.collate_text_batch, pad_index=ds.dataset.pad_index),
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        ds = self.train_ds
        return self.init_dataloader(ds)

    def val_dataloader(self):
        ds = self.valid_ds
        return self.init_dataloader(ds, shuffle=False)

    def test_dataloader(self):
        ds = self.test_ds
        return self.init_dataloader(ds, shuffle=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    @staticmethod
    def collate_text_batch(batch: List[Tuple[torch.Tensor, int]], pad_index: int) -> Tuple[torch.tensor, torch.tensor]:
        data = [x[0] for x in batch]
        lens = [x[1] for x in batch]
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_index), torch.tensor(lens)


def default_dataloader() -> TokenizedTextDataLoader:
    tokenizer_config = {"module": "password_generation.tokenizers.char_tokenizer", "class": "CharTokenizer", "args": {}}
    filter_config = {"module": "password_generation.filter", "class": "Filter", "args": {}}
    dataloader = TokenizedTextDataLoader(
        data_path=test_data_file,
        tokenizer_config=tokenizer_config,
        filter_config=filter_config,
        batch_size=4,
        max_sequence_length=13,
    )
    return dataloader


def main():
    dataloader = default_dataloader()
    for x in dataloader.train_dataloader():
        print(x.shape)


if __name__ == "__main__":
    main()
