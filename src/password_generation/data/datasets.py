import csv
import json
import logging
import math
import os
from typing import *

import h5py
import torch
from torch.utils.data import Dataset, IterableDataset, random_split

from password_generation import project_path, test_data_file
from password_generation.filter import Filter
from password_generation.tokenizers import Tokenizer
from password_generation.utils.file_operations import (
    get_disk_usage,
    get_file_size,
    hash_dict,
)
from password_generation.utils.helper import create_instance
from password_generation.utils.logging import tqdm

logger = logging.getLogger(__name__)


class RawTextDataset(IterableDataset):
    def __init__(self, data_path: str):
        self._len = None
        self.data_path: str = data_path
        self.open_data_file = open(self.data_path, "r", errors="ignore")

    def __iter__(self):
        for line in self.open_data_file:
            yield line.strip()
        self.reset_data_file()

    def reset_data_file(self):
        self.open_data_file.close()
        self.open_data_file = open(self.data_path, "r", errors="ignore")

    def get_length(self) -> int:
        i = 0
        with open(self.data_path, "r", errors="ignore") as f:
            for _ in f:
                i += 1
        return i

    def __len__(self):
        if self._len is None:
            self._len = self.get_length()
        return self._len

    def __getitem__(self, item):
        raise NotImplementedError


class TokenizedTextDataset:
    def __init__(
        self,
        data_path,
        max_sequence_length: int,
        tokenizer: Union[Tokenizer, Dict],
        filter: Optional[Union[Filter, Dict]] = None,
        use_cache: bool = True,
        in_memory: bool = True,
        max_size: Optional[int] = None,
    ):
        self.data_path = data_path

        assert in_memory or use_cache, "You need to set in_memory or use_cache or both True."

        if isinstance(tokenizer, dict):
            tokenizer = create_instance(tokenizer)
        self.tokenizer = tokenizer
        if isinstance(filter, dict):
            filter = create_instance(
                filter, max_length=max_sequence_length - 2
            )  # max length for pwd: max_sequence_length - 2 because of sos and eos
        self.filter = filter or Filter()

        self.max_size = max_size
        self.max_sequence_length = self.filter.max_length or max_sequence_length
        assert self.max_sequence_length is not None

        self.vocab_size = len(self.vocab)
        try:
            self.pad_index = self.tokenizer.pad_index
        except AttributeError:
            self.pad_index = len(self.vocab)
            self.vocab_size += 1
        try:
            self.sos_index = self.tokenizer.sos_index
        except AttributeError:
            self.sos_index = len(self.vocab)
            self.vocab_size += 1
        try:
            self.eos_index = self.tokenizer.eos_index
        except AttributeError:
            self.eos_index = len(self.vocab)
            self.vocab_size += 1
        try:
            self.unk_index = self.tokenizer.unk_index
        except AttributeError:
            self.unk_index = len(self.vocab)
            self.vocab_size += 1

        self.data, self.data_file, self.raw_text_dataset, cache_path = self.init_data(data_path, use_cache, in_memory)
        if cache_path is not None and not os.path.isfile(cache_path):
            self.cache_data(cache_path, cache_config=self.config, batch_size=1000)
            if self.data is None:  # reopen cache file to read data from there instead of raw_text_dataset
                self.data_file = h5py.File(cache_path, "r")

    def __getitem__(self, index) -> Tuple[torch.LongTensor, int]:
        if index > self.__len__():
            raise IndexError
        if self.data is not None:
            item = self.data[index]
        elif self.data_file is not None:
            item = torch.tensor(self.data_file["data"][index])
        else:
            for _item in self.raw_text_dataset:
                if self.filter(_item):
                    item = self.tokenize(_item)
                    break
        item = item[: self.max_sequence_length]
        return item.type(torch.LongTensor), len(item)

    def __len__(self):
        if self.data is not None:
            return min(len(self.data), self.max_size or math.inf)
        elif self.data_file is not None:
            return min(self.data_file["data"].shape[0], self.max_size or math.inf)
        elif self.raw_text_dataset is not None:
            return min(len(self.raw_text_dataset), self.max_size or math.inf)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def init_data(
        self, data_path, use_cache: bool = False, in_memory: bool = False
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[h5py.File], Optional[RawTextDataset], Optional[str]]:
        data: Optional[List[torch.Tensor]] = None
        data_file: Optional[h5py.File] = None
        raw_text_dataset: Optional[RawTextDataset] = None
        cache_path: Optional[str] = None

        if use_cache:
            cache_path = self.init_cache_path()
        if cache_path is not None and os.path.isfile(cache_path):
            data_file = h5py.File(cache_path, "r")
            if in_memory:
                data = self.read_from_cache(data_file)
                data_file.close()
                data_file = None
        else:
            raw_text_dataset = RawTextDataset(data_path)
            if in_memory:
                data = [item for item in self.read_data_from_raw_text(raw_text_dataset)]
        return data, data_file, raw_text_dataset, cache_path

    def read_data_from_raw_text(self, raw_text_dataset: RawTextDataset) -> Iterator[torch.Tensor]:
        total_length = min(len(raw_text_dataset), self.max_size or math.inf)
        for index, item in tqdm(
            enumerate(raw_text_dataset),
            total=total_length,
            desc="Reading and tokenizing all data from raw text into memory",
        ):
            if self.filter(item):
                yield self.tokenize(item)
                if index >= (self.max_size or math.inf):
                    break

    def init_cache_path(self) -> str:
        cache_file = hash_dict(self.config)
        cache_directory = os.path.join(os.path.dirname(self.data_path), "cache")
        os.makedirs(cache_directory, exist_ok=True)
        return os.path.join(cache_directory, cache_file)

    def read_from_cache(self, data_file, remove_padding: bool = True) -> List[torch.Tensor]:
        data = list(torch.tensor(data_file["data"][...]))
        lens = list(torch.tensor(data_file["lens"][...]))
        if remove_padding:
            data = [
                item[:len]
                for item, len in tqdm(
                    zip(data, lens), total=len(data), desc=__name__ + ": Removing padding from cached data"
                )
            ]
        return data

    @property
    def config(self):
        config = {
            "data_file": self.data_path,
            "max_sequence_length": self.max_sequence_length,
            "tokenizer": self.tokenizer.config,
            "filter_config": self.filter.config,
            "max_size": self.max_size,
        }
        return config

    @property
    def vocab(self):
        return self.tokenizer.vocab

    def tokenize(self, item: str) -> torch.Tensor:
        return self.tokenizer.encode(item)

    def create_cache_file(self, cache_path):
        logger.info("No cache file found. Writing all data to cache.")
        dir_path, total, used, free = get_disk_usage(cache_path)
        logger.info(f"Current disk usage in {dir_path}:\n" f"Total: {total}GB\n" f"Used: {used}GB\n" f"Free: {free}GB")
        self.cache_data(cache_path)
        dir_path, total, used, free = get_disk_usage(cache_path)
        logger.info(
            f"Written {get_file_size(cache_path)}GB to {cache_path}.\n"
            f"Current disk usage in {dir_path}:\n"
            f"Total: {total}GB\n"
            f"Used: {used}GB\n"
            f"Free: {free}GB"
        )

    @staticmethod
    def check_data_file(cache_path: str) -> None:
        file = h5py.File(cache_path, "r")
        try:
            data = file["data"][0]
        except (IndexError, KeyError) as e:
            logger.info(f"No cached data found in file {cache_path}")
            raise e
        logger.info(f"Found data of size {data.shape} in file {cache_path}")

    def cache_data(self, cache_path, cache_config: Dict = None, batch_size: int = 1000):
        def pad_to_tensor(batch_items: List[torch.Tensor], max_len: int, padding_value: int = -1):
            padded_batch = torch.zeros(len(batch_items), max_len).fill_(padding_value)
            for index, item in enumerate(batch_items):
                padded_batch[index, : len(item)] = item
            return padded_batch

        logger.info(f"Starting cache process for file {cache_path}.")
        with h5py.File(cache_path, "w") as file:
            for i in tqdm(range(0, len(self), batch_size), desc="Caching data in batches"):
                batch: List[Tuple] = [self[j] for j in range(i, min(len(self), i + batch_size))]
                batch_items: List[torch.Tensor] = [x[0] for x in batch]
                batch_lens: List[int] = [x[1] for x in batch]
                batch_padded: torch.Tensor = pad_to_tensor(batch_items, self.max_sequence_length, self.pad_index)

                if "data" not in file:
                    file.create_dataset(
                        "data", data=batch_padded, chunks=True, maxshape=(None,) + batch_padded.shape[1:]
                    )
                    file.create_dataset("lens", data=batch_lens, chunks=True, maxshape=(None,))
                else:
                    file["data"].resize(file["data"].shape[0] + batch_padded.shape[0], axis=0)
                    file["data"][-batch_padded.shape[0] :] = batch_padded
                    file["lens"].resize(file["lens"].shape[0] + len(batch_lens), axis=0)
                    file["lens"][-len(batch_lens) :] = batch_lens

            file.create_dataset("vocab", dtype=h5py.special_dtype(vlen=str), data=self.vocab)

        if cache_config is not None:
            cache_config_path = cache_path + ".json"
            if not os.path.exists(cache_config_path):
                with open(cache_path + ".json", "w") as f:
                    f.write(json.dumps(cache_config))
                logger.info(f"Wrote cache config to {cache_path + '.json'}")

    @classmethod
    def split(cls, *args, train_size: float = 0.8, valid_size: float = 0.1, test_size: float = 0.1, **kwargs):
        dataset = cls(*args, **kwargs)
        n = len(dataset)
        n_train = int(n * train_size)
        n_valid = int(n * valid_size)
        n_test = n - n_train - n_valid
        train, valid, test = random_split(dataset, lengths=[n_train, n_valid, n_test])
        return train, valid, test


def main():
    from password_generation.filter import Filter
    from password_generation.tokenizers.char_tokenizer import (
        default_tokenizer as default_char_tokenizer,
    )
    from password_generation.tokenizers.wordpiece_tokenizer import (
        default_tokenizer as default_wordpiece_tokenizer,
    )
    from password_generation.utils.valid_characters import password_characters

    tokenizer = default_char_tokenizer()
    filter = Filter()

    dataset = TokenizedTextDataset(
        test_data_file, tokenizer=tokenizer, filter=filter, max_sequence_length=15, use_cache=True, in_memory=True
    )
    train, valid = random_split(dataset, lengths=(len(dataset) - 10_000, 10_000))
    i = 0
    for x in train:
        print(x)
        if i > 5:
            break
        i += 1


if __name__ == "__main__":
    main()
