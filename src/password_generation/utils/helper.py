import copy
import itertools
from datetime import datetime as dt
from functools import partial, reduce
from importlib import import_module
from logging import Logger
from typing import Any, Dict, Generator, List, Union

import torch
import tqdm
import yaml

from .valid_characters import password_characters


def timestamp(format: str = "%Y-%m-%d-%H-%M") -> str:
    return dt.now().strftime(format)


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)


def create_class_instance(module_name, class_name, *args, **kwargs):
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    instance = clazz(*args, **kwargs)
    return instance


def create_instance(config, *args, **kwargs):
    return create_class_instance(config["module"], config["class"], *args, **config.get("args", {}), **kwargs)


def load_yaml_file(path: str):
    with open(path, "rb") as f:
        return load_yaml_string(f.read())


def load_yaml_string(stream: str) -> Dict:
    return yaml.full_load(stream)


class OneHotEncoding(object):
    def __init__(self, encoding_size: int):
        self.encoding_size = encoding_size

    def __call__(self, indexes: torch.LongTensor) -> torch.FloatTensor:
        one_hot = torch.nn.functional.one_hot(indexes, self.encoding_size)
        return one_hot

    @property
    def embedding_dim(self):
        return self.encoding_size


Embedding = Union[torch.nn.Embedding, OneHotEncoding]


def sample(dist: torch.Tensor, mode: str = "sample", unk_index: int = None) -> torch.Tensor:
    """
    Auxiliary sampling method.
    """
    if mode in ["sample-no-unk", "greedy-no-unk"] and unk_index is None:
        raise ValueError("Unknown index for the <unk> token!")
    if mode == "greedy":
        _, _sample = torch.topk(dist, 1, dim=-1)
    elif mode == "sample":
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == "sample-no-unk":
        # reduce chances for <unk>
        dist[:, :, unk_index] = dist.min()
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == "greedy-no-unk":
        # prevent <unk>
        dist[:, :, unk_index] = dist.min()
        _, _sample = torch.topk(dist, 1, dim=-1)
    else:
        raise ValueError(f"Unknown sampling mode = {mode}")

    _sample = _sample.squeeze()
    return _sample
