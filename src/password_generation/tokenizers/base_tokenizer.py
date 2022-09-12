from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class Padding(Enum):
    max_len = 1
    longest = 2


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, item: str) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, item: torch.Tensor) -> str:
        ...

    @property
    @abstractmethod
    def config(self):
        ...

    @property
    @abstractmethod
    def vocab(self):
        ...
