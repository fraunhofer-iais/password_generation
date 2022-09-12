from typing import Any, Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from password_generation.models import Model
from password_generation.tokenizers import Tokenizer


def generate_password_batch(model: Model, tokenizer: Tokenizer, n: int) -> List[str]:
    batch_indices, batch_lengths = model.generate(n)
    passwords: List[str] = [tokenizer.decode(indices)[:length] for indices, length in zip(batch_indices, batch_lengths)]
    return passwords


def generate_passwords(model: Model, tokenizer: Tokenizer, n: int, batch_size: int = 1000) -> Iterable[List[str]]:
    for i in tqdm(range(0, n, batch_size)):
        yield generate_password_batch(model, tokenizer, n=batch_size)
