import json
import os
from typing import Callable, Dict, List, Optional, Tuple, overload

import tokenizers
import torch

from password_generation import project_path
from password_generation.tokenizers.base_tokenizer import Tokenizer


class WordPieceTokenizer(Tokenizer):
    def __init__(self, config_path: str):
        self.config_path = os.path.join(project_path, config_path)
        self.tokenizer = tokenizers.Tokenizer.from_file(self.config_path)

        tokenizer_vocab = self.tokenizer.get_vocab()
        tokenizer_vocab = sorted([x for x in tokenizer_vocab.items()], key=lambda x: x[1])
        self._vocab: List[str] = [item[0] for item in tokenizer_vocab]

    @property
    def config(self):
        with open(self.config_path, "r") as f:
            return json.loads(f.read())

    def encode(self, item: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(item).ids)

    def decode(self, item: torch.Tensor) -> str:
        return self.tokenizer.decode(item.tolist())

    @property
    def vocab(self):
        return self._vocab

    @classmethod
    def train(
        cls,
        train_files: List[str],
        vocab_size: int,
        output_path: str = None,
        min_frequency: int = 2,
        special_tokens: List[str] = None,
        limit_alphabet: int = 1000,
        wordpieces_prefix: str = "##",
        **kwargs
    ):
        if special_tokens is None:
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        tokenizer = tokenizers.BertWordPieceTokenizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False,
        )
        tokenizer.train(
            train_files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            wordpieces_prefix=wordpieces_prefix,
            **kwargs
        )
        if output_path is not None:
            output_path = os.path.join(project_path, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tokenizer.save(output_path, pretty=True)
            return cls(config_path=output_path)
        return tokenizer

    @classmethod
    def from_vocab(cls, vocab_path: str, output_path: Optional[str] = None):
        tokenizer = tokenizers.BertWordPieceTokenizer(
            vocab_path, clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=False
        )
        if output_path is not None:
            output_path = os.path.join(project_path, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tokenizer.save(output_path, pretty=True)
        return cls(config_path=output_path)


def default_tokenizer():
    from password_generation import project_path

    return WordPieceTokenizer(config_path=os.path.join(project_path, "data", "tokenizers", "bert-base-cased.json"))


def main():
    from password_generation import project_path, test_data_file

    tokenizer_config_path = os.path.join(project_path, "data", "tokenizers", "bert-base-cased.json")

    tokenizer = WordPieceTokenizer(config_path=tokenizer_config_path)
    tokenizer.tokenizer.model.encode("this is a test")


if __name__ == "__main__":
    main()
