from typing import *

import torch

from password_generation.tokenizers.base_tokenizer import Tokenizer
from password_generation.utils.valid_characters import password_characters

char = str


class CharTokenizer(Tokenizer):
    def __init__(
        self,
        characters: str = password_characters,
        add_sos_and_eos: bool = True,
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        self.characters = characters
        self.char_to_index: Dict[char, int] = self.init_char_to_index_mapping(characters)

        self.add_sos_and_eos = add_sos_and_eos
        self.unk_token = unk_token
        self.unk_index = len(self.char_to_index)
        self.char_to_index[self.unk_token] = self.unk_index
        self.sos_token = sos_token
        self.sos_index = len(self.char_to_index)
        self.char_to_index[self.sos_token] = self.sos_index
        self.eos_token = eos_token
        self.eos_index = len(self.char_to_index)
        self.char_to_index[self.eos_token] = self.eos_index
        self.pad_token = pad_token
        self.pad_index = len(self.char_to_index)
        self.char_to_index[self.pad_token] = self.pad_index

        self.index_to_char: List[char] = self.init_index_to_char_mapping(self.char_to_index)

        self.vocab_size = len(self.char_to_index)

    @property
    def vocab(self):
        return self.index_to_char

    def init_char_to_index_mapping(self, characters: str) -> Dict[char, int]:
        return {c: i for i, c in enumerate(characters)}

    def init_index_to_char_mapping(self, char_to_index_mapping: Dict[char, int]) -> List[char]:
        return list(char_to_index_mapping.keys())

    def split(self, text: str) -> List[char]:
        return list(text)

    @property
    def config(self) -> Dict:
        return {"name": "char", "characters": self.characters, "unk_token": self.unk_token, "unk_index": self.unk_index}

    def encode(self, text: str) -> torch.Tensor:
        indices: List[int] = [self.char_to_index.get(c, self.unk_index) for c in self.split(text)]
        if self.add_sos_and_eos:
            indices = [self.sos_index] + indices + [self.eos_index]
        return torch.tensor(indices)

    def decode(self, indices: torch.Tensor) -> str:
        chars = [
            self.index_to_char[index]
            for index in indices
            if index not in [self.sos_index, self.eos_index, self.pad_index]
        ]
        return "".join(chars)


def main():
    characters = "abcdefg"

    tokenizer = CharTokenizer(characters=characters)
    print(f"pad_token: {tokenizer.pad_token}")
    print(f"unk_token: {tokenizer.unk_token}")
    print(f"vocab_size: {tokenizer.vocab_size}")

    texts = ["abdefg", "abcdefgh", "aaaaaaaaaaaaaaaaaaaaa", "", "this is a test"]

    for text in texts:
        indices = tokenizer.encode(text)
        reconstructed_text = tokenizer.decode(indices)
        print(indices.shape)
        print(reconstructed_text)


if __name__ == "__main__":
    main()
