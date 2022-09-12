from typing import Dict, Optional, Tuple

import torch

from password_generation.utils.helper import Embedding, OneHotEncoding


class Encoder(torch.nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        max_sequence_length: int,
        latent_dim: int,
        sos_index: int,
        eos_index: int,
        pad_index: int,
        unk_index: int,
        embedding: Optional[Embedding] = None,
        embedding_dim: Optional[int] = None,
        is_recurrent: bool = False,
    ):
        super().__init__()
        assert not (embedding is None and embedding_dim is None)
        self.embedding = embedding or torch.nn.Embedding(vocab_dim, embedding_dim)
        self.embedding_dim = self.embedding.embedding_dim

        self.latent_dim = latent_dim
        self.vocab_dim = vocab_dim
        self.max_sequence_length = max_sequence_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index

        self.is_recurrent = is_recurrent

    def forward(self, x: torch.LongTensor, lens: torch.LongTensor) -> torch.FloatTensor:
        ...


class Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        max_sequence_length: int,
        latent_dim: int,
        sos_index: int,
        eos_index: int,
        pad_index: int,
        unk_index: int,
        embedding: Optional[Embedding] = None,
        embedding_dim: Optional[int] = None,
        encoder: Optional[Encoder] = None,
        is_recurrent: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.vocab_dim = vocab_dim
        self.max_sequence_length = max_sequence_length
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index

        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.encoder = encoder

        self.is_recurrent = is_recurrent

    def forward(self, z: torch.FloatTensor, x: torch.LongTensor, lens: torch.LongTensor) -> torch.LongTensor:
        ...
