import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from password_generation.blocks import Decoder, Encoder
from password_generation.utils.helper import Embedding, OneHotEncoding
from password_generation.utils.helper import sample as sample_dist

logger = logging.getLogger()


class GPT2Encoder(Encoder):
    def __init__(
        self,
        vocab_dim: int,
        max_sequence_length: int,
        latent_dim: int,  # we ignore latent dim, encoder latent dim = 2 * embedding_dim
        sos_index: int,
        eos_index: int,
        pad_index: int,
        unk_index: int,
        embedding: Optional[Embedding] = None,
        embedding_dim: Optional[int] = None,
        n_layer: int = 12,
        n_head: int = 12,
        **kwargs,
    ):
        super().__init__(
            vocab_dim=vocab_dim,
            max_sequence_length=max_sequence_length,
            latent_dim=latent_dim,
            sos_index=sos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            unk_index=unk_index,
            embedding=embedding,
            embedding_dim=embedding_dim,
            is_recurrent=True,
        )

        transformer_config = init_gpt2config(
            vocab_size=self.vocab_dim,
            n_positions=self.max_sequence_length,
            n_ctx=self.max_sequence_length,
            n_embd=self.embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=self.sos_index,  # set by Encoder superclass
            eos_token_id=self.eos_index,
            pad_token_id=self.pad_index,
            **kwargs,
        )
        self.transformer_model = GPT2Model(transformer_config)

        self.hidden_state = None
        # output dimension of transformer module is self.embedding_dim
        # bidirectional lstm doubles output dim
        self.rnn_cell = nn.LSTM(
            input_size=self.embedding_dim, hidden_size=self.embedding_dim, bidirectional=True, batch_first=True
        )
        self.latent_dim = 2 * self.embedding_dim

    def forward(self, x: torch.LongTensor, lens: torch.LongTensor) -> torch.Tensor:
        input_ids = x  # [batch_size, sequence_length]

        # Use own embedding, do not embed with transformer model
        # so we can use the same embedding in the decoder
        if isinstance(self.embedding, OneHotEncoding):
            transformer_outputs = self.transformer_model(input_ids=input_ids)
        else:
            input_embeddings = self.embedding(input_ids)  # [batch_size, sequence_length, embedding_dim]
            transformer_outputs = self.transformer_model(inputs_embeds=input_embeddings)

        # get first element of output tuple
        x = transformer_outputs[0]  # [batch_size, sequence_length, embedding_dim]

        # RNN (bi-directional LSTM):
        out, self.hidden_state = self.rnn_cell(
            x, self.hidden_state
        )  # [batch_size, sequence_length, 2*embedding_dim], [2, 2, batch_size, embedding_dim]

        h = self.hidden_state[0]  # [2, batch_size, embedding_dim]
        h = torch.cat((h[0], h[1]), dim=1)  # Merge h for both directions [batch_size, 2*embedding_dim]
        return h

    def initialize_hidden_state(self, batch_size: int, device: torch.device):
        h = torch.randn(2, batch_size, self.latent_dim // 2).to(device)
        c = torch.randn(2, batch_size, self.latent_dim // 2).to(device)
        self.hidden_state = (h, c)

    def reset_history(self):
        self.hidden_state = tuple(x.detach() for x in self.hidden_state)

    @property
    def output_size(self):
        return self.__output_size


class GPT2Decoder(Decoder):
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
        encoder: Optional[Encoder] = None,
        n_layer: int = 12,
        n_head: int = 12,
        token_dropout_rate: float = 0.0,
        embedding_dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            vocab_dim=vocab_dim,
            max_sequence_length=max_sequence_length,
            latent_dim=latent_dim,
            sos_index=sos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            unk_index=unk_index,
            embedding=embedding,
            embedding_dim=None,
            encoder=encoder,
        )
        self.assert_embedding_not_onehot(self.embedding)

        self.token_dropout_rate = token_dropout_rate
        self.embedding_dropout_layer = (
            torch.nn.Dropout(p=embedding_dropout_rate) if embedding_dropout_rate > 0.0 else None
        )

        # If an encoder is passed to the init, we use the encoder weights in the
        # decoder and train both at the same time.
        # The encoder only takes input of dimension embedding_dim, but the
        # decoder receives embedding_dim + latent_dim. We therefore need another
        # mapping layer before passing to the transformer model.
        self.shared_encoder_weights: bool = encoder is not None
        if self.shared_encoder_weights:
            transformer_embedding_dim = self.embedding.embedding_dim  # We have to use the same dim as the encoder
            self.encoder_mapping_layer = nn.Linear(
                self.embedding.embedding_dim + self.latent_dim, self.embedding.embedding_dim
            )
        else:
            transformer_embedding_dim = self.embedding.embedding_dim + self.latent_dim
            self.encoder_mapping_layer = None

        transformer_config = init_gpt2config(
            vocab_size=vocab_dim,
            n_positions=max_sequence_length,
            n_ctx=max_sequence_length,
            n_embd=transformer_embedding_dim,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=self.sos_index,
            eos_token_id=self.eos_index,
            pad_token_id=self.pad_index,
            **kwargs,
        )
        # sets up a new transformer model and a linear mapping layer to the vocab
        self.transformer_model = GPT2LMHeadModel(transformer_config)

        if self.shared_encoder_weights:
            # replace the internal transformer model with the encoder
            self.transformer_model.transformer = encoder.transformer_model

    @staticmethod
    def assert_embedding_not_onehot(embedding):
        if isinstance(embedding, OneHotEncoding):
            raise AssertionError(
                "Embedding passed to GPT2Decoder can not be OneHotEncoding:\n"
                "If encoder embedding is OneHotEncoding, GPT2Encoder embedding is used. "
                "Pass this embedding to GPT2Decoder.\n"
                "If encoder embedding is custom model embedding, pass this embedding to GPT2Decoder."
            )

    def forward(self, z: torch.Tensor, x: torch.LongTensor, lens: torch.LongTensor):
        batch_size, sequence_length = x.shape
        x = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]

        if z is None:
            z = torch.zeros(batch_size, sequence_length, self.latent_dim, requires_grad=True).to(x.device)
        else:
            z = z.unsqueeze(1)
            z = z.repeat(1, sequence_length, 1)  # [batch_size, sequence_length, latent_dim]
        x = torch.cat((x, z), dim=-1)  # [batch_size, sequence_length, embedding_dim + latent_dim]

        if self.shared_encoder_weights:
            x = self.encoder_mapping_layer(x)  # [batch_size, sequence_length, D]

        if self.embedding_dropout_layer is not None:
            x = self.embedding_dropout_layer(x)

        # outputs = (loss), lm_logits, presents, (all hidden_states), (attentions)
        transformer_outputs = self.transformer_model(inputs_embeds=x)
        logits = transformer_outputs[0]  # [batch_size, sequence_length, V]
        return logits

    def generate(self, z: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Autoregressive sampling from the model.
        """
        batch_size = z.size(0)  # [batch_size, latent_dim]

        # accumulate tokens
        t = 0

        input_sequence = torch.LongTensor(batch_size, 1).fill_(self.sos_index).to(z.device)  # [batch_size, 1]
        running_sequences = torch.arange(0, batch_size)
        generated = torch.LongTensor(batch_size, self.max_sequence_length).fill_(self.pad_index)
        sequence_lengths = torch.zeros(batch_size, dtype=torch.long)

        while t < self.max_sequence_length and len(running_sequences) > 0:
            logits = self.forward(
                x=input_sequence, z=z, lens=[t + 1] * input_sequence.shape[0]
            )  # [batch_size, t, vocab_dim]
            logits = logits[:, t, :]  # [batch_size, 1, vocab_dim]  -  last prediction only

            new_indices: torch.LongTensor = sample_dist(logits, "sample").type_as(generated)
            generated[running_sequences, t] = new_indices
            sequence_lengths[running_sequences] += 1

            running_mask = new_indices != self.eos_index
            if not running_mask.shape:
                running_mask = running_mask.unsqueeze(0)  # only one element remaining, running mask was 0-dimensional
            running_sequences = running_sequences.masked_select(running_mask)

            input_sequence = torch.cat([input_sequence, new_indices.view(-1, 1).to(z.device)], dim=1)
            input_sequence = input_sequence[running_mask]
            z = z[running_mask]

            t += 1
        return generated, sequence_lengths

    @staticmethod
    def _save_sample(save_to: torch.LongTensor, sample: torch.LongTensor, running_seqs: torch.LongTensor, t: int):
        """
        Auxiliary sampling method.
        """
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:, t] = sample
        # save back
        save_to[running_seqs] = running_latest

        return save_to


def init_gpt2config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    activation_function="gelu_new",
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    summary_type="cls_index",
    summary_use_proj=True,
    summary_activation=None,
    summary_proj_to_labels=True,
    summary_first_dropout=0.1,
    bos_token_id=50256,
    eos_token_id=50256,
    pad_token_id=50256,
) -> GPT2Config:
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        activation_function=activation_function,
        resid_pdrop=resid_pdrop,
        embd_pdrop=embd_pdrop,
        attn_pdrop=attn_pdrop,
        layer_norm_epsilon=layer_norm_epsilon,
        initializer_range=initializer_range,
        summary_type=summary_type,
        summary_use_proj=summary_use_proj,
        summary_activation=summary_activation,
        summary_proj_to_labels=summary_proj_to_labels,
        summary_first_dropout=summary_first_dropout,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def main():
    batch_size = 4
    sequence_length = 10
    latent_dim = 16
    embedding_dim = 24
    vocab_dim = 20

    embedding = torch.nn.Embedding(vocab_dim, embedding_dim)
    encoder = GPT2Encoder(
        vocab_dim=vocab_dim,
        max_sequence_length=sequence_length,
        latent_dim=latent_dim,
        embedding=embedding,
        embedding_dim=None,
        pad_index=0,
        eos_index=1,
        sos_index=2,
        unk_index=3,
        n_layer=4,
        n_head=4,
    )

    x = torch.randint(low=0, high=vocab_dim - 1, size=(batch_size, sequence_length))

    decoder = GPT2Decoder(
        vocab_dim=vocab_dim,
        max_sequence_length=sequence_length,
        n_layer=12,
        n_head=12,
        latent_dim=latent_dim,
        pad_index=0,
        eos_index=1,
        sos_index=2,
        unk_index=3,
        encoder=encoder,
        embedding=embedding,
    )

    z = encoder.forward(x, lens=None)
    lens = torch.tensor([sequence_length] * batch_size)
    print(decoder(z, x, lens).shape)


if __name__ == "__main__":
    main()
