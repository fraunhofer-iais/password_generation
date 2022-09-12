import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

from password_generation.blocks import Decoder, Encoder
from password_generation.models.base_model import Model
from password_generation.models.loss import kl_divergence
from password_generation.utils.helper import OneHotEncoding, create_instance

logger = logging.getLogger()


class Output(TypedDict):
    loss: Optional[torch.FloatTensor]
    pred: Optional[torch.LongTensor]
    target: Optional[torch.LongTensor]
    samples: Optional[torch.LongTensor]


Loss = torch.FloatTensor
Predictions = torch.FloatTensor


class AE(Model):
    """
    Standard Deterministic Autoencoder
    """

    def __init__(
        self,
        vocab_dim: int,
        latent_dim: int,
        embedding_dim: int,
        max_sequence_length: int,
        encoder: Union[Dict, Encoder],
        decoder: Union[Dict, Decoder],
        sos_index: int,
        eos_index: int,
        pad_index: int,
        unk_index: int,
        train_embeddings: bool = True,
        coupled_decoder: bool = False,
        learning_rate: float = 0.01,
        learning_rate_scheduler: Optional[Dict] = None,
        parameter_schedulers: Optional[Dict[str, Dict]] = None,
        model_weights: Optional[os.PathLike] = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            parameter_schedulers=parameter_schedulers,
            model_weights=model_weights,
        )
        self.vocab_dim = vocab_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.train_embeddings = train_embeddings
        self.max_sequence_length = max_sequence_length

        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index
        self.unk_index = unk_index

        if not train_embeddings:
            self.embedding = OneHotEncoding(self.vocab_dim)
        else:
            self.embedding = torch.nn.Embedding(self.vocab_dim, self.embedding_dim, self.pad_index)
        self.embedding_dim = self.embedding.embedding_dim

        if isinstance(encoder, dict):
            self.encoder = create_instance(
                encoder,
                vocab_dim=self.vocab_dim,
                max_sequence_length=self.max_sequence_length,
                latent_dim=self.latent_dim,
                sos_index=self.sos_index,
                eos_index=self.eos_index,
                pad_index=self.pad_index,
                unk_index=self.unk_index,
                embedding_dim=self.embedding_dim,
                embedding=self.embedding,
            )
        else:
            self.encoder = encoder

        if isinstance(decoder, dict):
            self.decoder = create_instance(
                decoder,
                vocab_dim=self.vocab_dim,
                max_sequence_length=self.max_sequence_length,
                latent_dim=self.latent_dim,
                sos_index=self.sos_index,
                eos_index=self.eos_index,
                pad_index=self.pad_index,
                unk_index=self.unk_index,
                embedding=self.embedding,
                encoder=self.encoder if coupled_decoder else None,
            )
        else:
            self.decoder = decoder

    def forward(self, x: torch.Tensor, lens: torch.LongTensor):
        z = self.encoder(x, lens)
        logits = self.decoder(z, x, lens)  # [B, T, V]
        logits = logits.contiguous().view(-1, self.vocab_dim)  # [T * B, V]
        return logits

    def loss(self, y_pred: torch.Tensor, y_target: torch.Tensor, lens: torch.LongTensor) -> torch.Tensor:
        loss = self.cross_entropy_loss(y_pred, y_target, lens)
        return loss

    def cross_entropy_loss(
        self, y_pred: torch.Tensor, y_target: torch.LongTensor, lens: torch.LongTensor
    ) -> torch.Tensor:
        batch_size = lens.shape[0]
        max_len = torch.max(lens).item()
        loss = torch.nn.functional.cross_entropy(y_pred, y_target, ignore_index=self.pad_index, reduction="none")
        mask = (y_target.view(batch_size, -1) != self.pad_index).float()
        loss = loss.view(batch_size, -1) * (mask.float() / (lens.view(batch_size, 1).float()))
        loss = loss.sum() / batch_size
        return loss

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], logging_prefix: str = "train") -> Output:
        x, lens = batch
        batch_size, sequence_length = x.shape
        lens, sort_indices = torch.sort(lens, descending=True)
        x = x[sort_indices]
        y = x[1:].view(-1)

        self.initialize_hidden_state(batch_size)

        # Train loss
        logits = self.forward(x, lens)
        loss: torch.Tensor = self.loss(y_pred=logits.view(y.shape[0], -1), y_target=y, lens=lens)

        # Detach history from rnn models
        self.detach_history()

        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True)

        pred = logits.argmax(dim=1).view(batch_size, -1)
        return {"loss": loss, "pred": pred}

    def validate_step(self, batch: Tuple[torch.tensor, torch.tensor]) -> Output:
        return self.training_step(batch)

    def test_step(self, batch: torch.Tensor) -> Output:
        return self.training_step(batch)

    def on_train_batch_start(self, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        self.initialize_hidden_state(batch[0].shape[0])

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        # Detach history from rnn models
        self.detach_history()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.initialize_hidden_state(batch[0].shape[0])

    def on_validation_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.detach_history()

    def initialize_hidden_state(self, batch_size: int, enc: bool = True, dec: bool = True):
        if enc and self.encoder.is_recurrent:
            self.encoder.initialize_hidden_state(batch_size, self.device)
        if dec and self.decoder.is_recurrent:
            self.decoder.initialize_hidden_state(batch_size, self.device)

    def detach_history(self, enc: bool = True, dec: bool = True):
        if self.encoder.is_recurrent and enc:
            self.encoder.reset_history()
        if self.decoder.is_recurrent and dec:
            self.decoder.reset_history()

    def generate(self, n: int, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def predict_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass


class VAE(AE):
    def __init__(
        self,
        vocab_dim: int,
        latent_dim: int,
        embedding_dim: int,
        encoder: Dict,
        decoder: Dict,
        sos_index: int,
        eos_index: int,
        pad_index: int,
        unk_index: int,
        max_sequence_length: int,
        train_embeddings: bool = True,
        coupled_decoder: bool = False,
        learning_rate: float = 0.01,
        learning_rate_scheduler: Optional[Dict] = None,
        parameter_schedulers: Optional[Dict[str, Dict]] = None,
        model_weights: Optional[os.PathLike] = None,
        prior_mean: torch.Tensor = None,
        prior_var: torch.Tensor = None,
        temperature: float = 1.0,
        drop_char_rate: float = 0.0,
        beta: float = 1.0,
        min_logvar: float = -20,
        **kwargs,
    ):
        super().__init__(
            vocab_dim=vocab_dim,
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            train_embeddings=train_embeddings,
            coupled_decoder=coupled_decoder,
            encoder=encoder,
            decoder=decoder,
            sos_index=sos_index,
            eos_index=eos_index,
            pad_index=pad_index,
            unk_index=unk_index,
            max_sequence_length=max_sequence_length,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            parameter_schedulers=parameter_schedulers,
            model_weights=model_weights,
        )
        self.prior_mean = prior_mean or torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_sigma = prior_var or torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_sigma)
        self.temperature = temperature
        self.drop_char_rate = drop_char_rate

        self.latent_to_mean = torch.nn.Linear(self.encoder.latent_dim, self.latent_dim)
        self.latent_to_logvar = torch.nn.Linear(self.encoder.latent_dim, self.latent_dim)

        self.beta = beta
        self.min_logvar = min_logvar

        self.init_model_weights()

    def latent_to_sigma(self, z: torch.Tensor) -> torch.Tensor:
        logvar = self.latent_to_logvar(z)
        if self.min_logvar is not None:
            logvar = logvar.clamp(min=self.min_logvar)
        return torch.exp(0.5 * logvar)

    @property
    def hidden_dim(self):
        return self.decoder.embedding_dim

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        """
        returns: loss and KL divergences of VAE
        input & target shape: [B, T]
        Notation. B: batch size; T: seq len (== fix_len); V: voc size
        """
        if self.drop_char_rate > 0:
            x = self.drop_chars(x, lens)

        z: torch.Tensor = self.encoder(x, lens)
        mean, sigma = self.latent_to_mean(z), self.latent_to_sigma(z)
        z = torch.randn_like(mean, requires_grad=False)
        z = mean + z * sigma

        logits = self.decoder(z=z, x=x, lens=lens)  # [B, T, V]
        return logits, mean, sigma

    def drop_chars(self, x, lens):
        # randomly replace decoder input with unk
        # drop_char_rate gives probability to replace ~one index each batch with unk
        prob = torch.rand(*x.shape, device=self.device)
        prob[(x == self.sos_index) | (x == self.pad_index)] = 1.0

        x = x.clone()
        x[prob < (self.drop_char_rate / lens.unsqueeze(-1).float())] = self.unk_index
        return x

    def loss(
        self,
        y_pred: torch.FloatTensor,
        y_target: torch.LongTensor,
        lens: torch.LongTensor,
        mean: torch.FloatTensor,
        sigma: torch.FloatTensor,
        beta: Optional[torch.Tensor] = None,
    ):
        if "beta" in self.parameter_schedulers:
            beta = self.parameter_schedulers["beta"](self.global_step)
            self.log("beta", beta)
        else:
            beta = self.beta

        reconstruction_loss = self.cross_entropy_loss(y_pred, y_target, lens)
        distance_loss = kl_divergence(mean, sigma)
        loss = reconstruction_loss + beta * distance_loss

        return loss, reconstruction_loss, distance_loss

    def encode(self, input_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B = input_seq.size(0)
            self.initialize_hidden_state(B, self.device)
            z, m, s = self.encoder((input_seq, seq_len))
        return z, m, s

    def process_batch(self, batch, logging_prefix) -> Output:
        x, lens = batch
        batch_size, sequence_length = x.shape
        lens, sort_indices = torch.sort(lens, descending=True)
        x = x[sort_indices]
        y = x[:, 1:]
        x = x[:, :-1]

        # Train loss
        logits, mean, sigma = self.forward(x, lens)
        logits = logits.contiguous().view(-1, self.vocab_dim)  # [T * B, V]
        loss, reconstruction_loss, distance_loss = self.loss(
            y_pred=logits.view(-1, self.vocab_dim),
            y_target=y.reshape(-1),
            lens=lens.reshape(batch_size, 1),
            mean=mean,
            sigma=sigma,
            beta=None,
        )

        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True)
        self.log(f"{logging_prefix}/reconstruction_loss", reconstruction_loss, on_step=True, on_epoch=True)
        self.log(f"{logging_prefix}/distance_loss", distance_loss, on_step=True, on_epoch=True)

        pred = logits.argmax(dim=1).view(batch_size, -1)
        return {"loss": loss, "pred": pred, "target": y, "samples": None}

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], *args, **kwargs) -> Output:
        return self.process_batch(batch, logging_prefix="train")

    def validation_step(self, batch: Tuple[torch.tensor, torch.LongTensor], *args, **kwargs) -> Output:
        output = self.process_batch(batch, logging_prefix="valid")
        return output

    def sample_sequences(self, batch_size) -> torch.LongTensor:
        z: torch.Tensor = self.sample_latent(batch_size)
        generated_sequences, _ = self.decoder.generate(z)
        return generated_sequences

    def test_step(self, batch: torch.Tensor, *args, **kwargs) -> Output:
        output = self.process_batch(batch, logging_prefix="test")
        return output

    def predict_step(self, batch_size) -> Output:
        generated_sequences = self.sample_sequences(batch_size)
        return {"loss": None, "pred": None, "target": None, "samples": generated_sequences}

    def sample_latent(self, batch_size: int) -> torch.Tensor:
        return self.prior.sample((batch_size,)).to(self.device)

    def generate(self, n: int, **kwargs) -> torch.Tensor:
        z = self.sample_latent(n)
        return self.decoder.generate(z)


def main():
    embedding_dim = 16
    vocab_dim = 10
    latent_dim = 8
    batch_size = 4
    sequence_length = 13
    model = AE(
        vocab_dim=10,
        sos_index=0,
        eos_index=1,
        pad_index=2,
        unk_index=3,
        latent_dim=12,
        embedding_dim=16,
        train_embeddings=True,
        encoder={
            "module": "password_generation.models.mlp",
            "class": "MLPEncoder",
            "args": {"input_dim": embedding_dim, "output_dim": latent_dim},
        },
        decoder={
            "module": "password_generation.models.mlp",
            "class": "MLPDecoder",
            "args": {"latent_dim": latent_dim, "vocab_dim": vocab_dim, "sequence_length": sequence_length},
        },
        coupled_decoder=False,
    )

    x = torch.randint(low=0, high=vocab_dim - 1, size=(batch_size, sequence_length))
    print(model.forward(x).shape)


if __name__ == "__main__":
    main()

    # def generate(self, n: int, max_len: int = 12, tau: float = 1.0) -> torch.Tensor:
    #     if max_len is None:
    #         max_len = self.fix_len
    #
    #     normal = MultivariateNormal(torch.zeros(self.latent_dim), torch.eye(self.latent_dim))
    #     z = normal.sample([n])
    #
    #     output = torch.zeros(n, self.max_len, self.voc_dim)
    #     x = torch.tensor([self.SOS] * n * self.max_len).view(n, self.max_len)  # [n, T]
    #     eos = torch.tensor([self.EOS] * n).view(n, -1)
    #     is_end = torch.zeros_like(eos, dtype=torch.uint8)
    #     for i in range(0, max_len):
    #         logits, _ = self.decoder(x, z)  # [n, T, V]
    #         logits = logits[:, i, :]  # [n, V]
    #         indices = gumbel_softmax(logits, tau, self.device).argmax(dim=-1)  # [n]
    #         x[:, i + 1] = indices
    #         output[:, i, :] = logits
    #         is_end = is_end | (x == eos)
    #         if torch.sum(is_end) == n:
    #             break
    #     return output, _  # [B, T, V]
    #
    # def sample(self, n: int, temperature: Optional[float] = None, mcmc: int = 0) -> Tuple[
    #     torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Sample sentences from the prior
    #     :param n: number of sentences
    #     :param temperature: value [0, 1] lower value leads to greedy sampling
    #     :param mcmc: number markov chain montecarlo samples
    #     :return:
    #     """
    #     if temperature is None:
    #         temperature = self.temperature
    #     z = self.prior.sample((n,)) * temperature
    #     samples, z, length = self.decoder.inference(z)
    #     for i in range(mcmc):
    #         _, mean, sigma = self.encode(samples, length)
    #         z = torch.randn(n, self.latent_dim, device=self.device)
    #         z = mean + z * sigma
    #         samples, z, length = self.decoder.inference(z)
    #
    #     return samples, z, length
    #
    # def interpolate(self, start: Tuple[torch.Tensor, torch.Tensor], end: Tuple[torch.Tensor, torch.Tensor],
    #                 num_steps: int) -> List[torch.Tensor]:
    #     z_start, _, _ = self.encode(start[0], start[1])
    #     z_end, _, _ = self.encode(end[0], end[1])
    #     interpolations = []
    #     for ix, (z1, z2) in enumerate(zip(z_start, z_end)):
    #         t = torch.linspace(0, 1, num_steps + 2, device=self.device).unsqueeze(1)[1:-1]
    #         z_interp = z1 * (1 - t) + z2 * t
    #
    #         samples_interp, _, _ = self.decoder.inference(z_interp)
    #         interpolations.append(torch.cat([start[0][ix:ix + 1], samples_interp, end[0][ix:ix + 1]], dim=0))
    #     return torch.stack(interpolations)
    #
    # def set_prior_mean(self, prior_mean: str):
    #     tokens = char_tokenization(prior_mean, self.fix_len - 1)
    #     _input, seq_len = torch.tensor(tokens['input'], device=self.device), torch.tensor(tokens['length'],
    #                                                                                       device=self.device)
    #     z, _, _ = self.encode(_input.unsqueeze(0), seq_len.unsqueeze(0))
    #     self.prior.loc = z.squeeze()
    #
    # def set_prior_variance(self, prior_variance: float):
    #     self.prior.scale *= prior_variance

# class WAE(AModel):
#     """
#     Wasserstein autoencoder
#     """
#
#     def __init__(self, vocab, fix_len, **kwargs):
#         super(WAE, self).__init__(**kwargs)
#
#         self.voc_dim = len(vocab)
#         self.fix_len = fix_len
#         self.ignore_index = vocab.index('PAD')
#         self.SOS = vocab.index('SOS')
#         self.EOS = vocab.index('EOS')
#         self.PAD = vocab.index('PAD')
#         self.penalty_type = kwargs.get('penalty_type', 'gan')
#         if self.metrics is not None:
#             for m in self.metrics:
#                 m.PAD = self.ignore_index
#                 m.reduce = self.reduce
#
#         self.vocab = vocab
#         self.latent_dim = kwargs.get('latent_dim')
#         self.embedding_dim = kwargs.get('embedding_dim')
#         train_emb = kwargs.get('train_embeddings')
#         self.temperature = kwargs.get('temperature', 1.0)
#         self.drop_char_rate = kwargs.get('drop_char_rate', 0.0)
#         if train_emb is None or not train_emb:
#             self.UNK = self.voc_dim
#             self.embedding = OneHotEncoding(self.voc_dim, self.UNK)
#         else:
#             try:
#                 self.UNK = vocab.index('UNK')
#             except ValueError:
#                 self.UNK = self.voc_dim
#             self.embedding = torch.nn.Embedding(self.voc_dim, self.embedding_dim, self.ignore_index)
#         self.temperature = kwargs.get('temperature', 1.)
#
#         coupled_decoder = kwargs.get('coupled_decoder', False)
#         self._init_components(kwargs, vocab, fix_len, self.latent_dim, self.embedding, coupled_decoder)
#
#         # Prior model:
#         self.train_prior = kwargs.get('train_prior')
#         if self.train_prior:
#             self.prior = create_instance('prior', kwargs, self.latent_dim)
#         else:
#             self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
#             self.prior_var = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
#             self.prior = torch.distributions.Normal(self.prior_mean, self.prior_var)
#
#         # Wasserstein distance:
#         self.n_updates_critic = kwargs.get('n_updates_critic')
#         self.wasserstein_distance = create_instance("wasserstein_distance", kwargs, *(self.latent_dim,))
#         self.mmd_penalty = create_instance("mmd_penalty", kwargs, *(self.latent_dim,))
#
#     def _init_components(self, kwargs, vocab, fix_len, latent_dim, embedding, coupled_decoder: bool = False):
#         # Posterior model:
#         self.encoder = create_instance('encoder', kwargs, vocab, fix_len, latent_dim, embedding)
#         if coupled_decoder:
#             kwargs['decoder']['args']['encoder'] = self.encoder
#         # Likelihood model:
#         self.decoder = create_instance('decoder', kwargs, vocab, fix_len, latent_dim, embedding)
#
#     def encode(self, input_seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             B = input_seq.size(0)
#             self.initialize_hidden_state(B, self.device)
#             z, _, _ = self.encoder((input_seq, seq_len))
#         return z
#
#     def forward(self, input, critic_opt=False):
#         """
#         returns: loss function
#         input & target shape: [B, T]
#         Notation. B: batch size; T: seq len (== fix_len); V: voc size
#         """
#         # Sampling posterior
#         if self.drop_char_rate > 0:
#             x, l = input
#             # randomly replace decoder input with <unk>
#             prob = torch.rand(x.size(), device=self.device)
#             prob[(x - self.SOS) * (x - self.PAD) == 0] = 1.0
#
#             input_sequence = x.clone()
#             input_sequence[prob < self.drop_char_rate / l.unsqueeze(-1).float()] = self.UNK
#             input = input_sequence, l
#
#         z_post, _, _ = self.encoder(input)
#
#         # Sampling prior
#         z_prior = torch.randn(z_post.shape, device=self.device)
#         if self.train_prior:
#             z_prior = self.prior(z_prior)
#
#         # only decode if not critic optimization
#         if critic_opt:
#             return None, z_prior, z_post
#
#         # Decoding
#         logits, _ = self.decoder(input, z_post)  # [B, T, V]
#         logits = logits.contiguous().view(-1, self.voc_dim)  # [T * B, V]
#
#         # logits_from_prior, _ = self.decoder(input, z_prior)  # [B, T, V]
#         # logits_from_prior = logits_from_prior[:, :-1].contiguous().view(-1, self.voc_dim)  # [T * B, V]
#
#         return logits, z_prior, z_post  # , logits_from_prior
#
#     def loss(self, y, y_target, z_post, z_prior, stats, beta=None, seq_len=None):
#         """
#         returns the loss function of the Wasserstein autoencoder
#         """
#         beta = torch.tensor(1.0).to(self.device) if beta is None else beta
#         batch_size = z_prior.size(0)
#
#         # Mask loss for batch-dependent seq_len normalization
#         cost = self._calculate_nll(y, y_target, seq_len)
#         if self.penalty_type == 'gan':
#             distance = self.wasserstein_distance(z_prior, z_post)
#             distance = distance.sum() / float(batch_size)
#         elif self.penalty_type == 'mmd':
#             distance = self.mmd_penalty(z_prior, z_post)
#             distance /= float(batch_size)
#         else:
#             raise ValueError(f"Unknown value for the penalty type {self.penalty_type}")
#
#         loss = cost + beta * distance
#
#         stats['loss'] = loss
#         stats['NLL-Loss'] = cost
#         stats['KL-Loss'] = beta * distance
#
#         return stats
#
#     def _calculate_nll(self, y, y_target, seq_len):
#         max_len = torch.max(seq_len).item()
#         cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='none')
#         mask = (y_target.view(-1, max_len) != self.ignore_index).float()
#         cost = cost.view(-1, max_len) * (mask.float() / seq_len.view(-1, 1).float())
#         cost = cost.sum() / float(seq_len.size(0))
#         return cost
#
#     def decode_sentence(self, y, z):
#         """
#         y shape: [B, T]
#         z shape: [B, L, n_symbols] (indicator variables)
#         Notation. B: batch size; T: seq len (== fix_len); L: rw length
#         """
#
#         logits, _ = self.decoder(y, z)  # [B, T, V]
#         logits = logits.contiguous().view(-1, self.voc_dim)
#
#         return logits
#
#     def metric(self, input_seq, y, y_target, z_post, z_prior, seq_len=None):
#         """
#         returns a dictionary with metrics
#         """
#         with torch.no_grad():
#             stats = self.new_metric_stats()
#
#             z_prior /= self.latent_dim ** 0.5
#             logits_sim = self.decode_sentence((input_seq, seq_len), z_prior)
#             # Cross entropy
#             if seq_len is not None:
#                 # Mask loss for batch-dependent seq_len normalization
#                 cost = self._calculate_nll(y, y_target, seq_len)
#             else:
#                 cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index)
#
#             sim_cost = self._calculate_nll(logits_sim, y_target, seq_len)
#
#             stats['PPL'] = torch.exp(cost)
#             stats['PPL-Simulated'] = torch.exp(sim_cost)
#
#             if self.metrics is not None:
#                 for m in self.metrics:
#                     stats[type(m).__name__] = m(y, y_target)
#
#             return stats
#
#     def train_step(self, minibatch: dict, optimizer: dict, step: int, scheduler: Any = None):
#         input_seq, target_seq, seq_len = minibatch['input'], minibatch['target'], minibatch['length']
#         seq_len, _ix = torch.sort(seq_len, descending=True)
#         max_seq_len = torch.max(seq_len).item()
#         input_seq, target_seq = input_seq[_ix], target_seq[_ix, :max_seq_len].view(-1)
#
#         B = input_seq.size(0)
#
#         # statistics
#         stats = self.new_stats()
#
#         # schedulers
#         if scheduler is not None:
#             beta = torch.tensor(scheduler['beta_scheduler'](step), device=self.device)
#         else:
#             beta = torch.tensor(1.0, device=self.device)
#
#         ######## Training encoder-decoder pair ########
#
#         # (i) parameters:
#         frozen_params(self.wasserstein_distance)
#         free_params(self.encoder)
#         free_params(self.decoder)
#         if self.train_prior:
#             frozen_params(self.prior)
#
#         # (ii) loss:
#
#         # forward pass:
#         self.initialize_hidden_state(B, self.device)
#         logits, z_prior, z_post = self.forward((input_seq, seq_len))
#         loss_stats = self.loss(logits, target_seq, z_post, z_prior, stats, beta=beta, seq_len=seq_len)
#
#         # optimizers initialization and backward pass:
#         optimizer['optimizer']['opt'].zero_grad()  # encoder-decoder opt
#         if self.penalty_type == "gan":
#             loss_stats['NLL-Loss'].backward()
#         else:
#             loss_stats['loss'].backward()
#         clip_grad_norm(self.parameters(), optimizer['optimizer'])
#         optimizer['optimizer']['opt'].step()
#
#         if self.penalty_type == 'gan':
#             ######## Training critic ########
#
#             # (i) parameters:
#             free_params(self.wasserstein_distance)
#             frozen_params(self.encoder)
#             frozen_params(self.decoder)
#             if self.train_prior:
#                 frozen_params(self.prior)
#
#             # (ii) loss:
#             for _ in range(self.n_updates_critic):
#                 # forward pass:
#                 self.initialize_hidden_state(B, self.device)
#                 _, z_prior, z_post = self.forward((input_seq, seq_len), critic_opt=True)
#                 critic_loss = self.wasserstein_distance.get_critic_loss(z_prior, z_post)
#
#                 # optimizers initialization and backward pass:
#                 optimizer['critic_optimizer']['opt'].zero_grad()
#                 critic_loss.backward()
#                 clip_grad_norm(self.parameters(), optimizer['critic_optimizer'])
#                 optimizer['critic_optimizer']['opt'].step()
#
#             stats['Critic-Loss'] = critic_loss
#
#             ######## Training encoder-prior pair ########
#
#             # (i) parameters:
#             frozen_params(self.wasserstein_distance)
#             free_params(self.encoder)
#             frozen_params(self.decoder)
#             if self.train_prior:
#                 free_params(self.prior)
#
#             # (ii) loss:
#
#             # forward pass:
#             self.initialize_hidden_state(B, self.device)
#             logits, z_prior, z_post = self.forward((input_seq, seq_len))
#             loss_stats = self.loss(logits, target_seq, z_post, z_prior, stats, beta=beta, seq_len=seq_len)
#
#             # optimizers initialization and backward pass:
#             optimizer['optimizer']['opt'].zero_grad()  # encoder-decoder opt
#             loss_stats['KL-Loss'].backward()
#             clip_grad_norm(self.parameters(), optimizer['optimizer'])
#             optimizer['optimizer']['opt'].step()
#
#         # Metrics
#         metric_stats = self.metric(input_seq, logits, target_seq, z_post, z_prior, seq_len=seq_len)
#
#         prediction = logits.argmax(dim=1).view(B, -1)
#         target = target_seq.view(B, -1)
#
#         # monitor beta:
#         loss_stats['KL-Weight'] = beta
#
#         return {**loss_stats, **metric_stats, **{'reconstruction': (prediction, target)}}
#
#     def validate_step(self, minibatch: dict, scheduler: Any = None):
#         input_seq, target_seq, seq_len = minibatch['input'], minibatch['target'], minibatch['length']
#         seq_len, _ix = torch.sort(seq_len, descending=True)
#         input_seq, target_seq = input_seq[_ix], target_seq[_ix, :torch.max(seq_len).item()].view(-1)
#         B = input_seq.size(0)
#
#         # Statistics
#         stats = self.new_stats()
#
#         # Evaluate model
#         self.initialize_hidden_state(B, self.device)
#         logits, z_prior, z_post = self.forward((input_seq, seq_len))
#         loss_stats = self.loss(logits, target_seq, z_post, z_prior, stats, seq_len=seq_len)
#         metric_stats = self.metric(input_seq, logits, target_seq, z_post, z_prior, seq_len=seq_len)
#
#         prediction = logits.argmax(dim=1).view(B, -1)
#         target = target_seq.view(B, -1)
#
#         # monitor beta:
#         loss_stats['KL-Weight'] = torch.tensor(1.0, device=self.device)
#
#         return {**loss_stats, **metric_stats, **{'reconstruction': (prediction, target)}}
#
#     def sample(self, n: int, temperature: Optional[float] = None, **kwargs) -> Tuple[
#         torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Sample sentences from the prior
#         :param n: number of sentences
#         :param temperature: value [0, 1] lower value leads to greedy sampling
#         :return:
#         """
#         if temperature is None:
#             temperature = self.temperature
#         z = self.prior.sample((n,)) * temperature
#         samples, z, length = self.decoder.inference(z)
#
#         return samples, z, length
#
#     def interpolate(self, start: Tuple[torch.Tensor, torch.Tensor], end: Tuple[torch.Tensor, torch.Tensor],
#                     num_steps: int) -> torch.Tensor:
#         z_start = self.encode(start[0], start[1])
#         z_end = self.encode(end[0], end[1])
#         interpolations: List[torch.Tensor] = []
#         for ix, (z1, z2) in enumerate(zip(z_start, z_end)):
#             t = torch.linspace(0, 1, num_steps + 2, device=self.device).unsqueeze(1)[1:-1]
#             z_interp = z1 * (1 - t) + z2 * t
#
#             samples_interp, _, _ = self.decoder.inference(z_interp)
#             interpolations.append(torch.cat([start[0][ix:ix + 1], samples_interp, end[0][ix:ix + 1]], dim=0))
#         return torch.stack(interpolations)
#
#     def set_prior_mean(self, prior_mean: str):
#         tokens = char_tokenization(prior_mean, self.fix_len - 1)
#         _input, seq_len = torch.tensor(tokens['input'], device=self.device), torch.tensor(tokens['length'],
#                                                                                           device=self.device)
#         z = self.encode(_input.unsqueeze(0), seq_len.unsqueeze(0))
#         self.prior.loc = z.squeeze()
#
#     def set_prior_variance(self, prior_variance: float):
#         self.prior.scale *= prior_variance
#
#     def new_stats(self) -> dict:
#         stats = dict()
#         stats['loss'] = torch.tensor(0, device=self.device)
#         stats['NLL-Loss'] = torch.tensor(0, device=self.device)
#         stats['KL-Loss'] = torch.tensor(0, device=self.device)
#         stats['Critic-Loss'] = torch.tensor(0, device=self.device)
#         stats['KL-Weight'] = torch.tensor(0, device=self.device)
#         return stats
#
#     def new_metric_stats(self) -> dict:
#         stats = dict()
#         stats['PPL-Simulated'] = torch.tensor(0, device=self.device)
#         return stats
#
#     def initialize_hidden_state(self, batch_size, enc=True, dec=True):
#         if enc and self.encoder.is_recurrent:
#             self.encoder.initialize_hidden_state(batch_size, self.device)
#         if dec and self.decoder.is_recurrent:
#             self.decoder.initialize_hidden_state(batch_size, self.device)
#
#     def detach_history(self, enc=True, dec=True):
#         if self.encoder.is_recurrent and enc:
#             self.encoder.reset_history()
#         if self.decoder.is_recurrent and dec:
#             self.decoder.reset_history()
