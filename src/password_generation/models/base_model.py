import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

from password_generation.utils.helper import create_instance


class Model(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate: float = 0.01,
        learning_rate_scheduler: Optional[Dict] = None,
        parameter_schedulers: Optional[Dict[str, Dict]] = None,
        model_weights: Optional[os.PathLike] = None,
    ):
        super().__init__()

        self.model_weights = model_weights
        self.learning_rate: float = learning_rate
        self.learning_rate_scheduler: Optional[Dict] = learning_rate_scheduler

        self.parameter_schedulers = {}
        if parameter_schedulers is not None:
            self.parameter_schedulers = {key: create_instance(value) for key, value in parameter_schedulers.items()}

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def loss(self, *args, **kwargs):
        ...

    @abstractmethod
    def training_step(self, *args, **kwargs):
        ...

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        ...

    @abstractmethod
    def test_step(self, *args, **kwargs):
        ...

    @abstractmethod
    def predict_step(self, *args, **kwargs):
        ...

    @abstractmethod
    def generate(self, n: int, **kwargs) -> torch.Tensor:
        ...

    def init_model_weights(self) -> None:
        if self.model_weights is not None:
            data = torch.load(self.model_weights, map_location="cpu")
            self.load_state_dict(data["state_dict"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        if self.learning_rate_scheduler is not None:
            learning_rate_scheduler_config = self.init_learning_rate_scheduler(self.learning_rate_scheduler, optimizer)
            return {"optimizer": optimizer, "lr_scheduler": learning_rate_scheduler_config}
        return optimizer

    def init_learning_rate_scheduler(self, learning_rate_scheduler: Dict, optimizer: torch.optim.Optimizer):
        additional_kwargs = {}
        for key in ["interval", "frequency", "monitor", "strict"]:
            if key in learning_rate_scheduler["args"]:
                additional_kwargs[key] = learning_rate_scheduler["args"].pop(key)
        return {"scheduler": create_instance(learning_rate_scheduler, optimizer=optimizer), **additional_kwargs}
