from abc import ABC, abstractmethod
from torch import Tensor
import torch.nn as nn


class Block(nn.Module, ABC):
    @abstractmethod
    def forward(self, h: Tensor) -> Tensor: ...


class DataStream(ABC):
    @abstractmethod
    def get_batch(self) -> tuple[Tensor, Tensor]: ...
