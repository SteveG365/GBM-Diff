from abc import ABC, abstractmethod
from typing import Any
import torch

class ControlModule(ABC):
    """
    Base interface for Any Control Mechanism applied -
    to a backbone diffusion model.
    """
    @abstractmethod
    def get_control(
        self,
        x_t : torch.Tensor,
        t : torch.Tensor,
        condition : torch.Tensor
    ) -> Any:

        raise NotImplementedError