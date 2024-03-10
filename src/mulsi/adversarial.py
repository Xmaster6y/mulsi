"""Module for the Fast Gradient Sign Method (FGSM) adversarial attacks.
"""

from enum import Enum
from typing import Dict, Optional, Union

import torch
from torch.utils.data import Dataset


class LossType(int, Enum):
    TOKEN_PREDICTION = 0
    TEXT_EMBEDDING = 1
    IMAGE_EMBEDDING = 2


class AdversarialImage:
    """Class for producing adversarial images using FGSM"""

    def __init__(
        self,
        base_image: torch.Tensor,
        encoder,
        decoder,
    ) -> None:
        self.base_image = base_image
        self.delta = torch.zeros_like(base_image, dtype=float)
        self.delta.requires_grad = True
        self.encoder = encoder
        self.decoder = decoder

    @property
    def adv(self) -> torch.Tensor:
        _adv = self.base_image + self.delta
        return _adv.requires_grad_(False).int()

    def fgsm_(
        self,
        epsilon: int,
        losses: Dict[LossType, float],
        samples: Dict[LossType, Union[torch.Tensor, Dataset]],
        alpha: Optional[float] = None,
    ) -> None:
        """Perform the attack."""
        if losses.keys() != samples.keys():
            raise ValueError("Losses keys and samples keys must match")
        loss = torch.zeros(1)
        for loss_type, loss_coeff in losses.items():
            loss += loss_coeff * self._compute_loss(
                loss_type, samples[loss_type]
            )
        loss.backward()
        grad_mul = alpha or epsilon
        self.delta.add_(grad_mul * self.delta.grad.sign())
        self._ensure_valid_delta_(epsilon)

    def fgsm_iter_(
        self,
        epsilon: int,
        losses: Dict[LossType, float],
        samples: Dict[LossType, Union[torch.Tensor, Dataset]],
        n_iter: int,
        alpha: Optional[float] = None,
    ) -> None:
        for _ in range(n_iter):
            self.fgsm_(epsilon, losses, samples, alpha)

    @torch.no_grad
    def _ensure_valid_delta_(self, epsilon: int) -> None:
        self.delta.clip_(-epsilon, epsilon)
        max_ampl = 255 - self.base_image
        min_ampl = self.base_image
        self.delta.clip_(-min_ampl, max_ampl)

    def _compute_loss(
        self, loss_type: LossType, sample: Union[torch.Tensor, Dataset]
    ) -> torch.Tensor:
        if loss_type == LossType.TOKEN_PREDICTION:
            raise NotImplementedError
        elif loss_type == LossType.TEXT_EMBEDDING:
            raise NotImplementedError
        elif loss_type == LossType.IMAGE_EMBEDDING:
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
