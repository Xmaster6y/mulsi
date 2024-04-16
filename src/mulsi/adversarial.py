"""Module for the Fast Gradient Sign Method (FGSM) adversarial attacks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import CLIPForImageClassification, CLIPModel

from mulsi.preprocess import DiffCLIPImageProcessor, DiffCLIPProcessor


class Loss(torch.nn.Module, ABC):
    """Class for computing the loss."""

    @abstractmethod
    def forward(
        self,
        adv_image: torch.Tensor,
        data: Dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


class CLIPClfLoss(Loss):
    """Class for computing the CLIP classifier loss."""

    def __init__(
        self,
        model: CLIPForImageClassification,
        image_processor: DiffCLIPImageProcessor,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.reduction = reduction

    def forward(
        self, adv_image: torch.Tensor, data: Dict[str, Any]
    ) -> torch.Tensor:
        labels = data.get("labels")
        if labels is None:
            raise ValueError("Labels must be provided.")
        elif labels.ndim != 2:
            raise ValueError("Labels must be 2D.")
        encoded_inputs = self.image_processor(
            images=adv_image, return_tensors="pt"
        )
        outputs = self.model(**encoded_inputs)
        logits = outputs.logits
        logits = logits.expand(labels.shape[0], -1)
        return torch.nn.functional.cross_entropy(
            logits, data, reduction=self.reduction
        )


class CLIPContrastiveLoss(Loss):
    """Class for computing the CLIP contrastive loss."""

    def __init__(
        self,
        model: CLIPModel,
        processor: DiffCLIPProcessor,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor

    def forward(
        self, adv_image: torch.Tensor, data: Dict[str, Any]
    ) -> torch.Tensor:

        text = data.get("text")
        if text is None:
            raise ValueError("Text must be provided.")
        encoded_inputs = self.processor(
            images=adv_image, text=text, return_tensors="pt"
        )
        outputs = self.model(**encoded_inputs, return_loss=True)
        return outputs.loss


class CLIPEmbedsLoss(Loss):
    """Class for computing the CLIP image pooler loss."""

    def __init__(
        self,
        model: CLIPModel,
        processor: DiffCLIPProcessor,
        reduction: str = "mean",
        loss_fn: Union[str, torch.nn.Module] = "mse",
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.reduction = reduction
        if isinstance(loss_fn, str):
            self.loss_fn = torch.nn.__dict__[loss_fn.lower()]
        else:
            self.loss_fn = loss_fn

    def forward(
        self, adv_image: torch.Tensor, data: Dict[str, Any]
    ) -> torch.Tensor:
        images = data.get("images")
        text = data.get("text")
        encoded_inputs = self.processor(images=adv_image, return_tensors="pt")
        outputs = self.model(**encoded_inputs)
        adv_embeds = outputs.image_embeds
        encoded_inputs = self.processor(
            images=images, text=text, return_tensors="pt"
        )
        outputs = self.model(**encoded_inputs)
        n = 0
        total_loss = 0.0
        if images is not None:
            embeds = outputs.image_embeds
            total_loss += self.loss_fn(adv_embeds, embeds, reduction="sum")
            n += embeds.shape[0]
        if text is not None:
            embeds = outputs.text_embeds
            total_loss += self.loss_fn(adv_embeds, embeds, reduction="sum")
            n += embeds.shape[0]

        if self.reduction == "mean":
            total_loss /= n
        return total_loss


class AdversarialImage:
    """Class for producing adversarial images using FGSM"""

    def __init__(
        self,
        base_image: torch.Tensor,
    ) -> None:
        self.base_image = base_image
        self.delta = torch.zeros_like(base_image, dtype=float)
        self.delta.requires_grad = True

    @property
    def adv(self) -> torch.Tensor:
        _adv = self.base_image + self.delta
        return _adv.detach().to(torch.uint8)

    def fgsm_(
        self,
        epsilon: int,
        input_list: List[Tuple[Loss, Dict[str, Any], float]],
        alpha: Optional[float] = None,
        use_sign: bool = True,
    ) -> None:
        """Perform the attack."""
        total_loss = torch.zeros(1)
        for loss, data, loss_coeff in input_list:
            total_loss += loss_coeff * loss(self.base_image + self.delta, data)
        total_loss.backward()
        grad_mul = alpha or epsilon
        if use_sign:
            self.delta.add_(grad_mul * self.delta.grad.sign())
        else:
            self.delta.add_(grad_mul * self.delta.grad)
        self._ensure_valid_delta_(epsilon)

    def fgsm_iter_(
        self,
        epsilon: int,
        input_list: List[Tuple[Loss, Dict[str, Any], float]],
        n_iter: int,
        alpha: Optional[float] = None,
        use_sign: bool = True,
    ) -> None:
        for _ in range(n_iter):
            self.fgsm_(epsilon, input_list, alpha, use_sign)

    @torch.no_grad
    def _ensure_valid_delta_(self, epsilon: int) -> None:
        self.delta.clip_(-epsilon, epsilon)
        max_ampl = 255 - self.base_image
        min_ampl = self.base_image
        self.delta.clip_(-min_ampl, max_ampl)
