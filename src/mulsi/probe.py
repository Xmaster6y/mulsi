"""Module to implement generic probing.
"""

from abc import ABC, abstractmethod
from typing import Any

import einops
import torch

EPS = 1e-6


class Probe(ABC):
    """Abstract class for probes."""

    def __init__(self):
        self._trained = False

    @abstractmethod
    def train(
        self,
        activations: torch.Tensor,
        labels: Any,
        **kwargs,
    ):
        """Train the probe."""
        pass

    @abstractmethod
    def predict(self, activations: torch.Tensor, **kwargs):
        """Predict with the probe."""
        pass

    def score(self, activations: torch.Tensor, labels: Any, **kwargs):
        """Score the probe."""
        predictions = self.predict(activations, **kwargs)
        scores = {"mse": torch.nn.functional.mse_loss(predictions, labels)}
        scores["accuracy"] = (
            (predictions.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
        )
        return scores


class SignalCav(Probe):
    """Signal CAV probe."""

    def train(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        if len(activations) != len(labels):
            raise ValueError("Number of activations and labels must match")
        if len(activations.shape) != 2:
            raise ValueError("Activations must a batch of tensors")
        if len(labels.shape) != 2:
            raise ValueError("Labels must a batch of tensors")

        mean_activation = activations.mean(dim=0, keepdim=True)
        mean_label = labels.float().mean(dim=0, keepdim=True)
        scaled_activations = activations - mean_activation
        scaled_labels = labels - mean_label
        cav = einops.einsum(
            scaled_activations, scaled_labels, "b a, b d -> a d"
        )
        self._h = cav / (cav.norm(dim=0, keepdim=True) + EPS)
        self._trained = True

    def predict(self, activations: torch.Tensor, **kwargs):
        if not self._trained:
            raise ValueError("Probe not trained")

        if len(activations.shape) != 2:
            raise ValueError("Activations must a batch of tensors")

        dot_prod = einops.einsum(activations, self._h, "b a, a d -> b d")
        return dot_prod / (activations.norm(dim=1, keepdim=True) + EPS)


class LinearProbe(Probe):
    """Linear probe."""

    def train(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        if len(activations) != len(labels):
            raise ValueError("Number of activations and labels must match")
        if len(activations.shape) != 2:
            raise ValueError("Activations must a batch of tensors")
        if len(labels.shape) != 2:
            raise ValueError("Labels must a batch of tensors")

        self._h = torch.linalg.lstsq(activations, labels.float()).solution
        self._trained = True

    def predict(self, activations: torch.Tensor, **kwargs):
        if not self._trained:
            raise ValueError("Probe not trained")
        if len(activations.shape) != 2:
            raise ValueError("Activations must a batch of tensors")

        return activations @ self._h
