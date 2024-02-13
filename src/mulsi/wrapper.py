"""Abstract class for MULSI models.
"""

from abc import ABC, abstractmethod


class MulsiWrapper(ABC):
    """Abstract class for MULSI models."""

    @abstractmethod
    @classmethod
    def compute_representation(cls, inputs, **kwargs):
        """Computes the representation."""
        pass

    @abstractmethod
    @classmethod
    def compute_loss(cls, inputs, **kwargs):
        """Computes the loss."""
        pass
