"""Module for the Fast Gradient Sign Method (FGSM) attack.
"""

from abc import ABC, abstractmethod


class Attack(ABC):
    """Abstract class for attacks"""

    @abstractmethod
    def perform(self, **kwargs):
        """Perform the attacks on the model."""
        pass
