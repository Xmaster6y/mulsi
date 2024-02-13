"""Module for the Fast Gradient Sign Method (FGSM) attack.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type


class Attack(ABC):
    """Abstract class for attacks"""

    all_attacks: Dict[str, Type["Attack"]] = {}

    @abstractmethod
    def perform(self, **kwargs):
        """Perform the attacks on the model."""
        pass

    @classmethod
    def register(cls, name: str):
        """Registers the attack."""

        def decorator(subclass):
            cls.all_attacks[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Attack":
        """Returns the attack from the name."""
        return cls.all_attacks[name](**kwargs)
