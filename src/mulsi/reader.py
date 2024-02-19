"""Module for reading representations from a model.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type


class RepresentationReader(ABC):
    """Reads representations from a model."""

    all_readers: Dict[str, Type["RepresentationReader"]] = {}

    @abstractmethod
    def read(self, **kwargs):
        """Reads the representations."""
        pass

    @abstractmethod
    def compute_reading_vector(self, **kwargs):
        """Computes the reading vector."""
        pass

    @classmethod
    def register(cls, name: str):
        """Registers the reader."""

        def decorator(subclass):
            cls.all_readers[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "RepresentationReader":
        """Returns the reader from the name."""
        return cls.all_readers[name](**kwargs)
