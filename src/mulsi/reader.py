"""Module for reading representations from a model.
"""

from abc import ABC, abstractmethod


class RepresentationReader(ABC):
    """Reads representations from a model."""

    @abstractmethod
    def read(self, **kwargs):
        """Reads the representations."""
        pass
