"""Module to abstract the internal representation of the model.
"""

from typing import Callable, Union

import torch
from torch.types import Number
from typeguard import check_type


class Representation(dict):
    """Class for manipulating representations."""

    def __add__(
        self, other: Union[Number, torch.Tensor, "Representation"]
    ) -> "Representation":
        """Adds two representations."""
        check_type(other, Union[Number, torch.Tensor, "Representation"])
        if isinstance(other, Representation):
            if self.keys() != other.keys():
                raise ValueError(
                    "Incompatible representations (different keys)."
                )
            new_dict = {}
            for key in self.keys():
                new_dict[key] = self[key] + other[key]
            return Representation(new_dict)
        else:
            return self.apply(lambda value: value + other)

    def __mul__(
        self, other: Union[Number, torch.Tensor, "Representation"]
    ) -> "Representation":
        """Multiplies two representations."""
        check_type(other, Union[Number, torch.Tensor, "Representation"])
        if isinstance(other, Representation):
            if self.keys() != other.keys():
                raise ValueError(
                    "Incompatible representations (different keys)."
                )
            new_dict = {}
            for key in self.keys():
                new_dict[key] = self[key] * other[key]
            return Representation(new_dict)

        else:
            return Representation(self.apply(lambda value: value * other))

    def __neg__(self) -> "Representation":
        """Negates the representation."""
        return self.__mul__(-1)

    def __sub__(
        self, other: Union[Number, torch.Tensor, "Representation"]
    ) -> "Representation":
        """Subtracts two representations."""
        return self.__add__(-other)

    def dot_prod(self, other: "Representation") -> torch.Tensor:
        """Computes the dot product between two representations."""
        return (self * other).scalar_sum()

    def norm(self) -> torch.Tensor:
        """Computes the norm of the representation."""
        return torch.sqrt(self.dot_prod(self))

    def cosim(self, other: "Representation") -> torch.Tensor:
        """Computes the cosine similarity between two representations."""
        return self.dot_prod(other) / (self.norm() * other.norm())

    def scalar_sum(self) -> torch.Tensor:
        """Computes the average of the representation."""
        return sum([value.sum() for value in self.values()])

    def scalar_avg(self) -> torch.Tensor:
        """Computes the average of the representation."""
        total_size = sum([value.numel() for value in self.values()])
        return self.scalar_sum() / total_size

    def apply(self, func: Callable) -> "Representation":
        """Applies a function to the representation."""
        new_dict = {}
        for key in self.keys():
            new_dict[key] = func(self[key])
        return Representation(new_dict)

    def mean(self, dim) -> "Representation":
        """Computes the mean of the representation."""
        new_dict = {}
        for key in self.keys():
            new_dict[key] = self[key].mean(dim=dim)
        return Representation(new_dict)

    def flatten(self) -> "Representation":
        """Flattens the representation."""
        new_dict = {}
        for key in self.keys():
            new_dict[key] = self[key].flatten()
        return Representation(new_dict)

    @classmethod
    def mean_representation(
        cls, *representations: "Representation"
    ) -> "Representation":
        """Averages the representations."""
        if not representations:
            raise ValueError("No representations given.")
        new_dict = {}
        for key in representations[0].keys():
            new_dict[key] = torch.mean(
                torch.stack([rep[key] for rep in representations]), dim=0
            )
        return cls(new_dict)
