"""Module to abstract the internal representation of the model.
"""

from typing import Union

import torch
from tensordict import TensorDict
from torch.types import Number
from typeguard import check_type


class Representation(TensorDict):
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
            if self.batch_size != other.batch_size:
                raise ValueError(
                    "Incompatible representations (different batch sizes)."
                )
            new_dict = {}
            for key in self.keys():
                new_dict[key] = self[key] + other[key]
            return Representation(new_dict, batch_size=self.batch_size)
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
            if self.batch_size != other.batch_size:
                raise ValueError(
                    "Incompatible representations (different batch sizes)."
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

    def dot(self, other: "Representation") -> torch.Tensor:
        """Computes the dot product of two representations."""
        raise NotImplementedError()

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
        return cls(new_dict, batch_size=representations[0].batch_size)
