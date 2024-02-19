"""Abstract class for MULSI models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mulsi.hook import CacheHook, HookConfig
from mulsi.representation import Representation


class MulsiWrapper(ABC):
    """Abstract class for MULSI models."""

    @abstractmethod
    def compute_representation(cls, inputs, **kwargs):
        """Computes the representation."""
        pass

    @abstractmethod
    def compute_loss(cls, inputs, **kwargs):
        """Computes the loss."""
        pass


@dataclass
class LlmWrapper(MulsiWrapper):
    """Wrapper for a language model."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    _cache_hook = CacheHook(
        HookConfig(module_exp=r"^transformer\.h\.\d*\.mlp\.act$")
    )

    @torch.no_grad()
    def compute_representation(self, inputs, **kwargs):
        """Computes the representation."""
        self._cache_hook.register(self.model)
        encoded_inputs = self.tokenizer(inputs, return_tensors="pt")
        self.model(**encoded_inputs, **kwargs)
        representation = Representation(self._cache_hook.storage)
        representation = representation.mean(dim=(0, 1)).flatten()
        self._cache_hook.clear()
        return representation

    def compute_loss(self, inputs, **kwargs):
        """Computes the loss."""
        raise NotImplementedError
