"""Class for MULSI models."""

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

from mulsi.hook import CacheHook, HookConfig
from mulsi.representation import Representation


@dataclass
class LlmWrapper:
    """Wrapper for a language model."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    _cache_hook = CacheHook(HookConfig(module_exp=r"^transformer\.h\.\d*\.mlp\.act$"))

    @torch.no_grad()
    def compute_representation(self, inputs, **kwargs) -> Representation:
        """Computes the representation."""
        self._cache_hook.register(self.model)
        encoded_inputs = self.tokenizer(inputs, return_tensors="pt")
        self.model(**encoded_inputs, **kwargs)
        representation = Representation(self._cache_hook.storage)
        representation = representation.mean(dim=(0, 1)).flatten()
        self._cache_hook.clear()
        return representation

    def compute_loss(self, inputs, labels, **kwargs):
        """Computes the loss."""
        encoded_inputs = self.tokenizer(inputs, return_tensors="pt")
        outputs = self.model(**encoded_inputs, labels=labels, **kwargs)
        return outputs.loss


@dataclass
class CLIPModelWrapper:
    """Wrapper for CLIP model."""

    model: CLIPModel
    processor: CLIPProcessor
    _cache_hook = CacheHook(HookConfig(module_exp=r"*layers\.\d+\.mlp\.act$"))

    @torch.no_grad()
    def compute_representation(self, images, text=None, **kwargs) -> Representation:
        """Computes the representation."""
        self._cache_hook.register(self.model)
        encoded_inputs = self.processor(images=images, text=text, return_tensors="pt")
        self.model(**encoded_inputs, **kwargs)
        return Representation(self._cache_hook.storage)
