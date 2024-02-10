"""
Hook module for mulsi.
"""

from typing import List

from transformers import AutoModelForCausalLM

from .hook import Hook


class HookedModel:
    def __init__(self, model: AutoModelForCausalLM, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks: List[Hook] = []

    def register_hook(self, hook: Hook):
        """
        Registers a hook.
        """
        self.hooks.append(hook)
        return hook.register(self.model)

    def remove_hooks(self):
        """
        Removes all hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __call__(self, sentence: str):
        """
        Forward pass.
        """
        input_ids = {
            key: input_id.to(self.model.device)
            for key, input_id in self.tokenizer(
                sentence, return_tensors="pt"
            ).items()
        }
        return self.model(**input_ids)
