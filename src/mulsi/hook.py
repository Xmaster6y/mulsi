"""
Hook module for mulsi.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import PreTrainedModel


class RemovableHandleList(list):
    def remove(self):
        for handle in self:
            handle.remove()
        self.clear()


def cache_input_hook_factory(cache, key):
    def cache_input_hook(module, input, output):
        cache[key] = input

    return cache_input_hook


def cache_output_hook_factory(cache, key):
    def cache_output_hook(module, input, output):
        cache[key] = output

    return cache_output_hook


def add_vector_hook_factory(vector_cache, key):
    def add_vector_hook(module, input, output):
        return output + vector_cache[key]

    return add_vector_hook


def measure_vector_hook_factory(vector_cache, cache, key):
    def measure_vector_hook(module, input, output):
        cache[key] = (output * vector_cache[key]).sum(axis=-1)

    return measure_vector_hook


@dataclass
class RunConfig:
    cache_input: bool = False
    cache_output: bool = False
    add_vector: bool = False
    measure_vector: bool = False
    add_vector_cache: Optional[Dict[str, torch.Tensor]] = None
    measure_vector_cache: Optional[Dict[str, torch.Tensor]] = None
    module_exp: Optional[str] = None
    generate: bool = False
    debug: bool = False


class HookedModel(PreTrainedModel):
    def __init__(self, model):
        super().__init__(model.config)

        self.model = model
        self.handles = RemovableHandleList()

    def remove(self):
        self.handles.remove()

    def register(self, run_config):
        self.remove()
        returned_cache = {}
        if run_config is None:
            return returned_cache
        if run_config.add_vector and run_config.measure_vector:
            raise NotImplementedError(
                "Cannot measure and modify on the same run!"
            )
        if run_config.cache_input:
            returned_cache["input"] = {}
        if run_config.cache_output:
            returned_cache["output"] = {}
        if run_config.add_vector:
            if run_config.add_vector_cache is None:
                raise ValueError("Add cache not specified")
        if run_config.measure_vector:
            if run_config.measure_vector_cache is None:
                raise ValueError("Add cache not specified")
            returned_cache["measure"] = {}
            for vector in run_config.measure_vector_cache:
                returned_cache["measure"][vector] = {}

        for name, module in self.model.named_modules():
            if name == "":
                continue
            if run_config.module_exp is not None:
                m = re.match(run_config.module_exp, name)
                if m is None:
                    if run_config.debug:
                        print(name)
                    continue
            if run_config.cache_input:
                hook = cache_input_hook_factory(returned_cache["input"], name)
                handle = module.register_forward_hook(hook)
                self.handles.append(handle)
            if run_config.cache_output:
                hook = cache_output_hook_factory(
                    returned_cache["output"], name
                )
                handle = module.register_forward_hook(hook)
                self.handles.append(handle)
            if run_config.add_vector:
                for vector in run_config.add_vector_cache:
                    hook = add_vector_hook_factory(
                        run_config.add_vector_cache[vector], name
                    )
                    handle = module.register_forward_hook(hook)
                    self.handles.append(handle)
            if run_config.measure_vector:
                for vector in run_config.measure_vector_cache:
                    hook = measure_vector_hook_factory(
                        run_config.measure_vector_cache[vector],
                        returned_cache["measure"][vector],
                        name,
                    )
                    handle = module.register_forward_hook(hook)
                    self.handles.append(handle)
        return returned_cache

    def run_with_hooks(self, run_config, *args, **kwargs):
        try:
            returned_cache = self.register(run_config=run_config)
            if run_config.generate:
                out = self.model.generate(*args, **kwargs)
            else:
                out = self.model(*args, **kwargs)
        finally:
            self.remove()
        return out, returned_cache
