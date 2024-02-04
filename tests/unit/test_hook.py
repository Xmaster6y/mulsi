"""
Test of the hook module.
"""

import torch

from mulsi import CacheHook, HookConfig, HookedModel


class TestCacheInputHook:
    """
    Test the cache input hook.
    """

    def test_cache_input_hook(self, model, tokenizer):
        """
        Test the cache input hook.
        """
        cache_hook_config = HookConfig(module_exp=r"transformer.h.0.mlp.act")
        cache_hook = CacheHook(cache_hook_config)
        hooked_model = HookedModel(model, tokenizer)
        hooked_model.register_hook(cache_hook)
        sentence = "I hate this city"

        with torch.no_grad():
            _ = hooked_model(sentence)
        assert "transformer.h.0.mlp.act" in cache_hook.storage
