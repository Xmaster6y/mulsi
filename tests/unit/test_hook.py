"""Test of the hook module.
"""

import torch

from mulsi import CacheHook, HookConfig


class TestCacheInputHook:
    """Test the cache input hook."""

    def test_cache_input_hook(self, text_model, td_tokenizer):
        """Test the cache input hook."""
        cache_hook_config = HookConfig(module_exp=r"transformer.h.0.mlp.act")
        cache_hook = CacheHook(cache_hook_config)
        cache_hook.register(text_model)
        inputs = td_tokenizer("I love this city!")
        with torch.no_grad():
            _ = text_model(**inputs.to(text_model.device))
        assert "transformer.h.0.mlp.act" in cache_hook.storage
