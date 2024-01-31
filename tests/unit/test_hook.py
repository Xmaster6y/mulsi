"""
Test of the hook module.
"""

import torch

from mulsi import HookedModel, RunConfig

class TestCacheInputHook:
    """
    Test the cache input hook.
    """

    def test_cache_input_hook(self, model, tokenizer):
        """
        Test the cache input hook.
        """
        hooked_model = HookedModel(model)
        run_config = RunConfig(
            cache_output=True,
            module_exp = r"^transformer\.h\.\d*\.mlp\.act$"
        )
        sentence = "I hate this city"
        input_ids = {key: input_id.to(model.device) for key, input_id in tokenizer(sentence, return_tensors='pt').items()}
        with torch.no_grad():
            _, hate_returned_cache = hooked_model.run_with_hooks(run_config=run_config, **input_ids)

        assert "transformer.h.0.mlp.act" in hate_returned_cache