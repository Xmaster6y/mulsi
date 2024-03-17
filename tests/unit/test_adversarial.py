"""Test of the fgsm attack.
"""

import torch

from mulsi import AdversarialImage


class TestFgsm:
    def test_define_fgsm(self):
        """
        Test the fgsm attack.
        """
        base_image = torch.zeros(1, 3, 224, 224, dtype=torch.int)
        clip_wrapper = None
        llm_wrapper = None
        adv_image = AdversarialImage(base_image, clip_wrapper, llm_wrapper)
        adv = adv_image.adv
        assert torch.allclose(adv, base_image)
