"""Test of the fgsm attack.
"""

import torch

from mulsi import AdversarialImage


class TestFgsm:
    def test_define_fgsm(self):
        """
        Test the fgsm attack.
        """
        base_image = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)
        adv_image = AdversarialImage(base_image)
        adv = adv_image.adv
        assert torch.allclose(adv, base_image)
