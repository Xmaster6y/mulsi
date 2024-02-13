"""Test of the fgsm attack.
"""

import torch

from mulsi import Attack


class TestFgsm:
    def test_basic_fgsm(self):
        """
        Test the fgsm attack.
        """

        attack = Attack.from_name("fgsm", epsilon=0.1)
        assert attack.epsilon == 0.1
        inputs = torch.zeros(2, 4, 4)

        def compute_loss(inputs):
            return (inputs**2).sum()

        adv = attack.perform(inputs, compute_loss)
        assert torch.allclose(adv, inputs)
