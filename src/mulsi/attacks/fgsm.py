"""Module for the Fast Gradient Sign Method (FGSM) attack.
"""

from mulsi.attack import Attack


class Fgsm(Attack):
    """Abstract class"""

    def attack(self, inputs, labels):
        """Attacks the model."""
        pass
