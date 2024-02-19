"""Module for the Fast Gradient Sign Method (FGSM) attack.
"""

from dataclasses import dataclass

from mulsi.attack import Attack


@Attack.register("fgsm")
@dataclass
class Fgsm(Attack):
    """Abstract class"""

    epsilon: float

    def perform(self, inputs, compute_loss):
        """Perform the attack."""
        inputs.requires_grad = True
        loss = compute_loss(inputs)
        loss.backward()
        return inputs + self.epsilon * inputs.grad.sign()
