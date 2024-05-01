"""
Test of the representation module.
"""

import torch

from mulsi import Representation


class TestSimpleRepresentation:
    def test_add_representation(self):
        """
        Test representation.
        """
        representation = Representation(
            {
                "a": torch.tensor([-1.0, -2.0, -3.0]),
                "b": torch.tensor([1.0, 2.0, 3.0]),
            }
        )
        zero_representation = representation + representation
        expected_representation = Representation(
            {
                "a": torch.tensor([-2.0, -4.0, -6.0]),
                "b": torch.tensor([2.0, 4.0, 6.0]),
            }
        )
        for key in zero_representation.keys():
            assert (zero_representation[key] == expected_representation[key]).all()

    def test_mul_representation(self):
        """
        Test representation.
        """
        representation = Representation(
            {
                "a": torch.tensor([-1.0, -2.0, -3.0]),
                "b": torch.tensor([1.0, 2.0, 3.0]),
            }
        )
        zero_representation = representation * 0
        expected_representation = Representation(
            {
                "a": torch.tensor([0.0, 0.0, 0.0]),
                "b": torch.tensor([0.0, 0.0, 0.0]),
            }
        )
        for key in zero_representation.keys():
            assert (zero_representation[key] == expected_representation[key]).all()

    def test_sub_representation(self):
        """
        Test representation.
        """
        representation = Representation(
            {
                "a": torch.tensor([1.0, 2.0, 3.0]),
                "b": torch.tensor([1.0, 2.0, 3.0]),
            }
        )
        zero_representation = representation - representation
        expected_representation = Representation(
            {
                "a": torch.tensor([0.0, 0.0, 0.0]),
                "b": torch.tensor([0.0, 0.0, 0.0]),
            }
        )
        for key in zero_representation.keys():
            assert (zero_representation[key] == expected_representation[key]).all()
