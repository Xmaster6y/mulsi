"""Module for testing the attack.
"""

from mulsi.attack import Attack


class TestLoading:
    def test_register_attack(self):
        """
        Test register attack.
        """

        assert "test" not in Attack.all_attacks

        @Attack.register("test")
        class TestAttack(Attack):
            def perform(self, **kwargs):
                """Perform the attack."""
                pass

        assert "test" in Attack.all_attacks
        assert Attack.all_attacks["test"] == TestAttack
