"""Module for testing the reader.
"""

from mulsi.reader import RepresentationReader


class TestLoading:
    def test_register_reader(self):
        """
        Test register reader.
        """

        assert "test" not in RepresentationReader.all_readers

        @RepresentationReader.register("test")
        class TestReader(RepresentationReader):
            def read(self, **kwargs):
                """Read the representations."""
                pass

        assert "test" in RepresentationReader.all_readers
        assert RepresentationReader.all_readers["test"] == TestReader
