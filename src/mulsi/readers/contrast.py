"""Module implementing contrast vectors reading.
"""

from mulsi.reader import RepresentationReader
from mulsi.wrapper import MulsiWrapper


@RepresentationReader.register("contrast")
class ContrastReader:
    """Class to read a representation using contrast vectors."""

    def __init__(self, pros_inputs, cons_inputs):
        self.pros_inputs = pros_inputs
        self.cons_inputs = cons_inputs

    def read(self, wrapper: MulsiWrapper, **kwargs):
        """Reads the representation."""
        return wrapper.compute_representation(
            self.pros_inputs, **kwargs
        ) - wrapper.compute_representation(self.cons_inputs, **kwargs)
