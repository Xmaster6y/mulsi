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

    def read(
        self, wrapper: MulsiWrapper, inputs, reading_vector=None, **kwargs
    ):
        """Reads the representation."""
        if reading_vector is None:
            reading_vector = self.compute_reading_vector(wrapper, **kwargs)
        return wrapper.compute_representation(inputs, **kwargs).cosim(
            reading_vector
        )

    def compute_reading_vector(self, wrapper: MulsiWrapper, **kwargs):
        """Reads the representation."""
        return wrapper.compute_representation(
            self.pros_inputs, **kwargs
        ) - wrapper.compute_representation(self.cons_inputs, **kwargs)
