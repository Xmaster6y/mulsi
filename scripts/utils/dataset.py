"""Utils for building datasets.
"""

from typing import Callable, Generator, List


def empty_gen():
    return
    yield


def merge_gens(gens: List[Callable[[], Generator]]):
    def new_gen():
        for gen in gens:
            yield from gen()

    return new_gen


def make_generators(layers, splits, make_gen_list, **kwargs):
    gen_dict = {layer: {split: [] for split in splits} for layer in layers}
    make_gen_list(gen_dict, **kwargs)
    return {
        layer: {split: merge_gens(gen_dict[layer][split]) for split in splits}
        for layer in layers
    }
