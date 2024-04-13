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


def make_generators(configs, splits, make_gen_list, **kwargs):
    if configs is None:
        gen_dict = {split: [] for split in splits}
    else:
        gen_dict = {
            config: {split: [] for split in splits} for config in configs
        }
    make_gen_list(gen_dict, **kwargs)
    if configs is None:
        return {split: merge_gens(gen_dict[split]) for split in splits}
    else:
        return {
            config: {
                split: merge_gens(gen_dict[config][split]) for split in splits
            }
            for config in configs
        }
