import logging
import math
import os
import random

logging.getLogger().setLevel(logging.INFO)


def create_split_indices(dataset_length, splits, seed):
    random.seed(seed)
    indices = list(range(dataset_length))
    random.shuffle(indices)
    output = {}
    previous_index = 0
    # Ensure order is deterministic.
    for i, split_name in enumerate(sorted(splits)):
        if i == len(splits) - 1:
            # Leftover instances, as we ensure that the fractions sum to one this ensures we don't
            # miss any instances due to rounding issues.
            output[split_name] = indices[previous_index:]
        else:
            n_instances = int(math.floor(splits[split_name] * dataset_length))
            if n_instances == 0:
                logging.warn(
                    f"No data was assigned to split {split_name} with ratio {splits[split_name]}. "
                    "Make sure this is intentional."
                )
            next_index = previous_index + n_instances
            output[split_name] = indices[previous_index:next_index]
            previous_index = next_index

    return {split_name: set(indices) for split_name, indices in output.items()}


def get_shard_pattern(path: str):
    base_pattern: str = "shard-%06d.tar"
    return os.path.join(path, base_pattern)


def make_subdirs_and_patterns(output_path, split_indices):
    """Create subfolders and paths for splits and return the indices for the appropriate splits."""
    # Ensure order is deterministic.
    split_names = sorted(split_indices.keys())
    indices = [split_indices[split_name] for split_name in split_names]
    paths = [os.path.join(output_path, subdir) for subdir in split_names]
    for p in paths:
        os.makedirs(p, exist_ok=True)
    return [get_shard_pattern(p) for p in paths], indices


def get_index(element, shards):
    """Get the index of element in list of lists.

    Given a list of lists returns the index of the list which contains the element.
    """
    for index, l in enumerate(shards):
        if element in l:
            return index

    raise IndexError(f"Cannot find index of element {element}.")


class ContextList(list):
    """Allow usage of a list of context managers as a context manager.

    When entering the context, all context managers enter and when exiting all context managers
    exit.
    """

    def __enter__(self):
        for v in self:
            v.__enter__()
        return self

    def __exit__(self, *exc):
        for v in self:
            v.__exit__()


class FakeIndices:
    """Class that behaves like a set, but always returns true on the isin operation."""

    def __contains__(self, element):
        return True
