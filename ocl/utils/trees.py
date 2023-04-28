"""Utilities for working with our own version of PyTrees which focus on torch tensors.

PyTrees are any nested structure of dictionaries, lists, tuples, namedtuples or dataclasses.
"""
import copy
import dataclasses
from collections import OrderedDict, abc
from typing import Any, Callable, Dict, Generator, List, Mapping, Sequence, Tuple, Union

import torch

Tree = Union[Dict, List, Tuple]


def is_tensor_or_module(t: Any):
    """Check if input is a torch.Tensor or a torch.nn.Module."""
    return isinstance(t, (torch.Tensor, torch.nn.Module))


def is_namedtuple(obj) -> bool:
    """Check if input is a named tuple."""
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def get_tree_element(d: Tree, path: List[str]) -> Any:
    """Get element of a tree."""
    next_element = d

    for next_element_name in path:
        if isinstance(next_element, abc.Mapping) and next_element_name in next_element:
            next_element = next_element[next_element_name]
        elif hasattr(next_element, next_element_name):
            next_element = getattr(next_element, next_element_name)
        elif isinstance(next_element, (list, tuple)) and next_element_name.isnumeric():
            next_element = next_element[int(next_element_name)]
        else:
            try:
                next_element = getattr(next_element, next_element_name)
            except AttributeError:
                msg = f"Trying to access path {'.'.join(path)}, "
                if isinstance(next_element, abc.Mapping):
                    msg += f"but element {next_element_name} is not among keys {next_element.keys()}"
                elif isinstance(next_element, (list, tuple)):
                    msg += f"but cannot index into list with {next_element_name}"
                else:
                    msg += (
                        f"but element {next_element_name} cannot be used to access attribute of "
                        f"object of type {type(next_element)}"
                    )
                raise ValueError(msg)
    return next_element


def _build_walk_path(previous_element, new_element):
    return previous_element + [new_element]


def walk_tree_with_paths(
    next_element, path=None, instance_check=is_tensor_or_module
) -> Generator[Tuple[List[str], Any], None, None]:
    """Walk over all tensors + modules and their paths in a nested structure.

    This could lead to an infinite loop.
    """
    if path is None:
        path = []

    if instance_check(next_element):
        yield path, next_element
    elif isinstance(next_element, str):
        # Special handling for strings, as even a single element slice is a sequence. This leads to
        # infinite nesting.
        pass
    elif isinstance(next_element, torch.nn.Module):
        for key, value in next_element.named_children():
            yield from walk_tree_with_paths(
                value, path=_build_walk_path(path, key), instance_check=instance_check
            )
    elif isinstance(next_element, (dict, Mapping)):
        for key, value in next_element.items():
            yield from walk_tree_with_paths(
                value, path=_build_walk_path(path, key), instance_check=instance_check
            )
    elif dataclasses.is_dataclass(next_element):
        for field in dataclasses.fields(next_element):
            yield from walk_tree_with_paths(
                getattr(next_element, field.name),
                path=_build_walk_path(path, field.name),
                instance_check=instance_check,
            )
    elif is_namedtuple(next_element):
        for field_name in next_element._fields:
            yield from walk_tree_with_paths(
                getattr(next_element, field_name),
                path=_build_walk_path(path, field_name),
                instance_check=instance_check,
            )
    elif isinstance(next_element, (List, Sequence, tuple)):
        for index, el in enumerate(next_element):
            yield from walk_tree_with_paths(
                el, path=_build_walk_path(path, index), instance_check=instance_check
            )


def reduce_tree(outputs: List[Dict[str, Any]], fn: Callable[[List[torch.Tensor]], torch.Tensor]):
    """Apply reduction function to a list of nested dicts.

    This only considers tensors at the moment, for other data types are simply copied from the first
    element.
    """
    id_to_reduced_tensor = {}
    for path, tensor in walk_tree_with_paths(outputs[0]):
        stacked_tensor = fn([tensor] + [get_tree_element(output, path) for output in outputs[1:]])
        id_to_reduced_tensor[id(tensor)] = stacked_tensor

    # Replace all tensors with their stacked versions.
    return copy.deepcopy(outputs[0], memo=id_to_reduced_tensor)


def map_tree(d: Tree, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Apply a function to each element of a tree.

    This only considers tensors at the moment, for other data types are simply copied from the first
    element.
    """
    id_to_mapped_tensor = {}
    for _, tensor in walk_tree_with_paths(d):
        mapped_tensor = fn(tensor)
        id_to_mapped_tensor[id(tensor)] = mapped_tensor

    # Replace all tensors with their stacked versions.
    return copy.deepcopy(d, memo=id_to_mapped_tensor)


def split_tree(d: Tree, split_paths: List[List[str]], split_axis: int, chunk_size: int):
    # We essentially need a deep copy of the input dict that we then update with splitted
    # references. To avoid copies of tensors and thus memory duplication we want to use shallow
    # copies for tensors instead. We do this by defining the memo parameter used in deepcopy for
    # all tensors in the dict. This way deepcopy thinks that these where already copied and uses
    # the provided objects instead. We can further use this trick to replace the original
    # tensors with splitted counterparts when running deepcopy.

    # Create memo containing all tensors to avoid data duplication.
    memo = {id(tensor): tensor for path, tensor in walk_tree_with_paths(d)}

    # Gather tensors that should be replaced and note their id.
    tensors_to_split = [get_tree_element(d, path) for path in split_paths]
    splitted_memos = OrderedDict(
        (id(tensor), torch.split(tensor, chunk_size, dim=split_axis)) for tensor in tensors_to_split
    )

    for tensor_slices in zip(*splitted_memos.values()):
        # Replace entires in memo dict with splitted counterparts.
        if chunk_size == 1:
            # Additionally squeeze the input.
            memo_override = {
                orig_id: tensor_slice.squeeze(split_axis)
                for orig_id, tensor_slice in zip(splitted_memos.keys(), tensor_slices)
            }
        else:
            memo_override = {
                orig_id: tensor_slice
                for orig_id, tensor_slice in zip(splitted_memos.keys(), tensor_slices)
            }
        yield copy.deepcopy(d, {**memo, **memo_override})
