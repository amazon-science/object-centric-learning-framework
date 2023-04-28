from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch

from ocl.utils.trees import get_tree_element, split_tree, walk_tree_with_paths

TEST_TENSOR_A = torch.randn(10, 10)
TEST_TENSOR_B = torch.randn(10, 10)
TEST_TENSOR_C = torch.randn(10, 10)

test_namedtuple = namedtuple("test_namedtuple", ["a", "b"])


@dataclass
class test_dataclass:
    a: torch.Tensor
    b: torch.Tensor


TEST_COMBINATIONS = [
    pytest.param({"a": {"b": TEST_TENSOR_A}}, "a.b".split("."), id(TEST_TENSOR_A), id="dict"),
    pytest.param((TEST_TENSOR_A, TEST_TENSOR_B), "1", id(TEST_TENSOR_B), id="tuple"),
    pytest.param([TEST_TENSOR_A, TEST_TENSOR_B], "1", id(TEST_TENSOR_B), id="list"),
    pytest.param(
        test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_B), "b", id(TEST_TENSOR_B), id="namedtuple"
    ),
    pytest.param(
        test_dataclass(TEST_TENSOR_A, TEST_TENSOR_B), "b", id(TEST_TENSOR_B), id="dataclass"
    ),
    pytest.param(
        {
            "a": (
                12345,
                test_dataclass(TEST_TENSOR_A, [test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_C)]),
            )
        },
        "a.1.b.0.b".split("."),
        id(TEST_TENSOR_C),
        id="mixed",
    ),
]

TEST_COMBINATIONS_WALK = [
    pytest.param({"a": {"b": TEST_TENSOR_A}}, [id(TEST_TENSOR_A)], id="dict"),
    pytest.param((TEST_TENSOR_A, TEST_TENSOR_B), [id(TEST_TENSOR_A), id(TEST_TENSOR_B)], id="tuple"),
    pytest.param([TEST_TENSOR_A, TEST_TENSOR_B], [id(TEST_TENSOR_A), id(TEST_TENSOR_B)], id="list"),
    pytest.param(
        test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_B),
        [id(TEST_TENSOR_A), id(TEST_TENSOR_B)],
        id="namedtuple",
    ),
    pytest.param(
        test_dataclass(TEST_TENSOR_A, TEST_TENSOR_B),
        [id(TEST_TENSOR_A), id(TEST_TENSOR_B)],
        id="dataclass",
    ),
    pytest.param(
        {
            "a": (
                12345,
                test_dataclass(TEST_TENSOR_A, [test_namedtuple(TEST_TENSOR_B, TEST_TENSOR_C)]),
            )
        },
        [id(TEST_TENSOR_A), id(TEST_TENSOR_B), id(TEST_TENSOR_C)],
        id="mixed",
    ),
]

TEST_COMBINATIONS_SPLIT = [
    pytest.param({"a": {"b": TEST_TENSOR_A}}, ["a.b".split(".")], [], id="dict"),
    pytest.param((TEST_TENSOR_A, TEST_TENSOR_B), ["1"], ["0"], id="tuple"),
    pytest.param([TEST_TENSOR_A, TEST_TENSOR_B], ["1"], ["0"], id="list"),
    pytest.param(test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_B), ["b"], ["a"], id="namedtuple"),
    pytest.param(test_dataclass(TEST_TENSOR_A, TEST_TENSOR_B), ["b"], ["a"], id="dataclass"),
    pytest.param(
        {
            "a": (
                12345,
                test_dataclass(TEST_TENSOR_A, [test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_C)]),
            )
        },
        ["a.1.b.0.b".split(".")],
        ["a.1.a".split(".")],
        id="mixed",
    ),
]


@pytest.mark.parametrize("nested_object,path,expected_id", TEST_COMBINATIONS)
def test_get_tree_element(nested_object, path, expected_id):
    assert id(get_tree_element(nested_object, path)) == expected_id


@pytest.mark.parametrize("nested_object,tensor_ids", TEST_COMBINATIONS_WALK)
def test_walk_tree_with_paths(nested_object, tensor_ids):
    walked_tensors = [id(tensor) for path, tensor in walk_tree_with_paths(nested_object)]
    for tensor_id in tensor_ids:
        assert tensor_id in walked_tensors


@pytest.mark.parametrize("nested_object,split_paths,keep_paths", TEST_COMBINATIONS_SPLIT)
def test_split_tree(nested_object, split_paths, keep_paths):
    non_split_tensors = {
        ".".join(path): get_tree_element(nested_object, path) for path in split_paths + keep_paths
    }

    for i, splitted_object in enumerate(split_tree(nested_object, split_paths, 0, 1)):
        # Check the slices are correct.
        for split_path in split_paths:
            split = get_tree_element(splitted_object, split_path)
            assert torch.allclose(non_split_tensors[".".join(split_path)][i], split)

        # Check non_split_tensors are not copied.
        for path in keep_paths:
            assert id(non_split_tensors[".".join(path)]) == id(
                get_tree_element(splitted_object, path)
            )
