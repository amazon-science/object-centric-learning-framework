from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch

from ocl.utils.routing import DataRouter, Recurrent

TEST_TENSOR_A = torch.randn(10, 10)
TEST_TENSOR_B = torch.randn(10, 10)
TEST_TENSOR_C = torch.randn(10, 10)

ZERO_TENSOR = torch.zeros(10)

test_namedtuple = namedtuple("test_namedtuple", ["a", "b"])


@dataclass
class test_dataclass:
    a: torch.Tensor
    b: torch.Tensor


class Cumsum(torch.nn.Module):
    """Simple cumulative sum for testing."""

    def forward(self, inputs):
        input = inputs.get("input")
        previous_output = inputs.get("previous_output")
        return previous_output + input


TEST_COMBINATIONS_ROUTING = [
    pytest.param(
        {"a": {"b": TEST_TENSOR_A}}, {"input_1": "a.b"}, {"input_1": TEST_TENSOR_A}, id="dict"
    ),
    pytest.param(
        {"a": (TEST_TENSOR_A, TEST_TENSOR_B)},
        {"input_1": "a.1"},
        {"input_1": TEST_TENSOR_B},
        id="tuple",
    ),
    pytest.param(
        {"a": [TEST_TENSOR_A, TEST_TENSOR_B]},
        {"input_1": "a.1"},
        {"input_1": TEST_TENSOR_B},
        id="list",
    ),
    pytest.param(
        {"a": test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_B)},
        {"input_1": "a.b"},
        {"input_1": TEST_TENSOR_B},
        id="namedtuple",
    ),
    pytest.param(
        {"a": test_dataclass(TEST_TENSOR_A, TEST_TENSOR_B)},
        {"input_1": "a.b"},
        {"input_1": TEST_TENSOR_B},
        id="dataclass",
    ),
    pytest.param(
        {
            "a": (
                12345,
                test_dataclass(TEST_TENSOR_A, [test_namedtuple(TEST_TENSOR_A, TEST_TENSOR_C)]),
            )
        },
        {"input_1": "a.1.b.0.b", "input_2": "a.1.b.0.a"},
        {"input_1": TEST_TENSOR_C, "input_2": TEST_TENSOR_A},
        id="mixed",
    ),
]


@pytest.mark.parametrize("tree,input_mapping,expected_inputs", TEST_COMBINATIONS_ROUTING)
def test_data_router(tree, input_mapping, expected_inputs):
    # Having some problems with inspection and mock module, thus we use our own approach here.
    called_with = {}

    class MyModule(torch.nn.Module):
        def forward(self, input_1, input_2=None):
            called_with["input_1"] = input_1
            called_with["input_2"] = input_2
            return "return_value"

    mock_module = MyModule()
    data_router = DataRouter(mock_module, input_mapping)
    return_value = data_router(**tree)
    assert return_value == "return_value"
    for name, value in expected_inputs.items():
        assert id(called_with[name]) == id(value)


def test_recurrent():
    recurrent_module = Recurrent(Cumsum(), ["input"], {"": "initial"}, 0, 1)
    output = recurrent_module(inputs={"input": TEST_TENSOR_A, "initial": ZERO_TENSOR})
    assert torch.allclose(output, torch.cumsum(TEST_TENSOR_A, 0), rtol=1e-4, atol=1e-6)
