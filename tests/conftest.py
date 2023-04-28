import os

import pytest
import torch


@pytest.fixture
def assert_tensors_equal():
    def _assert_tensors_equal(tensor: torch.Tensor, expected_tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == expected_tensor.shape
        assert torch.allclose(tensor, expected_tensor)

    return _assert_tensors_equal


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_generate_tests(metafunc):
    # Set variable that tests are running.
    os.environ["RUNNING_TESTS"] = "true"
