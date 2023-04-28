import pytest
import torch

from ocl import datasets


def test_collate_with_autopadding(assert_tensors_equal):
    batch = [torch.tensor([1, 1]), torch.tensor([2, 2])]
    result = datasets.collate_with_autopadding(batch)
    assert_tensors_equal(result, torch.tensor([[1, 1], [2, 2]]))

    batch = [torch.tensor([1]), torch.tensor([2, 2]), torch.tensor([3, 3, 3])]
    result = datasets.collate_with_autopadding(batch)
    assert_tensors_equal(result, torch.tensor([[1, 0, 0], [2, 2, 0], [3, 3, 3]]))

    batch = [torch.ones(2, 1), torch.ones(1, 2)]
    result = datasets.collate_with_autopadding(batch)
    assert_tensors_equal(
        result, torch.tensor([[[1, 0], [1, 0]], [[1, 1], [0, 0]]], dtype=torch.float32)
    )

    # Collating entries of different dimensionalities should not work
    with pytest.raises(ValueError):
        batch = [torch.ones(1, 2), torch.ones(1, 2, 3)]
        result = datasets.collate_with_autopadding(batch)

    with pytest.raises(ValueError):
        batch = [torch.ones(1, 2, 3), torch.ones(1, 2)]
        result = datasets.collate_with_autopadding(batch)
