import pytest
import torch

# TODO(hornmax): After reorganizing some of the code, the tests should also be reorganized.
from ocl.metrics.utils import tensor_to_one_hot
from ocl.utils.masking import CreateSlotMask
from ocl.utils.resizing import Resize, resize_patches_to_image


@pytest.mark.parametrize(
    "n_channels,n_patches,size,scale_factor,expected_size,expect_error",
    [
        (3, 4, 10, None, 10, False),
        (2, 4, None, 1.5, 3, False),
        (2, 5, 10, None, None, True),  # Error: Number of patches is not squarable
        (2, 4, None, None, None, True),  # Error: None of size and scale_factor specified
        (2, 4, 8, 2, None, True),  # Error: Both of size and scale_factor specified
    ],
)
def test_resize_patches_to_image(
    n_channels, n_patches, size, scale_factor, expected_size, expect_error
):
    bs = 2
    patches = torch.randn(bs, n_channels, n_patches)

    if expect_error:
        with pytest.raises(ValueError):
            image = resize_patches_to_image(patches, size, scale_factor)
    else:
        image = resize_patches_to_image(patches, size, scale_factor)
        assert image.shape == (bs, n_channels, expected_size, expected_size)


def test_tensor_to_one_hot():
    inp = torch.tensor([[1.0, 0.0, -10.0], [0.0, 2.0, 1.99], [-1, 0, 0.01]])
    result = tensor_to_one_hot(inp, dim=1)
    assert torch.allclose(result, torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.long))


def test_create_slot_mask():
    max_slots = 5
    n_slots = torch.tensor([0, 2, 5])

    create_slot_mask = CreateSlotMask(max_slots)
    mask = create_slot_mask(n_slots)

    expected_mask = [[False] * ns + [True] * (max_slots - ns) for ns in n_slots]
    assert torch.allclose(mask, torch.tensor(expected_mask))


@pytest.mark.parametrize(
    "inp_shape,size,expected_shape,patch_mode,channels_last",
    [
        ((2, 3, 9), (10, 20), (2, 3, 10, 20), True, False),
        ((4, 2, 9, 3), (11, 21), (4, 2, 3, 11, 21), True, True),
        ((3, 2, 4, 6, 5), 10, (3, 2, 4, 10, 10), False, False),
        ((1, 6, 5, 4), (10, 20), (1, 4, 10, 20), False, True),
    ],
)
def test_resize(inp_shape, size, expected_shape, patch_mode, channels_last):
    # Explicitly pass size
    resizer = Resize(size=size, patch_mode=patch_mode, channels_last=channels_last)
    inp = torch.rand(*inp_shape)
    outp = resizer(inp)
    assert outp.shape == expected_shape

    # Use tensor specifying size
    resizer = Resize(patch_mode=patch_mode, channels_last=channels_last)
    size_as_list = [size, size] if isinstance(size, int) else list(size)
    size_tensors = [
        torch.ones(1, *size_as_list),
        torch.tensor(size_as_list),
        torch.tensor(size_as_list).unsqueeze(0),
        torch.tensor([size_as_list, size_as_list]),
    ]
    for size_tensor in size_tensors:
        outp = resizer(inp, size_tensor)
        assert outp.shape == expected_shape
