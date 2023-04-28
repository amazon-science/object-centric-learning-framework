import numpy as np
import pytest
import torch

from ocl import preprocessing


def test_add_segmentation_mask_from_instance_mask():
    inp = np.zeros((3, 3, 3, 1), dtype=np.uint8)
    inp[0, 0] = 1
    inp[1, 1] = 2
    inp[2, 2] = 1

    result = preprocessing.AddSegmentationMaskFromInstanceMask.convert(inp)

    expected_result = np.zeros((2, 3, 3, 1), dtype=np.uint8)
    expected_result[0, [0, 2]] = 1
    expected_result[1, 1] = 2

    assert np.allclose(result, expected_result)


@pytest.mark.parametrize(
    "input_shape, size, expected_shape, max_size",
    [
        ((1, 16, 16), 31, (1, 31, 31), None),
        ((5, 16, 16), (32, 20), (5, 32, 20), None),
        ((3, 1, 5, 15), 10, (3, 1, 10, 30), None),
        ((1, 3, 5, 15), 20, (1, 3, 10, 30), 30),
    ],
)
def test_resize_nearest_exact(input_shape, size, expected_shape, max_size):
    resize = preprocessing.ResizeNearestExact(size, max_size=max_size)

    inp = torch.rand(*input_shape)
    resized = resize(inp)
    assert resized.shape == expected_shape
    assert resized.dtype == inp.dtype

    resized_uint8 = resize(inp.to(torch.uint8))
    assert resized_uint8.shape == expected_shape
    assert resized_uint8.dtype == torch.uint8


def test_compressed_mask():
    n_frames, n_objects, w, h = 10, 8, 10, 10
    fake_mask = np.random.rand(n_frames, n_objects, w, h) > 0.5
    compress = preprocessing.CompressMask()
    decompress = preprocessing.CompressedMaskToTensor()

    compressed = compress(fake_mask)
    decompressed = (decompress(compressed) > 0.0).detach().numpy()

    assert (fake_mask == decompressed).all()
