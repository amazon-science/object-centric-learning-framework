import os

import decord
import numpy as np
import pytest
import torch
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper

from ocl.preprocessing import MaskInstances
from ocl.transforms import (
    DecodeRandomStridedWindow,
    DecodeRandomWindow,
    Map,
    SampleSlices,
    SplitConsecutive,
    VideoDecoder,
)

# This is temporarily deactivated as we cannot distribute the test videos with the code.
# try:
#     vr = decord.VideoReader(
#         os.path.join(os.path.dirname(__file__), "data", "color_red.mp4"), ctx=decord.gpu()
#     )
#     GPU_AVAILABLE = True
# except decord._ffi.base.DECORDError as e:
#     if "CUDA not enabled" in str(e):
#         GPU_AVAILABLE = False
#     else:
#         raise e
GPU_AVAILABLE = False
SKIP_VIDEO_DATA_TESTS = True


class FakeDataset(IterDataPipe):
    def __iter__(self):
        for i in range(10):
            yield {
                "__key__": str(i),
                "image": np.random.rand(15, 32, 32, 3),
                "mask": np.random.rand(15, 32, 32, 3),
                "do_not_touch": np.random.rand(15, 32, 32, 3),
                "consecutive_data": np.arange(15)[:, None],
            }


@pytest.mark.parametrize("n_frames,expected_len", [(1, 10), (2, 20), (-1, 150)])
def test_sample_slices(n_frames, expected_len):
    dataset = FakeDataset()
    transform = SampleSlices(
        n_slices_per_input=n_frames,
        fields=["image", "mask"],
        dim=0,
    )

    transformed_dataset = transform(dataset)
    transformed_data = list(transformed_dataset)
    assert len(transformed_data) == expected_len
    assert transformed_data[0]["image"].shape == (32, 32, 3)
    assert transformed_data[0]["mask"].shape == (32, 32, 3)
    assert transformed_data[0]["do_not_touch"].shape == (15, 32, 32, 3)


@pytest.mark.parametrize("n_frames", [1, 2, -1])
def test_sample_slices_reproducability(n_frames):
    dataset = FakeDataset()
    transform = SampleSlices(
        n_slices_per_input=n_frames,
        fields=["image", "mask"],
        dim=0,
        seed=123,
        shuffle_buffer_size=1,
    )

    transformed_dataset = transform(dataset)
    keys = [element["__key__"] for element in transformed_dataset]
    keys2 = [element["__key__"] for element in transformed_dataset]
    assert keys == keys2


@pytest.mark.parametrize(
    "fields,shape",
    [
        (("image",), (32, 32, 3)),
    ],
)
def test_sample_slices_shapes(fields, shape):
    dataset = FakeDataset()
    transform = SampleSlices(
        n_slices_per_input=2,
        fields=fields,
        dim=0,
        seed=123,
    )

    transformed_dataset = transform(dataset)
    transformed_data = next(transformed_dataset.__iter__())
    assert transformed_data["image"].shape == shape


@pytest.mark.parametrize("n_frames", [2, 5, 15])
def test_split_consecutive(n_frames):
    dataset = FakeDataset()
    transform = SplitConsecutive(n_frames, ["consecutive_data"])

    transformed_dataset = transform(dataset)
    for instance in transformed_dataset:
        consecutive_data = instance["consecutive_data"]
        assert consecutive_data.shape == (n_frames, 1)
        assert instance["image"].shape == (15, 32, 32, 3)
        assert (np.ediff1d(consecutive_data[:, 0]) == 1).all()


class MinimalVideoDataset(IterDataPipe):
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def __iter__(self):
        video_file = open(self.video_path, "rb")
        yield {"__key__": "0", "video.mp4": StreamWrapper(video_file)}


@pytest.mark.skipif(SKIP_VIDEO_DATA_TESTS, reason="Requires test video data.")
@pytest.mark.parametrize("ctx", [decord.cpu(), decord.gpu()] if GPU_AVAILABLE else [decord.cpu()])
def test_video_decoding_whole(ctx):
    dataset = MinimalVideoDataset(os.path.join(os.path.dirname(__file__), "data", "color_red.mp4"))
    transform = VideoDecoder(fields=["video.mp4"])

    transformed_dataset = transform(dataset)
    count = 0
    for instance in transformed_dataset:
        assert count == 0  # Should only return a single element
        assert instance["__key__"] == "0"
        assert instance["video"].shape == (10, 144, 176, 3)
        # Check the input is very red, strangely the ffmpeg definition of red is not 255.
        assert torch.all(instance["video"][..., 0] == 253)


@pytest.mark.skipif(SKIP_VIDEO_DATA_TESTS, reason="Requires test video data.")
@pytest.mark.skipif(
    not GPU_AVAILABLE, reason="No GPU available or decord not compiled with CUDA support."
)
def test_video_decoding_GPU_CPU_consistency():
    dataset = MinimalVideoDataset(
        os.path.join(os.path.dirname(__file__), "data", "example_video.mp4")
    )
    whole_video_cpu = next(
        VideoDecoder(fields=["video.mp4"], video_reader_kwargs={"ctx": decord.cpu()})(
            dataset
        ).__iter__()
    )["video"]
    whole_video_gpu = next(
        VideoDecoder(fields=["video.mp4"], video_reader_kwargs={"ctx": decord.gpu()})(
            dataset
        ).__iter__()
    )["video"]

    assert torch.all(whole_video_gpu.cpu() == whole_video_cpu)


@pytest.mark.skipif(SKIP_VIDEO_DATA_TESTS, reason="Requires test video data.")
@pytest.mark.parametrize("n_consecutive_frames", [2, 5])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("ctx", [decord.cpu(), decord.gpu()] if GPU_AVAILABLE else [decord.cpu()])
def test_video_decoding_random(n_consecutive_frames, stride, ctx):
    dataset = MinimalVideoDataset(
        os.path.join(os.path.dirname(__file__), "data", "example_video.mp4")
    )
    video_reader_kwargs = {"ctx": ctx}
    whole_video = next(
        VideoDecoder(fields=["video.mp4"], video_reader_kwargs=video_reader_kwargs)(
            dataset
        ).__iter__()
    )["video"]

    transform = DecodeRandomWindow(
        n_consecutive_frames,
        fields=["video.mp4"],
        stride=stride,
        video_reader_kwargs=video_reader_kwargs,
    )

    transformed_dataset = transform(dataset)
    count = 0
    for instance in transformed_dataset:
        indices = instance["decoded_indices"]
        video = instance["video"]
        assert count == 0  # Should only return a single element
        assert instance["__key__"].startswith("0")
        assert video.shape[0] == n_consecutive_frames
        assert len(indices) == n_consecutive_frames
        assert np.all(np.diff(indices) == stride)  # Check if frames are consecutive.

        assert torch.all(video == whole_video[indices])


@pytest.mark.skipif(SKIP_VIDEO_DATA_TESTS, reason="Requires test video data.")
@pytest.mark.parametrize("n_consecutive_frames", [2, 5])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("ctx", [decord.cpu(), decord.gpu()] if GPU_AVAILABLE else [decord.cpu()])
def test_video_decoding_strided_random(n_consecutive_frames, stride, ctx):
    dataset = MinimalVideoDataset(
        os.path.join(os.path.dirname(__file__), "data", "example_video.mp4")
    )
    video_reader_kwargs = {"ctx": ctx}
    whole_video = next(
        VideoDecoder(fields=["video.mp4"], video_reader_kwargs=video_reader_kwargs)(
            dataset
        ).__iter__()
    )["video"]

    transform = DecodeRandomStridedWindow(
        n_consecutive_frames,
        fields=["video.mp4"],
        stride=stride,
        video_reader_kwargs=video_reader_kwargs,
    )

    transformed_dataset = transform(dataset)
    count = 0
    for instance in transformed_dataset:
        indices = instance["decoded_indices"]
        video = instance["video"]
        assert count == 0  # Should only return a single element
        assert instance["__key__"].startswith("0")
        assert video.shape[0] == n_consecutive_frames
        assert len(indices) == n_consecutive_frames
        assert np.all(np.diff(indices) == stride)  # Check if frames are consecutive.

        assert torch.all(video == whole_video[indices])


# Strictly speaking this should be in test_preprocessing, we keep it here
# because we require a transformed dataset to test it.
@pytest.mark.parametrize("instances_to_keep", [("1", "5", "8")])
def test_filter_with_masking(instances_to_keep):
    dataset = FakeDataset()
    transform = Map(
        transform=MaskInstances(
            fields=["image"],
            keys_to_keep=instances_to_keep,
        ),
        fields=("image",),
        batch_transform=False,
    )
    transformed_dataset = transform(dataset)

    instances_to_keep = set(instances_to_keep)

    for instance in transformed_dataset:
        key = instance["__key__"]
        if key not in instances_to_keep:
            assert np.isnan(instance["image"]).all()
        else:
            assert not np.isnan(instance["image"]).any()
