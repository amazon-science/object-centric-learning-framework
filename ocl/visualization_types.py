"""Classes for handling different types of visualizations."""
import dataclasses
from typing import Any, List, Optional, Union

import matplotlib.pyplot
import torch
from torchtyping import TensorType


class SummaryWriter:
    """Placeholder class for SummaryWriter.

    Emulates interface of `torch.utils.tensorboard.SummaryWriter`.
    """

    def add_figure(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def add_video(self, *args, **kwargs):
        pass

    def add_embedding(self, *args, **kwargs):
        pass


def dataclass_to_dict(d):
    return {field.name: getattr(d, field.name) for field in dataclasses.fields(d)}


@dataclasses.dataclass
class Visualization:
    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        pass


@dataclasses.dataclass
class Figure(Visualization):
    """Matplotlib figure."""

    figure: matplotlib.pyplot.figure
    close: bool = True

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_figure(**dataclass_to_dict(self), tag=tag, global_step=global_step)


@dataclasses.dataclass
class Image(Visualization):
    """Single image."""

    img_tensor: torch.Tensor
    dataformats: str = "CHW"

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_image(**dataclass_to_dict(self), tag=tag, global_step=global_step)


@dataclasses.dataclass
class Images(Visualization):
    """Batch of images."""

    img_tensor: torch.Tensor
    dataformats: str = "NCHW"

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_images(**dataclass_to_dict(self), tag=tag, global_step=global_step)


@dataclasses.dataclass
class Video(Visualization):
    """Batch of videos."""

    vid_tensor: TensorType["batch_size", "frames", "channels", "height", "width"]  # noqa: F821
    fps: Union[int, float] = 4

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_video(**dataclass_to_dict(self), tag=tag, global_step=global_step)


class Embedding(Visualization):
    """Batch of embeddings."""

    mat: TensorType["batch_size", "feature_dim"]  # noqa: F821
    metadata: Optional[List[Any]] = None
    label_img: Optional[TensorType["batch_size", "channels", "height", "width"]] = None  # noqa: F821
    metadata_header: Optional[List[str]] = None

    def add_to_experiment(self, experiment: SummaryWriter, tag: str, global_step: int):
        experiment.add_embedding(**dataclass_to_dict(self), tag=tag, global_step=global_step)
