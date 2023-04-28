from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, make_grid

from ocl import visualization_types


def _flow_tensor_to_rgb_tensor(flow, flow_scaling_factor=50.0):
    """Visualizes flow motion image as an RGB image.

    Adapted from github.com/google-research/slot-attention-video/blob/main/savi/lib/preprocessing.py

    Args:
        flow: A tensor either of shape [..., 2, height, width].
        flow_scaling_factor: How much to scale flow for visualization.

    Returns:
        A visualization tensor with the same shape as flow, except with three channels.

    """
    hypot = lambda a, b: (a**2.0 + b**2.0) ** 0.5  # sqrt(a^2 + b^2)
    flow = torch.moveaxis(flow, -3, -1)
    height, width = flow.shape[-3:-1]
    scaling = flow_scaling_factor / hypot(height, width)
    x, y = flow[..., 0], flow[..., 1]
    motion_angle = torch.atan2(y, x)
    motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
    motion_magnitude = hypot(y, x)
    motion_magnitude = torch.clip(motion_magnitude * scaling, 0.0, 1.0)
    value_channel = torch.ones_like(motion_angle)
    flow_hsv = torch.stack([motion_angle, motion_magnitude, value_channel], dim=-1)
    flow_rbg = matplotlib.colors.hsv_to_rgb(flow_hsv.detach().numpy())
    flow_rbg = torch.moveaxis(torch.Tensor(flow_rbg), -1, -3)
    return flow_rbg


def _nop(arg):
    return arg


class VisualizationMethod(ABC):
    """Abstract base class of a visualization method."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> visualization_types.Visualization:
        """Comput visualization output.

        A visualization method takes some inputs and returns a Visualization.
        """
        pass


class Image(VisualizationMethod):
    """Visualize an image."""

    def __init__(
        self,
        n_instances: int = 8,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        as_grid: bool = True,
    ):
        """Initialize image visualization.

        Args:
            n_instances: Number of instances to visualize
            n_row: Number of rows when `as_grid=True`
            denormalization: Function to map from normalized inputs to unnormalized values
            as_grid: Output a grid of images
        """
        self.n_instances = n_instances
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.as_grid = as_grid

    def __call__(
        self, image: torch.Tensor
    ) -> Union[visualization_types.Image, visualization_types.Images]:
        """Visualize image.

        Args:
            image: Tensor to visualize as image

        Returns:
            Visualized image or images.
        """
        image = self.denormalization(image[: self.n_instances].cpu())
        if self.as_grid:
            return visualization_types.Image(make_grid(image, nrow=self.n_row))
        else:
            return visualization_types.Images(image)


class Video(VisualizationMethod):
    def __init__(
        self,
        n_instances: int = 8,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        as_grid: bool = True,
        fps: int = 10,
    ):
        """Initialize video visualization.

        Args:
            n_instances: Number of instances to visualize
            n_row: Number of rows when `as_grid=True`
            denormalization: Function to map from normalized inputs to unnormalized values
            as_grid: Output a grid of images
            fps: Frames per second
        """
        self.n_instances = n_instances
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.as_grid = as_grid
        self.fps = fps

    def __call__(self, video: torch.Tensor) -> visualization_types.Video:
        """Visualize video.

        Args:
            video: Tensor to visualize as video

        Returns:
            Visualized video.
        """
        video = video[: self.n_instances].cpu()
        if self.as_grid:
            video = torch.stack(
                [
                    make_grid(self.denormalization(frame.unsqueeze(1)).squeeze(1), nrow=self.n_row)
                    for frame in torch.unbind(video, 1)
                ],
                dim=0,
            ).unsqueeze(0)
        return visualization_types.Video(video, fps=self.fps)


class Mask(VisualizationMethod):
    def __init__(
        self,
        n_instances: int = 8,
        fps: int = 10,
    ):
        """Initialize mask visualization.

        Args:
            n_instances: Number of masks to visualize
            fps: Frames per second in the case of video input.
        """
        self.n_instances = n_instances
        self.fps = fps

    def __call__(
        self, mask: torch.Tensor
    ) -> Union[visualization_types.Image, visualization_types.Video]:
        """Visualize mask.

        Args:
            mask: Tensor to visualize as mask

        Returns:
            Visualized mask.
        """
        masks = mask[: self.n_instances].cpu().contiguous()
        image_shape = masks.shape[-2:]
        n_objects = masks.shape[-3]

        if masks.dim() == 5:
            # Handling video data.
            # bs x frames x objects x H x W
            mask_vis = masks.transpose(1, 2).contiguous()
            flattened_masks = mask_vis.flatten(0, 1).unsqueeze(2)

            # Draw masks inverted as they are easier to print.
            mask_vis = torch.stack(
                [
                    make_grid(1.0 - masks, nrow=n_objects)
                    for masks in torch.unbind(flattened_masks, 1)
                ],
                dim=0,
            )
            mask_vis = mask_vis.unsqueeze(0)
            return visualization_types.Video(mask_vis, fps=self.fps)
        elif masks.dim() == 4:
            # Handling image data.
            # bs x objects x H x W
            # Monochrome image with single channel.
            masks = masks.view(-1, 1, *image_shape)
            # Draw masks inverted as they are easier to print.
            return visualization_types.Image(make_grid(1.0 - masks, nrow=n_objects))
        else:
            raise RuntimeError("Unsupported tensor dimensions.")


class VisualObject(VisualizationMethod):
    def __init__(
        self,
        n_instances: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fps: int = 10,
    ):
        """Initialize VisualObject visualization.

        Args:
            n_instances: Number of masks to visualize
            denormalization: Function to map from normalized inputs to unnormalized values
            fps: Frames per second in the case of video input.
        """
        self.n_instances = n_instances
        self.denormalization = denormalization if denormalization else _nop
        self.fps = fps

    def __call__(
        self, object: torch.Tensor, mask: torch.Tensor
    ) -> Union[Dict[str, visualization_types.Image], Dict[str, visualization_types.Video]]:
        """Visualize a visual object.

        Args:
            object: Tensor of objects to visualize
            mask: Tensor of object masks

        Returns:
            Visualized objects as masked images and masks in the keys `reconstruction` and `mask`.
        """
        objects = object[: self.n_instances].cpu()
        masks = mask[: self.n_instances].cpu().contiguous()

        image_shape = objects.shape[-3:]
        n_objects = objects.shape[-4]

        if objects.dim() == 6:
            # Handling video data.
            # bs x frames x objects x C x H x W

            # We need to denormalize prior to constructing the grid, yet the denormalization
            # method assumes video input. We thus convert a frame into a single frame video and
            # remove the additional dimension prior to make_grid.
            # Switch object and frame dimension.
            object_vis = objects.transpose(1, 2).contiguous()
            mask_vis = masks.transpose(1, 2).contiguous()
            flattened_masks = mask_vis.flatten(0, 1).unsqueeze(2)
            object_vis = self.denormalization(object_vis.flatten(0, 1))
            # Keep object pixels and apply white background to non-objects parts.
            object_vis = object_vis * flattened_masks + (1.0 - flattened_masks)
            object_vis = torch.stack(
                [
                    make_grid(
                        object_vis_frame,
                        nrow=n_objects,
                    )
                    for object_vis_frame in torch.unbind(object_vis, 1)
                ],
                dim=0,
            )
            # Add batch dimension as this is required for video input.
            object_vis = object_vis.unsqueeze(0)

            # Draw masks inverted as they are easier to print.
            mask_vis = torch.stack(
                [
                    make_grid(1.0 - masks, nrow=n_objects)
                    for masks in torch.unbind(flattened_masks, 1)
                ],
                dim=0,
            )
            mask_vis = mask_vis.unsqueeze(0)
            return {
                "reconstruction": visualization_types.Video(object_vis, fps=self.fps),
                "mask": visualization_types.Video(mask_vis, fps=self.fps),
            }
        elif objects.dim() == 5:
            # Handling image data.
            # bs x objects x C x H x W
            object_reconstructions = self.denormalization(objects.view(-1, *image_shape))
            # Monochrome image with single channel.
            masks = masks.view(-1, 1, *image_shape[1:])
            # Save object reconstructions as RGBA image. make_grid does not support RGBA input, thus
            # we combine the channels later.  For the masks we need to pad with 1 as we want the
            # borders between images to remain visible (i.e. alpha value of 1.)
            masks_grid = make_grid(masks, nrow=n_objects, pad_value=1.0)
            object_grid = make_grid(object_reconstructions, nrow=n_objects)
            # masks_grid expands the image to three channels, which we don't need. Only keep one, and
            # use it as the alpha channel. After make_grid the tensor has the shape C X W x H.
            object_grid = torch.cat((object_grid, masks_grid[:1]), dim=0)

            return {
                "reconstruction": visualization_types.Image(object_grid),
                # Draw masks inverted as they are easier to print.
                "mask": visualization_types.Image(make_grid(1.0 - masks, nrow=n_objects)),
            }
        else:
            raise RuntimeError("Unsupported tensor dimensions.")


class Segmentation(VisualizationMethod):
    """Segmentaiton visualization."""

    def __init__(
        self,
        n_instances: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Initialize segmentation visualization.

        Args:
            n_instances: Number of masks to visualize
            denormalization: Function to map from normalized inputs to unnormalized values
        """
        self.n_instances = n_instances
        self.denormalization = denormalization if denormalization else _nop
        self._cmap_cache: Dict[int, List[Tuple[int, int, int]]] = {}

    def _get_cmap(self, num_classes: int) -> List[Tuple[int, int, int]]:
        if num_classes in self._cmap_cache:
            return self._cmap_cache[num_classes]

        from matplotlib import cm

        if num_classes <= 20:
            mpl_cmap = cm.get_cmap("tab20", num_classes)(range(num_classes))
        else:
            mpl_cmap = cm.get_cmap("turbo", num_classes)(range(num_classes))

        cmap = [tuple((255 * cl[:3]).astype(int)) for cl in mpl_cmap]
        self._cmap_cache[num_classes] = cmap
        return cmap

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Optional[visualization_types.Image]:
        """Visualize segmentation overlaying original image.

        Args:
            image: Image to overlay
            mask: Masks of individual objects
        """
        image = image[: self.n_instances].cpu()
        mask = mask[: self.n_instances].cpu().contiguous()
        if image.dim() == 4:  # Only support image data at the moment.
            input_image = self.denormalization(image)
            n_objects = mask.shape[1]

            masks_argmax = mask.argmax(dim=1)[:, None]
            classes = torch.arange(n_objects)[None, :, None, None].to(masks_argmax)
            masks_one_hot = masks_argmax == classes

            cmap = self._get_cmap(n_objects)
            masks_on_image = torch.stack(
                [
                    draw_segmentation_masks(
                        (255 * img).to(torch.uint8), mask, alpha=0.75, colors=cmap
                    )
                    for img, mask in zip(input_image.to("cpu"), masks_one_hot.to("cpu"))
                ]
            )

            return visualization_types.Image(make_grid(masks_on_image, nrow=8))
        return None


class Flow(VisualizationMethod):
    def __init__(
        self,
        n_instances: int = 8,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        as_grid: bool = True,
    ):
        self.n_instances = n_instances
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.as_grid = as_grid

    def __call__(self, flow: torch.Tensor):
        flow = self.denormalization(flow[: self.n_instances].cpu())
        flow = _flow_tensor_to_rgb_tensor(flow)
        if self.as_grid:
            return visualization_types.Image(make_grid(flow, nrow=self.n_row))
        else:
            return visualization_types.Images(flow)


color_list = [
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "black",
    "#614051",
    "#cd7f32",
    "#008b8b",
    "#556b2f",
    "#ffbf00",
]


class TrackedObject(VisualizationMethod):
    def __init__(
        self,
        n_clips: int = 3,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.n_clips = n_clips
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop

    def __call__(
        self, video: torch.Tensor, bbox: torch.Tensor, idx: torch.Tensor
    ) -> Optional[visualization_types.Visualization]:
        video = video[: self.n_clips].cpu()
        num_frames = video.shape[1]

        bbox = bbox[: self.n_clips].to(torch.uint8).cpu()
        idx = idx[: self.n_clips].cpu()

        rendered_video = torch.zeros_like(video)
        num_color = len(color_list)

        for cidx in range(self.n_clips):
            for fidx in range(num_frames):
                if cidx >= idx.shape[0] or fidx >= idx.shape[1]:
                    break
                cur_obj_idx = idx[cidx, fidx]
                valid_index = cur_obj_idx > -1
                cur_obj_idx = cur_obj_idx[valid_index].to(torch.int)
                cur_color_list = [
                    color_list[obj_idx % num_color] for obj_idx in cur_obj_idx.numpy().tolist()
                ]
                frame = (video[cidx, fidx] * 256).to(torch.uint8)
                frame = draw_bounding_boxes(
                    frame, bbox[cidx, fidx][valid_index], colors=cur_color_list
                )
                rendered_video[cidx, fidx] = frame

        rendered_video = (
            torch.stack(
                [
                    make_grid(self.denormalization(frame.unsqueeze(1)).squeeze(1), nrow=self.n_row)
                    for frame in torch.unbind(rendered_video, 1)
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(torch.float32)
        )

        return visualization_types.Video(rendered_video)


class TrackedObject_from_Mask(VisualizationMethod):
    def __init__(
        self,
        n_clips: int = 3,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        fps: int = 10,
    ):
        self.n_clips = n_clips
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop
        self.fps = fps

    def __call__(
        self,
        video: torch.Tensor,
        object_masks: torch.Tensor,
    ) -> Optional[visualization_types.Visualization]:
        video = video[: self.n_clips].cpu()
        num_frames = video.shape[1]

        masks = object_masks[: self.n_clips]
        B, F, C, h, w = masks.shape
        masks = masks > 0.5

        rendered_video = torch.zeros_like(video)

        for cidx in range(self.n_clips):
            for fidx in range(num_frames):

                idx = []
                for i in range(C):
                    if torch.sum(masks[cidx, fidx][i]) != 0:
                        idx.append(i)
                bbox = masks_to_boxes(masks[cidx, fidx][np.array(idx)]).cpu().contiguous()

                for id in idx:
                    pred_h = bbox[id][2] - bbox[id][0]
                    pred_w = bbox[id][3] - bbox[id][1]
                    thres = h * w * 0.2
                    if pred_h * pred_w >= thres:
                        idx.remove(id)
                cur_obj_idx = np.array(idx)
                cur_color_list = [color_list[obj_idx] for obj_idx in idx]
                frame = (video[cidx, fidx] * 256).to(torch.uint8)
                frame = draw_bounding_boxes(frame, bbox[cur_obj_idx], colors=cur_color_list)
                rendered_video[cidx, fidx] = frame

        rendered_video = (
            torch.stack(
                [
                    make_grid(self.denormalization(frame.unsqueeze(1)).squeeze(1), nrow=self.n_row)
                    for frame in torch.unbind(rendered_video, 1)
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(torch.float32)
            / 256
        )

        return visualization_types.Video(rendered_video, fps=self.fps)


def masks_to_bboxes_xyxy(masks: torch.Tensor, empty_value: float = -1.0) -> torch.Tensor:
    """Compute bounding boxes around the provided masks.

    Adapted from DETR: https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

    Args:
        masks: Tensor of shape (N, H, W), where N is the number of masks, H and W are the spatial
            dimensions.
        empty_value: Value bounding boxes should contain for empty masks.

    Returns:
        Tensor of shape (N, 4), containing bounding boxes in (x1, y1, x2, y2) format, where (x1, y1)
        is the coordinate of top-left corner and (x2, y2) is the coordinate of the bottom-right
        corner (inclusive) in pixel coordinates. If mask is empty, all coordinates contain
        `empty_value` instead.
    """
    masks = masks.bool()
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    large_value = 1e8
    inv_mask = ~masks

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    bboxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)  # x1y1x2y2
    bboxes[x_min == large_value] = empty_value
    return bboxes


class ObjectMOT(VisualizationMethod):
    def __init__(
        self,
        n_clips: int = 3,
        n_row: int = 8,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.n_clips = n_clips
        self.n_row = n_row
        self.denormalization = denormalization if denormalization else _nop

    def __call__(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[visualization_types.Visualization]:
        video = video[: self.n_clips].cpu()
        num_frames = video.shape[1]

        masks = mask[: self.n_clips].cpu().contiguous()
        B, F, C, h, w = masks.shape  # [5, 6, 11, 64, 64]
        masks = masks.flatten(0, 1)
        masks = masks > 0.5
        bbox = masks_to_bboxes_xyxy(masks.flatten(0, 1)).unflatten(0, (B, F, C))

        rendered_video = torch.zeros_like(video)

        color_list = [
            "red",
            "blue",
            "green",
            "yellow",
            "pink",
            "black",
            "#614051",
            "#cd7f32",
            "#008b8b",
            "#556b2f",
            "#ffbf00",
            "white",
            "orange",
            "gray",
            "#ffbf00",
        ]

        for cidx in range(self.n_clips):
            for fidx in range(num_frames):
                cur_obj_box = bbox[cidx, fidx][:, 0] != -1.0
                cur_obj_idx = cur_obj_box.nonzero()[:, 0].detach().cpu().numpy()
                idx = cur_obj_idx.tolist()
                cur_obj_idx = np.array(idx)
                cur_color_list = [color_list[obj_idx] for obj_idx in idx]
                frame = (video[cidx, fidx] * 256).to(torch.uint8)
                frame = draw_bounding_boxes(
                    frame, bbox[cidx, fidx][cur_obj_idx], colors=cur_color_list
                )
                rendered_video[cidx, fidx] = frame

        rendered_video = (
            torch.stack(
                [
                    make_grid(self.denormalization(frame.unsqueeze(1)).squeeze(1), nrow=self.n_row)
                    for frame in torch.unbind(rendered_video, 1)
                ],
                dim=0,
            )
            .unsqueeze(0)
            .to(torch.float32)
            / 256
        )

        return visualization_types.Video(rendered_video)


class TextToImageMatching(VisualizationMethod):
    def __init__(
        self,
        denormalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        row_is_image: bool = True,
        n_instances: Optional[int] = None,
    ):
        self.denormalization = denormalization if denormalization else _nop
        self.row_is_image = row_is_image
        self.n_instances = n_instances

    def __call__(
        self, image: torch.Tensor, text: List[str], similarities: torch.Tensor
    ) -> Optional[visualization_types.Visualization]:
        n_images = len(image)
        n_texts = len(text)

        image = image.detach()
        if self.row_is_image:
            # Code assumes that each rows in the similarity matrix correspond to a single text.
            similarities = similarities.T.detach()
        else:
            similarities = similarities.detach()

        assert n_texts == similarities.shape[0]
        assert n_images == similarities.shape[1]

        if self.n_instances:
            n_images = min(self.n_instances, n_images)
            n_texts = min(self.n_instances, n_texts)

        image = (
            torch.clamp((255 * self.denormalization(image[:n_images])), 0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
        )
        text = text[:n_texts]
        similarities = similarities[:n_texts, :n_images].cpu()

        fig, ax = plt.subplots(1, 1, figsize=(20, 14))
        ax.imshow(similarities, vmin=0.1, vmax=0.3)

        ax.set_yticks(range(n_texts), text, fontsize=18, wrap=True)
        ax.set_xticks([])
        for i, cur_image in enumerate(image):
            ax.imshow(cur_image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        for x in range(similarities.shape[1]):
            for y in range(similarities.shape[0]):
                ax.text(x, y, f"{similarities[y, x]:.2f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            ax.spines[side].set_visible(False)

        ax.set_xlim([-0.5, n_images - 0.5])
        ax.set_ylim([n_texts + 0.5, -2])
        return visualization_types.Figure(fig)
