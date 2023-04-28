"""Implementation of different types of decoders."""
import dataclasses
import math
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

from ocl.neural_networks.convenience import get_activation_fn
from ocl.neural_networks.positional_embedding import SoftPositionEmbed
from ocl.neural_networks.slate import Conv2dBlockWithGroupNorm
from ocl.utils.resizing import resize_patches_to_image


@dataclasses.dataclass
class SimpleReconstructionOutput:
    reconstruction: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class ReconstructionOutput:
    reconstruction: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821
    object_reconstructions: TensorType[
        "batch_size", "n_objects", "channels", "height", "width"  # noqa: F821
    ]
    masks: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class ReconstructionAmodalOutput:
    reconstruction: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821
    object_reconstructions: TensorType[
        "batch_size", "n_objects", "channels", "height", "width"  # noqa: F821
    ]
    masks: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    masks_vis: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    masks_eval: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class PatchReconstructionOutput:
    reconstruction: TensorType["batch_size", "n_patches", "n_patch_features"]  # noqa: F821
    masks: TensorType["batch_size", "n_objects", "n_patches"]  # noqa: F821
    masks_as_image: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    target: Optional[TensorType["batch_size", "n_patches", "n_patch_features"]] = None  # noqa: F821


@dataclasses.dataclass
class DepthReconstructionOutput(ReconstructionOutput):
    masks_amodal: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    depth_map: Optional[TensorType["batch_size", "height", "width"]] = None  # noqa: F821
    object_depth_map: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    densities: Optional[
        TensorType["batch_size", "n_objects", "n_depth", "height", "width"]  # noqa: F821
    ] = None
    colors: Optional[
        TensorType["batch_size", "n_objects", "n_depth", "channels", "height", "width"]  # noqa: F821
    ] = None


@dataclasses.dataclass
class OpticalFlowPredictionTaskOutput:
    predicted_flow: TensorType["batch_size", "channels", "height", "width"]  # noqa: F821
    object_flows: TensorType["batch_size", "n_objects", "channels", "height", "width"]  # noqa: F821
    masks: TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821


@dataclasses.dataclass
class BBoxOutput:
    bboxes: TensorType["batch_size", "n_objects", "box_dim"]  # noqa: F821
    classes: TensorType["batch_size", "n_objects", "num_classes"]  # noqa: F821
    ori_res_bboxes: TensorType["batch_size", "n_objects", "box_dim"]  # noqa: F821
    inference_obj_idxes: TensorType["batch_size", "n_objects"]  # noqa: F821


def build_grid_of_positions(resolution):
    """Build grid of positions which can be used to create positions embeddings."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    return grid


def get_slotattention_decoder_backbone(object_dim: int, output_dim: int = 4):
    """Get CNN decoder backbone form the original slot attention paper."""
    return nn.Sequential(
        nn.ConvTranspose2d(object_dim, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2, output_padding=0),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, output_dim, 3, stride=1, padding=1, output_padding=0),
    )


def get_savi_decoder_backbone(
    object_dim: int,
    output_dim: int = 4,
    larger_input_arch: bool = False,
    channel_multiplier: float = 1,
):
    """Get CNN decoder backbone form the slot attention for video paper."""
    channels = int(64 * channel_multiplier)
    if larger_input_arch:
        output_stride = 2
        output_padding = 1
    else:
        output_stride = 1
        output_padding = 0
    return nn.Sequential(
        nn.ConvTranspose2d(object_dim, channels, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(channels, channels, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(channels, channels, 5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            channels, channels, 5, stride=output_stride, padding=2, output_padding=output_padding
        ),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(
            channels,
            output_dim,
            1,
            stride=1,
            padding=0,
            output_padding=0,
        ),
    )


def get_dvae_decoder(vocab_size: int, output_dim: int = 3):
    """Get CNN decoder backbone for DVAE module in SLATE paper."""
    conv2d = nn.Conv2d(64, output_dim, 1)
    nn.init.xavier_uniform_(conv2d.weight)
    nn.init.zeros_(conv2d.bias)
    return nn.Sequential(
        Conv2dBlockWithGroupNorm(vocab_size, 64, 1),
        Conv2dBlockWithGroupNorm(64, 64, 3, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64 * 2 * 2, 1),
        nn.PixelShuffle(2),
        Conv2dBlockWithGroupNorm(64, 64, 3, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64 * 2 * 2, 1),
        nn.PixelShuffle(2),
        conv2d,
    )


def get_dvae_encoder(vocab_size: int, patch_size: int = 16, output_dim: int = 3):
    """Get CNN decoder backbone for DVAE module in SLATE paper."""
    conv2d = nn.Conv2d(64, vocab_size, 1)
    nn.init.xavier_uniform_(conv2d.weight)
    nn.init.zeros_(conv2d.bias)

    return nn.Sequential(
        Conv2dBlockWithGroupNorm(output_dim, 64, patch_size, patch_size),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        Conv2dBlockWithGroupNorm(64, 64, 1, 1),
        conv2d,
    )


class StyleGANv2Decoder(nn.Module):
    """CNN decoder as used in StyleGANv2 and GIRAFFE."""

    def __init__(
        self,
        object_feature_dim: int,
        output_dim: int = 4,
        min_features=32,
        input_size: int = 8,
        output_size: int = 128,
        activation_fn: str = "leaky_relu",
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        input_size_log2 = math.log2(input_size)
        assert math.floor(input_size_log2) == input_size_log2, "Input size needs to be power of 2"

        output_size_log2 = math.log2(output_size)
        assert math.floor(output_size_log2) == output_size_log2, "Output size needs to be power of 2"

        n_blocks = int(output_size_log2 - input_size_log2)

        self.convs = nn.ModuleList()
        cur_dim = object_feature_dim
        for _ in range(n_blocks):
            next_dim = max(cur_dim // 2, min_features)
            self.convs.append(nn.Conv2d(cur_dim, next_dim, 3, stride=1, padding=1))
            cur_dim = next_dim

        self.skip_convs = nn.ModuleList()
        cur_dim = object_feature_dim
        for _ in range(n_blocks + 1):
            self.skip_convs.append(nn.Conv2d(cur_dim, output_dim, 1, stride=1))
            cur_dim = max(cur_dim // 2, min_features)

        nn.init.zeros_(self.skip_convs[-1].bias)

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation_fn == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(leaky_relu_slope, inplace=True)
        else:
            raise ValueError(f"Unknown activation function {activation_fn}")

    def forward(self, inp):
        output = self.skip_convs[0](inp)

        features = inp
        for conv, skip_conv in zip(self.convs, self.skip_convs[1:]):
            features = F.interpolate(features, scale_factor=2, mode="nearest-exact")
            features = conv(features)
            features = self.activation_fn(features)

            output = F.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=False, antialias=True
            )
            output = output + skip_conv(features)

        return output


class SlotAttentionDecoder(nn.Module):
    """Decoder used in the original slot attention paper."""

    def __init__(
        self,
        decoder: nn.Module,
        final_activation: Union[str, Callable] = "identity",
        positional_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.final_activation = get_activation_fn(final_activation)
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.register_buffer("grid", build_grid_of_positions(self.initial_conv_size))

    def forward(self, object_features: torch.Tensor):
        assert object_features.dim() >= 3  # Image or video data.
        initial_shape = object_features.shape[:-1]
        object_features = object_features.flatten(0, -2)

        object_features = (
            object_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.initial_conv_size)
        )
        if self.positional_embedding:
            object_features = self.positional_embedding(object_features, self.grid.unsqueeze(0))

        # Apply deconvolution and restore object dimension.
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        # The decoder is assumed to output tensors in CNN order, i.e. * x C x H x W.
        rgb, alpha = output.split([3, 1], dim=-3)
        rgb = self.final_activation(rgb)
        alpha = alpha.softmax(dim=-4)

        return ReconstructionOutput(
            # Combine rgb weighted according to alpha channel.
            reconstruction=(rgb * alpha).sum(-4),
            object_reconstructions=rgb,
            masks=alpha.squeeze(-3),
        )


class SlotAttentionAmodalDecoder(nn.Module):
    """Decoder used in the original slot attention paper."""

    def __init__(
        self,
        decoder: nn.Module,
        final_activation: Union[str, Callable] = "identity",
        positional_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.final_activation = get_activation_fn(final_activation)
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.register_buffer("grid", build_grid_of_positions(self.initial_conv_size))

    def rescale_mask(self, mask):
        max = torch.max(mask)
        min = torch.min(mask)
        mask_new = (mask - min) / (max - min)
        return mask_new

    def forward(self, object_features: torch.Tensor):
        assert object_features.dim() >= 3  # Image or video data.
        initial_shape = object_features.shape[:-1]

        object_features_ori = object_features.clone()

        object_features = object_features.flatten(0, -2)
        object_features = (
            object_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.initial_conv_size)
        )
        if self.positional_embedding:
            object_features = self.positional_embedding(object_features, self.grid.unsqueeze(0))

        # Apply deconvolution and restore object dimension.
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        # The decoder is assumed to output tensors in CNN order, i.e. * x C x H x W.
        rgb, alpha = output.split([3, 1], dim=-3)
        rgb = self.final_activation(rgb)
        alpha1 = alpha.softmax(dim=-4)  # visible masks
        alpha2 = alpha.sigmoid()  # amodal masks

        masks_vis = torch.zeros(alpha1.shape).to(alpha1.device)
        for b in range(object_features_ori.shape[0]):
            index = torch.sum(object_features_ori[b], dim=-1).nonzero(as_tuple=True)[0]
            masks_vis[b][index] = alpha1[b][index]
            for i in index:
                masks_vis[b][i] = self.rescale_mask(alpha1[b][i])

        return ReconstructionAmodalOutput(
            # Combine rgb weighted according to alpha channel.
            reconstruction=(rgb * alpha1).sum(-4),
            object_reconstructions=rgb,
            masks=alpha2.squeeze(-3),
            masks_vis=alpha1.squeeze(-3),
            masks_eval=masks_vis.squeeze(-3),
        )


class SlotAttentionOpticalFlowDecoder(nn.Module):
    # TODO(flwenzel): for now use the same decoder as for rbg reconstruction. Might implement
    # improved/specialized decoder.
    # TODO(hornmax): Maybe we can merge this with the RGB decoder and generalize the task outputs.
    # This is something for a later time though.

    def __init__(
        self,
        decoder: nn.Module,
        positional_embedding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.decoder = decoder
        self.positional_embedding = positional_embedding
        if positional_embedding:
            self.register_buffer("grid", build_grid_of_positions(self.initial_conv_size))

    def forward(self, object_features: torch.Tensor):
        assert object_features.dim() >= 3  # Image or video data.
        initial_shape = object_features.shape[:-1]
        object_features = object_features.flatten(0, -2)

        object_features = (
            object_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *self.initial_conv_size)
        )
        if self.positional_embedding:
            object_features = self.positional_embedding(object_features, self.grid.unsqueeze(0))

        # Apply deconvolution and restore object dimension.
        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        # The decoder is assumed to output tensors in CNN order, i.e. * x C x H x W.
        flow, alpha = output.split([2, 1], dim=2)  # flow is assumed to be 2-dim.
        alpha = alpha.softmax(dim=-4)

        return OpticalFlowPredictionTaskOutput(
            # Combine rgb weighted according to alpha channel.
            predicted_flow=(flow * alpha).sum(-4),
            object_flows=flow,
            masks=alpha.squeeze(-3),
        )


class PatchDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes inputs of shape (B * K), P, N, and produce
            outputs of shape (B * K), P, M, where K is the number of objects, N is the number of
            input dimensions and M the number of output dimensions.
        decoder_input_dim: Input dimension to decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        decoder_input_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode

        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = object_dim

        self.decoder = decoder(decoder_input_dim, output_dim + 1)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, decoder_input_dim) * 0.02)

    def forward(
        self,
        object_features: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ):
        assert object_features.dim() >= 3  # Image or video data.
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),
                    scale_factor=self.upsample_target,
                    resize_mode=self.resize_mode,
                )
                .flatten(-2, -1)
                .transpose(-2, -1)
            )

        initial_shape = object_features.shape[:-1]
        object_features = object_features.flatten(0, -2)

        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        object_features = object_features.unsqueeze(1).expand(-1, self.num_patches, -1)

        # Simple learned additive embedding as in ViT
        object_features = object_features + self.pos_embed

        output = self.decoder(object_features)
        output = output.unflatten(0, initial_shape)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)
        alpha = alpha.softmax(dim=-3)

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)
        masks = alpha.squeeze(-1)

        if image is not None:
            masks_as_image = resize_patches_to_image(
                masks, size=image.shape[-1], resize_mode="bilinear"
            )
        else:
            masks_as_image = None

        return PatchReconstructionOutput(
            reconstruction=reconstruction,
            masks=alpha.squeeze(-1),
            masks_as_image=masks_as_image,
            target=target if target is not None else None,
        )


class AutoregressivePatchDecoder(nn.Module):
    """Decoder that takes object representations and reconstructs patches autoregressively.

    Args:
        object_dim: Dimension of objects representations.
        output_dim: Dimension of each patch.
        num_patches: Number of patches P to reconstruct.
        decoder: Function that returns backbone to use for decoding. Function takes input and output
            dimensions and should return module that takes autoregressive targets of shape B, P, M,
            conditioning of shape B, K, N, masks of shape P, P, and produces outputs of shape
            B, P, M, where K is the number of objects, N is the number of input dimensions and M the
            number of output dimensions.
        decoder_cond_dim: Dimension of conditioning input of decoder backbone. If specified, a linear
            transformation from object to decoder dimension is added. If not specified, the object
            dimension is used and no linear transform is added.
    """

    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches: int,
        decoder: Callable[[int, int], nn.Module],
        decoder_dim: Optional[int] = None,
        decoder_cond_dim: Optional[int] = None,
        upsample_target: Optional[float] = None,
        resize_mode: str = "bilinear",
        use_decoder_masks: bool = False,
        use_bos_token: bool = True,
        use_input_transform: bool = False,
        use_input_norm: bool = False,
        use_output_transform: bool = False,
        use_positional_embedding: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_patches = num_patches
        self.upsample_target = upsample_target
        self.resize_mode = resize_mode
        self.use_decoder_masks = use_decoder_masks

        if decoder_dim is None:
            decoder_dim = output_dim

        self.decoder = decoder(decoder_dim, decoder_dim)
        if use_bos_token:
            self.bos_token = nn.Parameter(torch.randn(1, 1, output_dim) * output_dim**-0.5)
        else:
            self.bos_token = None
        if decoder_cond_dim is not None:
            self.cond_transform = nn.Sequential(
                nn.Linear(object_dim, decoder_cond_dim, bias=False),
                nn.LayerNorm(decoder_cond_dim, eps=1e-5),
            )
            nn.init.xavier_uniform_(self.cond_transform[0].weight)
        else:
            decoder_cond_dim = object_dim
            self.cond_transform = nn.LayerNorm(decoder_cond_dim, eps=1e-5)

        if use_input_transform:
            self.inp_transform = nn.Sequential(
                nn.Linear(output_dim, decoder_dim, bias=False),
                nn.LayerNorm(decoder_dim, eps=1e-5),
            )
            nn.init.xavier_uniform_(self.inp_transform[0].weight)
        elif use_input_norm:
            self.inp_transform = nn.LayerNorm(decoder_dim, eps=1e-5)
        else:
            self.inp_transform = None

        if use_output_transform:
            self.outp_transform = nn.Linear(decoder_dim, output_dim)
            nn.init.xavier_uniform_(self.outp_transform.weight)
            nn.init.zeros_(self.outp_transform.bias)
        else:
            self.outp_transform = None

        if use_positional_embedding:
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_patches, decoder_dim) * decoder_dim**-0.5
            )
        else:
            self.pos_embed = None

        mask = torch.triu(torch.full((num_patches, num_patches), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        object_features: torch.Tensor,
        masks: torch.Tensor,
        target: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        empty_objects: Optional[torch.Tensor] = None,
    ) -> PatchReconstructionOutput:
        assert object_features.dim() >= 3  # Image or video data.
        if self.upsample_target is not None and target is not None:
            target = (
                resize_patches_to_image(
                    target.detach().transpose(-2, -1),
                    scale_factor=self.upsample_target,
                    resize_mode=self.resize_mode,
                )
                .flatten(-2, -1)
                .transpose(-2, -1)
            )
        # Squeeze frames into batch if present.
        object_features = object_features.flatten(0, -3)

        object_features = self.cond_transform(object_features)

        # Squeeze frame into batch size if necessary.
        initial_targets_shape = target.shape[:-2]
        targets = target.flatten(0, -3)
        if self.bos_token is not None:
            bs = len(object_features)
            inputs = torch.cat((self.bos_token.expand(bs, -1, -1), targets[:, :-1].detach()), dim=1)
        else:
            inputs = targets

        if self.inp_transform is not None:
            inputs = self.inp_transform(inputs)

        if self.pos_embed is not None:
            # Simple learned additive embedding as in ViT
            inputs = inputs + self.pos_embed

        if empty_objects is not None:
            outputs = self.decoder(
                inputs,
                object_features,
                self.mask,
                memory_key_padding_mask=empty_objects,
            )
        else:
            outputs = self.decoder(inputs, object_features, self.mask)

        if self.use_decoder_masks:
            decoded_patches, masks = outputs
        else:
            decoded_patches = outputs

        if self.outp_transform is not None:
            decoded_patches = self.outp_transform(decoded_patches)

        decoded_patches = decoded_patches.unflatten(0, initial_targets_shape)

        if image is not None:
            masks_as_image = resize_patches_to_image(
                masks, size=image.shape[-1], resize_mode="bilinear"
            )
        else:
            masks_as_image = None

        return PatchReconstructionOutput(
            reconstruction=decoded_patches, masks=masks, masks_as_image=masks_as_image, target=target
        )


class DensityPredictingSlotAttentionDecoder(nn.Module):
    """Decoder predicting color and densities along a ray into the scene."""

    def __init__(
        self,
        object_dim: int,
        decoder: nn.Module,
        depth_positions: int,
        white_background: bool = False,
        normalize_densities_along_slots: bool = False,
        initial_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.initial_conv_size = (8, 8)
        self.depth_positions = depth_positions
        self.white_background = white_background
        self.normalize_densities_along_slots = normalize_densities_along_slots
        self.register_buffer("grid", build_grid_of_positions(self.initial_conv_size))
        self.pos_embedding = SoftPositionEmbed(2, object_dim, cnn_channel_order=True)

        self.decoder = decoder
        if isinstance(self.decoder, nn.Sequential) and hasattr(self.decoder[-1], "bias"):
            nn.init.zeros_(self.decoder[-1].bias)

        if initial_alpha is not None:
            # Distance between neighboring ray points, currently assumed to be 1
            point_distance = 1
            # Value added to density output of network before softplus activation. If network outputs
            # are approximately zero, the initial mask value per voxel becomes `initial_alpha`. See
            # https://arxiv.org/abs/2111.11215 for a derivation.
            self.initial_density_offset = math.log((1 - initial_alpha) ** (-1 / point_distance) - 1)
        else:
            self.initial_density_offset = 0.0

    def _render_objectwise(self, densities, rgbs):
        """Render objects individually.

        Args:
            densities: Predicted densities of shape (B, S, Z, H, W), where S is the number of slots
                and Z is the number of depth positions.
            rgbs: Predicted color values of shape (B, S, 3, H, W), where S is the number of slots.
            background: Optional background to render on.
        """
        densities_objectwise = densities.flatten(0, 1).unsqueeze(2)
        rgbs_objectwise = rgbs.flatten(0, 1).unsqueeze(1)
        rgbs_objectwise = rgbs_objectwise.expand(-1, densities_objectwise.shape[1], -1, -1, -1)

        if self.white_background:
            background = torch.full_like(rgbs_objectwise[:, 0], 1.0)  # White color, i.e. 0xFFFFFF
        else:
            background = None

        object_reconstructions, _, object_masks_per_depth, p_ray_hits_points = volume_rendering(
            densities_objectwise, rgbs_objectwise, background=background
        )

        object_reconstructions = object_reconstructions.unflatten(0, rgbs.shape[:2])
        object_masks_per_depth = object_masks_per_depth.squeeze(2).unflatten(0, rgbs.shape[:2])
        p_ray_hits_points = p_ray_hits_points.squeeze(2).unflatten(0, rgbs.shape[:2])

        p_ray_hits_points_and_reflects = p_ray_hits_points * object_masks_per_depth
        object_masks, object_depth_map = p_ray_hits_points_and_reflects.max(2)

        return object_reconstructions, object_masks, object_depth_map

    def forward(self, object_features: torch.Tensor):
        # TODO(hornmax): Adapt this for video data.
        # Reshape object dimension into batch dimension and broadcast.
        bs, n_objects, object_feature_dim = object_features.shape
        object_features = object_features.view(bs * n_objects, object_feature_dim, 1, 1).expand(
            -1, -1, *self.initial_conv_size
        )
        object_features = self.pos_embedding(object_features, self.grid.unsqueeze(0))

        # Apply deconvolution and restore object dimension.
        output = self.decoder(object_features)
        output = output.view(bs, n_objects, *output.shape[-3:])

        # Split rgb and density channels and transform to appropriate ranges.
        rgbs, densities = output.split([3, self.depth_positions], dim=2)
        rgbs = torch.sigmoid(rgbs)  # B x S x 3 x H x W
        densities = F.softplus(densities + self.initial_density_offset)  # B x S x Z x H x W

        if self.normalize_densities_along_slots:
            densities_depthwise_sum = torch.einsum("bszhw -> bzhw", densities).unsqueeze(1)
            densities_weighted = densities * F.softmax(densities, dim=1)
            densities_weighted_sum = torch.einsum("bszhw -> bzhw", densities_weighted).unsqueeze(1)
            densities = densities_weighted * densities_depthwise_sum / densities_weighted_sum

        # Combine densities from different slots by summing over slot dimension
        density = torch.einsum("bszhw -> bzhw", densities).unsqueeze(2)
        # Combine colors from different slots by density-weighted mean
        rgb = torch.einsum("bszhw, bschw -> bzchw", densities, rgbs) / density

        if self.white_background:
            background = torch.full_like(rgb[:, 0], 1.0)  # White color, i.e. 0xFFFFFF
        else:
            background = None

        reconstruction, _, _, p_ray_hits_point = volume_rendering(
            density, rgb, background=background
        )

        if self.training:
            # Get object masks by taking the max density over all depth positions
            masks = 1 - torch.exp(-densities.detach().max(dim=2).values)
            object_reconstructions = rgbs.detach() * masks.unsqueeze(2)

            if background is not None:
                masks = torch.cat((masks, p_ray_hits_point[:, -1:, 0]), dim=1)
                object_reconstructions = torch.cat(
                    (object_reconstructions, background[:, None]), dim=1
                )

            return ReconstructionOutput(
                reconstruction=reconstruction,
                object_reconstructions=object_reconstructions,
                masks=masks,
            )
        else:
            object_reconstructions, object_masks, object_depth_map = self._render_objectwise(
                densities, rgbs
            )

            # Joint depth map results from taking minimum depth over objects per pixel, whereas
            # joint mask results from the index of the object with minimum depth
            depth_map, mask_dense = object_depth_map.min(1)

            if background is not None:
                object_reconstructions = torch.cat(
                    (object_reconstructions, background[:, None]), dim=1
                )
                # Assign designated background class wherever the depth map indicates background
                mask_dense[depth_map == self.depth_positions] = n_objects
                n_classes = n_objects + 1
            else:
                n_classes = n_objects

            masks = F.one_hot(mask_dense, num_classes=n_classes)
            masks = masks.squeeze(1).permute(0, 3, 1, 2).contiguous()  # B x C x H x W

            return DepthReconstructionOutput(
                reconstruction=reconstruction,
                object_reconstructions=object_reconstructions,
                masks=masks,
                masks_amodal=object_masks,
                depth_map=depth_map,
                object_depth_map=object_depth_map,
                densities=densities,
                colors=rgbs.unsqueeze(2).expand(-1, -1, self.depth_positions, -1, -1, -1),
            )


def volume_rendering(
    densities: torch.Tensor,
    colors: torch.Tensor,
    distances: Union[float, torch.Tensor] = None,
    background: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Volume render along camera rays (also known as alpha compositing).

    For each ray, assumes input of Z density and C color channels, corresponding to Z points along
    the ray from front to back of the scene.

    Args:
        densities: Tensor of shape (B, Z, 1, ...). Non-negative, real valued density values along
            the ray.
        colors: Tensor of shape (B, Z, C, ...). Color values along the ray.
        distances: Tensor of shape (B, Z, 1, ...). Optional distances between this ray point and
            the next. Can also be a single float value. If not given, distances between all points
            are assumed to be one. The last value corresponds to the distance between the last point
            and the background.
        background: Tensor of shape (B, C, ...). An optional background image that the rendering can
            be put on.

    Returns:
        - Rendered image of shape (B, C, ...)
        - Rendered images along different points of the ray with shape (B, Z, 1, ...), if background
            is not None, the background is included as the last ray point.
        - The alpha masks for each point of the ray with shape (B, Z, 1, ...)
        - The probabilities of reaching each point of the ray (the transmittances) with shape
            (B, Z, 1, ...)
    """
    if distances is None:
        transmittances = torch.exp(-torch.cumsum(densities, dim=1))
        p_ray_reflects = 1.0 - torch.exp(-densities)
    else:
        densities_distance_weighted = densities * distances
        transmittances = torch.exp(-torch.cumsum(densities_distance_weighted, dim=1))
        p_ray_reflects = 1.0 - torch.exp(-densities_distance_weighted)

    # First object has 100% probability of being hit as it cannot be occluded by other objects
    p_ray_hits_point = torch.cat((torch.ones_like(densities[:, :1]), transmittances), dim=1)

    if background is not None:
        background = background.unsqueeze(1)

        # All rays reaching the background reflect
        p_ray_reflects = torch.cat((p_ray_reflects, torch.ones_like(p_ray_reflects[:, :1])), dim=1)
        colors = torch.cat((colors, background), dim=1)
    else:
        p_ray_hits_point = p_ray_hits_point[:, :-1]

    z_images = p_ray_reflects * colors
    image = (p_ray_hits_point * z_images).sum(dim=1)

    return image, z_images, p_ray_reflects, p_ray_hits_point


class DVAEDecoder(nn.Module):
    """VQ Decoder used in the original SLATE paper."""

    def __init__(
        self,
        decoder: nn.Module,
        patch_size: int = 4,
    ):
        super().__init__()
        self.initial_conv_size = (patch_size, patch_size)
        self.decoder = decoder

    def forward(self, features: Dict[str, torch.Tensor]):
        rgb = self.decoder(features)
        return SimpleReconstructionOutput(reconstruction=rgb)
