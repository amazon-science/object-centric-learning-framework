"""Module implementing support for pretrained clip models.

Reference:
    Radford et al.,
    Learning transferable visual models from natural language supervision,
    ICML 2021
"""
from typing import Tuple

try:
    import clip
except ImportError:
    raise Exception("Using clip models requires installation with extra `clip`.")

import torch
from torch import nn

import ocl.typing
from ocl.feature_extractors.utils import (
    ImageFeatureExtractor,
    cnn_compute_positions_and_flatten,
    transformer_compute_positions,
)


class ClipImageModel(ImageFeatureExtractor):
    """Image part of pretrained clip model."""

    def __init__(
        self,
        model_type: str,
        freeze_model: bool = False,
        reset_weights: bool = False,
        remove_pooling: bool = False,
    ):
        """Initialize ClipImageModel.

        Args:
            model_type: Model type matching `clip.load`.
            freeze_model: Freeze weights of model.
            reset_weights: Reset model weights and dont used pretrained ones.
            remove_pooling: Remove final pooling layer and return features
                instead of single token.
        """
        super().__init__()
        self.freeze_model = freeze_model

        self.clip_vision_model = clip.load(
            model_type,
            # Initially force cpu to ensure tensors are float32 (load routine automatically converts
            # to half precision if GPUs are detected).  We can still do half-precision training via
            # pytorch lightning if we want to.
            device="cpu",
        )[0].visual
        if self.freeze_model:
            for parameter in self.clip_vision_model.parameters():
                parameter.requires_grad_(False)

        if reset_weights:

            def weight_reset(module):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            self.clip_vision_model.apply(weight_reset)
            self.clip_vision_model.initialize_parameters()

        if remove_pooling:
            if isinstance(self.clip_vision_model, clip.model.VisionTransformer):
                self.get_output = self._get_features_from_vision_transformer
            else:
                self.get_output = self._get_features_from_resnet
        else:
            self.get_output = self.clip_vision_model

    def _get_features_from_vision_transformer(self, x):
        # Commands from:
        # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L223
        model = self.clip_vision_model

        x = model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                model.class_embedding
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + model.positional_embedding
        x = model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x, transformer_compute_positions(x)

    def _get_features_from_resnet(self, x: ocl.typing.ImageData):
        # Commands from:
        # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138

        model = self.clip_vision_model
        # Apply "stem".
        x = model.relu1(model.bn1(model.conv1(x)))
        x = model.relu2(model.bn2(model.conv2(x)))
        x = model.relu3(model.bn3(model.conv3(x)))
        x = model.avgpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return cnn_compute_positions_and_flatten(x)

    def forward_images(
        self, image: ocl.typing.ImageData
    ) -> Tuple[ocl.typing.ImageFeatures, ocl.typing.Positions]:
        if self.freeze_model:
            with torch.no_grad():
                return self.get_output(image)
        else:
            return self.get_output(image)


class ClipTextModel(nn.Module):
    """Text part of pretrained clip model."""

    def __init__(
        self,
        model_type: str,
        freeze_model: bool = False,
        reset_weights: bool = False,
        remove_pooling: bool = False,
        remove_eot: bool = False,
    ):
        """Initialize ClipImageModel.

        Args:
            model_type: Model type matching `clip.load`.
            freeze_model: Freeze weights of model.
            reset_weights: Reset model weights and dont used pretrained ones.
            remove_pooling: Remove final pooling layer and return features
                instead of single token.
            remove_eot: Mask out any that are padding including the eot token.
        """
        super().__init__()
        self.freeze_model = freeze_model
        self.remove_pooling = remove_pooling

        clip_model = clip.load(
            model_type,
            # Initially force cpu to ensure tensors are float32 (load routine automatically converts
            # to half precision if GPUs are detected).  We can still do half-precision training via
            # pytorch lightning if we want to.
            device="cpu",
        )[0]
        if reset_weights:

            def weight_reset(module):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            clip_model.apply(weight_reset)
            clip_model.initialize_parameters()

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        if self.freeze_model:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        self.remove_pooling = remove_pooling
        self.remove_eot = remove_eot

    def get_output(self, text):
        # Based on:
        # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L343
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.remove_pooling:
            # Mask out tokens which are part of the padding.
            # Get position of eot token, it has the highest value of all tokens.
            lengths = text.argmax(dim=-1)
            if self.remove_eot:
                # Also mask out the eot token.
                lengths = lengths - 1
            indices = torch.arange(x.shape[1], device=text.device)
            mask = indices.unsqueeze(0) >= lengths
            x.masked_fill_(mask, 0.0)

            x = x @ self.text_projection
        else:
            # Do what is done in the standard clip text encoder.
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text: ocl.typing.TextData):
        if self.freeze_model:
            with torch.no_grad():
                return self.get_output(text)
        else:
            return self.get_output(text)
