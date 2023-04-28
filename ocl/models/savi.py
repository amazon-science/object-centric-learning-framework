from functools import partial
from typing import Any, Dict

import torch
from torch import nn

from ocl.utils.trees import get_tree_element, reduce_tree


class SAVi(nn.Module):
    """Code based implementation of SAVi model."""

    def __init__(
        self,
        conditioning: nn.Module,
        feature_extractor: nn.Module,
        perceptual_grouping: nn.Module,
        decoder: nn.Module,
        transition_model: nn.Module,
        input_path: str = "input.image",
    ):
        super().__init__()
        self.conditioning = conditioning
        self.feature_extractor = feature_extractor
        self.perceptual_grouping = perceptual_grouping
        self.decoder = decoder
        self.transition_model = transition_model
        self.input_path = input_path

    def forward(self, inputs: Dict[str, Any]):
        output = inputs
        video = get_tree_element(inputs, self.input_path)
        batch_size = video.shape[0]

        features = self.feature_extractor(video=video)
        output["feature_extractor"] = features
        conditioning = self.conditioning(batch_size=batch_size)
        output["initial_conditioning"] = conditioning

        # Loop over time.
        perceptual_grouping_outputs = []
        decoder_outputs = []
        transition_model_outputs = []
        for frame_features in features:
            perceptual_grouping_output = self.perceptual_grouping(
                feature=frame_features, conditioning=conditioning
            )
            slots = perceptual_grouping_output.objects
            decoder_output = self.decoder(object_features=slots)
            conditioning = self.transition_model(slots)
            # Store outputs.
            perceptual_grouping_outputs.append(slots)
            decoder_outputs.append(decoder_output)
            transition_model_outputs.append(conditioning)

        # Stack all recurrent outputs.
        stacking_fn = partial(torch.stack, dim=1)
        output["perceptual_grouping"] = reduce_tree(perceptual_grouping_outputs, stacking_fn)
        output["decoder"] = reduce_tree(decoder_outputs, stacking_fn)
        output["transition_model"] = reduce_tree(transition_model_outputs, stacking_fn)
        return output
