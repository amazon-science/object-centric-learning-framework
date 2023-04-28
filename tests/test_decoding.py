import functools

import pytest
import torch

from ocl import decoding, neural_networks


@pytest.mark.parametrize(
    "decoder_cls, decoder_kwargs",
    [
        (decoding.SlotAttentionDecoder, dict()),
        (
            decoding.DensityPredictingSlotAttentionDecoder,
            dict(object_dim=5, depth_positions=4),
        ),
        (
            decoding.DensityPredictingSlotAttentionDecoder,
            dict(object_dim=5, depth_positions=4, white_background=True),
        ),
        (
            decoding.DensityPredictingSlotAttentionDecoder,
            dict(
                object_dim=5,
                depth_positions=4,
                normalize_densities_along_slots=True,
            ),
        ),
    ],
)
def test_decoders(decoder_cls, decoder_kwargs):
    def check_result(result):
        if decoder_kwargs.get("white_background", False):
            expected_n_objects = n_objects + 1
        else:
            expected_n_objects = n_objects
        assert isinstance(result, decoding.ReconstructionOutput)
        # Can not check for image size, as it is currently not exposed to the outside of the decoder
        assert result.reconstruction.shape[:2] == (batch_size, 3)
        assert result.object_reconstructions.shape[:3] == (batch_size, expected_n_objects, 3)
        assert result.masks.shape[:2] == (batch_size, expected_n_objects)
        assert 0.0 <= result.masks.min() and result.masks.max() <= 1.0

    batch_size, n_objects, object_dim = 2, 3, 5
    depth_positions = decoder_kwargs.get("depth_positions", 1)
    backbone = decoding.get_slotattention_decoder_backbone(object_dim, 3 + depth_positions)
    decoder = decoder_cls(decoder=backbone, **decoder_kwargs)
    slots = torch.randn(batch_size, n_objects, object_dim)

    result = decoder(slots)
    check_result(result)

    # Decoder can have different behaviour in eval mode, check that it does not crash.
    decoder.eval()
    result = decoder(slots)
    check_result(result)


@pytest.mark.parametrize(
    "backbone_fn",
    [
        functools.partial(neural_networks.build_mlp, features=[20, 10]),
        functools.partial(neural_networks.build_transformer_encoder, n_layers=2, n_heads=1),
    ],
)
@pytest.mark.parametrize("decoder_input_dim", [None, 8])
def test_patch_decoder(backbone_fn, decoder_input_dim):
    batch_size, n_objects, object_dim, feature_dim, n_patches = 2, 3, 5, 10, 8
    decoder = decoding.PatchDecoder(
        object_dim,
        feature_dim,
        n_patches,
        decoder=backbone_fn,
        decoder_input_dim=decoder_input_dim,
    )

    slots = torch.randn(batch_size, n_objects, object_dim)
    result = decoder(slots)

    assert result.reconstruction.shape == (batch_size, n_patches, feature_dim)
    assert result.masks.shape == (batch_size, n_objects, n_patches)


def test_stylegan_decoder():
    batch_size, feature_dim, output_dim, input_size, output_size = 2, 32, 5, 4, 16
    decoder = decoding.StyleGANv2Decoder(
        feature_dim, output_dim, min_features=4, input_size=input_size, output_size=output_size
    )
    inp = torch.randn(batch_size, feature_dim, input_size, input_size)
    outp = decoder(inp)
    assert outp.shape == (batch_size, output_dim, output_size, output_size)


def test_volume_rendering(assert_tensors_equal):
    def check_rendering_result(result, image, z_images, masks):
        assert_tensors_equal(result[0], image)
        assert_tensors_equal(result[1], z_images)
        assert_tensors_equal(result[2], masks)

    B, H, W = 2, 2, 3
    zero_density = torch.zeros(B, 1, 1, H, W)
    one_half_density = -torch.log(torch.full((B, 1, 1, H, W), fill_value=0.5))
    random_color = torch.rand(B, 1, 3, H, W)
    black_color = torch.ones(B, 1, 3, H, W)
    white_background = torch.ones(B, 3, H, W)

    black_image = torch.zeros(B, 3, H, W)
    gray_image = torch.full((B, 3, H, W), fill_value=0.5)
    white_image = torch.ones(B, 3, H, W)
    zero_mask = torch.zeros(B, 1, 1, H, W)
    one_mask = torch.ones(B, 1, 1, H, W)
    one_half_mask = torch.full((B, 1, 1, H, W), fill_value=0.5)

    # Single point with zero density renders black
    result = decoding.volume_rendering(zero_density, random_color)
    check_rendering_result(result, black_image, black_image[:, None], zero_mask)

    # Single point with zero density and white background renders white
    result = decoding.volume_rendering(zero_density, random_color, background=white_background)
    check_rendering_result(
        result,
        white_image,
        torch.cat((black_image[:, None], white_background[:, None]), dim=1),
        torch.cat((zero_mask, one_mask), dim=1),
    )

    # Single point with "0.5" density and black color renders gray
    result = decoding.volume_rendering(one_half_density, black_color)
    check_rendering_result(result, gray_image, gray_image[:, None], one_half_mask)

    # Two points with [zero, "0.5"] densities and [random, black] colors render gray
    result = decoding.volume_rendering(
        torch.cat((zero_density, one_half_density), dim=1),
        torch.cat((random_color, black_color), dim=1),
    )
    check_rendering_result(
        result,
        gray_image,
        torch.cat((black_image[:, None], gray_image[:, None]), dim=1),
        torch.cat((zero_mask, one_half_mask), dim=1),
    )
