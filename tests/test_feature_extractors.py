import pytest
import torch

from ocl import feature_extractors


@pytest.mark.parametrize(
    "model_name,feature_level,aux_feature_levels,inp_size,outp_size",
    [
        ("resnet18", 4, None, 224, 7),
        ("resnet34_savi", 4, None, 128, 16),
        ("vit_tiny_patch16_224", None, None, 224, 14),
        ("vit_tiny_patch16_224", 2, None, 224, 14),
        ("vit_tiny_patch16_224", (2, 12), ["block1", "block2"], 224, 14),
        ("vit_tiny_patch16_224", "block1", None, 224, 14),
        ("vit_tiny_patch16_224", "key1", None, 224, 14),
        ("vit_tiny_patch16_224", "value1", None, 224, 14),
        ("vit_tiny_patch16_224", None, ["query1"], 224, 14),
        ("vit_tiny_patch16_224", "query1", ["query1"], 224, 14),
    ],
)
def test_timm_feature_extractors(model_name, feature_level, aux_feature_levels, inp_size, outp_size):
    extractor = feature_extractors.TimmFeatureExtractor(
        model_name,
        feature_level=feature_level,
        aux_features=aux_feature_levels,
        pretrained=False,
        freeze=True,
    )

    bs = 2
    image = torch.rand(bs, 3, inp_size, inp_size)
    features, _, aux_features = extractor.forward_images(image)
    assert features.shape[0] == bs
    assert features.shape[1] == outp_size**2
    assert features.shape[2] == extractor.feature_dim

    if aux_feature_levels is None:
        assert aux_features is None
    else:
        assert list(aux_features.keys()) == aux_feature_levels


@pytest.mark.parametrize(
    "model_name,feature_level,freeze,n_blocks_to_unfreeze",
    [
        ("resnet18", 2, False, 0),
        ("resnet18", 2, True, 0),
        ("vit_tiny_patch16_224", None, False, 0),
        ("vit_tiny_patch16_224", None, True, 0),
        ("vit_tiny_patch16_224", None, True, 4),
    ],
)
def test_timm_feature_extractors_freeze(model_name, feature_level, freeze, n_blocks_to_unfreeze):
    extractor = feature_extractors.TimmFeatureExtractor(
        model_name,
        feature_level=feature_level,
        pretrained=False,
        freeze=freeze,
        n_blocks_to_unfreeze=n_blocks_to_unfreeze,
    )

    bs = 2
    image = torch.rand(bs, 3, 224, 224)
    features, *_ = extractor.forward_images(image)

    loss = features.mean()

    if freeze and n_blocks_to_unfreeze == 0:
        assert loss.grad_fn is None
    else:
        loss.backward()
        if n_blocks_to_unfreeze == 0:
            for param in extractor.parameters():
                assert param.grad is not None
        else:
            if extractor.is_vit:
                for param in extractor.model.blocks[:-n_blocks_to_unfreeze].parameters():
                    assert param.grad is None
                for param in extractor.model.blocks[-n_blocks_to_unfreeze:].parameters():
                    assert param.grad is not None
