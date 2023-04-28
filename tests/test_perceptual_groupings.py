import pytest
import torch

from ocl import perceptual_grouping


@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("kvq_dim", [None, 16])
@pytest.mark.parametrize("use_implicit_differentiation", [False, True])
@pytest.mark.parametrize("test_masking", [False, True])
def test_slot_attention(n_heads, kvq_dim, use_implicit_differentiation, test_masking):
    bs, n_inputs, n_slots = 2, 5, 3
    inp_dim, slot_dim = 12, 8
    slot_attention = perceptual_grouping.SlotAttention(
        dim=slot_dim,
        feature_dim=inp_dim,
        iters=2,
        n_heads=n_heads,
        kvq_dim=kvq_dim,
        use_implicit_differentiation=use_implicit_differentiation,
        eps=0.0,
    )

    inputs = torch.randn(bs, n_inputs, inp_dim)
    slots = torch.randn(bs, n_slots, slot_dim)

    if test_masking:
        mask = torch.zeros(bs, n_slots, dtype=torch.bool)
        mask[0, 1] = True
        mask[0, 2] = True
        mask[1, 0] = True
    else:
        mask = None

    upd_slots, attn = slot_attention(inputs, slots, mask)

    assert upd_slots.shape == (bs, n_slots, slot_dim)
    assert attn.shape == (bs, n_slots, n_inputs)

    if test_masking:
        # First slot should get all attention (averaged over heads)
        assert torch.allclose(attn[0, 0], torch.ones_like(attn[0, 0]) / n_heads)
        assert torch.allclose(attn[0, 1], torch.zeros_like(attn[0, 1]))
        assert torch.allclose(attn[0, 2], torch.zeros_like(attn[0, 2]))
        # Second and third slot should get all attention (averaged over heads)
        assert torch.allclose(attn[1, 0], torch.zeros_like(attn[1, 0]))
        assert torch.allclose(attn[1, 1] + attn[1, 2], torch.ones_like(attn[1, 1]) / n_heads)
