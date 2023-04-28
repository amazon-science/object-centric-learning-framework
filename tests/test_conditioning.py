import pytest

from ocl import conditioning


@pytest.mark.parametrize(
    "conditioning_cls,additional_args",
    [
        (conditioning.RandomConditioning, {}),
        (conditioning.RandomConditioning, {"learn_mean": False, "learn_std": False}),
        (conditioning.RandomConditioningWithQMCSampling, {}),
        (conditioning.RandomConditioningWithQMCSampling, {"learn_mean": False, "learn_std": False}),
        (conditioning.LearntConditioning, {}),
        (conditioning.SlotwiseLearntConditioning, {}),
    ],
)
def test_slot_conditionings(conditioning_cls, additional_args):
    batch_size, n_slots, object_dim = 3, 5, 8
    conditioner = conditioning_cls(object_dim=object_dim, n_slots=n_slots, **additional_args)

    slots = conditioner(batch_size)

    assert slots.shape == (batch_size, n_slots, object_dim)
