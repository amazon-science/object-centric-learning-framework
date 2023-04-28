import pytest
import torch

from ocl.callbacks import FreezeParameters


def test_parameter_freezing():
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.ones(1))
            self.param2 = torch.nn.Parameter(torch.ones(1))

        def forward(self):
            return self.param1 * self.param2

    model = MyModel()
    optim = torch.optim.SGD(model.parameters(), 0.001)
    callback = FreezeParameters([{"params": "param1"}])
    callback.on_fit_start(None, model)

    output = model()
    loss = (2 - output) ** 2
    loss.backward()
    optim.step()

    assert model.param1 == 1.0
    assert model.param2 != 1.0

    # Ensure incorrect paths raise an error.
    with pytest.raises(ValueError):
        callback = FreezeParameters([{"params": "param_fake"}])
        callback.on_fit_start(None, model)
