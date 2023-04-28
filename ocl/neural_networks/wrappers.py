"""Wrapper modules with allow the introduction or residuals or the combination of other modules."""
from torch import nn


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


class Sequential(nn.Module):
    """Extended sequential module that supports multiple inputs and outputs to layers.

    This allows a stack of layers where for example the first layer takes two inputs and only has
    a single output or where a layer has multiple outputs and the downstream layer takes multiple
    inputs.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *inputs):
        outputs = inputs
        for layer in self.layers:
            if isinstance(outputs, (tuple, list)):
                outputs = layer(*outputs)
            else:
                outputs = layer(outputs)
        return outputs
