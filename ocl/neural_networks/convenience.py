"""Convenience functions for the construction neural networks using config."""
from typing import Callable, List, Optional, Union

from torch import nn

from ocl.neural_networks.extensions import TransformerDecoderWithAttention
from ocl.neural_networks.wrappers import Residual


class ReLUSquared(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu(x, inplace=self.inplace) ** 2


def get_activation_fn(name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = None):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu_squared":
        return ReLUSquared(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def build_two_layer_mlp(
    input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False
):
    """Build a two layer MLP, with optional initial layer norm.

    Separate class as this type of construction is used very often for slot attention and
    transformers.
    """
    return build_mlp(
        input_dim, output_dim, [hidden_dim], initial_layer_norm=initial_layer_norm, residual=residual
    )


def build_transformer_encoder(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation_fn: Union[str, Callable] = "relu",
    layer_norm_eps: float = 1e-5,
    use_output_transform: bool = True,
):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    layers = []
    for _ in range(n_layers):
        layers.append(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation_fn,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=True,
            )
        )

    if use_output_transform:
        layers.append(nn.LayerNorm(input_dim, eps=layer_norm_eps))
        output_transform = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(output_transform.weight)
        nn.init.zeros_(output_transform.bias)
        layers.append(output_transform)

    return nn.Sequential(*layers)


def build_transformer_decoder(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation_fn: Union[str, Callable] = "relu",
    layer_norm_eps: float = 1e-5,
    return_attention_weights: bool = False,
    attention_weight_type: Union[int, str] = -1,
):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=input_dim,
        nhead=n_heads,
        dim_feedforward=hidden_dim,
        dropout=dropout,
        activation=activation_fn,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=True,
    )

    if return_attention_weights:
        return TransformerDecoderWithAttention(
            decoder_layer,
            n_layers,
            return_attention_weights=True,
            attention_weight_type=attention_weight_type,
        )
    else:
        return nn.TransformerDecoder(decoder_layer, n_layers)
