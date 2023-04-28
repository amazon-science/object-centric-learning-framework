"""Neural networks used for the implemenation of SLATE."""
import torch
from torch import nn


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


class Conv2dBlockWithGroupNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        weight_init="xavier",
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        if weight_init == "kaiming":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(self.conv2d.weight)

        if bias:
            nn.init.zeros_(self.conv2d.bias)
        self.group_norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        return nn.functional.relu(self.group_norm(x))
