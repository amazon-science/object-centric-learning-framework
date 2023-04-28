"""Extensions of existing layers to implement additional functionality."""
from typing import Optional, Union

import torch
from torch import nn


class TransformerDecoderWithAttention(nn.TransformerDecoder):
    """Modified nn.TransformerDecoder class that returns attention weights over memory."""

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_attention_weights=False,
        attention_weight_type: Union[int, str] = "mean",
    ):
        super(TransformerDecoderWithAttention, self).__init__(decoder_layer, num_layers, norm)

        if return_attention_weights:
            self.attention_hooks = []
            for layer in self.layers:
                self.attention_hooks.append(self._prepare_layer(layer))
        else:
            self.attention_hooks = None

        if isinstance(attention_weight_type, int):
            if attention_weight_type >= num_layers or attention_weight_type < -num_layers:
                raise ValueError(
                    f"Index {attention_weight_type} exceeds number of layers {num_layers}"
                )
        elif attention_weight_type != "mean":
            raise ValueError("`weights` needs to be a number or 'mean'.")
        self.weights = attention_weight_type

    def _prepare_layer(self, layer):
        assert isinstance(layer, nn.TransformerDecoderLayer)

        def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal):
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=True,
            )[0]
            return self.dropout2(x)

        # Patch _mha_block method to compute attention weights
        layer._mha_block = _mha_block.__get__(layer, nn.TransformerDecoderLayer)

        class AttentionHook:
            def __init__(self):
                self._attention = None

            def pop(self) -> torch.Tensor:
                assert self._attention is not None, "Forward was not called yet!"
                attention = self._attention
                self._attention = None
                return attention

            def __call__(self, module, inp, outp):
                self._attention = outp[1]

        hook = AttentionHook()
        layer.multihead_attn.register_forward_hook(hook)
        return hook

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        if self.attention_hooks is not None:
            attentions = []
            for hook in self.attention_hooks:
                attentions.append(hook.pop())

            if self.weights == "mean":
                attentions = torch.stack(attentions, dim=-1)
                # Take mean over all layers
                attention = attentions.mean(dim=-1)
            else:
                attention = attentions[self.weights]

            return output, attention.transpose(1, 2)
        else:
            return output
