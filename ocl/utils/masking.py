"""Utilities related to masking."""
import math
from typing import Optional

import torch
from torch import nn


class CreateSlotMask(nn.Module):
    """Module intended to create a mask that marks empty slots.

    Module takes a tensor holding the number of slots per batch entry, and returns a binary mask of
    shape (batch_size, max_slots) where entries exceeding the number of slots are masked out.
    """

    def __init__(self, max_slots: int):
        super().__init__()
        self.max_slots = max_slots

    def forward(self, n_slots: torch.Tensor) -> torch.Tensor:
        (batch_size,) = n_slots.shape

        # Create mask of shape B x K where the first n_slots entries per-row are false, the rest true
        indices = torch.arange(self.max_slots, device=n_slots.device)
        masks = indices.unsqueeze(0).expand(batch_size, -1) >= n_slots.unsqueeze(1)

        return masks


class CreateRandomMaskPatterns(nn.Module):
    """Create random masks.

    Useful for showcasing behavior of metrics.
    """

    def __init__(self, pattern: str, n_slots: Optional[int] = None, n_cols: int = 2):
        super().__init__()
        if pattern not in ("random", "blocks"):
            raise ValueError(f"Unknown pattern {pattern}")
        self.pattern = pattern
        self.n_slots = n_slots
        self.n_cols = n_cols

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if self.pattern == "random":
            rand_mask = torch.rand_like(masks)
            return rand_mask / rand_mask.sum(1, keepdim=True)
        elif self.pattern == "blocks":
            n_slots = masks.shape[1] if self.n_slots is None else self.n_slots
            height, width = masks.shape[-2:]
            new_masks = torch.zeros(
                len(masks), n_slots, height, width, device=masks.device, dtype=masks.dtype
            )
            blocks_per_col = int(n_slots // self.n_cols)
            remainder = n_slots - (blocks_per_col * self.n_cols)
            slot = 0
            for col in range(self.n_cols):
                rows = blocks_per_col if col < self.n_cols - 1 else blocks_per_col + remainder
                for row in range(rows):
                    block_width = math.ceil(width / self.n_cols)
                    block_height = math.ceil(height / rows)
                    x = col * block_width
                    y = row * block_height
                    new_masks[:, slot, y : y + block_height, x : x + block_width] = 1
                    slot += 1
            assert torch.allclose(new_masks.sum(1), torch.ones_like(masks[:, 0]))
            return new_masks
