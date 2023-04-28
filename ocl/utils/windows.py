"""Utility functions related to windows of inputs."""
import torch


class JoinWindows(torch.nn.Module):
    """Join individual windows to single output."""

    def __init__(self, n_windows: int, size):
        super().__init__()
        self.n_windows = n_windows
        self.size = size

    def forward(self, masks: torch.Tensor, keys: str) -> torch.Tensor:
        assert len(masks) == self.n_windows
        keys_split = [key.split("_") for key in keys]
        pad_left = [int(elems[1]) for elems in keys_split]
        pad_top = [int(elems[2]) for elems in keys_split]

        target_height, target_width = self.size
        n_masks = masks.shape[0] * masks.shape[1]
        height, width = masks.shape[2], masks.shape[3]
        full_mask = torch.zeros(n_masks, *self.size).to(masks)
        x = 0
        y = 0
        for idx, mask in enumerate(masks):
            elems = masks.shape[1]
            x_start = 0 if pad_left[idx] >= 0 else -pad_left[idx]
            x_end = min(width, target_width - pad_left[idx])
            y_start = 0 if pad_top[idx] >= 0 else -pad_top[idx]
            y_end = min(height, target_height - pad_top[idx])
            cropped = mask[:, y_start:y_end, x_start:x_end]
            full_mask[
                idx * elems : (idx + 1) * elems, y : y + cropped.shape[-2], x : x + cropped.shape[-1]
            ] = cropped
            x += cropped.shape[-1]
            if x > target_width:
                y += cropped.shape[-2]
                x = 0

        assert torch.all(torch.abs(torch.sum(full_mask, axis=0) - 1) <= 1e-2)

        return full_mask.unsqueeze(0)
