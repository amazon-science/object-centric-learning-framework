from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ocl.utils.routing import RoutableMixin


class FeaturePyramidDecoder(nn.Module, RoutableMixin):
    def __init__(
        self,
        slot_dim: int,
        feature_dim: int,
        mask_path: Optional[str] = None,
        slots_path: Optional[str] = None,
        features_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "slots": slots_path,
                "mask": mask_path,
                "features": features_path,
            },
        )

        inter_dims = [slot_dim, slot_dim // 2, slot_dim // 4, slot_dim // 8, slot_dim // 16]
        # Depth dimension is slot dimension, no padding there and kernel size 1.
        self.lay1 = torch.nn.Conv3d(inter_dims[0], inter_dims[0], (1, 3, 3), padding=(0, 1, 1))
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[0])
        self.lay2 = torch.nn.Conv3d(inter_dims[0], inter_dims[1], (1, 3, 3), padding=(0, 1, 1))
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv3d(inter_dims[1], inter_dims[2], (1, 3, 3), padding=(0, 1, 1))
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv3d(inter_dims[2], inter_dims[3], (1, 3, 3), padding=(0, 1, 1))
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv3d(inter_dims[3], inter_dims[4], (1, 3, 3), padding=(0, 1, 1))
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.ConvTranspose3d(
            inter_dims[4],
            1,
            stride=(1, 2, 2),
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            output_padding=(0, 1, 1),
        )

        upsampled_dim = feature_dim // 8
        self.upsampling = nn.ConvTranspose2d(
            feature_dim, upsampled_dim, kernel_size=8, stride=8
        )  # 112 x 112
        self.adapter1 = nn.Conv2d(
            upsampled_dim, inter_dims[0], kernel_size=5, padding=2, stride=8
        )  # Should downsample 112 to 14
        self.adapter2 = nn.Conv2d(
            upsampled_dim, inter_dims[1], kernel_size=5, padding=2, stride=4
        )  # 28x28
        self.adapter3 = nn.Conv2d(
            upsampled_dim, inter_dims[2], kernel_size=5, padding=2, stride=2
        )  # 56 x 56
        self.adapter4 = nn.Conv2d(
            upsampled_dim, inter_dims[3], kernel_size=5, padding=2, stride=1
        )  # 112 x 112

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, slots: torch.Tensor, mask: torch.Tensor, features: torch.Tensor):
        # Bring features into image format with channels first
        features = features.unflatten(1, (14, 14)).permute(0, 3, 1, 2)
        mask = mask.unflatten(-1, (14, 14))
        # Use depth dimension for slots
        x = slots.transpose(1, 2)[..., None, None] * mask.unsqueeze(1)
        bs, n_channels, n_slots, width, height = x.shape

        upsampled_features = self.upsampling(features)

        # Add fake depth dimension for broadcasting and upsample representation.
        x = self.lay1(x) + self.adapter1(upsampled_features).unsqueeze(2)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(upsampled_features)
        # Add fake depth dimension for broadcasting and upsample representation.
        x = cur_fpn.unsqueeze(2) + F.interpolate(
            x, size=(n_slots,) + cur_fpn.shape[-2:], mode="nearest"
        )
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(upsampled_features)
        x = cur_fpn.unsqueeze(2) + F.interpolate(
            x, size=(n_slots,) + cur_fpn.shape[-2:], mode="nearest"
        )
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter4(upsampled_features)
        x = cur_fpn.unsqueeze(2) + F.interpolate(
            x, size=(n_slots,) + cur_fpn.shape[-2:], mode="nearest"
        )
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        # Squeeze channel dimension.
        x = self.out_lay(x).squeeze(1).softmax(1)
        return x
