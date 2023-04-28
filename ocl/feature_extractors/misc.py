"""Feature extractors from diverse papers.

In particular this module implements feature extractors from

 - [SlotAttention][ocl.feature_extractors.misc.SlotAttentionFeatureExtractor]
   Locatello et al., Object-Centric Learning with Slot Attention, NeurIPS 2020
 - [SAVi][ocl.feature_extractors.misc.SAViFeatureExtractor]
   Kipf et al., Conditional Object-Centric Learning from Video, ICLR 2020
 - [SLATE][ocl.feature_extractors.misc.DVAEFeatureExtractor]
   Singh et al., Simple Unsupervised Object-Centric Learning for Complex and
   Naturalistic Videos, NeurIPS 2022
"""
import torch
from torch import nn

from ocl.feature_extractors.utils import ImageFeatureExtractor, cnn_compute_positions_and_flatten


class SlotAttentionFeatureExtractor(ImageFeatureExtractor):
    """CNN-based feature extractor as used in the slot attention paper.

    Reference: Locatello et al., Object-Centric Learning with Slot Attention, NeurIPS 2020
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )

    @property
    def feature_dim(self):
        return 64

    def forward_images(self, images: torch.Tensor):
        features = self.layers(images)
        flattened, positions = cnn_compute_positions_and_flatten(features)
        return flattened, positions


class SAViFeatureExtractor(ImageFeatureExtractor):
    """CNN-based feature extractor as used in the slot attention for video paper.

    Reference: Kipf et al., Conditional Object-Centric Learning from Video, ICLR 2020
    """

    def __init__(self, larger_input_arch: bool = False):
        """Initialize SAVi feature extractor.

        Args:
            larger_input_arch: Use the architecture for larger image datasets such as MOVi++, which
                contains more a stride in the first layer and a higher number of feature channels in
                the CNN backbone.
        """
        super().__init__()
        self.larger_input_arch = larger_input_arch
        if larger_input_arch:
            self.layers = nn.Sequential(
                # Pytorch does not support stride>1 with padding=same.
                # Implement tensorflow behaviour manually.
                # See: https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(3, out_channels=64, kernel_size=5, stride=2, padding="valid"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(3, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
            )

    @property
    def feature_dim(self):
        return 64 if self.larger_input_arch else 32

    def forward_images(self, images: torch.Tensor):
        features = self.layers(images)
        flattened, positions = cnn_compute_positions_and_flatten(features)
        return flattened, positions


class DVAEFeatureExtractor(ImageFeatureExtractor):
    """DVAE VQ Encoder as used in SLATE.

    Reference:
        Singh et al., Simple Unsupervised Object-Centric Learning for Complex and
        Naturalistic Videos, NeurIPS 2022
    """

    def __init__(
        self,
        encoder: nn.Module,
        positional_encoder: nn.Module,
        dictionary: nn.Module,
        tau: float = 1.0,
        hard: bool = False,
    ):
        """Feature extractor as used in the SLATE paper.

        Args:
            encoder: torch Module that transforms image to the patch representations.
            positional_encoder: torch Module that adds pos encoding.
            dictionary: map from onehot vectors to embeddings.
            tau: temporature for gumbel_softmax.
            hard: hard gumbel_softmax if True.
        """
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dictionary = dictionary
        self.positional_encoder = positional_encoder
        self.encoder = encoder

    @property
    def feature_dim(self):
        return 64

    def forward_images(self, images: torch.Tensor):
        z_logits = nn.functional.log_softmax(self.encoder(images), dim=1)
        _, _, H_enc, W_enc = z_logits.size()
        z = nn.functional.gumbel_softmax(z_logits, float(self.tau), self.hard, dim=1)
        z_hard = nn.functional.gumbel_softmax(z_logits, float(self.tau), True, dim=1).detach()

        # add beginning of sequence (BOS) token
        # [1, 0, 0, 0, ...] is encoding for BOS token
        # and each sequence starts from such token
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        # add first zeros column to the z_hard matrix
        z_transformer_input = torch.cat([torch.zeros_like(z_hard[..., :1]), z_hard], dim=-1)
        # add first zeros row to the z_hard matrix
        z_transformer_input = torch.cat(
            [torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2
        )
        # fill new row and column with one,
        # so that we added [1, 0, 0, 0, ...] token
        z_transformer_input[:, 0, 0] = 1.0

        # tokens to embeddings
        features = self.dictionary(z_transformer_input)
        features = self.positional_encoder(features)

        slot_attention_features = features[:, 1:]

        transformer_input = features[:, :-1]
        aux_features = {
            "z": z,
            "targets": transformer_input,
            "z_hard": z_hard,
        }
        return slot_attention_features, None, aux_features
