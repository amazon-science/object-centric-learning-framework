"""Metrics related to the evaluation of masks."""
from typing import Optional

import scipy.optimize
import torch
import torchmetrics

from ocl.metrics.utils import adjusted_rand_index, fg_adjusted_rand_index, tensor_to_one_hot
from ocl.utils.resizing import resize_patches_to_image


class ARIMetric(torchmetrics.Metric):
    """Computes ARI metric."""

    def __init__(
        self,
        foreground: bool = True,
        convert_target_one_hot: bool = False,
        ignore_overlaps: bool = False,
    ):
        super().__init__()
        self.foreground = foreground
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0

        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # For empty pixels (all values zero), one-hot assigns 1 to the first class, correct for
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"

        if self.foreground:
            ari = fg_adjusted_rand_index(prediction, target)
        else:
            ari = adjusted_rand_index(prediction, target)

        self.values += ari.sum()
        self.total += len(ari)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class PatchARIMetric(ARIMetric):
    """Computes ARI metric assuming patch masks as input."""

    def __init__(
        self,
        foreground=True,
        resize_masks_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(foreground=foreground, **kwargs)
        self.resize_masks_mode = resize_masks_mode

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, P) or (B, F, C, P), where C is the
                number of classes and P the number of patches.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        h, w = target.shape[-2:]
        assert h == w

        prediction_resized = resize_patches_to_image(
            prediction, size=h, resize_mode=self.resize_masks_mode
        )

        return super().update(prediction=prediction_resized, target=target)


class UnsupervisedMaskIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for segmentation masks when correspondences to ground truth are not known.

    Uses Hungarian matching to compute the assignment between predicted classes and ground truth
    classes.

    Args:
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: Approach to match predicted to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class. Using "best_overlap"
            leads to the "average best overlap" metric.
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth class was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
        ignore_background: If true, assume class at index 0 of ground truth masks is background class
            that is removed before computing IoU.
        ignore_overlaps: If true, remove points where ground truth masks has overlappign classes from
            predictions and ground truth masks.
    """

    def __init__(
        self,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
    ):
        super().__init__()
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.use_threshold:
            predictions = predictions > self.threshold
        else:
            indices = torch.argmax(predictions, dim=1)
            predictions = torch.nn.functional.one_hot(indices, num_classes=predictions.shape[1])
            predictions = predictions.transpose(1, 2)

        if self.ignore_background:
            targets = targets[:, 1:]

        targets = targets > 0  # Ensure masks are binary

        if self.ignore_overlaps:
            overlaps = targets.sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            predictions[ignore.expand_as(predictions)] = 0
            targets[ignore.expand_as(targets)] = 0

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(targets.sum(dim=1) < 2), "Issues with target format, mask non-exclusive"

        for pred, target in zip(predictions, targets):
            nonzero_classes = torch.sum(target, dim=-1) > 0
            target = target[nonzero_classes]  # Remove empty (e.g. padded) classes
            if len(target) == 0:
                continue  # Skip elements without any target mask

            iou_per_class = unsupervised_mask_iou(
                pred, target, matching=self.matching, reduction="none"
            )

            if self.compute_discovery_fraction:
                discovered = iou_per_class > self.discovery_threshold
                self.values += discovered.sum() / len(discovered)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_class > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_class.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


class MaskCorLocMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", correct_localization=True, **kwargs)


class AverageBestOverlapMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", **kwargs)


class BestOverlapObjectRecoveryMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", compute_discovery_fraction=True, **kwargs)


def unsupervised_mask_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    matching: str = "hungarian",
    reduction: str = "mean",
    iou_empty: float = 0.0,
) -> torch.Tensor:
    """Compute intersection-over-union (IoU) between masks with unknown class correspondences.

    This metric is also known as Jaccard index. Note that this is a non-batched implementation.

    Args:
        pred_mask: Predicted mask of shape (C, N), where C is the number of predicted classes and
            N is the number of points. Masks are assumed to be binary.
        true_mask: Ground truth mask of shape (K, N), where K is the number of ground truth
            classes and N is the number of points. Masks are assumed to be binary.
        matching: How to match predicted classes to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class (each predicted class
            can be assigned to multiple ground truth classes). Empty ground truth classes are
            assigned IoU of zero.
        reduction: If "mean", return IoU averaged over classes. If "none", return per-class IoU.
        iou_empty: IoU for the case when a class does not occur, but was also not predicted.

    Returns:
        Mean IoU over classes if reduction is `mean`, tensor of shape (K,) containing per-class IoU
        otherwise.
    """
    assert pred_mask.ndim == 2
    assert true_mask.ndim == 2
    n_gt_classes = len(true_mask)
    pred_mask = pred_mask.unsqueeze(1).to(torch.bool)
    true_mask = true_mask.unsqueeze(0).to(torch.bool)

    intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
    union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
    pairwise_iou = intersection / union

    # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
    pairwise_iou[union == 0] = iou_empty

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
    elif matching == "best_overlap":
        non_empty_gt = torch.sum(true_mask.squeeze(0), dim=1) > 0
        pred_idxs = torch.argmax(pairwise_iou, dim=0)[non_empty_gt]
        true_idxs = torch.arange(pairwise_iou.shape[1])[non_empty_gt]
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs, true_idxs]
    iou = torch.zeros(n_gt_classes, dtype=torch.float64, device=pairwise_iou.device)
    iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou
