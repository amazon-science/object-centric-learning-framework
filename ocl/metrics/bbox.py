"""Metrics related to the evaluation of bounding boxes."""
import scipy.optimize
import torch
import torchmetrics
import torchvision

from ocl.metrics.utils import masks_to_bboxes


class UnsupervisedBboxIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for bounding boxes when correspondences to ground truth are not known.

    Currently, assumes segmentation masks as input for both prediction and targets.

    Args:
        target_is_mask: If `True`, assume input is a segmentation mask, in which case the masks are
            converted to bounding boxes before computing IoU. If `False`, assume the input for the
            targets are already bounding boxes.
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: How to match predicted boxes to ground truth boxes. For "hungarian", computes
            assignment that maximizes total IoU between all boxes. For "best_overlap", uses the
            predicted box with maximum overlap for each ground truth box (each predicted box
            can be assigned to multiple ground truth boxes).
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold. This
            is recall, or sometimes called the detection rate metric.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth bounding box was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
    """

    def __init__(
        self,
        target_is_mask: bool = False,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
    ):
        super().__init__()
        self.target_is_mask = target_is_mask
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

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of instances. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of instance, if using masks as input, or bounding boxes of shape (B, K, 4)
                or (B, F, K, 4).
        """
        if prediction.ndim == 5:
            # Merge batch and frame dimensions
            prediction = prediction.flatten(0, 1)
            target = target.flatten(0, 1)
        elif prediction.ndim != 4:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        bs, n_pred_classes = prediction.shape[:2]
        n_gt_classes = target.shape[1]

        if self.use_threshold:
            prediction = prediction > self.threshold
        else:
            indices = torch.argmax(prediction, dim=1)
            prediction = torch.nn.functional.one_hot(indices, num_classes=n_pred_classes)
            prediction = prediction.permute(0, 3, 1, 2)

        pred_bboxes = masks_to_bboxes(prediction.flatten(0, 1)).unflatten(0, (bs, n_pred_classes))
        if self.target_is_mask:
            target_bboxes = masks_to_bboxes(target.flatten(0, 1)).unflatten(0, (bs, n_gt_classes))
        else:
            assert target.shape[-1] == 4
            # Convert all-zero boxes added during padding to invalid boxes
            target[torch.all(target == 0.0, dim=-1)] = -1.0
            target_bboxes = target

        for pred, target in zip(pred_bboxes, target_bboxes):
            valid_pred_bboxes = pred[:, 0] != -1.0
            valid_target_bboxes = target[:, 0] != -1.0
            if valid_target_bboxes.sum() == 0:
                continue  # Skip data points without any target bbox

            pred = pred[valid_pred_bboxes]
            target = target[valid_target_bboxes]

            if valid_pred_bboxes.sum() > 0:
                iou_per_bbox = unsupervised_bbox_iou(
                    pred, target, matching=self.matching, reduction="none"
                )
            else:
                iou_per_bbox = torch.zeros_like(valid_target_bboxes, dtype=torch.float32)

            if self.compute_discovery_fraction:
                discovered = iou_per_bbox > self.discovery_threshold
                self.values += discovered.sum() / len(iou_per_bbox)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_bbox > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_bbox.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


class BboxCorLocMetric(UnsupervisedBboxIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", correct_localization=True, **kwargs)


class BboxRecallMetric(UnsupervisedBboxIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", compute_discovery_fraction=True, **kwargs)


def unsupervised_bbox_iou(
    pred_bboxes: torch.Tensor,
    true_bboxes: torch.Tensor,
    matching: str = "best_overlap",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute IoU between two sets of bounding boxes.

    Args:
        pred_bboxes: Predicted bounding boxes of shape N x 4.
        true_bboxes: True bounding boxes of shape M x 4.
        matching: Method to assign predicted to true bounding boxes.
        reduction: Whether to average the computes IoUs per true box.
    """
    n_gt_bboxes = len(true_bboxes)

    pairwise_iou = torchvision.ops.box_iou(pred_bboxes, true_bboxes)

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
    elif matching == "best_overlap":
        pred_idxs = torch.argmax(pairwise_iou, dim=0)
        true_idxs = torch.arange(pairwise_iou.shape[1], device=pairwise_iou.device)
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs, true_idxs]

    iou = torch.zeros(n_gt_bboxes, dtype=torch.float32, device=pairwise_iou.device)
    iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou
