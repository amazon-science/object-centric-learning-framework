"""Metrics computed on a whole dataset."""
from typing import Any, Dict, Optional

import numpy as np
import scipy.optimize
import torch
import torchmetrics

from ocl.metrics.utils import _remap_one_hot_mask


class DatasetSemanticMaskIoUMetric(torchmetrics.Metric):
    """Unsupervised IoU metric for semantic segmentation using dataset-wide matching of classes.

    The input to this metric is an instance-level mask with objects, and a class id for each object.
    This is required to convert the mask to semantic classes. The number of classes for the
    predictions does not have to match the true number of classes.

    Note that contrary to the other metrics in this module, this metric is not supposed to be added
    in the online metric computation loop, which is why it does not inherit from `RoutableMixin`.

    Args:
        n_predicted_classes: Number of predictable classes, i.e. highest prediction class id that can
            occur.
        n_classes: Total number of classes, i.e. highest class id that can occur.
        threshold: Value to use for thresholding masks.
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using an argmax instead.
        matching: Method to produce matching between clusters and ground truth classes. If
            "hungarian", assigns each class one cluster such that the total IoU is maximized. If
            "majority", assigns each cluster to the class with the highest IoU (each class can be
            assigned multiple clusters).
        ignore_background: If true, pixels labeled as background (class zero) in the ground truth
            are not taken into account when computing IoU.
        use_unmatched_as_background: If true, count predicted classes not selected after Hungarian
            matching as the background predictions.
    """

    def __init__(
        self,
        n_predicted_classes: int,
        n_classes: int,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        ignore_background: bool = False,
        use_unmatched_as_background: bool = False,
    ):
        super().__init__()
        matching_methods = {"hungarian", "majority"}
        if matching not in matching_methods:
            raise ValueError(
                f"Unknown matching method {matching}. Valid values are {matching_methods}."
            )

        self.matching = matching
        self.n_predicted_classes = n_predicted_classes
        self.n_predicted_classes_with_bg = n_predicted_classes + 1
        self.n_classes = n_classes
        self.n_classes_with_bg = n_classes + 1
        self.matching = matching
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.ignore_background = ignore_background
        self.use_unmatched_as_background = use_unmatched_as_background
        if use_unmatched_as_background and ignore_background:
            raise ValueError(
                "Option `use_unmatched_as_background` not compatible with option `ignore_background`"
            )
        if use_unmatched_as_background and matching == "majority":
            raise ValueError(
                "Option `use_unmatched_as_background` not compatible with matching `majority`"
            )

        confusion_mat = torch.zeros(
            self.n_predicted_classes_with_bg, self.n_classes_with_bg, dtype=torch.int64
        )
        self.add_state("confusion_mat", default=confusion_mat, dist_reduce_fx="sum", persistent=True)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prediction_class_ids: torch.Tensor,
        ignore: Optional[torch.Tensor] = None,
    ):
        """Update metric by computing confusion matrix between predicted and target classes.

        Args:
            predictions: Probability mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of object instances in the image.
            targets: Mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the number of object
                instances in the image. Class ID of objects is encoded as the value, i.e. densely
                represented.
            prediction_class_ids: Tensor of shape (B, K), containing the class id of each predicted
                object instance in the image. Id must be 0 <= id <= n_predicted_classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        predictions = self.preprocess_predicted_mask(predictions)
        predictions = _remap_one_hot_mask(
            predictions, prediction_class_ids, self.n_predicted_classes, strip_empty=False
        )
        assert predictions.shape[-1] == self.n_predicted_classes_with_bg

        targets = self.preprocess_ground_truth_mask(targets)
        assert targets.shape[-1] == self.n_classes_with_bg

        if ignore is not None:
            if ignore.ndim == 5:  # Video case
                ignore = ignore.flatten(0, 1)
            assert ignore.ndim == 4 and ignore.shape[1] == 1
            ignore = ignore.to(torch.bool).flatten(-2, -1).squeeze(1)  # B x P
            predictions[ignore] = 0
            targets[ignore] = 0

        # We are doing the multiply in float64 instead of int64 because it proved to be significantly
        # faster on GPU. We need to use 64 bits because we can easily exceed the range of 32 bits
        # if we aggregate over a full dataset.
        confusion_mat = torch.einsum(
            "bpk,bpc->kc", predictions.to(torch.float64), targets.to(torch.float64)
        )
        self.confusion_mat += confusion_mat.to(torch.int64)

    def preprocess_predicted_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Preprocess predicted masks for metric computation.

        Args:
            mask: Probability mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the number
                of object instances in the prediction.

        Returns:
            Binary tensor of shape (B, P, K), where P is the number of points. If `use_threshold` is
            True, overlapping objects for the same point are possible.
        """
        if mask.ndim == 5:  # Video case
            mask = mask.flatten(0, 1)
        mask = mask.flatten(-2, -1)

        if self.use_threshold:
            mask = mask > self.threshold
            mask = mask.transpose(1, 2)
        else:
            maximum, indices = torch.max(mask, dim=1)
            mask = torch.nn.functional.one_hot(indices, num_classes=mask.shape[1])
            mask[:, :, 0][maximum == 0.0] = 0

        return mask

    def preprocess_ground_truth_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Preprocess ground truth mask for metric computation.

        Args:
            mask: Mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the number of object
                instances in the image. Class ID of objects is encoded as the value, i.e. densely
                represented.

        Returns:
            One-hot tensor of shape (B, P, J), where J is the number of the classes and P the number
            of points, with object instances with the same class ID merged together. In the case of
            an overlap of classes for a point, the class with the highest ID is assigned to that
            point.
        """
        if mask.ndim == 5:  # Video case
            mask = mask.flatten(0, 1)
        mask = mask.flatten(-2, -1)

        # Pixels which contain no object get assigned the background class 0. This also handles the
        # padding of zero masks which is done in preprocessing for batching.
        mask = torch.nn.functional.one_hot(
            mask.max(dim=1).values.to(torch.long), num_classes=self.n_classes_with_bg
        )

        return mask

    def compute(self):
        """Compute per-class IoU using matching."""
        if self.ignore_background:
            n_classes = self.n_classes
            confusion_mat = self.confusion_mat[:, 1:]
        else:
            n_classes = self.n_classes_with_bg
            confusion_mat = self.confusion_mat

        pairwise_iou, _, _, area_gt = self._compute_iou_from_confusion_mat(confusion_mat)

        if self.use_unmatched_as_background:
            # Match only in foreground
            pairwise_iou = pairwise_iou[1:, 1:]
            confusion_mat = confusion_mat[1:, 1:]
        else:
            # Predicted class zero is not matched against anything
            pairwise_iou = pairwise_iou[1:]
            confusion_mat = confusion_mat[1:]

        if self.matching == "hungarian":
            cluster_idxs, class_idxs = scipy.optimize.linear_sum_assignment(
                pairwise_iou.cpu(), maximize=True
            )
            cluster_idxs = torch.as_tensor(
                cluster_idxs, dtype=torch.int64, device=self.confusion_mat.device
            )
            class_idxs = torch.as_tensor(
                class_idxs, dtype=torch.int64, device=self.confusion_mat.device
            )
            matched_iou = pairwise_iou[cluster_idxs, class_idxs]
            true_pos = confusion_mat[cluster_idxs, class_idxs]

            if self.use_unmatched_as_background:
                cluster_oh = torch.nn.functional.one_hot(
                    cluster_idxs, num_classes=pairwise_iou.shape[0]
                )
                matched_clusters = cluster_oh.max(dim=0).values.to(torch.bool)
                bg_pred = self.confusion_mat[:1]
                bg_pred += self.confusion_mat[1:][~matched_clusters].sum(dim=0)
                bg_iou, _, _, _ = self._compute_iou_from_confusion_mat(bg_pred, area_gt)
                class_idxs = torch.cat((torch.zeros_like(class_idxs[:1]), class_idxs + 1))
                matched_iou = torch.cat((bg_iou[0, :1], matched_iou))
                true_pos = torch.cat((bg_pred[0, :1], true_pos))

        elif self.matching == "majority":
            max_iou, class_idxs = torch.max(pairwise_iou, dim=1)
            # Form new clusters by merging old clusters which are assigned the same ground truth
            # class. After merging, the number of clusters equals the number of classes.
            _, old_to_new_cluster_idx = torch.unique(class_idxs, return_inverse=True)

            confusion_mat_new = torch.zeros(
                n_classes, n_classes, dtype=torch.int64, device=self.confusion_mat.device
            )
            for old_cluster_idx, new_cluster_idx in enumerate(old_to_new_cluster_idx):
                if max_iou[old_cluster_idx] > 0.0:
                    confusion_mat_new[new_cluster_idx] += confusion_mat[old_cluster_idx]

            # Important: use previously computed area_gt because it includes background predictions,
            # whereas the new confusion matrix does not contain the bg predicted class anymore.
            pairwise_iou, _, _, _ = self._compute_iou_from_confusion_mat(confusion_mat_new, area_gt)
            max_iou, class_idxs = torch.max(pairwise_iou, dim=1)
            valid = max_iou > 0.0  # Ignore clusters without any kind of overlap
            class_idxs = class_idxs[valid]
            cluster_idxs = torch.arange(pairwise_iou.shape[1])[valid]
            matched_iou = pairwise_iou[cluster_idxs, class_idxs]
            true_pos = confusion_mat_new[cluster_idxs, class_idxs]

        else:
            raise RuntimeError(f"Unsupported matching: {self.matching}")

        iou = torch.zeros(n_classes, dtype=torch.float64, device=pairwise_iou.device)
        iou[class_idxs] = matched_iou

        accuracy = true_pos.sum().to(torch.float64) / area_gt.sum()
        empty_classes = area_gt == 0

        return iou, accuracy, empty_classes

    @staticmethod
    def _compute_iou_from_confusion_mat(
        confusion_mat: torch.Tensor, area_gt: Optional[torch.Tensor] = None
    ):
        area_pred = torch.sum(confusion_mat, axis=1)
        if area_gt is None:
            area_gt = torch.sum(confusion_mat, axis=0)
        union = area_pred.unsqueeze(1) + area_gt.unsqueeze(0) - confusion_mat
        pairwise_iou = confusion_mat.to(torch.float64) / union

        # Ignore classes that occured on no image.
        pairwise_iou[union == 0] = 0.0

        return pairwise_iou, union, area_pred, area_gt


class SklearnClustering:
    """Wrapper around scikit-learn clustering algorithms.

    Args:
        n_clusters: Number of clusters.
        method: Clustering method to use.
        clustering_kwargs: Dictionary of additional keyword arguments to pass to clustering object.
        use_l2_normalization: Whether to L2 normalize the representations before clustering (but
            after PCA).
        use_pca: Whether to apply PCA before fitting the clusters.
        pca_dimensions: Number of dimensions for PCA dimensionality reduction. If `None`, do not
            reduce dimensions with PCA.
        pca_kwargs: Dictionary of additional keyword arguments to pass to PCA object.
    """

    def __init__(
        self,
        n_clusters: int,
        method: str = "kmeans",
        clustering_kwargs: Optional[Dict[str, Any]] = None,
        use_l2_normalization: bool = False,
        use_pca: bool = False,
        pca_dimensions: Optional[int] = None,
        pca_kwargs: Optional[Dict[str, Any]] = None,
    ):
        methods = ("kmeans", "spectral")
        if method not in methods:
            raise ValueError(f"Unknown clustering method {method}. Valid values are {methods}.")

        self._n_clusters = n_clusters
        self.method = method
        self.clustering_kwargs = clustering_kwargs
        self.use_l2_normalization = use_l2_normalization
        self.use_pca = use_pca
        self.pca_dimensions = pca_dimensions
        self.pca_kwargs = pca_kwargs

        self._clustering = None
        self._pca = None

    @property
    def n_clusters(self):
        return self._n_clusters

    def _init(self):
        from sklearn import cluster, decomposition

        kwargs = self.clustering_kwargs if self.clustering_kwargs is not None else {}
        if self.method == "kmeans":
            self._clustering = cluster.KMeans(n_clusters=self.n_clusters, **kwargs)
        elif self.method == "spectral":
            self._clustering = cluster.SpectralClustering(n_clusters=self.n_clusters, **kwargs)
        else:
            raise NotImplementedError(f"Clustering {self.method} not implemented.")

        if self.use_pca:
            kwargs = self.pca_kwargs if self.pca_kwargs is not None else {}
            self._pca = decomposition.PCA(n_components=self.pca_dimensions, **kwargs)

    def fit_predict(self, features: torch.Tensor):
        self._init()
        features = features.detach().cpu().numpy()
        if self.use_pca:
            features = self._pca.fit_transform(features)
        if self.use_l2_normalization:
            features /= np.maximum(np.linalg.norm(features, ord=2, axis=1, keepdims=True), 1e-8)
        cluster_ids = self._clustering.fit_predict(features).astype(np.int64)
        return torch.from_numpy(cluster_ids)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self._clustering is None:
            raise ValueError("Clustering was not fitted. Call `fit_predict` first.")

        features = features.detach().cpu().numpy()
        if self.use_pca:
            features = self._pca.transform(features)
        if self.use_l2_normalization:
            features /= np.maximum(np.linalg.norm(features, ord=2, axis=1, keepdims=True), 1e-8)
        cluster_ids = self._clustering.predict(features).astype(np.int64)
        return torch.from_numpy(cluster_ids)
