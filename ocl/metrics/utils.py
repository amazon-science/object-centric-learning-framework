"""Utility functions used in metrics computation."""
import torch


def tensor_to_one_hot(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert tensor to one-hot encoding by using maximum across dimension as one-hot element."""
    assert 0 <= dim
    max_idxs = torch.argmax(tensor, dim=dim, keepdim=True)
    shape = [1] * dim + [-1] + [1] * (tensor.ndim - dim - 1)
    one_hot = max_idxs == torch.arange(tensor.shape[dim], device=tensor.device).view(*shape)
    return one_hot.to(torch.long)


def adjusted_rand_index(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
    """Computes adjusted Rand index (ARI), a clustering similarity score.

    This implementation ignores points with no cluster label in `true_mask` (i.e. those points for
    which `true_mask` is a zero vector). In the context of segmentation, that means this function
    can ignore points in an image corresponding to the background (i.e. not to an object).

    Implementation adapted from https://github.com/deepmind/multi_object_datasets and
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_pred_clusters = pred_mask.shape[-1]
    pred_cluster_ids = torch.argmax(pred_mask, axis=-1)

    # Convert true and predicted clusters to one-hot ('oh') representations. We use float64 here on
    # purpose, otherwise mixed precision training automatically casts to FP16 in some of the
    # operations below, which can create overflows.
    true_mask_oh = true_mask.to(torch.float64)  # already one-hot
    pred_mask_oh = torch.nn.functional.one_hot(pred_cluster_ids, n_pred_clusters).to(torch.float64)

    n_ij = torch.einsum("bnc,bnk->bck", true_mask_oh, pred_mask_oh)
    a = torch.sum(n_ij, axis=-1)
    b = torch.sum(n_ij, axis=-2)
    n_fg_points = torch.sum(a, axis=1)

    rindex = torch.sum(n_ij * (n_ij - 1), axis=(1, 2))
    aindex = torch.sum(a * (a - 1), axis=1)
    bindex = torch.sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clamp(n_fg_points * (n_fg_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both true_mask and pred_mask assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == n_fg_points * (n_fg_points-1))
    # 2. If both true_mask and pred_mask assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator > 0, ari, torch.ones_like(ari))


def fg_adjusted_rand_index(
    pred_mask: torch.Tensor, true_mask: torch.Tensor, bg_dim: int = 0
) -> torch.Tensor:
    """Compute adjusted random index using only foreground groups (FG-ARI).

    Args:
        pred_mask: Predicted cluster assignment encoded as categorical probabilities of shape
            (batch_size, n_points, n_pred_clusters).
        true_mask: True cluster assignment encoded as one-hot of shape (batch_size, n_points,
            n_true_clusters).
        bg_dim: Index of background class in true mask.

    Returns:
        ARI scores of shape (batch_size,).
    """
    n_true_clusters = true_mask.shape[-1]
    assert 0 <= bg_dim < n_true_clusters
    if bg_dim == 0:
        true_mask_only_fg = true_mask[..., 1:]
    elif bg_dim == n_true_clusters - 1:
        true_mask_only_fg = true_mask[..., :-1]
    else:
        true_mask_only_fg = torch.cat(
            (true_mask[..., :bg_dim], true_mask[..., bg_dim + 1 :]), dim=-1
        )

    return adjusted_rand_index(pred_mask, true_mask_only_fg)


def _all_equal_masked(values: torch.Tensor, mask: torch.Tensor, dim=-1) -> torch.Tensor:
    """Check if all masked values along a dimension of a tensor are the same.

    All non-masked values are considered as true, i.e. if no value is masked, true is returned
    for this dimension.
    """
    assert mask.dtype == torch.bool
    _, first_non_masked_idx = torch.max(mask, dim=dim)

    comparison_value = values.gather(index=first_non_masked_idx.unsqueeze(dim), dim=dim)

    return torch.logical_or(~mask, values == comparison_value).all(dim=dim)


def masks_to_bboxes(masks: torch.Tensor, empty_value: float = -1.0) -> torch.Tensor:
    """Compute bounding boxes around the provided masks.

    Adapted from DETR: https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

    Args:
        masks: Tensor of shape (N, H, W), where N is the number of masks, H and W are the spatial
            dimensions.
        empty_value: Value bounding boxes should contain for empty masks.

    Returns:
        Tensor of shape (N, 4), containing bounding boxes in (x1, y1, x2, y2) format, where (x1, y1)
        is the coordinate of top-left corner and (x2, y2) is the coordinate of the bottom-right
        corner (inclusive) in pixel coordinates. If mask is empty, all coordinates contain
        `empty_value` instead.
    """
    masks = masks.bool()
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    large_value = 1e8
    inv_mask = ~masks

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    bboxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    bboxes[x_min == large_value] = empty_value

    return bboxes


def _remap_one_hot_mask(
    mask: torch.Tensor, new_classes: torch.Tensor, n_new_classes: int, strip_empty: bool = False
):
    """Remap classes from binary mask to new classes.

    In the case of an overlap of classes for a point, the new class with the highest ID is
    assigned to that point. If no class is assigned to a point, the point will have no class
    assigned after remapping as well.

    Args:
        mask: Binary mask of shape (B, P, K) where K is the number of old classes and P is the
            number of points.
        new_classes: Tensor of shape (B, K) containing ids of new classes for each old class.
        n_new_classes: Number of classes after remapping, i.e. highest class id that can occur.
        strip_empty: Whether to remove the empty pixels mask

    Returns:
        Tensor of shape (B, P, J), where J is the new number of classes.
    """
    assert new_classes.shape[1] == mask.shape[2]
    mask_dense = (mask * new_classes.unsqueeze(1)).max(dim=-1).values
    mask = torch.nn.functional.one_hot(mask_dense.to(torch.long), num_classes=n_new_classes + 1)

    if strip_empty:
        mask = mask[..., 1:]

    return mask
