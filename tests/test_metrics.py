import warnings

import numpy as np
import pytest
import sklearn.metrics
import torch
from torch.nn import functional as F

import ocl.metrics.masks
import ocl.metrics.utils
from ocl import metrics


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_log_tensor(reduction):
    def check(metric, tensor):
        if reduction == "mean":
            expected = tensor.to(torch.float64).mean(-1).mean()
        elif reduction == "sum":
            expected = tensor.to(torch.float64).sum(-1).mean()

        assert torch.allclose(metric.compute(), expected)

    metric = metrics.TensorStatistic(reduction=reduction)

    tensor1 = torch.randn(1, 3)
    metric.update(tensor1)
    check(metric, tensor1)

    tensor2 = torch.tensor([[1, 4, 3], [-2, 1, 5], [30, 1, 2]])
    metric.update(tensor2)
    check(metric, torch.cat([tensor1, tensor2]))


def test_ari_metric():
    metric = metrics.ARIMetric(foreground=False)
    # All clusters match
    pred = _make_one_hot([[0, 0, 0, 0], [2, 2, 1, 1]], 3)
    target = _make_one_hot([[0, 0, 0, 0], [2, 2, 1, 1]], 3)
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),  # ARIMetric wants image data
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    metric = metrics.ARIMetric(foreground=True)
    # All clusters in foreground match
    pred = _make_one_hot([[1, 2, 1, 1], [1, 1, 1, 0]], 3)
    target = _make_one_hot([[0, 0, 1, 1], [2, 2, 0, 1]], 3)
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    # Test ignore overlaps
    metric = metrics.ARIMetric(
        foreground=False,
        ignore_overlaps=True,
    )
    # All clusters at non-overlapping locations match
    pred = _make_one_hot([[0, 0, 0, 1], [2, 2, 1, 2]], 3)
    target = _make_one_hot([[0, 0, 0, 0], [2, 2, 1, 1]], 3)
    target[:, -1, :] = 1
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    # Test ignore mask
    metric = metrics.ARIMetric(
        foreground=False,
    )
    # All clusters at non-ignored locations match
    pred = _make_one_hot([[0, 0, 0, 1], [2, 2, 1, 2]], 3)
    target = _make_one_hot([[0, 0, 0, 0], [2, 2, 1, 1]], 3)
    ignore = torch.zeros_like(target)[..., :1]
    ignore[:, -1] = 1
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
        ignore.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))


def test_ari_metric_video():
    # TODO (hornmax): Add test case where batch-wise and frame-wise flattening differ.
    metric = metrics.ARIMetric(foreground=False)
    # All clusters match
    pred = _make_one_hot([[0, 0, 0, 0] * 2, [2, 2, 1, 1] * 2], 3)
    target = _make_one_hot([[0, 0, 0, 0] * 2, [2, 2, 1, 1] * 2], 3)

    # Reshape to video data.
    pred = pred.permute(0, 2, 1).unflatten(-1, (2, 2, 2)).transpose(1, 2)
    target = target.permute(0, 2, 1).unflatten(-1, (2, 2, 2)).transpose(1, 2)
    metric.update(pred, target)
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    metric = metrics.ARIMetric(foreground=True)
    # All clusters in foreground match
    pred = _make_one_hot([[1, 2, 1, 1] * 2, [1, 1, 1, 0] * 2], 3)
    target = _make_one_hot([[0, 0, 1, 1] * 2, [2, 2, 0, 1] * 2], 3)
    pred = pred.permute(0, 2, 1).unflatten(-1, (2, 2, 2)).transpose(1, 2)
    target = target.permute(0, 2, 1).unflatten(-1, (2, 2, 2)).transpose(1, 2)
    metric.update(pred, target)
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))


def test_adjusted_rand_index(assert_tensors_equal):
    # All clusters match
    assert_tensors_equal(
        ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([[0, 0, 0, 0], [1, 1, 1, 1]], 2),
            _make_one_hot([[0, 0, 0, 0], [1, 1, 1, 1]], 2),
        ),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )

    # No clusters match
    assert_tensors_equal(
        ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([[0, 1, 2, 3], [0, 1, 2, 3]], 4),
            _make_one_hot([[0, 0, 0, 0], [1, 1, 1, 1]], 2),
        ),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
    )

    # Some clusters match (expected result from sklearn.metrics.adjusted_rand_score)
    assert_tensors_equal(
        ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([[0, 0, 1, 1, 2, 2]], 3), _make_one_hot([[0, 0, 0, 1, 1, 1]], 3)
        ),
        torch.tensor([0.24242424], dtype=torch.float64),
    )
    # Some clusters match, different class indices
    assert_tensors_equal(
        ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([[1, 1, 2, 2, 3, 3]], 4), _make_one_hot([[0, 0, 0, 1, 1, 1]], 3)
        ),
        torch.tensor([0.24242424], dtype=torch.float64),
    )

    # A single point
    assert_tensors_equal(
        ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([[0]], 1),
            _make_one_hot([[0]], 1),
        ),
        torch.tensor([1.0], dtype=torch.float64),
    )

    # One cluster per data point
    assert_tensors_equal(
        ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([[0, 1]], 2), _make_one_hot([[1, 0]], 2)
        ),
        torch.tensor([1.0], dtype=torch.float64),
    )


def test_adjusted_rand_index_with_sklearn():
    def get_scores(labels_true, labels_pred, n_classes):
        sklearn_score = sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
        our_score = ocl.metrics.masks.adjusted_rand_index(
            _make_one_hot([labels_pred], n_classes),
            _make_one_hot([labels_true], n_classes),
        ).numpy()
        return sklearn_score, our_score

    labels_pred = [0, 0, 0, 0]
    labels_true = [0, 0, 0, 0]
    sklearn_score, our_score = get_scores(labels_true, labels_pred, n_classes=1)
    assert our_score == pytest.approx(sklearn_score, rel=1e-5, abs=1e-8)

    labels_pred = [1, 1, 1, 1]
    labels_true = [1, 1, 1, 1]
    sklearn_score, our_score = get_scores(labels_true, labels_pred, n_classes=2)
    assert our_score == pytest.approx(sklearn_score, rel=1e-5, abs=1e-8)

    labels_pred = [0, 1, 2, 3]
    labels_true = [0, 1, 2, 3]
    sklearn_score, our_score = get_scores(labels_true, labels_pred, n_classes=4)
    assert our_score == pytest.approx(sklearn_score, rel=1e-5, abs=1e-8)

    # Fuzzy test with random data
    n_classes = 4
    for _ in range(20):
        labels_true = np.random.randint(n_classes, size=10)
        labels_pred = np.random.randint(n_classes, size=10)
        sklearn_score, our_score = get_scores(labels_true, labels_pred, n_classes)
        assert our_score == pytest.approx(sklearn_score, rel=1e-5, abs=1e-8)


def test_fg_adjusted_rand_index(assert_tensors_equal):
    # All clusters in foreground match
    assert_tensors_equal(
        ocl.metrics.masks.fg_adjusted_rand_index(
            _make_one_hot([[1, 2, 1, 1, 2], [1, 1, 1, 0, 0]], 3),
            _make_one_hot([[0, 0, 1, 1, 2], [2, 2, 0, 1, 1]], 3),
            bg_dim=0,
        ),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    assert_tensors_equal(
        ocl.metrics.masks.fg_adjusted_rand_index(
            _make_one_hot([[0, 0, 1, 1, 1], [0, 0, 1, 0, 0]], 3),
            _make_one_hot([[0, 0, 1, 1, 2], [2, 2, 0, 1, 1]], 3),
            bg_dim=1,
        ),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    assert_tensors_equal(
        ocl.metrics.masks.fg_adjusted_rand_index(
            _make_one_hot([[0, 0, 1, 1, 1], [0, 1, 1, 2, 2]], 3),
            _make_one_hot([[0, 0, 1, 1, 2], [2, 2, 0, 1, 1]], 3),
            bg_dim=2,
        ),
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )

    # Tricky case where implementation should not return NaN: only one true and predicted cluster
    assert torch.allclose(
        ocl.metrics.masks.fg_adjusted_rand_index(
            _make_one_hot([[2, 2, 1, 1]], 3),
            _make_one_hot([[0, 0, 2, 2]], 4),
            bg_dim=0,
        ),
        torch.tensor([1.0], dtype=torch.float64),
    )


def test_all_equal_masked():
    values = torch.tensor([[1, 2, 1, 1], [1, 1, 1, 2]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]).to(torch.bool)
    assert torch.all(
        ocl.metrics.utils._all_equal_masked(values, mask, dim=-1) == torch.tensor([True, False])
    )

    values = torch.tensor([[1, 1], [1, 2]])
    mask = torch.tensor([[0, 0], [0, 0]]).to(torch.bool)
    assert torch.all(
        ocl.metrics.utils._all_equal_masked(values, mask, dim=-1) == torch.tensor([True, True])
    )


@pytest.mark.parametrize("matching", ["hungarian", "best_overlap"])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_unsupervised_mask_iou_hungarian(matching, reduction, assert_tensors_equal):
    def make_tensor(tensor, reduction):
        tensor = torch.tensor(tensor, dtype=torch.float64)
        if reduction == "mean":
            return tensor.mean()
        else:
            return tensor

    hungarian = matching == "hungarian"

    # All clusters match, with one class not occurring and not predicted
    assert_tensors_equal(
        ocl.metrics.masks.unsupervised_mask_iou(
            _make_one_hot([0, 0, 1, 1], 3).transpose(0, 1),
            _make_one_hot([1, 1, 0, 0], 3).transpose(0, 1),
            matching=matching,
            reduction=reduction,
            iou_empty=1.0,
        ),
        make_tensor([1.0, 1.0, 1.0] if hungarian else [1.0, 1.0, 0.0], reduction),
    )

    # All clusters match, more predicted classes than ground truth classes
    assert_tensors_equal(
        ocl.metrics.masks.unsupervised_mask_iou(
            _make_one_hot([0, 0, 2, 2], 3).transpose(0, 1),
            _make_one_hot([1, 1, 0, 0], 2).transpose(0, 1),
            matching=matching,
            reduction=reduction,
        ),
        make_tensor([1.0, 1.0], reduction),
    )

    # All clusters match, but two ground truth classes missing
    assert_tensors_equal(
        ocl.metrics.masks.unsupervised_mask_iou(
            _make_one_hot([0, 0, 1, 1], 2).transpose(0, 1),
            _make_one_hot([1, 1, 2, 2], 4).transpose(0, 1),
            matching=matching,
            reduction=reduction,
        ),
        make_tensor([0.0, 1.0, 1.0, 0.0], reduction),
    )

    # Partial overlap, first class does not occur but was also not predicted
    assert_tensors_equal(
        ocl.metrics.masks.unsupervised_mask_iou(
            _make_one_hot([0, 0, 2, 3], 4).transpose(0, 1),
            _make_one_hot([1, 1, 1, 1], 2).transpose(0, 1),
            matching=matching,
            reduction=reduction,
            iou_empty=1.0,
        ),
        make_tensor([1.0, 0.5] if hungarian else [0.0, 0.5], reduction),
    )

    # Partial overlap, predicted class assigned to both ground truth classes in best_overlap matching
    assert_tensors_equal(
        ocl.metrics.masks.unsupervised_mask_iou(
            _make_one_hot([0, 0, 0, 0], 2).transpose(0, 1),
            _make_one_hot([0, 0, 1, 1], 2).transpose(0, 1),
            matching=matching,
            reduction=reduction,
        ),
        make_tensor([0.5, 0.0] if hungarian else [0.5, 0.5], reduction),
    )


@pytest.mark.parametrize("use_threshold", [False, True])
@pytest.mark.parametrize("compute_discovery_fraction", [False, True])
def test_unsupervised_mask_iou_metric(use_threshold, compute_discovery_fraction):
    metric = metrics.UnsupervisedMaskIoUMetric(
        use_threshold=use_threshold,
        compute_discovery_fraction=compute_discovery_fraction,
        discovery_threshold=0.4,
    )

    # All clusters match
    pred = _make_one_hot([[0, 0, 0, 0], [2, 2, 1, 1]], 4)
    target = _make_one_hot([[0, 0, 0, 0], [0, 0, 1, 1]], 3)
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    # Partial matches
    metric.reset()
    pred = _make_one_hot([[0, 1, 2, 3], [0, 1, 2, 3]], 4)
    target = _make_one_hot([[0, 0, 0, 0], [0, 0, 1, 1]], 3)
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    if compute_discovery_fraction:
        # First element has IoU 0.25 (<=0.4, not discovered), second element has 2x IoU 0.5 (>0.4,
        # discovered), i.e. fraction discovered is 0 for first element, 1 for second.
        assert torch.allclose(metric.compute(), torch.tensor((0.0 + 1.0) / 2, dtype=torch.float64))
    else:
        assert torch.allclose(metric.compute(), torch.tensor((0.25 + 0.5) / 2, dtype=torch.float64))

    # Fully empty targets should be skipped in computation
    metric.reset()
    pred = _make_one_hot([[0, 0, 0, 0], [1, 1, 1, 1]], 4)
    target_empty = torch.zeros_like(pred)
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target_empty.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(0.0, dtype=torch.float64))

    # Partially empty targets should be skipped in computation
    metric.reset()
    pred = _make_one_hot([[0, 0, 0, 0], [1, 1, 2, 2]], 4)
    target_empty = torch.zeros_like(pred[0])
    target = torch.stack((target_empty, _make_one_hot([2, 2, 0, 0], 4)))
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    # Test ignore overlaps
    metric = metrics.UnsupervisedMaskIoUMetric(
        use_threshold=use_threshold,
        compute_discovery_fraction=compute_discovery_fraction,
        discovery_threshold=0.4,
        ignore_overlaps=True,
    )
    # All clusters at non-overlapping locations match
    pred = _make_one_hot([[0, 0, 0, 0], [2, 2, 1, 2]], 4)
    target = _make_one_hot([[0, 0, 0, 0], [0, 0, 1, 1]], 3)
    target[:, -1, :] = 1
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))

    # Test ignore mask with ignore overlaps
    metric = metrics.UnsupervisedMaskIoUMetric(
        use_threshold=use_threshold,
        compute_discovery_fraction=compute_discovery_fraction,
        discovery_threshold=0.4,
        ignore_overlaps=True,
    )
    # All clusters at non-ignored locations match
    pred = _make_one_hot([[0, 0, 0, 0], [2, 2, 2, 2]], 4)
    target = _make_one_hot([[0, 0, 1, 2], [0, 0, 1, 1]], 3)
    target[:, -1, :] = 1
    ignore = torch.zeros_like(target)[..., :1]
    ignore[:, -2:-1] = 1
    metric.update(
        pred.permute(0, 2, 1).unflatten(-1, (2, 2)),
        target.permute(0, 2, 1).unflatten(-1, (2, 2)),
        ignore.permute(0, 2, 1).unflatten(-1, (2, 2)),
    )
    assert torch.allclose(metric.compute(), torch.tensor(1.0, dtype=torch.float64))


def test_masks_to_bboxes(assert_tensors_equal):
    masks = _make_one_hot([0, 0, 0, 1], 2)
    assert_tensors_equal(
        ocl.metrics.utils.masks_to_bboxes(masks.transpose(0, 1).unflatten(-1, (2, 2))),
        torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.float32),
    )

    masks = _make_one_hot([0, 0, 0, 0], 2)
    assert_tensors_equal(
        ocl.metrics.utils.masks_to_bboxes(
            masks.transpose(0, 1).unflatten(-1, (2, 2)), empty_value=-2.0
        ),
        torch.tensor([[0, 0, 1, 1], [-2] * 4], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "matching,ignore_background,use_unmatched_as_background",
    [
        ("hungarian", False, False),
        ("hungarian", False, True),
        ("hungarian", True, False),
        ("majority", False, False),
        ("majority", True, False),
    ],
)
def test_dataset_semantic_mask_iou_metric(matching, ignore_background, use_unmatched_as_background):
    n_pred_classes, n_classes = 4, 3
    metric = metrics.DatasetSemanticMaskIoUMetric(
        n_predicted_classes=n_pred_classes,
        n_classes=n_classes,
        matching=matching,
        ignore_background=ignore_background,
        use_unmatched_as_background=use_unmatched_as_background,
    )

    conf_mat = [
        [5, 0, 0, 0],
        [0, 0, 0, 50],
        [0, 75, 0, 0],
        [0, 0, 0, 0],
        [5, 25, 0, 0],
    ]
    metric.confusion_mat += torch.tensor(conf_mat)
    n_pixels = int(
        metric.confusion_mat[:, 1:].sum() if ignore_background else metric.confusion_mat.sum()
    )

    if matching == "hungarian":
        expected_iou = [0.75, 0.0, 1.0]
        if use_unmatched_as_background:
            expected_acc = (10 + 50 + 75) / n_pixels
        elif ignore_background:
            expected_acc = (50 + 75) / n_pixels
        else:
            expected_acc = (5 + 50 + 75) / n_pixels
    elif matching == "majority":
        first = 1.0 if ignore_background else 100 / 105
        expected_iou = [first, 0.0, 1.0]
        expected_acc = (50 + 100) / n_pixels

    if ignore_background:
        expected_empty = [False, True, False]
    else:
        if use_unmatched_as_background:
            background = 10 / (25 + 5 + 5)
        else:
            if matching == "hungarian":
                background = 5 / (25 + 5 + 5)
            else:
                background = 0.0
        expected_iou = [background] + expected_iou
        expected_empty = [False, False, True, False]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iou, accuracy, empty_classes = metric.compute()
    assert torch.allclose(iou, torch.tensor(expected_iou, dtype=torch.float64))
    assert torch.allclose(accuracy, torch.tensor(expected_acc, dtype=torch.float64))
    assert torch.allclose(empty_classes, torch.tensor(expected_empty, dtype=torch.bool))


def _make_one_hot(values, num_classes):
    return F.one_hot(torch.tensor(np.array(values), dtype=torch.long), num_classes=num_classes)
