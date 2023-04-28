"""Metrics related to tracking."""
import io

import motmetrics as mm
import numpy as np
import pandas as pd
import torch
import torchmetrics

from ocl.metrics.utils import masks_to_bboxes


class MOTMetric(torchmetrics.Metric):
    """Multiple object tracking metric."""

    def __init__(
        self,
        target_is_mask: bool = True,
        use_threshold: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize MOTMetric.

        Args:
            target_is_mask: Is the metrics evaluated on masks
            use_threshold: Use threshold to binarize predicted mask
            threshold: Threshold value

        """
        super().__init__()
        self.target_is_mask = target_is_mask
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.reset_accumulator()
        self.accuracy = []

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        # Merge batch and frame dimensions
        B, F = prediction.shape[:2]
        prediction = prediction.flatten(0, 1)
        target = target.flatten(0, 1)

        n_pred_classes = prediction.shape[1]
        n_gt_classes = target.shape[1]

        if self.use_threshold:
            prediction = prediction > self.threshold
        else:
            indices = torch.argmax(prediction, dim=1)
            prediction = torch.nn.functional.one_hot(indices, num_classes=n_pred_classes)
            prediction = prediction.permute(0, 3, 1, 2)

        pred_bboxes = masks_to_bboxes(prediction.flatten(0, 1)).unflatten(0, (B, F, n_pred_classes))
        if self.target_is_mask:
            target_bboxes = masks_to_bboxes(target.flatten(0, 1)).unflatten(0, (B, F, n_gt_classes))
        else:
            assert target.shape[-1] == 4
            # Convert all-zero boxes added during padding to invalid boxes
            target[torch.all(target == 0.0, dim=-1)] = -1.0
            target_bboxes = target

        self.reset_accumulator()
        for preds, targets in zip(pred_bboxes, target_bboxes):
            # seq evaluation
            self.reset_accumulator()
            for pred, target, mask in zip(preds, targets, prediction):
                valid_track_box = pred[:, 0] != -1.0
                valid_target_box = target[:, 0] != -1.0

                track_id = valid_track_box.nonzero()[:, 0].detach().cpu().numpy()
                target_id = valid_target_box.nonzero()[:, 0].detach().cpu().numpy()

                # move background
                idx = track_id.tolist()
                for id in idx:
                    h, w = mask[id].shape
                    thres = h * w * 0.25
                    if pred[id][2] * pred[id][3] >= thres:
                        idx.remove(id)
                cur_obj_idx = np.array(idx)

                if valid_target_box.sum() == 0:
                    continue  # Skip data points without any target bbox

                pred = pred[cur_obj_idx].detach().cpu().numpy()
                target = target[valid_target_box].detach().cpu().numpy()
                # frame evaluation
                self.eval_frame(pred, target, cur_obj_idx, target_id)
            self.accuracy.append(self.acc)

        self.total += 1

    def eval_frame(self, trk_tlwhs, tgt_tlwhs, trk_ids, tgt_ids):
        # get distance matrix
        trk_tlwhs = np.copy(trk_tlwhs)
        tgt_tlwhs = np.copy(tgt_tlwhs)
        trk_ids = np.copy(trk_ids)
        tgt_ids = np.copy(tgt_ids)
        iou_distance = mm.distances.iou_matrix(tgt_tlwhs, trk_tlwhs, max_iou=0.5)
        # acc
        self.acc.update(tgt_ids, trk_ids, iou_distance)

    def convert_motmetric_to_value(self, res):
        dp = res.replace(" ", ";").replace(";;", ";").replace(";;", ";").replace(";;", ";")
        tmp = list(dp)
        tmp[0] = "-"
        dp = "".join(tmp)
        return io.StringIO(dp)

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            metrics = mm.metrics.motchallenge_metrics
            mh = mm.metrics.create()
            summary = mh.compute_many(
                self.accuracy, metrics=metrics, names=None, generate_overall=True
            )
            strsummary = mm.io.render_summary(
                summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
            )
            res = self.convert_motmetric_to_value(strsummary)
            df = pd.read_csv(res, sep=";", engine="python")

            mota = df.iloc[-1]["MOTA"]
            self.values = torch.tensor(float(mota[:-1]), dtype=torch.float64).to(self.values.device)
            self.reset_accumulator()
            self.accuracy = []
            return self.values
