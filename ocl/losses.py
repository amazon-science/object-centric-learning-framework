from math import log
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou

import ocl.typing
from ocl.matching import CPUHungarianMatcher
from ocl.utils.bboxes import box_cxcywh_to_xyxy


class ReconstructionLoss(nn.Module):
    """Simple reconstruction loss."""

    def __init__(
        self,
        loss_type: str,
        weight: float = 1.0,
        normalize_target: bool = False,
    ):
        """Initialize ReconstructionLoss.

        Args:
            loss_type: One of `mse`, `mse_sum`, `l1`, `cosine_loss`, `cross_entropy_sum`.
            weight: Weight of loss, output is multiplied with this value.
            normalize_target: Normalize target using mean and std of last dimension
                prior to computing output.
        """
        super().__init__()
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "mse_sum":
            # Used for slot_attention and video slot attention.
            self.loss_fn = (
                lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
            )
        elif loss_type == "l1":
            self.loss_name = "l1_loss"
            self.loss_fn = nn.functional.l1_loss
        elif loss_type == "cosine":
            self.loss_name = "cosine_loss"
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        elif loss_type == "cross_entropy_sum":
            # Used for SLATE, average is over the first (batch) dim only.
            self.loss_name = "cross_entropy_sum_loss"
            self.loss_fn = (
                lambda x1, x2: nn.functional.cross_entropy(
                    x1.reshape(-1, x1.shape[-1]), x2.reshape(-1, x2.shape[-1]), reduction="sum"
                )
                / x1.shape[0]
            )
        else:
            raise ValueError(
                f"Unknown loss {loss_type}. Valid choices are (mse, l1, cosine, cross_entropy)."
            )
        # If weight is callable use it to determine scheduling otherwise use constant value.
        self.weight = weight
        self.normalize_target = normalize_target

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        """Compute reconstruction loss.

        Args:
            input: Prediction / input tensor.
            target: Target tensor.

        Returns:
            The reconstruction loss.
        """
        target = target.detach()
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = self.loss_fn(input, target)
        return self.weight * loss


class LatentDupplicateSuppressionLoss(nn.Module):
    """Latent Dupplicate Suppression Loss.

    Inspired by: Li et al, Duplicate latent representation suppression
      for multi-object variational autoencoders, BMVC 2021
    """

    def __init__(
        self,
        weight: float,
        eps: float = 1e-08,
    ):
        """Initialize LatentDupplicateSuppressionLoss.

        Args:
            weight: Weight of loss, output is multiplied with this value.
            eps: Small value to avoid division by zero in cosine similarity computation.
        """
        super().__init__()
        self.weight = weight
        self.similarity = nn.CosineSimilarity(dim=-1, eps=eps)

    def forward(self, grouping: ocl.typing.PerceptualGroupingOutput) -> float:
        """Compute latent dupplicate suppression loss.

        This also takes into account the `is_empty` tensor of
        [ocl.typing.PerceptualGroupingOutput][].

        Args:
            grouping: Grouping to use for loss computation.

        Returns:
            The weighted loss.
        """
        if grouping.objects.dim() == 4:
            # Build large tensor of reconstructed video.
            objects = grouping.objects
            bs, n_frames, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, :, off_diag_indices[0], :], objects[:, :, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            return self.weight * losses.sum() / (bs * n_frames)
        elif grouping.objects.dim() == 3:
            # Build large tensor of reconstructed image.
            objects = grouping.objects
            bs, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, off_diag_indices[0], :], objects[:, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            return self.weight * losses.sum() / bs
        else:
            raise ValueError("Incompatible input format.")


def _focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mean_in_dim1=True
):
    """Loss used in RetinaNet for dense detection. # noqa: D411.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if mean_in_dim1:
        return loss.mean(1).sum() / num_boxes
    else:
        return loss.sum() / num_boxes


def _compute_detr_cost_matrix(
    outputs,
    targets,
    use_focal=True,
    class_weight: float = 1,
    bbox_weight: float = 1,
    giou_weight: float = 1,
):
    """Compute cost matrix between outputs instances and target instances.

    Params:
        outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes]
                            with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the
                            predicted box coordinates

        targets: a list of targets (len(targets) = batch_size), where each target is a instance:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                        ground-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

    Returns:
        costMatrix: A iter of tensors of size [num_outputs, num_targets].
    """
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1)
        else:
            AssertionError("only support focal for now.")
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if use_focal:
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(  # noqa: F821
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)  # noqa: F821
        )

        # Final cost matrix
        C = bbox_weight * cost_bbox + class_weight * cost_class + giou_weight * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        return C.split(sizes, -1)


class CLIPLoss(nn.Module):
    """Contrastive CLIP loss.

    Reference:
        Radford et al.,
        Learning transferable visual models from natural language supervision,
        ICML 2021
    """

    def __init__(
        self,
        normalize_inputs: bool = True,
        learn_scale: bool = True,
        max_temperature: Optional[float] = None,
    ):
        """Initiailize CLIP loss.

        Args:
            normalize_inputs: Normalize both inputs based on mean and variance.
            learn_scale: Learn scaling factor of dot product.
            max_temperature: Maximum temperature of scaling.
        """
        super().__init__()
        self.normalize_inputs = normalize_inputs
        if learn_scale:
            self.logit_scale = nn.Parameter(torch.zeros([]) * log(1 / 0.07))  # Same init as CLIP.
        else:
            self.register_buffer("logit_scale", torch.zeros([]))  # exp(0) = 1, i.e. no scaling.
        self.max_temperature = max_temperature

    def forward(
        self,
        first: ocl.typing.PooledFeatures,
        second: ocl.typing.PooledFeatures,
        model: Optional[pl.LightningModule] = None,
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Compute CLIP loss.

        Args:
            first: First tensor.
            second: Second tensor.
            model: Pytorch lighting model. This is needed in order to perform
                multi-gpu / multi-node communication independent of the backend.

        Returns:
            - Computed loss
            - Dict with keys `similarity` (containing local similarities)
                and `temperature` (containing the current temperature).
        """
        # Collect all representations.
        if self.normalize_inputs:
            first = first / first.norm(dim=-1, keepdim=True)
            second = second / second.norm(dim=-1, keepdim=True)

        temperature = self.logit_scale.exp()
        if self.max_temperature:
            temperature = torch.clamp_max(temperature, self.max_temperature)

        if model is not None and hasattr(model, "trainer") and model.trainer.world_size > 1:
            # Running on multiple GPUs.
            global_rank = model.global_rank
            all_first_rep, all_second_rep = model.all_gather([first, second], sync_grads=True)
            world_size, batch_size = all_first_rep.shape[:2]
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=first.device)
                + batch_size * global_rank
            )
            # Flatten the GPU dim into batch.
            all_first_rep = all_first_rep.flatten(0, 1)
            all_second_rep = all_second_rep.flatten(0, 1)

            # Compute inner product for instances on the current GPU.
            logits_per_first = temperature * first @ all_second_rep.t()
            logits_per_second = temperature * second @ all_first_rep.t()

            # For visualization purposes, return the cosine similarities on the local batch.
            similarities = (
                1
                / temperature
                * logits_per_first[:, batch_size * global_rank : batch_size * (global_rank + 1)]
            )
            # shape = [local_batch_size, global_batch_size]
        else:
            batch_size = first.shape[0]
            labels = torch.arange(batch_size, dtype=torch.long, device=first.device)
            # When running with only a single GPU we can save some compute time by reusing
            # computations.
            logits_per_first = temperature * first @ second.t()
            logits_per_second = logits_per_first.t()
            similarities = 1 / temperature * logits_per_first

        return (
            (F.cross_entropy(logits_per_first, labels) + F.cross_entropy(logits_per_second, labels))
            / 2,
            {"similarities": similarities, "temperature": temperature},
        )


def _compute_detr_seg_const_matrix(
    predicts,
    targets,
):
    """Compute cost matrix between outputs instances and target instances.

    Returns:
        costMatrix: A iter of tensors of size [num_outputs, num_targets].
    """
    # filter out valid targets
    npr, h, w = predicts.shape
    nt = targets.shape[0]

    predicts = repeat(predicts, "npr h w -> (npr repeat) h w", repeat=nt)
    targets = repeat(targets, "nt h w -> (repeat nt) h w", repeat=npr)

    cost = F.binary_cross_entropy(predicts, targets.float(), reduction="none").mean(-1).mean(-1)
    cost = rearrange(cost, "(npr nt) -> npr nt", npr=npr, nt=nt)
    return cost


class DETRSegLoss(nn.Module):
    """DETR inspired loss for segmentation.

    This loss computes a hungarian matching of segmentation masks between a prediction and
    a target.  The loss is then a linear combination of the CE loss between matched masks
    and a foreground prediction classification.

    Reference:
        Carion et al., End-to-End Object Detection with Transformers, ECCV 2020
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        ignore_background: bool = True,
        foreground_weight: float = 1.0,
        foreground_matching_weight: float = 1.0,
        global_loss: bool = True,
    ):
        """Initialize DETRSegLoss.

        Args:
            loss_weight: Loss weight
            ignore_background: Ignore background masks.
            foreground_weight: Contribution weight of foreground classification loss.
            foreground_matching_weight: Contribution weight of foreground classification
                to matching.
            global_loss: Use average loss over all instances of all gpus.  This is
                particularly useful when training with sparse labels.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_background = ignore_background
        self.foreground_weight = foreground_weight
        self.foreground_matching_weight = foreground_matching_weight
        self.global_loss = global_loss
        self.matcher = CPUHungarianMatcher()

    def forward(
        self,
        input_mask: ocl.typing.ObjectFeatureAttributions,
        target_mask: ocl.typing.ObjectFeatureAttributions,
        foreground_logits: Optional[torch.Tensor] = None,
        model: Optional[pl.LightningModule] = None,
    ) -> float:
        """Compute DETR segmentation loss.

        Args:
            input_mask: Input/predicted masks
            target_mask: Target masks
            foreground_logits: Forground prediction logits
            model: Pytorch lighting model. This is needed in order to perform
                multi-gpu / multi-node communication independent of the backend.

        Returns:
            The computed loss.
        """
        target_mask = target_mask.detach() > 0
        device = target_mask.device

        # A nan mask is not considered.
        valid_targets = ~(target_mask.isnan().all(-1).all(-1)).any(-1)
        # Discard first dimension mask as it is background.
        if self.ignore_background:
            # Assume first class in masks is background.
            if len(target_mask.shape) > 4:  # Video data (bs, frame, classes, w, h).
                target_mask = target_mask[:, :, 1:]
            else:  # Image data (bs, classes, w, h).
                target_mask = target_mask[:, 1:]

        targets = target_mask[valid_targets]
        predictions = input_mask[valid_targets]
        if foreground_logits is not None:
            foreground_logits = foreground_logits[valid_targets]

        total_loss = torch.tensor(0.0, device=device)
        num_samples = 0

        # Iterate through each clip. Might think about if parallelable
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            # Filter empty masks.
            target = target[target.sum(-1).sum(-1) > 0]

            # Compute matching.
            costMatrixSeg = _compute_detr_seg_const_matrix(
                prediction,
                target,
            )
            # We cannot rely on the matched cost for computing the loss due to
            # normalization issues between segmentation component (normalized by
            # number of matches) and classification component (normalized by
            # number of predictions). Thus compute both components separately
            # after deriving the matching matrix.
            if foreground_logits is not None and self.foreground_matching_weight != 0.0:
                # Positive classification component.
                logits = foreground_logits[i]
                costMatrixTotal = (
                    costMatrixSeg
                    + self.foreground_weight
                    * F.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits), reduction="none"
                    ).detach()
                )
            else:
                costMatrixTotal = costMatrixSeg

            # Matcher takes a batch but we are doing this one by one.
            matching_matrix = self.matcher(costMatrixTotal.unsqueeze(0))[0].squeeze(0)
            n_matches = min(predictions.shape[0], target.shape[0])
            if n_matches > 0:
                instance_cost = (costMatrixSeg * matching_matrix).sum(-1).sum(-1) / n_matches
            else:
                instance_cost = torch.tensor(0.0, device=device)

            if foreground_logits is not None:
                ismatched = (matching_matrix > 0).any(-1)
                logits = foreground_logits[i].squeeze(-1)
                instance_cost += self.foreground_weight * F.binary_cross_entropy_with_logits(
                    logits, ismatched.float(), reduction="mean"
                )

            total_loss += instance_cost
            # Normalize by number of matches.
            num_samples += 1

        if (
            model is not None
            and hasattr(model, "trainer")
            and model.trainer.world_size > 1
            and self.global_loss
        ):
            # As data is sparsely labeled return the average loss over all GPUs.
            # This should make the loss a mit more smooth.
            all_losses, sample_counts = model.all_gather([total_loss, num_samples], sync_grads=True)
            total_count = sample_counts.sum()
            if total_count > 0:
                total_loss = all_losses.sum() / total_count
            else:
                total_loss = torch.tensor(0.0, device=device)

            return total_loss * self.loss_weight
        else:
            if num_samples == 0:
                # Avoid division by zero if a batch does not contain any labels.
                return torch.tensor(0.0, device=targets.device)

            total_loss /= num_samples
            total_loss *= self.loss_weight
            return total_loss


class EM_rec_loss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 20,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="none")

    def forward(
        self,
        segmentations: torch.Tensor,  # rollout_decode.masks
        masks: torch.Tensor,  # decoder.masks
        reconstructions: torch.Tensor,
        rec_tgt: torch.Tensor,
        masks_vis: torch.Tensor,
        attn_index: torch.Tensor,
    ):
        b, f, c, h, w = segmentations.shape
        _, _, n_slots, n_buffer = attn_index.shape

        segmentations = (
            segmentations.reshape(-1, n_buffer, h, w).unsqueeze(1).repeat(1, n_slots, 1, 1, 1)
        )
        masks = masks.reshape(-1, n_slots, h, w).unsqueeze(2).repeat(1, 1, n_buffer, 1, 1)
        masks = masks > 0.5
        masks_vis = (
            masks_vis.reshape(-1, n_slots, h, w)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, n_buffer, 3, 1, 1)
        )
        masks_vis = masks_vis > 0.5
        attn_index = attn_index.reshape(-1, n_slots, n_buffer)
        rec_tgt = (
            rec_tgt.reshape(-1, 3, h, w)
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, n_slots, n_buffer, 1, 1, 1)
        )
        reconstructions = (
            reconstructions.reshape(-1, n_buffer, 3, h, w)
            .unsqueeze(1)
            .repeat(1, n_slots, 1, 1, 1, 1)
        )
        rec_pred = reconstructions * masks_vis
        rec_tgt_ = rec_tgt * masks_vis
        loss = torch.sum(
            F.binary_cross_entropy(segmentations, masks.float(), reduction="none"), (-1, -2)
        ) / (h * w) + 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3, -2, -1))
        total_loss = torch.sum(attn_index * loss, (0, 1, 2)) / (b * f * n_slots * n_buffer)
        return (total_loss) * self.loss_weight
