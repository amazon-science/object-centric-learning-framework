#!/usr/bin/env python
"""Evaluate a trained model for object discovery by clustering object representations.

Given a set of images, each with a set of ground truth masks and a set of object masks and
representations, we perform the following steps:
    1) Assign each object a cluster id by clustering the corresponding representations over all
    images.
    2) Merge object masks with the same cluster id on the same image to form a semantic mask.
    3) Compute IoU between masks of predicted clusters and ground truth classes over all images.
    4) Assign clusters to classes based on the IoU and a matching strategy.
"""
import dataclasses
import enum
import json
import logging
import os
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional

import hydra
import hydra_zen
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
import tqdm

from ocl import feature_extractors, metrics
from ocl.cli import cli_utils, eval_utils, train

logger = logging.getLogger("eval_cluster_metrics")


class RepresentationType(enum.Enum):
    NONE = enum.auto()
    SLOTS = enum.auto()
    MASK_WEIGHTED_FEATURES = enum.auto()
    CLS_ON_MASKED_INPUT = enum.auto()


# --8<-- [start:EvaluationClusteringConfig]
@dataclasses.dataclass
class EvaluationClusteringConfig:
    """Configuration for evaluation."""

    # Path to training configuration file or configuration dir. If dir, train_config_name
    # needs to be set as well.
    train_config_path: str

    # Number of classes. Note that on COCO, this should be one larger than the maximum class ID that
    # can appear, which does not correspond to the real number of classes.
    n_classes: int
    # Clustering methods to get cluster ID per object by clustering representations
    # This only supports clustering metrics.
    clusterings: Optional[Dict[str, Any]] = None
    # Paths for model outputs to get cluster ID per object
    model_clusterings: Optional[Dict[str, str]] = None

    train_config_overrides: Optional[List[str]] = None
    train_config_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None
    report_filename: str = "clustering_metrics.json"

    batch_size: int = 25
    class_discovery_threshold: float = 0.02
    use_mask_threshold: bool = False
    mask_threshold: float = 0.5
    ignore_background: bool = False
    use_unmatched_as_background: bool = False
    use_ignore_masks: bool = False
    n_min_mask_pixels: int = 1  # Minimum number of pixels a mask must occupy to be considered valid
    n_min_max_mask_values: float = 1e-4  # Mask must have at least one value above threshold

    # Type of representation to use for clustering.
    representation_type: RepresentationType = RepresentationType.SLOTS

    # Setting this allows to add modules to the model that are executed during evaluation
    modules: Optional[Dict[str, Any]] = None
    # Setting this allows to evaluate on a different dataset than the model was trained on
    dataset: Optional[Any] = None

    # Path to slot representations
    slots_path: str = "perceptual_grouping.objects"
    # Path to feature representations
    features_path: str = "feature_extractor.features"
    # Path to slot masks, image shaped
    masks_path: str = "object_decoder.masks_as_image"
    # Path to slot masks, but flattened to match the size of features
    masks_flat_path: str = "object_decoder.masks"
    # Path to reference masks
    target_masks_path: str = "input.segmentation_mask"
    # Path to ignore masks
    ignore_masks_path: str = "input.ignore_mask"
    # Path under which representation to cluster is stored
    cluster_representation_path: str = "representation"
    # Path under which empty slot mask is stored
    empty_slots_path: str = "empty_slots"

    class_name_by_category_id: Optional[Dict[int, str]] = None


# --8<-- [end:EvaluationClusteringConfig]


@dataclasses.dataclass
class Results:
    iou_per_class: np.ndarray
    accuracy: np.ndarray
    empty_classes: np.ndarray
    n_classes: int
    n_clusters: int
    has_background: bool
    matching: str
    clustering: Optional[Any] = None
    model_clustering: Optional[str] = None

    def mean_iou(self) -> float:
        return np.mean(self.iou_per_class[~self.empty_classes])

    def num_discovered(self, threshold) -> int:
        iou_non_empty = self.iou_per_class[~self.empty_classes]
        return len(iou_non_empty[iou_non_empty > threshold])

    def mean_iou_discovered(self, threshold) -> float:
        iou_non_empty = self.iou_per_class[~self.empty_classes]
        iou_discovered = iou_non_empty[iou_non_empty > threshold]
        if len(iou_discovered) > 0:
            return np.mean(iou_discovered)
        else:
            return 0.0

    def mean_iou_without_bg(self) -> float:
        if self.has_background:
            return np.mean(self.iou_per_class[1:][~self.empty_classes[1:]])
        else:
            return self.mean_iou()


hydra.core.config_store.ConfigStore.instance().store(
    name="evaluation_clustering_config",
    node=EvaluationClusteringConfig,
)


def get_transform_for_representation_type(
    representation_type: RepresentationType, config: EvaluationClusteringConfig, model
):
    def cast_masks(mask: torch.Tensor):
        # Transform float masks to uint8 masks to save memory
        if config.use_mask_threshold:
            mask = mask > config.mask_threshold
        else:
            maximum, indices = torch.max(mask, dim=1)
            mask = torch.nn.functional.one_hot(indices, num_classes=mask.shape[1])
            mask[:, :, :, 0][maximum == 0.0] = 0
            mask = mask.transpose(-1, -2).transpose(-2, -3)

        return mask.to(torch.uint8)

    def filter_empty_masks(masks: torch.Tensor):
        # Count how many pixels each mask occupies
        max_per_class = masks.max(dim=2).values
        max_per_pixel = masks.max(dim=1, keepdim=True).values
        n_maxs = torch.sum((max_per_pixel == masks) & (max_per_pixel > 0), dim=-1)
        return (n_maxs < config.n_min_mask_pixels) | (max_per_class < config.n_min_max_mask_values)

    if representation_type == RepresentationType.MASK_WEIGHTED_FEATURES:
        required_paths = [config.masks_path, config.features_path, config.masks_flat_path]

        def transform(data):
            data[config.masks_path] = cast_masks(data[config.masks_path])

            # Compute a weighted mean of the features using the slot masks as weights.
            features = data[config.features_path]  # batch, [frames,] n_tokens, n_features
            masks = data[config.masks_flat_path]  # batch, [frames,] n_slots, n_tokens

            if features.ndim == 4:
                features = features.flatten(0, 1)
                masks = masks.flatten(0, 1)

            objects = torch.einsum("btd, bst -> bsd", features, masks)
            data[config.cluster_representation_path] = objects
            data[config.empty_slots_path] = filter_empty_masks(masks)
            return data

    elif representation_type == RepresentationType.SLOTS:
        required_paths = [config.masks_path, config.slots_path, config.masks_flat_path]

        def transform(data):
            data[config.masks_path] = cast_masks(data[config.masks_path])

            objects = data[config.slots_path]
            masks = data[config.masks_flat_path]
            if objects.ndim == 4:
                objects = objects.flatten(0, 1)
                masks = masks.flatten(0, 1)

            data[config.cluster_representation_path] = objects
            data[config.empty_slots_path] = filter_empty_masks(masks)
            return data

    elif representation_type == RepresentationType.CLS_ON_MASKED_INPUT:
        required_paths = [
            "input.image",
            config.masks_path,
            config.masks_flat_path,
            "object_decoder.masks_as_image",
        ]
        # TODO: Make this more flexible.
        feature_extractor = None
        for module in model.modules():
            if isinstance(module, feature_extractors.TimmFeatureExtractor):
                feature_extractor = module

        if feature_extractor is None:
            raise RuntimeError("Unable to find TimmFeatureExtractor in model.")

        def transform(data):
            data[config.masks_path] = cast_masks(data[config.masks_path])

            # Mask out part of the input image.
            input_image = data["input.image"]  # batch, C, H, W
            masks = cast_masks(data["object_decoder.masks_as_image"])

            masked_input = masks.unsqueeze(2).to(dtype=input_image.dtype) * input_image.unsqueeze(1)
            bs, n_objects = masked_input.shape[:2]
            masked_input = masked_input.flatten(0, 1)  # Merge object dim into batch dim.

            feature_extractor.eval()
            with torch.no_grad():
                aux_features = feature_extractor.forward_images(masked_input)[2]
            cls_tokens = aux_features[config.features_path].unflatten(0, (bs, n_objects))

            data[config.cluster_representation_path] = cls_tokens
            data[config.empty_slots_path] = filter_empty_masks(data[config.masks_flat_path])
            return data

    elif representation_type == RepresentationType.NONE:
        required_paths = [config.masks_path, config.masks_flat_path]

        def transform(data):
            data[config.masks_path] = cast_masks(data[config.masks_path])
            masks = data[config.masks_flat_path]
            if masks.ndim == 4:
                masks = masks.flatten(0, 1)

            data[config.empty_slots_path] = filter_empty_masks(masks)

    else:
        raise ValueError(f"Unknown representation type {representation_type}.")

    return transform, required_paths


def get_data_from_model(model, datamodule, input_paths, output_paths, transform=None):
    data_extractor = eval_utils.ExtractDataFromPredictions(input_paths, output_paths, transform)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        callbacks=[data_extractor],
        logger=False,
    )
    trainer.predict(model, datamodule.val_dataloader(), return_predictions=False)
    outputs = data_extractor.outputs

    return outputs


def get_cluster_ids_from_clustering(outputs, clustering, cluster_repr_path, empty_slots_path):
    objects = torch.cat(outputs[cluster_repr_path])
    objects_flat = objects.flatten(0, 1)

    if outputs[empty_slots_path] is not None:
        nonempty_objects = ~torch.cat(outputs[empty_slots_path]).flatten(0, 1).to(torch.bool)
    else:
        nonempty_objects = torch.ones_like(objects_flat[:, 0]).to(torch.bool)

    objects_selected = objects_flat[nonempty_objects]
    if len(objects_selected) > 0:
        object_ids_selected = clustering.fit_predict(objects_selected)
        object_ids = torch.zeros_like(objects_flat[:, 0], dtype=torch.int64)
        object_ids[nonempty_objects] = object_ids_selected + 1
    else:
        object_ids = torch.zeros_like(objects_flat[:, 0], dtype=torch.int64)

    object_ids = object_ids.unflatten(0, objects.shape[:2])
    object_ids = object_ids.split([len(obj) for obj in outputs[cluster_repr_path]])
    return object_ids


def get_cluster_ids_from_outputs(outputs, ids_path):
    return outputs[ids_path]


def report_from_results(results: Dict[str, Results], config: EvaluationClusteringConfig):
    output = OrderedDict()
    output["train_config_path"] = config.train_config_path
    output["checkpoint_path"] = config.checkpoint_path
    if config.train_config_overrides is not None:
        output["train_config_overrides"] = [str(s) for s in config.train_config_overrides]
    output["settings"] = OrderedDict(
        use_mask_threshold=config.use_mask_threshold,
        mask_threshold=config.mask_threshold,
        representation_type=config.representation_type.name,
        ignore_background=config.ignore_background,
        use_unmatched_as_background=config.use_unmatched_as_background,
    )

    for method in sorted(results):
        result = results[method]
        method_outputs = OrderedDict()
        settings = OrderedDict(
            n_classes=result.n_classes,
            n_clusters=result.n_clusters,
            matching=result.matching,
        )
        if result.clustering is not None:
            clustering = result.clustering
            settings["clustering"] = (clustering.method,)
            if clustering.clustering_kwargs is not None:
                clustering_kwargs = clustering.clustering_kwargs
                if omegaconf.OmegaConf.is_config(clustering_kwargs):
                    clustering_kwargs = omegaconf.OmegaConf.to_container(clustering_kwargs)
                settings["clustering_kwargs"] = clustering_kwargs
            if clustering.use_l2_normalization:
                settings["clustering_use_l2_normalization"] = True
            if clustering.get("use_pca", False):
                settings["clustering_use_pca"] = True
                settings["clustering_pca_dims"] = clustering.pca_dimensions
                if clustering.pca_kwargs is not None:
                    pca_kwargs = clustering.pca_kwargs
                    if omegaconf.OmegaConf.is_config(pca_kwargs):
                        pca_kwargs = omegaconf.OmegaConf.to_container(pca_kwargs)
                    settings["clustering_pca_kwargs"] = pca_kwargs
        elif result.model_clustering is not None:
            settings["model_clustering"] = result.model_clustering

        method_outputs["settings"] = settings
        method_outputs["iou_mean"] = result.mean_iou()
        method_outputs["iou_without_background"] = result.mean_iou_without_bg()
        method_outputs["accuracy"] = float(result.accuracy)
        method_outputs["num_discovered"] = result.num_discovered(config.class_discovery_threshold)
        method_outputs["iou_discovered"] = result.mean_iou_discovered(
            config.class_discovery_threshold
        )

        start_id = 1 if config.ignore_background else 0
        per_class_iou = result.iou_per_class
        class_name_by_category_id = config.class_name_by_category_id
        if class_name_by_category_id is None:
            class_name_by_category_id = {i: f"{i}" for i in range(start_id, config.n_classes + 1)}

        named_per_class_iou = OrderedDict()
        for category_id, class_name in class_name_by_category_id.items():
            if (config.ignore_background and category_id == 0) or category_id > config.n_classes:
                continue
            named_per_class_iou[class_name] = per_class_iou[category_id - start_id]

        method_outputs["iou_per_class"] = named_per_class_iou
        output[method] = method_outputs

    return output


def generate_and_save_report(
    config: EvaluationClusteringConfig, results_by_method: Dict[str, Results]
):
    report = report_from_results(results_by_method, config)
    metrics_file = os.path.join(config.output_dir, config.report_filename)
    with open(metrics_file, "w") as f:
        json.dump(report, f, indent=2)

    return metrics_file


@hydra.main(
    config_name="evaluation_clustering_config", config_path="../../configs", version_base="1.1"
)
def evaluate(config: EvaluationClusteringConfig):
    config.train_config_path = hydra.utils.to_absolute_path(config.train_config_path)
    if config.train_config_path.endswith(".yaml"):
        config_dir, config_name = os.path.split(config.train_config_path)
    else:
        config_dir, config_name = config.train_config_path, config.train_config_name

    if not os.path.exists(config_dir):
        raise ValueError(f"Inferred config dir at {config_dir} does not exist.")

    if config.checkpoint_path is None:
        try:
            run_dir = os.path.dirname(config_dir)
            checkpoint_path = cli_utils.find_checkpoint(run_dir)
            config.checkpoint_path = checkpoint_path
            logger.info(f"Automatically derived checkpoint path: {checkpoint_path}")
        except (TypeError, IndexError):
            raise ValueError(
                "Unable to automatically derive checkpoint from command line provided config file "
                "path. You can manually specify a checkpoint using the `checkpoint_path` argument."
            )
    else:
        config.checkpoint_path = hydra.utils.to_absolute_path(config.checkpoint_path)
        if not os.path.exists(config.checkpoint_path):
            raise ValueError(f"Checkpoint at {config.checkpoint_path} does not exist.")

    if config.output_dir is None:
        config.output_dir = run_dir
        if not os.path.exists(config.output_dir):
            os.mkdir(config.output_dir)
        logger.info(f"Using {config.output_dir} as output directory.")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=config_dir):
        overrides = config.train_config_overrides if config.train_config_overrides else []
        train_config = hydra.compose(os.path.splitext(config_name)[0], overrides=overrides)
        train_config.dataset.batch_size = config.batch_size

        datamodule, model = eval_utils.build_from_train_config(train_config, config.checkpoint_path)

    if config.modules is not None:
        modules = hydra_zen.instantiate(config.modules, _convert_="all")
        for key, module in modules.items():
            model.models[key] = module

    if config.dataset is not None:
        datamodule = train.build_and_register_datamodule_from_config(
            config,
            batch_size=train_config.dataset.batch_size,
            eval_batch_size=config.batch_size,
        )

    config.clusterings = config.clusterings if config.clusterings else {}
    config.model_clusterings = config.model_clusterings if config.model_clusterings else {}

    clusterings = hydra_zen.instantiate(config.clusterings)

    transform, transform_paths = get_transform_for_representation_type(
        config.representation_type, config, model
    )

    model_clustering_paths = list(config.model_clusterings.values())
    pretransform_paths = (
        model_clustering_paths + transform_paths + [config.masks_path, config.target_masks_path]
    )
    output_paths = model_clustering_paths + [
        config.masks_path,
        config.target_masks_path,
        config.empty_slots_path,
    ]
    if config.representation_type != RepresentationType.NONE:
        output_paths.append(config.cluster_representation_path)
    if config.use_ignore_masks:
        pretransform_paths.append(config.ignore_masks_path)
        output_paths.append(config.ignore_masks_path)

    outputs = get_data_from_model(model, datamodule, pretransform_paths, output_paths, transform)

    if not config.use_ignore_masks:
        outputs[config.ignore_masks_path] = [None] * len(outputs[config.target_masks_path])

    cluster_fns = {
        name: partial(
            get_cluster_ids_from_clustering,
            clustering=clustering,
            cluster_repr_path=config.cluster_representation_path,
            empty_slots_path=config.empty_slots_path,
        )
        for name, clustering in clusterings.items()
    }
    cluster_fns.update(
        {
            name: partial(get_cluster_ids_from_outputs, ids_path=path)
            for name, path in config.model_clusterings.items()
        }
    )

    n_clusters_per_clustering = {
        name: clustering.n_clusters for name, clustering in clusterings.items()
    }
    n_clusters_per_clustering.update({name: None for name in config.model_clusterings})

    results_by_method = {}
    for clustering_name, cluster_fn in cluster_fns.items():
        logging.info(f"Fitting clustering {clustering_name}.")
        object_ids = cluster_fn(outputs)

        if n_clusters_per_clustering[clustering_name] is not None:
            n_clusters = n_clusters_per_clustering[clustering_name]
        else:
            # Get number of clusters from data: use maximum cluster ID
            n_clusters = max(int(ids.max()) for ids in object_ids)

        logging.info("Computing IoU over dataset.")
        metric_hung = metrics.DatasetSemanticMaskIoUMetric(
            n_predicted_classes=n_clusters,
            n_classes=config.n_classes,
            matching="hungarian",
            use_threshold=config.use_mask_threshold,
            threshold=config.mask_threshold,
            ignore_background=config.ignore_background,
            use_unmatched_as_background=config.use_unmatched_as_background,
        )
        for predictions, targets, ids, ignore in tqdm.tqdm(
            zip(
                outputs[config.masks_path],
                outputs[config.target_masks_path],
                object_ids,
                outputs[config.ignore_masks_path],
            ),
            total=len(object_ids),
        ):
            metric_hung.update(predictions, targets, ids, ignore)

        per_class_iou, accuracy, empty_classes = metric_hung.compute()
        results = Results(
            per_class_iou.cpu().numpy(),
            accuracy.cpu().numpy(),
            empty_classes.cpu().numpy(),
            n_classes=config.n_classes,
            n_clusters=n_clusters,
            has_background=not config.ignore_background,
            clustering=config.clusterings.get(clustering_name),
            model_clustering=config.model_clusterings.get(clustering_name),
            matching="hungarian",
        )
        results_by_method[f"{clustering_name}_hungarian"] = results
        logging.info(
            f"{clustering_name}_hungarian: mIoU {results.mean_iou():.4f}, Acc {results.accuracy:.4f}"
        )

        metric_maj = metrics.DatasetSemanticMaskIoUMetric(
            n_predicted_classes=n_clusters,
            n_classes=config.n_classes,
            matching="majority",
            use_threshold=config.use_mask_threshold,
            threshold=config.mask_threshold,
            ignore_background=config.ignore_background,
        )
        metric_maj.load_state_dict(metric_hung.state_dict())  # Load state to avoid recomputing
        # metric_maj._update_called = True  # Silence warning about calling compute before update

        per_class_iou, accuracy, empty_classes = metric_maj.compute()
        results = Results(
            per_class_iou.cpu().numpy(),
            accuracy.cpu().numpy(),
            empty_classes.cpu().numpy(),
            n_classes=config.n_classes,
            n_clusters=n_clusters,
            has_background=not config.ignore_background,
            clustering=config.clusterings.get(clustering_name),
            model_clustering=config.model_clusterings.get(clustering_name),
            matching="majority",
        )
        results_by_method[f"{clustering_name}_majority"] = results
        logging.info(
            f"{clustering_name}_majority: mIoU {results.mean_iou():.4f}, Acc {results.accuracy:.4f}"
        )

        # Already save intermediate reports.
        report_path = generate_and_save_report(config, results_by_method)

    logging.info(f"Report saved to {report_path}.")


if __name__ == "__main__":
    evaluate()
