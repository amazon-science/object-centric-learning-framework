import json
import pathlib
from typing import Any, Dict, List

import hydra
import pytest
import pytorch_lightning as pl

import ocl.cli.eval
import ocl.cli.eval_cluster_metrics
import ocl.cli.train

# Names of configurations that should be executed for a test run
CONFIGS_TO_TEST = [
    "experiment/slot_attention/clevr6",
    "experiment/occluded_slot_attention/clevr6",
    "experiment/projects/bridging/dinosaur/coco_feat_rec_resnet50",
]

IMAGE_DATASET_SHAPES = {"image": [256, 256, 3], "mask": [11, 256, 256, 1]}
IMAGE_DATASET_DTYPES = {"image": "image", "mask": "mask", "instance_mask": "categorical_10"}

COCO_DATASET_SHAPES = {
    "image": [256, 256, 3],
    "instance_mask": [5, 256, 256, 1],
    "instance_category": [5],
    "stuffthings_mask": [256, 256, 3],
}
COCO_DATASET_DTYPES = {
    "image": "image",
    "instance_mask": "mask",
    "instance_category": "categorical_1_91",
    "stuffthings_mask": "categorical_2",
}

# Shape of data created for test dataset
SHAPES_BY_CONFIG = {
    "experiment/slot_attention/clevr6": IMAGE_DATASET_SHAPES,
    "experiment/occluded_slot_attention/clevr6": IMAGE_DATASET_SHAPES,
    "experiment/projects/bridging/dinosaur/coco_feat_rec_resnet50": COCO_DATASET_SHAPES,
}
# Data types of data created for test dataset
DTYPES_BY_CONFIG = {
    "experiment/slot_attention/clevr6": IMAGE_DATASET_DTYPES,
    "experiment/occluded_slot_attention/clevr6": IMAGE_DATASET_DTYPES,
    "experiment/projects/bridging/dinosaur/coco_feat_rec_resnet50": COCO_DATASET_DTYPES,
}

TRAIN_OVERRIDES = [
    "trainer.devices=1",
    "trainer.fast_dev_run=True",
    "dataset=dummy_dataset",
    "++dataset.batch_size=4",
    "++dataset.train_size=null",
    "++dataset.val_size=null",
    "++dataset.test_size=null",
    "++dataset.data_shapes={shapes}",
    "++dataset.data_types={dtypes}",
]


def _format_dict(d: Dict[str, Any]) -> str:
    """Format dict as string such that Hydra accepts it."""
    values = ", ".join(f"{k}: {v}" for k, v in d.items())
    return "{" + values + "}"


def _format_list(ls: List[Any], quote=False) -> str:
    """Format list as string such that Hydra accepts it."""
    if quote:
        str_list = [f"'{x}'" for x in ls]
    else:
        str_list = [f"{x}" for x in ls]
    return "[" + ", ".join(str_list) + "]"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:LightningDeprecationWarning")
@pytest.mark.parametrize("config_name", CONFIGS_TO_TEST)
def test_train(config_name):
    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register TrainingConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="training_config",
            node=ocl.cli.train.TrainingConfig,
        )

        shapes = _format_dict(SHAPES_BY_CONFIG[config_name])
        dtypes = _format_dict(DTYPES_BY_CONFIG[config_name])
        overrides = [
            arg.format(config_name=config_name, shapes=shapes, dtypes=dtypes)
            for arg in TRAIN_OVERRIDES
        ]

        config = hydra.compose(config_name, overrides=overrides)
        ocl.cli.train.train(config)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:LightningDeprecationWarning")
@pytest.mark.parametrize("train_config_name", CONFIGS_TO_TEST)
def test_eval(train_config_name, tmp_path: pathlib.Path):
    from ocl.cli.eval import EvaluationConfig

    checkpoint_path = str(tmp_path / "checkpoint")
    report_filename = "test.json"
    output_path = str(tmp_path / report_filename)

    shapes = _format_dict(SHAPES_BY_CONFIG[train_config_name])
    dtypes = _format_dict(DTYPES_BY_CONFIG[train_config_name])
    dataset_config = [
        "dataset=dummy_dataset",
        "++dataset.batch_size=4",
        "++dataset.train_size=null",
        "++dataset.val_size=4",
        "++dataset.test_size=null",
        f"++dataset.data_shapes={shapes}",
        f"++dataset.data_types={dtypes}",
    ]
    train_config_overrides = ["trainer.devices=1"] + dataset_config

    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register TrainingConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="training_config",
            node=ocl.cli.train.TrainingConfig,
        )
        config = hydra.compose(train_config_name, overrides=train_config_overrides)
        _save_dummy_checkpoint(config, checkpoint_path)

    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register EvaluationConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="evaluation_config",
            node=EvaluationConfig,
        )

        train_config_path = pathlib.Path(__file__).absolute().parent.parent / "configs/"
        overrides = [
            f"train_config_path={train_config_path}",
            f"train_config_name={train_config_name}",
            f"train_config_overrides={_format_list(train_config_overrides, quote=True)}",
            f"checkpoint_path={checkpoint_path}",
            f"output_dir={tmp_path}",
            f"report_filename={report_filename}",
            "eval_train=False",
            "eval_val=True",
            "eval_test=False",
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
        ]
        config = hydra.compose("evaluation/eval", overrides=overrides)

        ocl.cli.eval.evaluate(config)

    with open(output_path, "r") as f:
        report = json.load(f)

    metrics = report["metrics"]
    assert "val" in metrics
    for value in metrics["val"].values():
        assert isinstance(value, float)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:LightningDeprecationWarning")
@pytest.mark.parametrize(
    "config_name, train_config_name",
    [
        (
            "evaluation/projects/bridging/clustering_coco",
            "experiment/projects/bridging/dinosaur/coco_feat_rec_resnet50",
        )
    ],
)
def test_eval_cluster_metrics(config_name, train_config_name, tmp_path: pathlib.Path):
    from ocl.cli.eval_cluster_metrics import EvaluationClusteringConfig

    checkpoint_path = str(tmp_path / "checkpoint")
    shapes = _format_dict(SHAPES_BY_CONFIG[train_config_name])
    dtypes = _format_dict(DTYPES_BY_CONFIG[train_config_name])
    dataset_config = [
        "dataset=dummy_dataset",
        "++dataset.batch_size=5",
        "++dataset.train_size=null",
        "++dataset.val_size=25",
        "++dataset.test_size=null",
        f"++dataset.data_shapes={shapes}",
        f"++dataset.data_types={dtypes}",
    ]
    train_config_overrides = ["trainer.devices=1"] + dataset_config

    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register TrainingConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="training_config",
            node=ocl.cli.train.TrainingConfig,
        )
        config = hydra.compose(train_config_name, overrides=train_config_overrides)
        _save_dummy_checkpoint(config, checkpoint_path)

    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register EvaluationConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="evaluation_clustering_config",
            node=EvaluationClusteringConfig,
        )

        train_config_path = pathlib.Path(__file__).absolute().parent.parent / "configs/"
        overrides = dataset_config + [
            f"train_config_path={train_config_path}",
            f"train_config_name={train_config_name}",
            f"output_dir={tmp_path}",
            f"checkpoint_path={checkpoint_path}",
            f"train_config_overrides={_format_list(dataset_config, quote=True)}",
            "n_min_mask_pixels=0",
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
        ]
        config = hydra.compose(config_name, overrides=overrides)

        ocl.cli.eval_cluster_metrics.evaluate(config)

        output_path = str(tmp_path / EvaluationClusteringConfig.report_filename)
        with open(output_path, "r") as f:
            json.load(f)


def _save_dummy_checkpoint(config, checkpoint_path):
    """Save a dummy checkpoint with random weights for the model specified by the configuration."""
    datamodule = ocl.cli.train.build_and_register_datamodule_from_config(config)
    model = ocl.cli.train.build_model_from_config(config)
    trainer = pl.Trainer(logger=False, limit_predict_batches=1)

    # Pytorch Lightning only allows to save a checkpoint after a function that does something with
    # the model is called on the trainer. Thus we run the predict method on the validation data here.
    trainer.predict(model, datamodule.val_dataloader(), return_predictions=False)

    trainer.save_checkpoint(checkpoint_path, weights_only=True)
