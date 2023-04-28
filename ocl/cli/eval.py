#!/usr/bin/env python
"""Evaluate a trained slot attention type model."""
import dataclasses
import json
import logging
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import hydra
import hydra_zen
import pytorch_lightning as pl
import torch

import ocl.cli._config  # noqa: F401
from ocl.cli import cli_utils, eval_utils, train

logger = logging.getLogger("eval")


# --8<-- [start:EvaluationConfig]
@dataclasses.dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Path to training configuration file or configuration dir. If dir, train_config_name
    # needs to be set as well.
    train_config_path: str
    train_config_overrides: Optional[List[str]] = None
    train_config_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None
    report_filename: str = "metrics.json"

    # Setting this allows to add modules to the model that are executed during evaluation
    modules: Optional[Dict[str, Any]] = None
    # Setting this allows to evaluate on a different dataset than the model was trained on
    dataset: Optional[Any] = None
    # Setting this allows to evaluate on different metrics than the model was trained on
    evaluation_metrics: Optional[Dict[str, Any]] = None

    save_outputs: bool = False
    skip_metrics: bool = False
    outputs_dirname: str = "outputs"
    outputs_to_store: Optional[List[str]] = None
    n_samples_to_store: Optional[int] = None

    eval_train: bool = False
    eval_val: bool = True
    eval_test: bool = False
    eval_batch_size: Optional[int] = None


# --8<-- [end:EvaluationConfig]

hydra.core.config_store.ConfigStore.instance().store(
    name="evaluation_config",
    node=EvaluationConfig,
)


def report_from_results(metrics: Dict[str, Any], config: EvaluationConfig):
    output = OrderedDict()
    output["train_config_path"] = config.train_config_path
    output["checkpoint_path"] = config.checkpoint_path
    output["metrics"] = metrics
    return output


@hydra.main(config_name="evaluation_config", config_path="../../configs", version_base="1.1")
def evaluate(config: EvaluationConfig):
    os.environ["WDS_EPOCH"] = str(0)

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
        train_config.dataset.eval_batch_size = config.eval_batch_size

        datamodule, model = eval_utils.build_from_train_config(train_config, config.checkpoint_path)

    if config.modules is not None:
        modules = hydra_zen.instantiate(config.modules, _convert_="all")
        for key, module in modules.items():
            model.models[key] = module

    if config.dataset is not None:
        datamodule = train.build_and_register_datamodule_from_config(
            config,
            batch_size=train_config.dataset.batch_size,
            eval_batch_size=config.eval_batch_size,
        )

    if config.evaluation_metrics is not None:
        model.evaluation_metrics = torch.nn.ModuleDict(
            hydra_zen.instantiate(config.evaluation_metrics)
        )
    if config.skip_metrics:
        model.evaluation_metrics = torch.nn.ModuleDict()

    if config.save_outputs:
        if config.outputs_to_store is None:
            raise ValueError("Need to specify which outputs to store using `outputs_to_store`")
        data_extractor = eval_utils.ExtractDataFromPredictions(
            config.outputs_to_store, max_samples=config.n_samples_to_store, flatten_batches=True
        )
        callbacks = [data_extractor]
        model.return_outputs_on_validation = True
    else:
        callbacks = None

    if config.n_samples_to_store is not None:
        limit = int(math.ceil(config.n_samples_to_store // config.eval_batch_size))
    else:
        limit = None

    trainer: pl.Trainer = hydra_zen.instantiate(
        train_config.trainer,
        _convert_="all",
        devices=1,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=True,
        limit_val_batches=limit,
    )

    metrics = {}
    if config.eval_train:
        logger.info("Running evaluation on train set.")
        trainer.validate(model, datamodule.train_dataloader())
        metrics["train"] = {
            key.replace("train/", ""): float(value) for key, value in trainer.logged_metrics.items()
        }
        if config.save_outputs:
            dir_path = os.path.join(config.output_dir, config.outputs_dirname, "train")
            logger.info(f"Saving train outputs to {dir_path}.")
            eval_utils.save_outputs(dir_path, data_extractor.get_outputs())
    if config.eval_val:
        logger.info("Running evaluation on validation set.")
        trainer.validate(model, datamodule.val_dataloader())
        metrics["val"] = {
            key.replace("val/", ""): float(value) for key, value in trainer.logged_metrics.items()
        }
        if config.save_outputs:
            dir_path = os.path.join(config.output_dir, config.outputs_dirname, "val")
            logger.info(f"Saving validation outputs to {dir_path}.")
            eval_utils.save_outputs(dir_path, data_extractor.get_outputs())
    if config.eval_test:
        logger.info("Running evaluation on test set.")
        trainer.test(model, datamodule.test_dataloader())
        metrics["test"] = {
            key.replace("test/", ""): float(value) for key, value in trainer.logged_metrics.items()
        }
        if config.save_outputs:
            raise NotImplementedError("Saving outputs not implemented with `trainer.test`.")

    metrics_file = os.path.join(config.output_dir, config.report_filename)
    report = report_from_results(metrics, config)
    with open(metrics_file, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to {metrics_file}.")


if __name__ == "__main__":
    evaluate()
