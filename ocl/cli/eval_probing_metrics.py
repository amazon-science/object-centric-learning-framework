import dataclasses
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import hydra
import hydra_zen
import pytorch_lightning as pl
import torch

from ocl import datasets, tree_utils
from ocl.cli import cli_utils, train
from ocl.combined_model import CombinedModel
from ocl.config.datasets import DataModuleConfig
from ocl.config.metrics import MetricConfig
from ocl.config.plugins import PluginConfig

# import json
# import omegaconf

logger = logging.getLogger("eval_probing_metrics")


@dataclasses.dataclass
class EvaluationProbingConfig:
    # Path to training configuration file or configuration dir. If dir, train_config_name
    # needs to be set as well.
    train_config_path: str

    dataset: DataModuleConfig
    models: Dict[str, Any]
    losses: Dict[str, Any]
    trainer: train.TrainerConf = train.TrainerConf
    visualizations: Dict[str, Any] = dataclasses.field(default_factory=dict)
    training_metrics: Optional[Dict[str, MetricConfig]] = None
    evaluation_metrics: Optional[Dict[str, MetricConfig]] = None
    plugins: Dict[str, PluginConfig] = dataclasses.field(default_factory=dict)

    train_config_overrides: Optional[List[str]] = None
    train_config_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    output_dir: Optional[str] = None
    report_filename: str = "probing_metrics.json"

    cache_model_predictions: bool = False
    predict_trainer: train.TrainerConf = train.TrainerConf
    predict_mapping: Optional[Dict[str, str]] = None

    pred_batch_size: int = 20
    train_batch_size: int = 20
    eval_batch_size: int = 20
    training_vis_frequency: int = 100

    seed: Optional[int] = None


hydra.core.config_store.ConfigStore.instance().store(
    name="evaluation_probing_config",
    node=EvaluationProbingConfig,
)


class ExtractDataFromPredictions(pl.callbacks.Callback):
    def __init__(self, mapping: Dict[str, str]):
        assert mapping is not None
        self.mapping = {k: (k if v is None else v) for k, v in mapping.items()}
        self.outputs = defaultdict(list)

    def on_predict_start(self, trainer, pl_module):
        self.outputs = defaultdict(list)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        data = {path: tree_utils.get_tree_element(outputs, path.split(".")) for path in self.mapping}

        for in_path, out_path in self.mapping.items():
            self.outputs[out_path].append(data[in_path].cpu())

    def extract(self):
        outputs = self.outputs
        self.outputs = defaultdict(list)
        return outputs


def get_data_from_model(
    trainer_conf: train.TrainerConf,
    model: CombinedModel,
    datamodule: pl.LightningDataModule,
    mapping: Dict[str, str],
):
    data_extractor = ExtractDataFromPredictions(mapping)
    trainer = hydra_zen.instantiate(
        trainer_conf,
        _convert_="all",
        accelerator="auto",
        devices=1,
        callbacks=[data_extractor],
        logger=False,
    )

    # Necessary for shuffling in webdatasets
    os.environ["WDS_EPOCH"] = "0"

    logging.info("Running predict on training data.")
    trainer.predict(model, datamodule.train_dataloader(), return_predictions=False)
    train_data = data_extractor.extract()

    logging.info("Running predict on validation data.")
    trainer.predict(model, datamodule.val_dataloader(), return_predictions=False)
    val_data = data_extractor.extract()

    return train_data, val_data


def load_from_train_config(config: train.TrainingConfig, checkpoint_path: str):
    pl.seed_everything(config.seed, workers=True)
    pm = train.create_plugin_manager()
    datamodule = train.build_and_register_datamodule_from_config(config, pm.hook, pm)
    train.build_and_register_plugins_from_config(config, pm)
    model = train.build_model_from_config(config, pm.hook, checkpoint_path)

    return datamodule, model


def extend_model(config: EvaluationProbingConfig, model: CombinedModel, pm):
    model.requires_grad_(False)

    additional_models = hydra_zen.instantiate(config.models, _convert_="all")
    for key, module in additional_models.items():
        model.models[key] = module

    model.losses = hydra_zen.instantiate(config.losses, _convert_="all")
    model.visualizations = hydra_zen.instantiate(config.visualizations, _convert_="all")
    model.training_metrics = torch.nn.ModuleDict(hydra_zen.instantiate(config.training_metrics))
    model.evaluation_metrics = torch.nn.ModuleDict(hydra_zen.instantiate(config.evaluation_metrics))

    model.hooks = pm.hook

    return model


@hydra.main(config_name="evaluation_probing_config", config_path="../../configs", version_base="1.1")
def evaluate(config: EvaluationProbingConfig):
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
        train_config.dataset.batch_size = config.eval_batch_size

        _, model = load_from_train_config(train_config, config.checkpoint_path)

    pm = train.create_plugin_manager()
    batch_size = (
        config.pred_batch_size if config.cache_model_predictions else config.train_batch_size
    )
    datamodule = train.build_and_register_datamodule_from_config(
        config, pm.hook, pm, batch_size=batch_size
    )
    train.build_and_register_plugins_from_config(config, pm)

    if config.cache_model_predictions:
        # Use trained model to get model predictions, then train probing modules on top of them
        train_data, val_data = get_data_from_model(
            config.predict_trainer, model, datamodule, config.predict_mapping
        )
        datamodule = datasets.InMemoryDataModule(
            train_data,
            val_data,
            train_batch_size=config.train_batch_size,
            val_batch_size=config.eval_batch_size,
        )
        _, model = load_from_train_config(config, checkpoint_path=None)
    else:
        # Reuse existing training model and add new probing modules on top
        model = extend_model(config, model, pm)

    # Train the probing modules
    logging.info("Training probing modules.")
    trainer: pl.Trainer = hydra_zen.instantiate(
        config.trainer, _convert_="all", default_root_dir=f"{config.output_dir}/eval_probing"
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    evaluate()
