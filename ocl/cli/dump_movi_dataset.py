"""Script to dump MOVi datasets."""
import dataclasses
import logging
import pathlib
from typing import Any, Dict

import hydra
import hydra_zen
import numpy as np
import tqdm
from pluggy import PluginManager

import ocl.hooks
import ocl.plugins


@dataclasses.dataclass
class ComputeSizeConfig:
    """Configuration of a training run."""

    dataset: Any
    plugins: Dict[str, Any] = dataclasses.field(default_factory=dict)
    output_dir: str = "./data"


hydra.core.config_store.ConfigStore.instance().store(
    name="compute_size_config",
    node=ComputeSizeConfig,
)


@hydra.main(config_name="compute_size_config", config_path="../../configs", version_base="1.1")
def compute_size(config: ComputeSizeConfig):
    pm = PluginManager("ocl")
    pm.add_hookspecs(ocl.hooks)

    datamodule = hydra_zen.instantiate(
        config.dataset, hooks=pm.hook, batch_size=1, eval_batch_size=1, num_workers=0
    )
    pm.register(datamodule)

    plugins = hydra_zen.instantiate(config.plugins)
    for plugin in plugins.values():
        pm.register(plugin)

    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    images = []
    for data in tqdm.tqdm(datamodule.train_data_iterator()):
        images.append(data["image"])
    logging.info("Writing train split...")
    np.save(str(output_dir / "train_images.npy"), np.stack(images))

    images, masks = [], []
    for data in tqdm.tqdm(datamodule.val_data_iterator()):
        images.append(data["image"])
        masks.append(data["mask"].astype(np.uint8))
    logging.info("Writing val split...")
    np.save(str(output_dir / "val_images.npy"), np.stack(images))
    np.save(str(output_dir / "val_labels.npy"), np.stack(masks))


if __name__ == "__main__":
    compute_size()
