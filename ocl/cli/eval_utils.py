import pathlib
import pickle
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy
import pytorch_lightning as pl
import torch

from ocl.cli import train
from ocl.utils.trees import get_tree_element


def build_from_train_config(
    config: train.TrainingConfig, checkpoint_path: Optional[str], seed: bool = True
):
    if seed:
        pl.seed_everything(config.seed, workers=True)

    datamodule = train.build_and_register_datamodule_from_config(config)
    model = train.build_model_from_config(config, checkpoint_path)

    return datamodule, model


class ExtractDataFromPredictions(pl.callbacks.Callback):
    """Callback used for extracting model outputs during validation and prediction."""

    def __init__(
        self,
        paths: List[str],
        output_paths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        flatten_batches: bool = False,
    ):
        self.paths = paths
        self.output_paths = output_paths if output_paths is not None else paths
        self.transform = transform
        self.max_samples = max_samples
        self.flatten_batches = flatten_batches

        self.outputs = defaultdict(list)
        self._n_samples = 0

    def _start(self):
        self._n_samples = 0
        self.outputs = defaultdict(list)

    def _process_outputs(self, outputs, batch):
        if self.max_samples is not None and self._n_samples >= self.max_samples:
            return

        data = {"input": batch, **outputs}
        data = {path: get_tree_element(outputs, path.split(".")) for path in self.paths}

        if self.transform:
            data = self.transform(data)

        first_path = True
        for path in self.output_paths:
            elems = data[path].detach().cpu()
            if not self.flatten_batches:
                elems = [elems]

            for idx in range(len(elems)):
                self.outputs[path].append(elems[idx])
                if first_path:
                    self._n_samples += 1

            first_path = False

    def on_validation_start(self, trainer, pl_module):
        self._start()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        assert (
            outputs is not None
        ), "Model returned no outputs. Set `model.return_outputs_on_validation=True`"
        self._process_outputs(outputs, batch)

    def on_predict_start(self, trainer, pl_module):
        self._start()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._process_outputs(outputs, batch)

    def get_outputs(self) -> List[Dict[str, Any]]:
        state = []
        for idx in range(self._n_samples):
            state.append({})
            for key, values in self.outputs.items():
                state[-1][key] = values[idx]

        return state


def save_outputs(dir_path: str, outputs: List[Dict[str, Any]], verbose: bool = False):
    """Save outputs to disk in numpy or pickle format."""
    dir_path = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    def get_path(path, prefix, key, extension):
        return str(path / f"{prefix}.{key}.{extension}")

    idx_fmt = "{:0" + str(len(str(len(outputs)))) + "d}"  # Get number of total digits
    for idx, entry in enumerate(outputs):
        idx_prefix = idx_fmt.format(idx)
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()

            if isinstance(value, numpy.ndarray):
                path = get_path(dir_path, idx_prefix, key, "npy")
                if verbose:
                    print(f"Saving numpy array to {path}.")
                numpy.save(path, value)
            else:
                path = get_path(dir_path, idx_prefix, key, "pkl")
                if verbose:
                    print(f"Saving pickle to {path}.")
                with open(path, "wb") as f:
                    pickle.dump(value, f)
