import os
from typing import Any, Dict, Iterable, List, Optional

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch import nn

from ocl.scheduling import HPScheduler
from ocl.utils.trees import get_tree_element, walk_tree_with_paths


class FreezeParameters(Callback):
    """Freeze parameters of model prior to training."""

    def __init__(self, parameter_groups: List[Dict[str, Any]]):
        """Initialize FreezeParameters callback.

        Args:
            parameter_groups: Parameter groups that should be frozen.
                Uses same syntax as [ocl.optimization.OptimizationWrapper][]
        """
        super().__init__()
        self.parameter_group_specs = parameter_groups
        for idx, param_group_spec in enumerate(self.parameter_group_specs):
            if "params" not in param_group_spec:
                raise ValueError(f'Parameter group {idx + 1} does not contain key "params"')
            param_spec = param_group_spec["params"]
            if isinstance(param_spec, str):
                param_group_spec["params"] = [param_spec]
            elif isinstance(param_spec, Iterable):
                param_group_spec["params"] = list(param_spec)
            else:
                raise ValueError(
                    f'"params" for parameter group {idx + 1} is not of type str or iterable'
                )

            if "predicate" in param_group_spec:
                if not callable(param_group_spec["predicate"]):
                    raise ValueError(f'"predicate" for parameter group {idx + 1} is not a callable')

    def _get_parameters_to_freeze(self, model):
        """Build parameter groups from specification."""
        parameters_to_freeze = []
        for param_group_spec in self.parameter_group_specs:
            for current_params in param_group_spec["params"]:
                param_path = current_params.split(".")
                # Default predicate includes all parameters
                predicate = param_group_spec.get("predicate", lambda name, param: True)
                param = get_tree_element(model, param_path)
                if isinstance(param, nn.Module):
                    parameters_to_freeze.extend(
                        param for name, param in param.named_parameters() if predicate(name, param)
                    )
                elif isinstance(param, nn.Parameter):
                    parameters_to_freeze.append(param)
                else:
                    raise ValueError(
                        "Object at path {'.'.join(param_path)} is neither nn.Module nor nn.Parameter"
                    )
        return parameters_to_freeze

    def on_fit_start(self, trainer, model: nn.Module):
        parameters_to_freeze = self._get_parameters_to_freeze(model)
        for param in parameters_to_freeze:
            param.requires_grad_(False)


class RestoreParameterSubset(Callback):
    """Restore a subset of parameters using a checkpoint form a different model."""

    def __init__(self, checkpoint_file: str, target_path: str, source_path: Optional[str] = None):
        """Initialize RestoreParameterSubset callback.

        Args:
            checkpoint_file: File from which the model weights should be loaded.
            target_path: The path in the model where the model weights should be
                restored.  This should follow a dot separated syntax, such a `encoder.layer1`.
            source_path: The path in the checkpoint_file that should be used to restore weights.
                If none provided assumes to be the same as `target_path`.

        """
        self.checkpoint_file = checkpoint_file
        self.target_path = target_path
        self.source_path = source_path if source_path else self.target_path

    def on_fit_start(self, trainer, model: nn.Module):
        if model.global_step != 0:
            # Don't restore when we are resuming training.
            rank_zero_warn("Not restoring parameter subset as training is being resumed")
            return
        # Get parameters from state dict
        state_dict = torch.load(self.checkpoint_file, map_location=model.device)["state_dict"]
        # Add offset of 1 to remove potential dot.
        offset_keys = len(self.source_path) + 1
        state_dict = {
            key[offset_keys:]: value
            for key, value in state_dict.items()
            if key.startswith(self.source_path)
        }

        # Get module from model
        model_component: nn.Module = get_tree_element(model, self.target_path.split("."))
        result = model_component.load_state_dict(state_dict)
        if len(result.missing_keys):
            rank_zero_warn(
                f"Mismatch between state dict and model. Missing keys: {result.missing_keys}"
            )
        if len(result.unexpected_keys):
            rank_zero_warn(
                f"Mismatch between state dict and model. Unexpected keys: {result.missing_keys}"
            )
        rank_zero_info(f"Restored subset of model parameters from {self.checkpoint_file}")


class UpdateHyperparameterScheduling(Callback):
    """Callback to update hyperparameter schedulers found `ocl.scheduling`."""

    def __init__(self):
        self._hyperparameter_schedulers: List[HPScheduler] = []

    def on_fit_start(self, trainer, pl_module):
        del trainer
        self._hyperparameter_schedulers = list(
            map(
                lambda a: a[1],
                walk_tree_with_paths(pl_module, instance_check=lambda t: isinstance(t, HPScheduler)),
            )
        )
        # Set global step to 0 for pretraining evaluation routine.
        self._update_schedulers(0)

    def _update_schedulers(self, step):
        if len(self._hyperparameter_schedulers) == 0:
            rank_zero_warn(
                "UpdateHyperparameterScheduling: "
                "No schedulable hyperparameters where found in model."
            )
        for hparam in self._hyperparameter_schedulers:
            hparam.update_global_step(step)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, batch, batch_idx
        global_step = pl_module.global_step
        self._update_schedulers(global_step)


class SetEpochEnvironmentVariable(Callback):
    """Sets environment variable `EPOCH` which is used by [ocl.transforms.SampleSlices][]."""

    def on_train_epoch_start(self, trainer, pl_module):
        os.environ["EPOCH"] = str(pl_module.current_epoch)
