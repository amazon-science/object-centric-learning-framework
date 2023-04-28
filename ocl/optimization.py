"""Convenience functions that allow defining optimization via config."""
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer


class OptimizationWrapper:
    """Optimize (a subset of) the parameters using a optimizer and a LR scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_scheduler: Optional[Callable[[Optimizer], Dict[str, Any]]] = None,
        parameter_groups: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize OptimizationWrapper.

        Args:
            optimizer: The oprimizer that should be used to optimize the parameters.
            lr_scheduler: The LR scheduling callable that should be used.  This should
                be a callable that returns a dict for updating the optimizer output in
                pytorch_lightning. See [ocl.scheduling.exponential_decay_with_optional_warmup][]
                for an example of such a callable.
            parameter_groups: Define parameter groups which have different optimizer parameters.
                Each element of the list should at least one of two keys `params` (for defining
                parameters based on their path in the model) or `predicate` (for defining parameters
                using a predicate function which returns true if the parameter should be included).
                For an example on how to use this parameter_groups, see
                `configs/experiment/examples/parameter_groups.yaml`.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.parameter_group_specs = parameter_groups
        if self.parameter_group_specs:
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
                        raise ValueError(
                            f'"predicate" for parameter group {idx + 1} is not a callable'
                        )

    def _get_parameter_groups(self, model):
        """Build parameter groups from specification."""
        if not self.parameter_group_specs:
            return model.parameters()
        parameter_groups = []
        for param_group_spec in self.parameter_group_specs:
            param_spec = param_group_spec["params"]
            # Default predicate includes all parameters
            predicate = param_group_spec.get("predicate", lambda name, param: True)

            parameters = []
            for parameter_path in param_spec:
                root = model
                for child in parameter_path.split("."):
                    root = getattr(root, child)
                parameters.extend(
                    param for name, param in root.named_parameters() if predicate(name, param)
                )

            param_group = {
                k: v for k, v in param_group_spec.items() if k not in ("params", "predicate")
            }
            param_group["params"] = parameters
            parameter_groups.append(param_group)

        return parameter_groups

    def __call__(self, model: torch.nn.Module):
        """Called in configure optimizers."""
        params_or_param_groups = self._get_parameter_groups(model)
        optimizer = self.optimizer(params_or_param_groups)
        output = {"optimizer": optimizer}
        if self.lr_scheduler:
            output.update(self.lr_scheduler(optimizer))
        return output
