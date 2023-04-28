"""Magic routed module which allows dynamically routing.

This module is used to wrap classes from arbitrary packages into Routable classes.
A routable class is augmented with additional constructor parameters that
determine on which elements of a PyTree the methods

 - `update` (for classes of type [torchmetrics.Metric][])
 - `forward` (for clases of type [torch.nn.Module][]) or
 - `__call__` (for all other cases)

should be applied.

This is acomplished using a simple trick: Instead of passing the individual
parameters to methods of the class, the original method is wrapped. This wrapped method,
then selects the desired input arguments from a `inputs` argument and forwards these to
the original class implementation of the method.

Example:
    ```python
    import torch
    import routed

    non_routed_class = torch.nn.Sigmoid()
    routed_class = routed.torch.nn.Sigmoid(input_path="my_sigmoid_source")

    example_tensor = torch.randn(100)
    inputs = {
        "my_sigmoid_source": example_tensor
    }
    assert torch.allclose(non_routed_class(example_tensor), routed_class(inputs=inputs))
    ```

"""
import functools
import importlib
import inspect
import types
from typing import Any, Dict, List

import torch as _torch
import torchmetrics as _torchmetrics

import ocl.utils.trees as tree_utils

_CLASS_TO_ROUTED_METHODS_MAP = {_torchmetrics.Metric: ["update"], _torch.nn.Module: ["forward"]}


def build_routed_method(
    method: types.MethodType, filter_parameters: bool = True
) -> types.MethodType:
    """Pass arguments to a function based on the mapping defined in `self.input_mapping`.

    This method supports both filtering for parameters that match the arguments of the wrapped
    method and passing all arguments defined in `input_mapping`.  If a non-optional argument is
    missing this will raise an exception.  Additional arguments can also be passed to the method
    to override entries in the input dict.  Non-keyword arguments are always directly passed to
    the method.

    Args:
        method: The method to pass the arguments to.
        filter_parameters: Only pass arguments to wrapped method that match the methods
            signature.  This is practical if different methods require different types of input.

    """
    # Run inspection here to reduce compute time when calling method.
    signature = inspect.signature(method)
    valid_parameters = list(signature.parameters)  # Returns the parameter names.
    valid_parameters = valid_parameters[1:]  # Discard "self".
    # Keep track of default parameters. For these we should not fail if they are not in
    # the input dict.
    with_defaults = [
        name
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    ]

    @functools.wraps(method)
    def method_with_routing(self: RoutedClass, *args, inputs=None, **kwargs):
        if not inputs:
            inputs = {}
        if self.input_mapping:
            if not inputs:  # Empty dict.
                inputs = kwargs

            routed_inputs = {}
            for input_field, input_path in self.input_mapping.items():
                if filter_parameters and input_field not in valid_parameters:
                    # Skip parameters that are not the function signature.
                    continue
                if input_field in kwargs.keys():
                    # Skip parameters that are directly provided as kwargs.
                    continue
                try:
                    element = tree_utils.get_tree_element(inputs, input_path)
                    routed_inputs[input_field] = element
                except ValueError as e:
                    if input_field in with_defaults:
                        continue
                    else:
                        raise e
            # Support for additional parameters passed via keyword arguments.
            # TODO(hornmax): This is not ideal as it mixes routing args from the input dict
            # and explicitly passed kwargs and thus could lead to collisions.
            for name, element in kwargs.items():
                if filter_parameters and name not in valid_parameters:
                    continue
                else:
                    routed_inputs[name] = element
            return method(self, *args, **routed_inputs)
        else:
            return method(self, *args, **kwargs)

    return method_with_routing


def _get_routed_methods(cls):
    for key, value in _CLASS_TO_ROUTED_METHODS_MAP.items():
        if issubclass(cls, key):
            return value
    return ["__call__"]


class RoutedClass:
    """Class used to dynamically subclass routed classes.

    Any subclasses of this class are automatically patched to support routing of input arguments.

    Attributes:
        input_mapping: Mapping from parameters of routed functions to paths in the inputs dict.

    """

    input_mapping: Dict[str, List[str]]

    def __init__(self, *args, **kwargs):
        self._remove_routed_parameters(kwargs)
        super().__init__(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        # Patch routed methods.
        # This needs to be done here as they are otherwise not considered methods.
        _routed_methods = _get_routed_methods(cls)
        input_mapping = {}
        for method_name in _routed_methods:
            org_method = getattr(cls, method_name)
            for name in inspect.signature(org_method).parameters:
                path_name = f"{name}_path"
                if path_name in kwargs:
                    input_mapping[name] = kwargs[path_name].split(".")

            setattr(cls, method_name, build_routed_method(org_method))
        instance = super().__new__(cls)
        instance.input_mapping = input_mapping
        return instance

    def _remove_routed_parameters(self, kwargs: Dict[str, Any]):
        for param in self.input_mapping:
            path = f"{param}_path"
            if path in kwargs:
                del kwargs[path]


class WrappedModule(types.ModuleType):
    """Module which automatically patches all classes within it to support routing."""

    def __init__(self, path: str, module):
        super().__init__(path, f"Module with routed versions of {path}")
        self.path = path
        self.module = module

    def __getattr__(self, name):
        try:
            imported = getattr(self.module, name)
        except AttributeError:
            imported = importlib.import_module(f"{self.path}.{name}")
        if isinstance(imported, types.ModuleType):
            return WrappedModule(f"{self.path}.{name}", imported)
        return type(f"{self.path}.Routed{name}", (RoutedClass, imported), {})


# Dynamically create modules that implement routing.
def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError()
    module = importlib.import_module(name)
    return WrappedModule(f"{name}", module)
