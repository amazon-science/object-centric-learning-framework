"""Utility function related to routing of information.

These utility functions allow dynamical routing between modules and allow the specification of
complex models using config alone.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from torch import nn

import ocl.utils.trees as tree_utils


class RoutableMixin:
    """Mixin class that allows to connect any element of a (nested) dict with a module input."""

    def __init__(self, input_mapping: Mapping[str, Optional[str]]):
        self.input_mapping = {
            key: value.split(".") for key, value in input_mapping.items() if value is not None
        }

    def _route(method, filter_parameters=True):
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
        def method_with_routing(self, *args, inputs=None, **kwargs):
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

    # This is needed in order to allow the decorator to be used in child classes. The documentation
    # looks a bit hacky but I didn't find an alternative approach on how to do it.
    route = staticmethod(functools.partial(_route, filter_parameters=True))
    route.__doc__ = (
        """Route input arguments according to input_mapping and filter non-matching arguments."""
    )
    route_unfiltered = staticmethod(functools.partial(_route, filter_parameters=False))
    route_unfiltered.__doc__ = """Route all input arguments according to input_mapping."""


class DataRouter(nn.Module, RoutableMixin):
    """Data router for modules that don't support the RoutableMixin.

    This allows the usage of modules without RoutableMixin support in the dynamic information flow
    pattern of the code.
    """

    def __init__(self, module: nn.Module, input_mapping: Mapping[str, str]):
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, input_mapping)
        self.module = module
        self._cached_valid_parameters = None

    @RoutableMixin.route_unfiltered
    def forward(self, *args, **kwargs):
        # We need to filter parameters at runtime as we cannot know them prior to initialization.
        if not self._cached_valid_parameters:
            try:
                signature = inspect.signature(self.module.forward)
            except AttributeError:
                if callable(self.module):
                    signature = inspect.signature(self.module.__call__)
                else:
                    signature = inspect.signature(self.module)

            self._cached_valid_parameters = list(signature.parameters)

        kwargs = {
            name: param for name, param in kwargs.items() if name in self._cached_valid_parameters
        }
        return self.module(*args, **kwargs)


class Combined(nn.ModuleDict):
    """Module to combine multiple modules and store their outputs.

    A combined module groups together multiple model components and allows them to access any
    information that was returned in processing steps prior to their own application.

    It functions similarly to `nn.ModuleDict` yet for modules of type `RoutableMixin` and
    additionally implements a forward routine which will return a dict of the outputs of the
    submodules.

    """

    def __init__(self, **modules: Dict[str, Union[RoutableMixin, Combined, Recurrent]]):
        super().__init__(modules)

    def forward(self, inputs: Dict[str, Any]):
        # The combined module does not know where it is positioned and thus also does not know in
        # which sub-path results should be written. As we want different modules of a combined
        # module to be able access previous outputs using their global path in the dictionary, we
        # need to somehow keep track of the nesting level and then directly write results into the
        # input dict at the right path.  The prefix variable keeps track of the nesting level.
        prefix: List[str]
        if "prefix" in inputs.keys():
            prefix = inputs["prefix"]
        else:
            prefix = []
            inputs["prefix"] = prefix

        outputs = tree_utils.get_tree_element(inputs, prefix)
        for name, module in self.items():
            # Update prefix state such that nested calls of combined return dict in the correct
            # location.
            prefix.append(name)
            outputs[name] = {}
            # If module is a Combined module, it will return the same dict as set above. If not the
            # dict will be overwritten with the output of the module.
            outputs[name] = module(inputs=inputs)
            # Remove last component of prefix after execution.
            prefix.pop()
        return outputs


class Recurrent(nn.Module):
    """Module to apply another module in a recurrent fashion over a axis.

    This module takes a set of input tensors and applies a module recurrent over them.  The output
    of the previous iteration is kept in the `previous_output` key of input dict and thus can be
    accessed using data routing. After applying the module to the input slices, the outputs are
    stacked along the same axis as the inputs where split.


    """

    def __init__(
        self,
        module: nn.Module,
        inputs_to_split: List[str],
        initial_input_mapping: Dict[str, str],
        split_axis: int = 1,
        chunk_size: int = 1,
    ):
        """Initialize recurrent module.

        Args:
            module: The module that should be applied recurrently along input tensors.
            inputs_to_split: List of paths that should be split for recurrent application.
            initial_input_mapping: Mapping that constructs the first `previous_output` element.  If
                `previous_output` should just be a tensor, use a mapping of the format
                `{"": "input_path"}`.
            split_axis: Axis along which to split the tensors defined by inputs_to_split.
            chunk_size: The size of each slice, when set to 1, the slice dimension is squeezed prior
                to passing to the module.
        """
        super().__init__()
        self.module = module
        self.inputs_to_split = [path.split(".") for path in inputs_to_split]
        self.initial_input_mapping = {
            output: input.split(".") for output, input in initial_input_mapping.items()
        }
        self.split_axis = split_axis
        self.chunk_size = chunk_size

    def _build_initial_dict(self, inputs):
        # This allows us to bing the initial input and previous_output into a similar format.
        output_dict = {}
        for output_path, input_path in self.initial_input_mapping.items():
            source = tree_utils.get_tree_element(inputs, input_path)
            if output_path == "":
                # Just the object itself, no dict nesting.
                return source

            output_path = output_path.split(".")
            cur_search = output_dict
            for path_part in output_path[:-1]:
                # Iterate along path and create nodes that do not exist yet.
                try:
                    # Get element prior to last.
                    cur_search = tree_utils.get_tree_element(cur_search, [path_part])
                except ValueError:
                    # Element does not yet exist.
                    cur_search[path_part] = {}
                    cur_search = cur_search[path_part]

            cur_search[output_path[-1]] = source
        return output_dict

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Come up with a better way of handling the initial input without putting restrictions
        # on modules being run recurrently.
        outputs = [self._build_initial_dict(inputs)]
        for split_dict in tree_utils.split_tree(
            inputs, self.inputs_to_split, self.split_axis, self.chunk_size
        ):
            split_dict["previous_output"] = outputs[-1]
            outputs.append(self.module(inputs=split_dict))

        # TODO: When chunk size is larger than 1 then this should be cat and not stack. Otherwise an
        # additional axis would be added. Evtl. this should be configurable.
        stack_fn = functools.partial(torch.stack, dim=self.split_axis)
        # Ignore initial input.
        return tree_utils.reduce_tree(outputs[1:], stack_fn)
