import ast
import os
from distutils.util import strtobool
from typing import Any, Callable

import yaml
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def _torchvision_interpolation_mode(mode):
    import torchvision

    return torchvision.transforms.InterpolationMode[mode.upper()]


def lambda_string_to_function(function_string: str) -> Callable[..., Any]:
    """Convert string of the form "lambda x: x" into a callable Python function."""
    # This is a bit hacky but ensures that the syntax of the input is correct and contains
    # a valid lambda function definition without requiring to run `eval`.
    parsed = ast.parse(function_string)
    is_lambda = isinstance(parsed.body[0], ast.Expr) and isinstance(parsed.body[0].value, ast.Lambda)
    if not is_lambda:
        raise ValueError(f"'{function_string}' is not a valid lambda definition.")

    return eval(function_string)


class ConfigDefinedLambda:
    """Lambda function defined in the config.

    This allows lambda functions defined in the config to be pickled.
    """

    def __init__(self, function_string: str):
        self.__setstate__(function_string)

    def __getstate__(self) -> str:
        return self.function_string

    def __setstate__(self, function_string: str):
        self.function_string = function_string
        self._fn = lambda_string_to_function(function_string)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def eval_lambda(function_string, *args):
    lambda_fn = lambda_string_to_function(function_string)
    return lambda_fn(*args)


def make_slice(expr):
    if isinstance(expr, int):
        return expr

    pieces = [s and int(s) or None for s in expr.split(":")]
    if len(pieces) == 1:
        return slice(pieces[0], pieces[0] + 1)
    else:
        return slice(*pieces)


def slice_string(string: str, split_char: str, slice_str: str) -> str:
    """Split a string according to a split_char and slice.

    If the output contains more than one element, join these using the split char again.
    """
    sl = make_slice(slice_str)
    res = string.split(split_char)[sl]
    if isinstance(res, list):
        res = split_char.join(res)
    return res


def read_yaml(path):
    with open(to_absolute_path(path), "r") as f:
        return yaml.safe_load(f)


def when_testing(output_testing, output_otherwise):
    running_tests = bool(strtobool(os.environ.get("RUNNING_TESTS", "false")))
    return output_testing if running_tests else output_otherwise


OmegaConf.register_new_resolver("torchvision_interpolation_mode", _torchvision_interpolation_mode)
OmegaConf.register_new_resolver("lambda_fn", ConfigDefinedLambda)
OmegaConf.register_new_resolver("eval_lambda", eval_lambda)
OmegaConf.register_new_resolver("slice", slice_string)
OmegaConf.register_new_resolver("read_yaml", read_yaml)
OmegaConf.register_new_resolver("when_testing", when_testing)
