"""Implementation of feature extractors that can be used for object centric learning.

These are grouped into 3 modules

 - [ocl.feature_extractors.misc][] Feature extractors implemented in object
   centric learning papers
 - [ocl.feature_extractors.timm][] Feature extractors based on timm models
 - [ocl.feature_extractors.clip][] Feature extractors for multi-modal data
   using CLIP

Utilities used by all modules are found in [ocl.feature_extractors.utils][].

**Important note:** In order to use feature extractors in
[timm][ocl.feature_extractors.timm] and [clip][ocl.feature_extractors.clip]
this package has to be installed with the `timm` and/or `clip` extras (see
[Installation][installation] for further information on installing extras).
"""

import importlib

from ocl.feature_extractors.misc import (
    DVAEFeatureExtractor,
    SAViFeatureExtractor,
    SlotAttentionFeatureExtractor,
)

_EXTRA_CLASS_TO_MODULE_MAP = {
    "TimmFeatureExtractor": "timm",
    "ClipImageModel": "clip",
    "ClipTextModel": "clip",
    "ClipFeatureExtractor": "clip",
}


def __getattr__(name):
    # Only import these when needed as they require additional dependencies.
    try:
        module_path = _EXTRA_CLASS_TO_MODULE_MAP[name]
        module = importlib.import_module(f"{__name__}.{module_path}")
        return getattr(module, name)
    except KeyError:
        raise AttributeError()


__all__ = [
    "SlotAttentionFeatureExtractor",
    "SAViFeatureExtractor",
    "DVAEFeatureExtractor",
    "TimmFeatureExtractor",
    "ClipImageModel",
    "ClipTextModel",
    "ClipFeatureExtractor",
]
