"""Code to decode input data from file streams.

The code in this file is adapted from the torchdata and webdatasets packages.
It implements an extension based decoder, which selects a different
decoding function based on the extension.  In contrast to the
implementations in torchdata and webdatasets, the extension is
removed from the data field after decoding.  This ideally makes
the output format invariant to the exact decoding strategy.

Example:
    image.jpg will be decoded into a numpy array and will be accessable in the field `image`.
    image.npy.gz will be decoded into a numpy array which can also be accessed under `image`.

"""
import gzip
import json
import os
import pickle
from io import BytesIO
from typing import Any, Callable, Dict, Optional

import numpy
import torch
from torch.utils.data.datapipes.utils.decoder import imagespecs
from torchdata.datapipes.utils import StreamWrapper


class ExtensionBasedDecoder:
    """Decode key/data based on extension using a list of handlers.

    The input fields are assumed to be instances of
    [StreamWrapper][torchdata.datapipes.utils.StreamWrapper],
    which wrap an underlying file like object.
    """

    def __init__(self, *handler: Callable[[str, StreamWrapper], Optional[Any]]):
        self.handlers = list(handler) if handler else []

    def decode1(self, name, data):
        if not data:
            return data

        new_name, extension = os.path.splitext(name)
        if not extension:
            return name, data

        for f in self.handlers:
            result = f(extension, data)
            if result is not None:
                # Remove decoded part of name.
                data = result
                name = new_name
                # Try to decode next part of name.
                new_name, extension = os.path.splitext(name)
                if extension == "":
                    # Stop decoding if there are no further extensions to be handled.
                    break
        return name, data

    def decode(self, data: dict):
        result = {}

        if data is not None:
            for k, v in data.items():
                if k[0] == "_":
                    if isinstance(v, StreamWrapper):
                        data_bytes = v.file_obj.read()
                        v.autoclose()
                        v = data_bytes
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                decoded_key, decoded_data = self.decode1(k, v)
                result[decoded_key] = decoded_data
        return result

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decode input dictionary."""
        return self.decode(data)


def basic_handlers(extension: str, data: StreamWrapper):

    if extension in {".txt", ".text", ".transcript"}:
        data_bytes = data.file_obj.read()
        data.autoclose()
        return data_bytes.decode("utf-8")

    if extension in {".cls", ".cls2", ".class", ".count", ".index", ".inx", ".id"}:
        data_bytes = data.file_obj.read()
        data.autoclose()
        try:
            return int(data_bytes)
        except ValueError:
            return None

    if extension in {".json", ".jsn"}:
        output = json.load(data.file_obj)
        data.autoclose()
        return output

    if extension in {".pyd", ".pickle"}:
        output = pickle.load(data.file_obj)
        data.autoclose()
        return output

    if extension == ".pt":
        output = torch.load(data.file_obj)
        data.autoclose()
        return output

    if extension in {".npy", ".npz"}:
        with BytesIO(data.file_obj.read()) as f:
            data.autoclose()
            output = numpy.load(f, allow_pickle=False)
        return output
    return None


def compression_handler(extension: str, data: StreamWrapper):
    if extension not in [".gzip", ".gz"]:
        return None

    return StreamWrapper(gzip.GzipFile(fileobj=data.file_obj), parent_stream=data)


class ImageHandler:
    def __init__(self, imagespec):
        assert imagespec in list(imagespecs.keys()), "unknown image specification: {}".format(
            imagespec
        )
        self.imagespec = imagespecs[imagespec.lower()]

    def __call__(self, extension: str, data: StreamWrapper):
        if extension.lower() not in {".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".pbm", ".pnm"}:
            return None

        try:
            import numpy as np
        except ImportError as e:
            del e
            raise ModuleNotFoundError(
                "Package `numpy` is required to be installed for default image decoder."
                "Please use `pip install numpy` to install the package"
            )

        try:
            import PIL.Image
        except ImportError as e:
            del e
            raise ModuleNotFoundError(
                "Package `PIL` is required to be installed for default image decoder."
                "Please use `pip install Pillow` to install the package"
            )

        atype, etype, mode = self.imagespec
        img = PIL.Image.open(data.file_obj)
        # TODO: This could be a problem, check if we run into issue with StreamWrapper.
        img.load()
        data.autoclose()
        img = img.convert(mode.upper())
        if atype == "pil":
            return img
        elif atype == "numpy":
            result = np.asarray(img)
            assert (
                result.dtype == np.uint8
            ), "numpy image array should be type uint8, but got {}".format(result.dtype)
            if etype == "uint8":
                return result
            else:
                return result.astype("f") / 255.0
        elif atype == "torch":
            result = np.asarray(img)
            assert (
                result.dtype == np.uint8
            ), "numpy image array should be type uint8, but got {}".format(result.dtype)

            if etype == "uint8":
                result = np.array(result.transpose(2, 0, 1))
                return torch.tensor(result)
            else:
                result = np.array(result.transpose(2, 0, 1))
                return torch.tensor(result) / 255.0
        return None


default_image_handler = ImageHandler("rgb8")

default_decoder = ExtensionBasedDecoder(compression_handler, default_image_handler, basic_handlers)
