"""Convert a multi-object records dataset into a webdataset."""
import argparse
import collections
import gzip
import io
import json
import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tfds_extended_voc  # noqa: F401
import tqdm
import webdataset
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nested_convert_to_numpy(d):
    if isinstance(d, dict):
        out = {}
        for name, val in d.items():
            res = nested_convert_to_numpy(val)
            if res is None:
                continue
            out[name] = res
        return out
    if isinstance(d, tf.Tensor):
        return d.numpy()
    if isinstance(d, tf.RaggedTensor):
        # Currently not supported.
        return None
    raise ValueError(f"Unsupported type {type(d)}")


def nested_convert_to_python(d):
    if isinstance(d, dict):
        out = {}
        for name, val in d.items():
            res = nested_convert_to_python(val)
            if res is None:
                continue
            out[name] = res
        return out
    if isinstance(d, tf.Tensor):
        if tf.rank(d) == 0:
            element = d.numpy()
            if d.dtype == tf.string:
                element = element.decode("utf-8")
            else:
                element = element.item()
            return element
        else:
            elements = d.numpy().tolist()
            if d.dtype == tf.string:
                elements = [el.decode("utf-8") for el in elements]
            return elements

    if isinstance(d, tf.RaggedTensor):
        return [el.tolist() for el in d.numpy()]
    raise ValueError(f"Unsupported type {type(d)}")


def serialize_sequeuce(data, info):
    if isinstance(info, (tfds.features.Image, tfds.features.BBoxFeature)):
        with io.BytesIO() as stream:
            np.save(stream, data.numpy())
            image_bytes = gzip.compress(stream.getvalue(), compresslevel=5)
        return "npy.gz", image_bytes
    elif info == tf.dtypes.string or isinstance(info, tfds.features.Text):
        return "json", json.dumps(data.numpy().tolist()).encode("utf-8")
    elif info == tf.dtypes.int64 or isinstance(info, tfds.features.ClassLabel):
        return "json", json.dumps(data.numpy().tolist()).encode("utf-8")
    elif isinstance(info, tfds.features.Tensor):
        with io.BytesIO() as stream:
            np.save(stream, data.numpy())
            tensor_bytes = gzip.compress(stream.getvalue(), compresslevel=5)
        return "npy.gz", tensor_bytes
    else:
        raise ValueError


def serialize_dict(data):
    numpy_dict = nested_convert_to_numpy(data)
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **numpy_dict)
        dict_bytes = stream.getvalue()
    return "npz", dict_bytes


def serialize_instance(instance, info):
    """Simple but not entirely flexible routine to convert tfds outputs to webdataset outputs.

    This needs more work if we want it to cover all possible cases. Currently it should work on the
    object detection datasets of tfds.
    """
    output_dict = {}
    for name, data in instance.items():
        # Our dict elements correspond to file names, thus we need to replace some characters.
        cur_info = info[name]
        if isinstance(cur_info, tfds.features.Sequence):
            nested_info = cur_info.feature
            if isinstance(nested_info, tfds.features.FeaturesDict):
                extension, serialized = serialize_dict(data)
                output_dict[f"{name}.{extension}"] = serialized
            else:
                try:
                    extension, output_bytes = serialize_sequeuce(data, nested_info)
                    output_dict[f"{name}.{extension}"] = output_bytes
                except ValueError:
                    pass
        elif isinstance(cur_info, tfds.features.FeaturesDict):
            numpy_data = nested_convert_to_python(data)
            json_bytes = json.dumps(numpy_data).encode("utf-8")
            output_dict[f"{name}.json"] = json_bytes
        elif isinstance(cur_info, (tfds.features.Image, tfds.features.BBoxFeature)):
            with io.BytesIO() as stream:
                np.save(stream, data.numpy())
                image_bytes = gzip.compress(stream.getvalue(), compresslevel=5)
            output_dict[f"{name}.npy.gz"] = image_bytes
        elif cur_info == tf.string or isinstance(cur_info, tfds.features.Text):
            output_dict[f"{name}.txt"] = data.numpy()
        elif cur_info == tf.int64 or isinstance(cur_info, tfds.features.ClassLabel):
            output_dict[f"{name}.cls"] = str(data.numpy())
        else:
            logging.debug(f"Unable to convert field {name}.")
    return output_dict


def main(
    tfds_spec,
    split_spec,
    output_path,
    dataset_path=None,
):
    dataset, info = tfds.load(tfds_spec, split=split_spec, with_info=True, data_dir=dataset_path)
    shard_writer_params = {
        "maxsize": 100 * 1024 * 1024,  # 100 MB
        "maxcount": 5000,
        "keep_meta": True,
        "encoder": False,
    }
    feature_info = info.features

    with webdataset.ShardWriter(get_shard_pattern(output_path), **shard_writer_params) as writer:
        for i, instance in tqdm.tqdm(
            enumerate(dataset), total=len(dataset), desc=f"Processing {split_spec}"
        ):
            serialized = serialize_instance(instance, feature_info)
            flattened = flatten_dict(serialized, sep=".")
            flattened["__key__"] = f"{i:07d}"
            for key in list(flattened.keys()):
                if "/" in key:
                    flattened[key.replace("/", "-")] = flattened[key]
                    del flattened[key]

            writer.write(flattened)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tfds_dataset_spec", type=str)
    parser.add_argument("tfds_split_spec", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--dataset_path", type=str, required=False, default=None)
    args = parser.parse_args()
    main(
        args.tfds_dataset_spec,
        args.tfds_split_spec,
        args.output_path,
        dataset_path=args.dataset_path,
    )
