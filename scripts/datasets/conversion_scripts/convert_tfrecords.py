"""Convert a multi-object records dataset into a webdataset."""
import argparse
import gzip
import importlib
import io
import logging
import os
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
import tqdm
import webdataset
from utils import (
    ContextList,
    FakeIndices,
    create_split_indices,
    get_index,
    get_shard_pattern,
    make_subdirs_and_patterns,
)

logging.getLogger().setLevel(logging.INFO)


def determine_dataset_length(records, compression_type=None):
    """Determine the length of a tensorflow records dataset dataset.

    Avoids decoding the content for efficiency reasons.
    """
    dataset = tf.data.TFRecordDataset(records, compression_type=compression_type)
    return sum(
        1 for _ in tqdm.tqdm(dataset, desc="Reading dataset for length estimation", unit="instances")
    )


def tftensor_to_numpy_compressed_bytes(tftensor):
    """Convert a tensorflow tensor into a serialized numpy array."""
    with io.BytesIO() as stream:
        np.save(stream, tftensor.numpy())
        return gzip.compress(stream.getvalue(), compresslevel=5)


def main(
    input_dataset_name,
    dataset_path,
    output_path,
    splits=None,
    n_instances=None,
    seed=423234,
):
    dataset_module = importlib.import_module("multi_object_datasets." + input_dataset_name)
    records = tf.io.matching_files(dataset_path)
    if len(records) == 0:
        logging.error(f"No files matching {dataset_path}")
        return 1
    d = dataset_module.dataset(records)

    if splits:
        # Need dataset length for computing splits.
        if n_instances:
            dataset_length = n_instances
        else:
            dataset_length = determine_dataset_length(records, dataset_module.COMPRESSION_TYPE)
        split_indices = create_split_indices(dataset_length, splits, seed)
        patterns, list_of_indices = make_subdirs_and_patterns(output_path, split_indices)
        for split_name, indices in split_indices.items():
            logging.info(f"Split {split_name} will contain {len(indices)} instances.")
    else:
        # Same as above yet with a single split containing all the indices.
        patterns = [get_shard_pattern(output_path)]
        os.makedirs(output_path, exist_ok=True)
        list_of_indices = [FakeIndices()]
        dataset_length = None

    # Setup parameters for shard writers.
    shard_writer_params = {
        "maxsize": 100 * 1024 * 1024,  # 100 MB
        "maxcount": 5000,
        "keep_meta": True,
    }

    instance_count = 0
    # Create shards of data.
    with ContextList(webdataset.ShardWriter(p, **shard_writer_params) for p in patterns) as writers:
        for index, instance in tqdm.tqdm(enumerate(d), total=dataset_length):
            if dataset_length and index >= dataset_length:
                logging.warn(
                    "Stopped itterating over dataset due to dataset_length. This indicates that "
                    "not all samples where processed."
                )
                break
            writer_index = get_index(index, list_of_indices)
            writer = writers[writer_index]
            output = {
                name + ".npy.gz": tftensor_to_numpy_compressed_bytes(tensor)
                for name, tensor in instance.items()
            }
            output["__key__"] = str(index)
            writer.write(output)
            instance_count += 1

    logging.info(f"Wrote {instance_count} instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", choices=["clevr_with_masks", "cater_with_masks"])
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--split_names", nargs="+", type=str, default=None)
    parser.add_argument("--split_ratios", nargs="+", type=float, default=None)
    parser.add_argument("--n_instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    splits: Optional[Dict]

    if args.split_names and args.split_ratios:
        assert len(args.split_names) == len(args.split_ratios)
        assert sum(args.split_ratios) == 1.0
        splits = dict(zip(args.split_names, args.split_ratios))
    else:
        splits = None

    main(
        args.dataset_name,
        args.dataset_path,
        args.output_path,
        splits,
        n_instances=args.n_instances,
        seed=args.seed,
    )
