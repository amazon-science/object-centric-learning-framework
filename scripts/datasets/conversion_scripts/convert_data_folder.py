"""Convert a dataset folder into a webdataset.

Folder structure for dataset_path:
dataset_path/base_name_a_0000000.suffix
dataset_path/base_name_b_0000000.suffix
dataset_path/base_name_c_0000000.suffix
dataset_path/base_name_a_0000001.suffix
dataset_path/base_name_b_0000001.suffix
dataset_path/base_name_c_0000001.suffix
...

"""


import argparse
import logging
import os
from typing import Dict, Optional

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


def determine_dataset_length(base_names, files):
    """Determine the length of a data folder dataset."""
    n_files = len(files)
    n_base_names = len(base_names)
    if not n_files / n_base_names == n_files // n_base_names:
        raise ValueError("Either some files are missing or non-related files in path.")
    return n_files // n_base_names


def parse_datafolder(dataset_path):
    """Assumed file name structure, for example: base_name_0000000.npy.gz."""
    files = os.listdir(dataset_path)
    base_names = set(
        [("_".join(f.split("_")[:-1]), ".".join(f.split(".")[1:])) for f in files]
    )
    indices = set([int(f.split(".")[0].split("_")[-1]) for f in files])
    return base_names, indices, files


def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def main(dataset_path, output_path, splits=None, n_instances=None, seed=423234):
    base_names, data_indices, files = parse_datafolder(dataset_path)

    if splits:
        # Need dataset length for computing splits.
        if n_instances:
            dataset_length = n_instances
        else:
            dataset_length = determine_dataset_length(base_names, files)
        split_indices = create_split_indices(dataset_length, splits, seed)
        patterns, list_of_indices = make_subdirs_and_patterns(
            output_path, split_indices
        )
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
    with ContextList(
        webdataset.ShardWriter(p, **shard_writer_params) for p in patterns
    ) as writers:
        for index, instance_idx in tqdm.tqdm(
            enumerate(data_indices), total=dataset_length
        ):
            instance = {
                f"{base_name[0]}.{base_name[1]}": read_file(
                    os.path.join(
                        dataset_path,
                        f"{base_name[0]}_{instance_idx:07d}.{base_name[1]}",
                    )
                )
                for base_name in base_names
            }
            if dataset_length and index >= dataset_length:
                logging.warn(
                    "Stopped itterating over dataset due to dataset_length. "
                    "This indicates that not all samples where processed."
                )
                break
            writer_index = get_index(index, list_of_indices)
            writer = writers[writer_index]
            output = instance
            output["__key__"] = str(index)
            writer.write(output)
            instance_count += 1
    logging.info(f"Wrote {instance_count} instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        args.dataset_path,
        args.output_path,
        splits,
        n_instances=args.n_instances,
        seed=args.seed,
    )
