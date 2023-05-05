import argparse
import gzip
import io
import json
import logging
import os

import numpy as np
import tqdm
import webdataset
from pycocotools import _mask as coco_mask
from utils import ContextList, FakeIndices, make_subdirs_and_patterns

logging.getLogger().setLevel(logging.INFO)


def get_numpy_compressed_bytes(np_array):
    """Convert into a serialized numpy array."""
    with io.BytesIO() as stream:
        np.save(stream, np_array)
        return gzip.compress(stream.getvalue(), compresslevel=5)


def get_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def decode2binarymask(masks):
    mask = coco_mask.decode(masks)
    binary_masks = mask.astype("bool")  # (320,480,128)
    binary_masks = binary_masks.transpose(2, 0, 1)
    return binary_masks


def compress_mask(mask):
    non_empty = np.any(mask != 0, axis=(0, 2, 3))
    # Preserve first object being empty. This is often considered the
    # foreground mask and sometimes ignored.
    last_nonempty_index = len(non_empty) - non_empty[::-1].argmax()
    input_arr = mask[:, :last_nonempty_index]
    n_objects = input_arr.shape[1]
    dtype = np.uint8
    if n_objects > 8:
        dtype = np.uint16
    if n_objects > 16:
        dtype = np.uint32
    if n_objects > 32:
        dtype = np.uint64
    if n_objects > 64:
        raise RuntimeError("We do not support more than 64 objects at the moment.")

    object_flag = (1 << np.arange(n_objects, dtype=dtype))[None, :, None, None]
    output_arr = np.sum(input_arr.astype(dtype) * object_flag, axis=1).astype(dtype)
    return output_arr


def main(
    split,
    dataset_path,
    output_path,
    max_number_of_objects,
):
    # Get tf dataset.
    split_names = [split]

    # list all the folders
    vm_path = os.path.join(dataset_path, split + "_data", split)
    flow_path = os.path.join(dataset_path, split + "_flow")
    fm_path = os.path.join(dataset_path, split + "_data", split + "_objects")
    elements_in_fm = list(os.listdir(fm_path))
    video_dirs = [
        d for d in os.listdir(vm_path) if os.path.isdir(os.path.join(vm_path, d))
    ]

    # too large, use a small subset to train first
    # video_dirs = video_dirs[:200]

    # Setup parameters for shard writers.
    split_indices = {split_name: FakeIndices() for split_name in split_names}
    patterns, _ = make_subdirs_and_patterns(output_path, split_indices)

    shard_writer_params = {
        "maxsize": 100 * 1024 * 1024,  # 100 MB
        "maxcount": 50,
        "keep_meta": True,
    }

    # Create shards of the data.
    valid_count = 0
    with ContextList(
        webdataset.ShardWriter(p, **shard_writer_params) for p in patterns
    ) as writers:
        for split_idx, split_name in enumerate(split_names):
            for _index, scene_dir in tqdm.tqdm(
                enumerate(video_dirs), total=len(video_dirs)
            ):
                print(scene_dir)
                writer = writers[split_idx]
                # read all images

                # init masks
                n_frames, h, w = 128, 2 * 160, 2 * 240
                fm_array = np.zeros([n_frames, max_number_of_objects, h, w])
                vm_array = np.zeros([n_frames, max_number_of_objects, h, w])

                # read fm and create object map
                id_to_channel_map = {}
                num_obj = 0
                for obj in elements_in_fm:
                    if obj.startswith(scene_dir + "_object"):
                        # Note: object start from channel 1 since in oc-codebase
                        num_obj += 1
                        obj_data = json.load(
                            open(os.path.join(fm_path, obj, "objects.json"))
                        )
                        fish_id = obj_data[0]["id"]
                        id_to_channel_map[fish_id] = num_obj
                        # full masks
                        binary_masks = decode2binarymask(obj_data[0]["masks"])
                        fm_array[:, num_obj, :, :] = binary_masks

                        # ignore objects >  max_num
                        if num_obj >= max_number_of_objects:
                            break

                visible_data = json.load(
                    open(os.path.join(vm_path, scene_dir, "objects.json"))
                )

                for i in range(len(visible_data)):
                    fish_id = visible_data[i]["id"]
                    visible_bms = decode2binarymask(visible_data[i]["masks"])
                    obj_channel = id_to_channel_map[fish_id]
                    vm_array[:, obj_channel, :, :] = visible_bms

                output = {
                    "video.mp4": get_bytes(
                        os.path.join(vm_path, scene_dir, "video.mp4")
                    ),
                    "vm_mask.npy.gz": get_numpy_compressed_bytes(
                        compress_mask(vm_array)
                    ),
                    "fm_mask.npy.gz": get_numpy_compressed_bytes(
                        compress_mask(fm_array)
                    ),
                    "flow.mp4": get_bytes(
                        os.path.join(flow_path, f"{scene_dir}.flow.mp4")
                    ),
                }
                output["__key__"] = scene_dir
                writer.write(output)

            print(valid_count)

            logging.info(f"Wrote {_index} instances to {split_name} split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_number_of_objects", type=int, default=None)

    args = parser.parse_args()

    main(
        args.split,
        args.dataset_path,
        args.output_path,
        args.max_number_of_objects,
    )
