import argparse
import gzip
import io
import logging
import os
from typing import Any, Dict

import imageio
import numpy as np
import torch
import tqdm
import webdataset
from torchvision.ops import masks_to_boxes
from utils import ContextList, FakeIndices, make_subdirs_and_patterns

logging.getLogger().setLevel(logging.INFO)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class AddBBoxFromInstanceMasks:
    """Convert instance mask to bounding box.

    Args:
        instance_mask_key: mask key name.
        target_key: target key name.
    """

    def __init__(
        self,
        instance_mask_key: str = "mask",
        video_id_key: str = "__key__",  # not quite sure if this is the best key
        target_box_key: str = "instance_bbox",
        target_cls_key: str = "instance_cls",
        target_id_key: str = "instance_id",
    ):
        self.instance_mask_key = instance_mask_key
        self.video_id_key = video_id_key
        self.target_box_key = target_box_key
        self.target_cls_key = target_cls_key
        self.target_id_key = target_id_key

    @staticmethod
    def convert(instance_mask: np.ndarray, video_id: np.ndarray) -> np.ndarray:
        num_frame, num_instance, height, width, _ = instance_mask.shape

        # Convert to binary mask
        binary_mask = instance_mask > 0
        # Filter background. TODO: now we assume the first mask for each video is background.
        # Might not apply to every dataset
        binary_mask = binary_mask[:, 1:]
        num_instance -= 1
        binary_mask = (
            torch.tensor(binary_mask)
            .squeeze()
            .view(num_frame * num_instance, height, width)
        )
        # Filter empty masks
        non_empty_mask_idx = torch.where(binary_mask.sum(-1).sum(-1) > 0)[0]
        empty_mask_idx = torch.where(binary_mask.sum(-1).sum(-1) == 0)[0]
        non_empty_binary_mask = binary_mask[non_empty_mask_idx]
        non_empty_bboxes = masks_to_boxes(non_empty_binary_mask)

        # Turn box into cxcyhw
        bboxes = torch.zeros(num_frame * num_instance, 4)
        non_empty_bboxes = box_xyxy_to_cxcywh(non_empty_bboxes)
        bboxes[non_empty_mask_idx] = non_empty_bboxes
        # normalized to 0,1
        # Make sure width and height are correct
        bboxes[:, 0::2] = bboxes[:, 0::2] / width
        bboxes[:, 1::2] = bboxes[:, 1::2] / height
        bboxes = bboxes.view(num_frame, num_instance, 4).squeeze(-1).to(torch.float32)

        # class
        # -1 is background or no object, 0 is the first object class
        instance_cls = torch.ones(num_frame * num_instance, 1) * -1
        instance_cls[non_empty_mask_idx] = 0
        instance_cls = (
            instance_cls.view(num_frame, num_instance, 1).squeeze(-1).to(torch.long)
        )

        # ID
        instance_id = torch.range(0, num_instance - 1)[None, :, None].repeat(
            num_frame, 1, 1
        )
        instance_id = instance_id.view(num_frame * num_instance, 1)
        instance_id[empty_mask_idx] = -1
        instance_id = (
            instance_id.view(num_frame, num_instance, 1).squeeze(-1).to(torch.long)
        )

        return bboxes, instance_cls, instance_id

    def __call__(self, data: Dict[str, Any]):
        if self.instance_mask_key not in data:
            return data

        bboxes, instance_cls, instance_id = self.convert(
            data[self.instance_mask_key], data[self.video_id_key]
        )
        data[self.target_box_key] = bboxes
        data[self.target_cls_key] = instance_cls
        data[self.target_id_key] = instance_id
        return data


def get_numpy_compressed_bytes(np_array):
    """Convert into a serialized numpy array."""
    with io.BytesIO() as stream:
        np.save(stream, np_array)
        return gzip.compress(stream.getvalue(), compresslevel=5)


def main(
    dataset_path,
    output_path,
    max_number_of_objects,
):
    # list all the folders
    dir_list = []
    device_dir_list = [
        os.path.join(dataset_path, d)
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    for subfolder in device_dir_list:
        dir_list.extend(
            [
                os.path.join(subfolder, d)
                for d in os.listdir(subfolder)
                if os.path.isdir(os.path.join(subfolder, d))
            ]
        )

    # Get tf dataset.
    split_names = ["val"]

    # Setup parameters for shard writers.
    split_indices = {split_name: FakeIndices() for split_name in split_names}
    patterns, _ = make_subdirs_and_patterns(output_path, split_indices)

    shard_writer_params = {
        "maxsize": 5 * 1024 * 1024,  # 5 MB
        "maxcount": 50,
        "keep_meta": True,
    }

    uniq_val = None
    # Create shards of the data.
    valid_count = 0
    with ContextList(
        webdataset.ShardWriter(p, **shard_writer_params) for p in patterns
    ) as writers:
        for split_idx, split_name in enumerate(split_names):
            for index, scene_dir in tqdm.tqdm(enumerate(dir_list), total=len(dir_list)):
                writer = writers[split_idx]
                # read all images
                filelist = [
                    val
                    for val in os.listdir(scene_dir)
                    if (val.startswith("rgba") and val.endswith(".png"))
                ]
                filelist = sorted(
                    filelist,
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                    reverse=False,
                )
                # print (scene_dir, filelist)
                if len(filelist) == 0:
                    continue
                valid_count += 1

                video_numpy = np.vstack(
                    [
                        np.expand_dims(
                            imageio.imread(os.path.join(scene_dir, filename))[:, :, :3],
                            axis=0,
                        )
                        for filename in filelist
                    ]
                )

                # read_all_mask
                filelist = [
                    val
                    for val in os.listdir(scene_dir)
                    if val.startswith("segmentation")
                ]
                filelist = sorted(
                    filelist,
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                    reverse=False,
                )
                # converting rgb segmentation to one-hot masks
                # add 1 since cater consider the first channel is background
                one_hot_masks = np.zeros(
                    [
                        len(filelist),
                        max_number_of_objects + 1,
                        video_numpy.shape[-3],
                        video_numpy.shape[-2],
                    ],
                    dtype=np.uint8,
                )
                for fidx, f in enumerate(filelist):
                    mask_img = np.array(
                        imageio.imread(os.path.join(scene_dir, f))
                    ).astype(np.int64)
                    uniq_rgb_mask = (
                        mask_img[:, :, 0] * 256 * 256
                        + mask_img[:, :, 1] * 256
                        + mask_img[:, :, 2]
                    )
                    # We assume the color means ID and won't change
                    if uniq_val is None:
                        uniq_val = np.sort(np.unique(uniq_rgb_mask))[1:]
                    for i in range(uniq_val.shape[0]):
                        one_hot_masks[fidx, i + 1] = (
                            uniq_rgb_mask == uniq_val[i]
                        ).astype(np.uint8)

                one_hot_masks = np.expand_dims(one_hot_masks, axis=-1)
                # print (one_hot_masks.shape)
                output = {
                    "image.npy.gz": get_numpy_compressed_bytes(video_numpy),
                    "mask.npy.gz": get_numpy_compressed_bytes(one_hot_masks),
                }
                output["__key__"] = str(index)
                writer.write(output)

            print(valid_count)

            logging.info(f"Wrote {index} instances to {split_name} split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_number_of_objects", type=int, default=None)

    args = parser.parse_args()

    main(
        args.dataset_path,
        args.output_path,
        args.max_number_of_objects,
    )
