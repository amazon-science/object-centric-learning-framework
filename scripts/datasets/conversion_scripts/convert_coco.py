"""Convert a multi-object records dataset into a webdataset."""
import abc
import argparse
import gzip
import io
import json
import logging
import os
import random
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm
import webdataset
from pycocotools.coco import COCO
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)


class Handler(abc.ABC):
    @abc.abstractmethod
    def is_responsible(self, instance: Any) -> bool:
        pass

    @abc.abstractmethod
    def __call__(self, name: str, objs: Any) -> Tuple[str, bytes]:
        pass


class NumpyHandler(Handler):
    def is_responsible(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def __call__(self, name: str, obj: np.ndarray) -> bool:
        with io.BytesIO() as stream:
            np.save(stream, obj)
            return f"{name}.npy", stream.getvalue()


class JsonHandler(Handler):
    def is_responsible(self, obj: Any) -> bool:
        return isinstance(obj, (list, dict))

    def __call__(self, name: str, obj: np.ndarray) -> bool:
        return f"{name}.json", json.dumps(obj).encode("utf-8")


class GzipHandler(Handler):
    def __init__(self, compresslevel=5):
        self.compresslevel = compresslevel

    def is_responsible(self, obj: Any) -> bool:
        # Only compress if file is larger than a block, otherwise compression
        # is useless because the file will anyway use 512 bytes of space.
        return isinstance(obj, bytes) and len(obj) > tarfile.BLOCKSIZE

    def __call__(self, name: str, obj: bytes) -> bool:
        return f"{name}.gz", gzip.compress(obj, compresslevel=self.compresslevel)


DEFAULT_HANDLERS = [NumpyHandler(), JsonHandler(), GzipHandler()]


class AnnotationAggregator:
    """Class to aggregate COCO annotations from multiple COCO tasks."""

    def __init__(
        self, instance_annotation: str, stuff_annotation: str, caption_annotation: str
    ):
        self.instance_annotation = (
            COCO(instance_annotation) if instance_annotation else None
        )
        self.stuff_annotation = COCO(stuff_annotation) if stuff_annotation else None
        self.caption_annotation = (
            COCO(caption_annotation) if caption_annotation else None
        )
        if self.caption_annotation:
            self.image_ids = sorted(list(self.caption_annotation.imgs.keys()))
        if self.instance_annotation:
            self.image_ids = sorted(list(self.instance_annotation.imgs.keys()))
        if self.stuff_annotation:
            self.image_ids = sorted(list(self.stuff_annotation.imgs.keys()))

    def _get_filename(self, image_id):
        if self.caption_annotation:
            return self.caption_annotation.loadImgs(image_id)[0]["file_name"]
        if self.instance_annotation:
            return self.instance_annotation.loadImgs(image_id)[0]["file_name"]
        if self.stuff_annotation:
            return self.stuff_annotation.loadImgs(image_id)[0]["file_name"]
        raise RuntimeError()

    def _get_segmentation_annotations(self, coco_object, prefix, image_id):
        ann_ids = coco_object.getAnnIds(image_id)
        annotations = coco_object.loadAnns(ann_ids)
        masks = [coco_object.annToMask(annotation) for annotation in annotations]
        output = {}
        if len(masks) > 0:
            # Stack to single array and add final dimension for compatibility with clevr dataset.
            output[f"{prefix}mask"] = np.stack(masks, axis=0)[..., None]
            output[f"{prefix}bbox"] = np.array(
                [annotation["bbox"] for annotation in annotations], dtype=np.float32
            )
            output[f"{prefix}category"] = np.array(
                [annotation["category_id"] for annotation in annotations],
                dtype=np.uint8,
            )
            output[f"{prefix}area"] = np.array(
                [annotation["area"] for annotation in annotations], dtype=np.float32
            )
        return output

    def _get_caption_annotations(self, image_id):
        ann_ids = self.caption_annotation.getAnnIds(image_id)
        annotations = self.caption_annotation.loadAnns(ann_ids)
        # Stack to single array and add final dimension for compatibility with clevr dataset.
        return {"caption": [annotation["caption"] for annotation in annotations]}

    def __getitem__(self, image_id) -> Tuple[str, Dict]:
        """Get file name and aggregated annotations for image id."""
        filename = self._get_filename(image_id)
        annotations = {}
        if self.caption_annotation:
            annotations.update(self._get_caption_annotations(image_id))
        if self.instance_annotation:
            annotations.update(
                self._get_segmentation_annotations(
                    self.instance_annotation, "instance_", image_id
                )
            )
        if self.stuff_annotation:
            annotations.update(
                self._get_segmentation_annotations(
                    self.stuff_annotation, "stuff_", image_id
                )
            )

        return filename, annotations


class TestAnnotations:
    def __init__(self, test_annotation: str):
        self.test_annotation = COCO(test_annotation)
        self.image_ids = sorted(list(self.test_annotation.imgs.keys()))

    def _get_filename(self, image_id):
        return self.test_annotation.loadImgs(image_id)[0]["file_name"]

    def __getitem__(self, image_id) -> Tuple[str, Dict]:
        """Get file name and aggregated annotations for image id."""
        filename = self._get_filename(image_id)
        return filename, {}


def convert_to_bytes(name, obj, handlers: List[Handler] = DEFAULT_HANDLERS):
    for handler in handlers:
        if handler.is_responsible(obj):
            name, obj = handler(name, obj)
    return name, obj


def main(
    dataset_path: str,
    output_path: str,
    instance_annotation: str,
    stuff_annotation: str,
    caption_annotation: str,
    test_annotation: str,
    subset_list: Optional[str] = None,
    seed: Optional[int] = None,
):
    if caption_annotation:
        annotator = AnnotationAggregator(
            instance_annotation, stuff_annotation, caption_annotation
        )
    elif test_annotation:
        annotator = TestAnnotations(test_annotation)
    else:
        raise RuntimeError(
            "Either instance, stuff and caption annotations or test annotations must be provided."
        )

    if subset_list:
        with open(subset_list, "r") as f:
            subset_list = set(line.strip() for line in f.readlines())

    # Setup parameters for shard writers.
    shard_writer_params = {
        "maxsize": 50 * 1024 * 1024,  # 50 MB
        "maxcount": 1000,
        "keep_meta": True,
        "encoder": False,
    }

    image_ids = list(annotator.image_ids)  # Make copy.
    print("Number of instances:", len(image_ids))
    random.seed(seed)
    random.shuffle(image_ids)  # Ensure instances are shuffled.

    instance_count = 0
    with webdataset.ShardWriter(
        get_shard_pattern(output_path), **shard_writer_params
    ) as writer:
        for image_id in tqdm.tqdm(image_ids):
            filename, annotations = annotator[image_id]
            if subset_list:
                if filename in subset_list:
                    subset_list.remove(filename)
                else:
                    # Skip samples not in subset list
                    continue

            output = dict(
                [convert_to_bytes(name, obj) for name, obj in annotations.items()]
            )
            output["__key__"] = str(image_id)
            _, ext = os.path.splitext(filename)
            image_path = os.path.join(dataset_path, filename)
            with open(image_path, "rb") as f:
                output[f"image{ext}"] = f.read()
            writer.write(output)
            instance_count += 1
    if subset_list is not None and len(subset_list) != 0:
        # We did not process all images in the list.
        logging.error(
            f"{len(subset_list)} samples in the subset list where not processed"
        )

    logging.info(f"Wrote {instance_count} instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--caption", type=str, default=None)
    parser.add_argument("--stuff", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--subset_list", type=str, default=None)
    parser.add_argument("--seed", type=int, default=23894734)

    args = parser.parse_args()

    main(
        args.dataset_path,
        args.output_path,
        args.instance,
        args.stuff,
        args.caption,
        args.test,
        args.subset_list,
        args.seed,
    )
