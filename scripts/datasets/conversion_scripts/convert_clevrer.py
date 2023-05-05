"""Convert clevrer data to webdataset.

Folder structure for videos:
video_dir/video_subdir/video_10001.mp4
video_dir/video_subdir/video_10002.mp4
video_dir/video_subdir/video_10003.mp4
...

Folder structure for annotations:
annotation_dir/annotation_subdir/annotation_10001.json
annotation_dir/annotation_subdir/annotation_10002.json
annotation_dir/annotation_subdir/annotation_10003.json
...
"""

import argparse
import logging
import os

import webdataset
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)


def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def main(video_dir, annotation_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # list of all filenames in videos directory
    video_fnames = []
    for video_subdir in os.listdir(video_dir):
        # catch hidden subdirs
        if not video_subdir.startswith("."):
            for video_fname in os.listdir(os.path.join(video_dir, video_subdir)):
                # store full paths of video files
                video_fnames.append(os.path.join(video_dir, video_subdir, video_fname))
    # list of all filenames in annotations directory
    annotation_fnames = []
    for annotation_subdir in os.listdir(annotation_dir):
        # catch hidden subdirs
        if not annotation_subdir.startswith("."):
            for annotation_fname in os.listdir(os.path.join(annotation_dir, annotation_subdir)):
                # store full paths of annotation files
                annotation_fnames.append(
                    os.path.join(annotation_dir, annotation_subdir, annotation_fname)
                )

    # Check if terminal part of video and annotation filenames (basically the IDs) match in
    # both lists
    # video_fname format -> ../../.../../video_10000.mp4 -> ID: 10000
    video_ids = [int(video_fname.split("/")[-1][6:11]) for video_fname in video_fnames]
    # annotation_fname format -> ../../.../../annotation_10000.json -> ID: 10000
    annotation_ids = [
        int(annotation_fname.split("/")[-1][11:16]) for annotation_fname in annotation_fnames
    ]
    # Sort based on Terminal IDs for videos & annotations
    sorted_v_ids_idxs = sorted(range(len(video_ids)), key=lambda k: video_ids[k])
    sorted_a_ids_idxs = sorted(range(len(annotation_ids)), key=lambda k: annotation_ids[k])
    # Permute elements of both lists using sorted idxs
    sorted_video_fnames = [video_fnames[sorted_v_ids_idx] for sorted_v_ids_idx in sorted_v_ids_idxs]
    sorted_annotation_fnames = [
        annotation_fnames[sorted_a_ids_idx] for sorted_a_ids_idx in sorted_a_ids_idxs
    ]
    sorted_video_ids = [int(video_fname.split("/")[-1][6:11]) for video_fname in sorted_video_fnames]
    sorted_annotation_ids = [
        int(annotation_fname.split("/")[-1][11:16]) for annotation_fname in sorted_annotation_fnames
    ]
    assert sorted_video_ids == sorted_annotation_ids
    # Setup parameters for shard writers.
    shard_writer_params = {
        "maxsize": 100 * 1024 * 1024,  # 100 MB
        "maxcount": 5000,
        "keep_meta": True,
    }
    instance_count = 0
    # Create shards of data.
    with webdataset.ShardWriter(get_shard_pattern(output_dir), **shard_writer_params) as writer:
        for index, (video_file, json_file) in enumerate(
            zip(sorted_video_fnames, sorted_annotation_fnames)
        ):
            output = {}
            output["__key__"] = str(index)
            output["video.mp4"] = read_file(video_file)
            output["annotation.json"] = read_file(json_file)
            writer.write(output)
            instance_count += 1
    logging.info(f"Wrote {instance_count} instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    main(args.video_dir, args.annotation_dir, args.output_dir)
