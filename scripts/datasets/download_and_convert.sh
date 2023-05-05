#!/bin/bash

case $1 in
  COCO)
    echo "Downloading COCO2017 and COCO20k data to data/coco"
    # ./download_scripts/download_coco_data.sh

    echo "Converting COCO2017 to webdataset stored at outputs/coco2017"
    SEED=23894734
    mkdir -p outputs/coco2017/train
    python conversion_scripts/convert_coco.py data/coco/train2017 outputs/coco2017/train --instance data/coco/annotations/instances_train2017.json --stuff data/coco/annotations/stuff_train2017.json --caption data/coco/annotations/captions_train2017.json --seed $SEED
    mkdir -p outputs/coco2017/val
    python conversion_scripts/convert_coco.py data/coco/val2017 outputs/coco2017/val --instance data/coco/annotations/instances_val2017.json --stuff data/coco/annotations/stuff_val2017.json --caption data/coco/annotations/captions_val2017.json --seed $SEED
    mkdir -p outputs/coco2017/test
    python conversion_scripts/convert_coco.py data/coco/test2017 outputs/coco2017/test --test data/coco/annotations/image_info_test2017.json --seed $SEED
    mkdir -p outputs/coco2017/unlabeled
    python conversion_scripts/convert_coco.py data/coco/unlabeled2017 outputs/coco2017/unlabeled --test data/coco/annotations/image_info_unlabeled2017.json --seed $SEED

    echo "Converting COCO20k to webdataset stored at outputs/coco2014/20k"
    mkdir -p outputs/coco2014/20k
    python conversion_scripts/convert_coco.py data/coco/train2014 outputs/coco2014/20k --instance data/coco/annotations/instances_train2014.json --caption data/coco/annotations/captions_train2014.json --seed $SEED --subset_list misc/coco20k_files.txt
    ;;


  clevr+cater)
    echo "Downloading clevr and cater data to data/multi-object-datasets"
    mkdir -p data/multi-object-datasets/clevr_with_masks
    gsutil -m rsync -r gs://multi-object-datasets/clevr_with_masks data/multi-object-datasets/clevr_with_masks
    mkdir -p data/multi-object-datasets/cater_with_masks
    gsutil -m rsync -r gs://multi-object-datasets/cater_with_masks data/multi-object-datasets/cater_with_masks
    export CUDA_VISIBLE_DEVICES=""
    export TF_FORCE_GPU_ALLOW_GROWTH=True
    SEED=837452923

    echo "Converting clevr to webdataset stored at outputs/clevr_with_masks"
    python conversion_scripts/convert_tfrecords.py clevr_with_masks data/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords outputs/clevr_with_masks --split_names train val test --split_ratios 0.8 0.1 0.1 --n_instances 100000 --seed $SEED
    python conversion_scripts/convert_tfrecords.py clevr_with_masks data/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords outputs/clevr_with_masks --split_names train val test --split_ratios 0.7 0.15 0.15 --n_instances 100000 --seed $SEED

    echo "Converting cater to webdataset stored at outputs/cater_with_masks"
    python conversion_scripts/convert_tfrecords.py cater_with_masks "data/multi-object-datasets/cater_with_masks/cater_with_masks_train.tfrecords-*-of-*" outputs/cater_with_masks --split_names train val --split_ratios 0.9 0.1 --n_instances 39364 --seed $SEED
    python conversion_scripts/convert_tfrecords.py cater_with_masks "data/multi-object-datasets/cater_with_masks/cater_with_masks_test.tfrecords-*-of-*" outputs/cater_with_masks/test --n_instances 17100 --seed $SEED
    ;;


  clevrer)
    echo "Downloading clevrer data"
    ./download_scripts/download_clevrer_data.sh

    echo "Converting clevrer to webdataset stored at outputs/clevrer"
    python conversion_scripts/convert_clevrer.py --video_dir="data/clevrer/video_train/" --annotation_dir="data/clevrer/annotation_train/" --output_dir="outputs/clevrer/train/"
    python conversion_scripts/convert_clevrer.py --video_dir="data/clevrer/video_validation/" --annotation_dir="data/clevrer/annotation_validation/" --output_dir="outputs/clevrer/validation/"
    ;;


  voc2007)
    echo "Creating voc2007 webdataset in outputs/voc2007"
    # Ensure downloaded data is stored in data folder.
    export TFDS_DATA_DIR=data/tensorflow_datasets
    mkdir -p outputs/voc2007/train
    python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation train outputs/voc2007/train
    mkdir -p outputs/voc2007/val
    python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation validation outputs/voc2007/val
    mkdir -p outputs/voc2007/test
    python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation test outputs/voc2007/test
    ;;


  voc2012)
    # Augmented pascal voc dataset with segmentations and additional instances.
    echo "Creating voc2012 webdataset in outputs/voc2012"
    # Ensure downloaded data is stored in data folder.
    export TFDS_DATA_DIR=data/tensorflow_datasets
    mkdir -p outputs/voc2012/trainaug
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation train+sbd_train+sbd_validation outputs/voc2012/trainaug
    # Regular pascal voc splits.
    mkdir -p outputs/voc2012/train
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation train outputs/voc2012/train
    mkdir -p outputs/voc2012/val
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation validation outputs/voc2012/val
    ;;


  movi_c)
    echo "Creating movi_c webdataset in outputs/movi_c"
    mkdir -p outputs/movi_c
    mkdir -p outputs/movi_c/train
    python conversion_scripts/convert_tfds.py movi_c/128x128:1.0.0 train outputs/movi_c/train --dataset_path gs://kubric-public/tfds
    mkdir -p outputs/movi_c/val
    python conversion_scripts/convert_tfds.py movi_c/128x128:1.0.0 validation outputs/movi_c/val --dataset_path gs://kubric-public/tfds
    mkdir -p outputs/movi_c/test
    python conversion_scripts/convert_tfds.py movi_c/128x128:1.0.0 test outputs/movi_c/test --dataset_path gs://kubric-public/tfds
    ;;


  movi_e)
    echo "Creating movi_e webdataset in outputs/movi_e"
    mkdir -p outputs/movi_e
    mkdir -p outputs/movi_e/train
    python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 train outputs/movi_e/train --dataset_path gs://kubric-public/tfds
    mkdir -p outputs/movi_e/val
    python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 validation outputs/movi_e/val --dataset_path gs://kubric-public/tfds
    mkdir -p outputs/movi_e/test
    python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 test outputs/movi_e/test --dataset_path gs://kubric-public/tfds
    ;;


  *)
    echo "Unknown dataset $1"
    echo "Only COCO, clevr+cater, clevrer, voc2007, voc2012, movi_c and movi_e are supported."
    ;;
esac
