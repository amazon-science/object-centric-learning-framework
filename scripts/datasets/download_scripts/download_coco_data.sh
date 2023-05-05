mkdir -p data/coco
# COCO 2017
# Google storage hosted files seem to be down, use http instead.
set -x
# # Data
wget -qO- http://images.cocodataset.org/zips/train2017.zip | bsdtar -xf- -C data/coco
wget -qO- http://images.cocodataset.org/zips/val2017.zip | bsdtar -xf- -C data/coco
wget -qO- http://images.cocodataset.org/zips/test2017.zip | bsdtar -xf- -C data/coco
wget -qO- http://images.cocodataset.org/zips/unlabeled2017.zip | bsdtar -xf- -C data/coco

# Annotations
wget -qO- http://images.cocodataset.org/annotations/annotations_trainval2017.zip | bsdtar -xf- -C data/coco
wget -qO- http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip | bsdtar -xf- -C data/coco
wget -qO- http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip | bsdtar -xf- -C data/coco
# Test annotations
wget -qO- http://images.cocodataset.org/annotations/image_info_test2017.zip | bsdtar -xf- -C data/coco
# Unlabeled annotations
wget -qO- http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip | bsdtar -xf- -C data/coco

# COCO 2014, only train for generating 20k dataset
wget -qO- http://images.cocodataset.org/zips/train2014.zip | bsdtar -xf- -C data/coco
wget -qO- http://images.cocodataset.org/annotations/annotations_trainval2014.zip | bsdtar -xf- -C data/coco
