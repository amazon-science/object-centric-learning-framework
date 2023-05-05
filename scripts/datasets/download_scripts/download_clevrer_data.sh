# Make base directory for storing clevrer dataset
mkdir -p data/clevrer

# Download training/validation set videos hosted on MIT CSAIL project page
# training split videos
mkdir -p data/clevrer/video_train
wget -qO- http://data.csail.mit.edu/clevrer/videos/train/video_train.zip | bsdtar -xf- -C data/clevrer/video_train/
# validation split videos
mkdir -p data/clevrer/video_validation
wget -qO- http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip | bsdtar -xf- -C data/clevrer/video_validation/

# Download training/validation set annotations of videos hosted on MIT CSAIL project page
# training split annotations
mkdir -p data/clevrer/annotation_train
wget -qO- http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip | bsdtar -xf- -C data/clevrer/annotation_train/
# validation split annotations
mkdir -p data/clevrer/annotation_validation
wget -qO- http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip | bsdtar -xf- -C data/clevrer/annotation_validation/
