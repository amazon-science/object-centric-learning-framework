# Dataset configurations

A dataset is expected to be a pytorch lightning `LightningDataModule`, where
the constructor at least accepts the following parameters which are used in the
experiment config tests.

 - `batch_size: int`
 - `num_workers: int`
 - `shuffle_buffer_size: int`

Check out the [datasets section][ocl.datasets] of the api for some examples on
how a dataset can be implemented.

## Composition of datasets

Dataset configurations can be composed, such that it is straight forward to create
derived versions of datasets for example by sampling images from a video or by
filtering out some instances.  This is possible as transforms are stored in
dictionaries and thus can be composed in hydra.

Check out [configs/dataset/clevr6.yaml][configsdatasetclevr6yaml] and
[configs/dataset/movi_c_image.yaml][configsdatasetmovi_c_imageyaml] for some examples.
