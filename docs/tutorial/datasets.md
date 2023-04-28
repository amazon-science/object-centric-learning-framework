# Datasets

Most datasets are not suited for large-scale multi-GPU and multi-node training
by default, as they are either composed of many small files.  In OCLF datasets
are generally stored in the [webdataset format](https://github.com/webdataset/webdataset)
which bundles together files into tar archives.  In order to use the unmodified
configurations in oclf it is thus required to first convert the datasets into
this format which is explained below.

## Webdatasets / torchdata datapipes
We provide scripts to easily download and convert the datasets used in the
codebase into the right format. These scripts can be found in the subfolder
`scripts/datasets` and come with their own set of dependencies. We recommend to
install these is a separate virtual environment as they will not be needed for
training models but solely for the dataset generation.  When poetry is
installed you can simply install the dependencies using using the following
commands:

```bash
cd scripts/datasets
poetry install
```

???+ note "Installation issues with `pycocotools`"
    If you encounter any issue when installing `pycocotools` please be sure to
    have a compiler (for instance `gcc`) and the python development headers for
    your python version installed.  On rhel based systems this would look
    something like this:

    ```bash
    sudo yum install gcc python-devel
    ```

    If the python version you are using is installed via pyenv you might need
    to set environment variables such as `CPATH` for the correct headers to be
    found.

After the dependencies are installed you can convert a dataset by calling the
`download_and_convert.sh` script with the name of the dataset you would like
to convert.  It will download the necessary data to `data` and store converted
versions of the data in the `output` directory.  For example, to create
a webdataset version of the `movi_c` dataset run the following command:

```bash
bash download_and_convert.sh movi_c
```

### Changing the dataset path
As shown below, the dataset configurations instantiate
a [WebdatasetDataModule][ocl.datasets.WebdatasetDataModule] with the parameters
`train_shards`, `train_size`, `val_shards`, etc..

```yaml title="configs/dataset/clevr.yaml"
--8<-- "configs/dataset/clevr.yaml"
```

The shard paths contain a [resolver](https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html)
`${oc.env:DATASET_PREFIX}` which looks up the value of the environment variable
`DATASET_PREFIX`.

Thus, by setting the environment variable `DATASET_PREFIX` the path from which
the datasets are read can be changed.
[WebdatasetDataModule][ocl.datasets.WebdatasetDataModule] supports multiple
protocols for reading data from cloud storage or from local paths.  For further
information take a look at [torchdata.datapipes.iter.FSSpecFileOpener][].

Alternatively, you can of course create your own configuration file for the
dataset or replace the path of one of the predefined datasets directly in the
existing configuration files.  For more information on using your own
configuration files, see [Configuration][configuration].


## Custom datasets
OCLF command line tools are invariant to the exact dataset specification, thus
you can define your own dataset by implementing a pytorch lightning datamodule.

After creating your own datamodule you can simply add a dataset to OCLF by
creating an appropriate configuration files.  Please check out [Configuration][configuration]
for further information.

## Dataset transforms
Preprocessing steps are applied using dataset [Transforms][ocl.transforms]
which operate on the level of a pytorch data
[IterDataPipe][torchdata.datapipes.iter.IterDataPipe].  The transforms are
provided to the [WebdatasetDataModule][ocl.datasets.WebdatasetDataModule] via
the constructor arguments `train_transforms` and `eval_transforms`.  In OCLF
each IterDataPipe yields (potentially nested) dictionaries where each
represents an individual element or batch of the dataset. Further, each
transform takes a IterDataPipe as input to their call method and returns a
IterDataPipe.  Check out [Transforms][ocl.transforms] to see which
transformations are available.

Importantly, each [Transform][ocl.transforms.Transform] has a property
`is_batch_transform` which determines if it should be applied prior to or after
batching. This determines if the input will be a dictionary containing a single
element or a dictionary with concatenated tensors from multiple elements.  In
general, it is beneficial for performance to use batch transformation whenever
possible.

Often, a transformation does not need to be applied to the whole dictionary,
but only to a single input element (for instance an image) of the input dict.
Code that cover such functionality is stored in
[ocl.preprocessing][ocl.preprocessing].  Generally, any standard preprocessing
function can be used when combined with the right transform.  For instance, if
the input image should be converted to a tensor, resized and normalized the
following configuration will do the trick:

```yaml
dataset:
  train_transforms:
    resize_and_normalize:
      _target_: ocl.transforms.SimpleTransform
      transforms:
        image:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Resize
              size: 128
            - _target_: torchvision.transforms.Normalize
              mean: [0.5, 0.5, 0.5]
              std: [0.5, 0.5, 0.5]
      batch_transform: false
```
