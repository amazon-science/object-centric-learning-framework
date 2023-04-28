# Configuration

OCLF command line tools are configured using configuration files written in
yaml via the [hydra configuration framework](https://hydra.cc/docs/intro/).
The base configuration is defined in [training_config][training-config],
datasets and experiments are defined in  the corresponding subfolders.


## FAQs

??? quote "Why are some elements of the configuration files defined as dictionaries?"
    Hydra does not support merging of lists from multiple configurations.  We
    thus instead rely on a workaround of using dictionaries which are later
    converted to lists.  Examples of this are `trainer.callbacks` which is
    initialized using a `${oc.dict.values:experiment.callbacks}` and thus
    derives callbacks from the dictionariy `experiment.callbacks` and the
    `train_transforms` and `eval_transforms` arguments of
    [ocl.datasets.WebdatasetDataModule][].

??? quote "Can I add configurations in a separate location to those in OCLF?"
    Yes, this is possible with the hydra using the `--config-dir` command line
    argument. See
    [here](https://hydra.cc/docs/advanced/hydra-command-line-flags/) for
    further information.
