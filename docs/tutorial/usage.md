# Usage

OCLF relies heavily on configuration via the [hydra configuration
framework](https://hydra.cc/docs/intro/) and the command line interface of OCLF
scripts supports all overrides that hydra supports.  Thus is is always possible
to change **any value** that can be defined in the configuration.  As all
models and experiments are constructed with this in mind it is essentially
possible to override any parameter via the command line.  The basic syntax of
overriding parameters is explained [here](https://hydra.cc/docs/advanced/override_grammar/basic/).

OCLF provides three command line scripts: [ocl_train][ocl.cli.train] for
training models, [ocl_eval][ocl.cli.eval] for evaluating them and
[ocl_compute_dataset_size][ocl.cli.compute_dataset_size] for deriving the
number of instances in a dataset for correct progress monitoring.

!!! note
    The commands below assume you are running OCLF in a development
    installation and not as a package dependency.  They thus are prefixed with
    the `poetry run` command such that the command is run in the virtual
    environment of the project.  In a package dependency type installation this
    prefix is not needed.


## Running a simple experiment
For training a model in OCLF a experiment configuration is needed which
defines, model, dataset and training parameters.  Existing experiment
configurations are stored in the `configs/experiment` path of the repository
and can be selected using the command line.  For instance, in order to run an
training experiment based on the experiment configuration file
`configs/experiment/slot_attention/movi_c.yaml` we can call the following command:

```bash
poetry run ocl_train +experiment=slot_attention/movi_c
```

Using the cli we can change any parameter of this run. For instance we could use

```bash
poetry run ocl_train +experiment=slot_attention/movi_c dataset.batch_size=64
```
to train with a larger batch size.

In order to see all parameters that can be overwritten consult the experiment
config or have a look at the complete configuration by adding the flag `--cfg`
as shown below.

```bash
poetry run ocl_train +experiment=slot_attention/movi_c --cfg
```


## Experiment outputs
For all configuration that are part of OCLF, the result is saved in a
timestamped subdirectory in `outputs/<experiment_name>`, i.e.
`outputs/slot_attention/movi_c/<date>_<time>` in the above case. The prefix
path `outputs` can be configured using the `experiment.root_output_path`
variable.  This behavior implemented in the
`configs/experiment/_output_path.yaml`.


## Combining configurations
Some settings might only be applicable in certain contexts, for instance when
running a model on a cluster or when running a debug run.  These can simply be
grouped into a configuration file in the `configs` folder.  One such example is
shown below.

```yaml title="configs/tricks/debug.yaml"
# @package _global_
# Settings that implement a debug mode for quick testing locally on CPU.

trainer:
  devices: 1
  accelerator: cpu
  fast_dev_run: 5  # Only run 5 batches.

dataset:
  batch_size: 2    # Use small batch size to speed things up.
  num_workers: 0
  shuffle_buffer_size: 1
```

The previous call to run a training experiment can then be augmented to

```bash
poetry run ocl_train +experiment=slot_attention/movi_c +tricks=debug
```

## Separate codebase
When running OCLF cli tools installed as dependencies, one does not have direct
access to the `configs` folder.  Nevertheless, it is possible to add your own
configurations by additionally passing `--config-dir` on the command line.
This will augment the provided configuration folder to the configuration search
path.  You can thus still access configurations that are provided with OCLF.
An example is shown below.


```yaml title="my_configs/tricks/debug.yaml"
# @package _global_
# Settings that implement a debug mode for quick testing locally on CPU.

trainer:
  devices: 1
  accelerator: cpu
  fast_dev_run: 5  # Only run 5 batches.

dataset:
  batch_size: 2    # Use small batch size to speed things up.
  num_workers: 0
  shuffle_buffer_size: 1
```

```bash
poetry run ocl_train +experiment=slot_attention/movi_c +tricks=debug  --config-dir my_configs
```
