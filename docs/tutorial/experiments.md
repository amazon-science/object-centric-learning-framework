# Experiments

A experiment is a configuration that is applied to the global configuration
tree by adding `# @package _global_` to the beginning of the configuration
file.  A experiment is thus intended to define dataset, model, losses and
metrics that should be used during a training run.  The options which can be
configured in a training run are defined in the base configuration
[training_config][training-config] and shown below for convenience.

::: ocl.cli.train.TrainingConfig
    options:
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false


## Using routed classes
Some elements of the training config (especially, losses, metrics and
visualizations) expect dictionary elements to be able to handle a whole
dictionary that contains all information of the forward pass.  Instead of
coding up this support explicitly in your metric and loss implementations, it
is recommended to used [routed][] subclasses of your code.  This is allows
using external code for example from pytorch or torchmetrics.  Below you see an
example of this

```yaml title="configs/experiments/my_test_experiment.yaml"
training_metrics:
  classification_accuracy:
    _target_: routed.torchmetrics.BinaryAccuracy
    preds_path: my_model.prediction
    target_path: inputs.target

losses:
  bce:
    _target_: routed.torch.nn.BCEWithLogitsLoss
    input_path: my_model.prediction
    target_path: inputs.target
```

For further information take a look at [Models/How does this
work?](models.md#how-does-this-work) and the [routed][] module.


## Creating your own experiments - Example
Below an example of how it looks to adapt an existing experiment configuration
[/experiment/slot_attention/movi_c][configsexperimentslot_attentionmovi_cyaml]
to additionally reconstruct an optical flow signal.

```yaml title="configs/experiment/examples/composition.yaml"
--8<-- "configs/experiment/examples/composition.yaml"
```
