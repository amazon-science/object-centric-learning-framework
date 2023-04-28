# Experiment configurations

An experiment configuration is typically applied to the global
[training_config][training-config] by using the hydra command
`@package _global_` in the header.  They are used to ensure reproducibility
of experiments by including all relevant parameters for an experiment to run.

For an introduction to the concept of experiments please consider reading up on
this design pattern in the [hydra docs](https://hydra.cc/docs/patterns/configuring_experiments/).
