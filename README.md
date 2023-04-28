# Object Centric Learning Framework (OCLF)

[![Linting and Testing Status](https://github.com/amazon-science/object-centric-learning-framework/actions/workflows/lint_and_test.yaml/badge.svg?branch=main)](https://github.com/amazon-science/object-centric-learning-framework/actions/workflows/lint_and_test.yaml)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://amazon-science.github.io/object-centric-learning-framework/)


## What is OCLF?
OCLF (Object Centric Learning framework) is a framework designed to ease running
experiments for object centric learning research, yet is not limited to this
use case.  At its heart lies the idea that while code is not typically
composable many experiments in machine learning very similar with minor changes
and only represent minor changes.

One such example is multi-task training where a model might be trained to solve
multiple tasks at the same time.  Different ablations of said model would then
contain different model components but largely remain the same.

OCLF allows for such ablations without creating duplicate code by defining
models and experiments in configuration files and allowing their composition in
configuration space via [hydra](https://hydra.cc/).


## Quickstart - Development setup
Installing OCLF requires at least python3.8. Installation can be done using
[poetry](https://python-poetry.org/docs/#installation).  After installing
`poetry`, check out the repo and setup a development environment:

```bash
git clone https://github.com/amazon-science/object-centric-learning-framework.git
cd object-centric-learning-framework
poetry install
```

This installs the `ocl` package and the cli scripts used for running
experiments in a poetry managed virtual environment.

Next we need to prepare a dataset.  For this follow the steps below
to install the dependencies needed for dataset conversion and creation.

```bash
cd scripts/datasets
poetry install
bash download_and_convert.sh movi_c
```

This should create a webdataset in the path `scripts/datasets/outputs/movi_c`.

After exposing this dataset to OCLF, a first experiment can be run:

```bash
cd ../..   # Go back to root folder
export DATASET_PREFIX=scripts/datasets/outputs  # Expose dataset path
poetry run ocl_train +experiment=slot_attention/movi_c # Run training exeriment
```

The output of the training run should be stored at `outputs/slot_attention/movi_c/<timestamp>`.

For a more detailed guide on how to install, setup, and use OCLF check out
the Tutorial in the docs.


## Citation
If you use OCLF to run experiments in your work please cite it using the bibtex entry below

```bibtex
@misc{oclf,
  author = {Max Horn and Maximilian Seitzer and Andrii Zadaianchuk and Zixu Zhao and Dominik Zietlow and Florian Wenzel and Tianjun Xiao},
  title = {Object Centric Learning Framework (version 0.1)},
  year  = {2023},
  url   = {https://github.com/amazon-science/object-centric-learning-framework},
}
```

## Publications
Experiments for the following publications where run using OCLF. Please feel
free to add your own experiments via pull requests and to list them below.

 * M.Seitzer et al., Bridging the Gap to Real-World Object-Centric Learning
   [![arXiv](https://img.shields.io/badge/arXiv-2209.14860-b31b1b.svg)](https://arxiv.org/abs/2209.14860)
   [training configurations](https://amazon-science.github.io/object-centric-learning-framework/configs/experiment/projects/bridging/)
   [evaluation configurations](https://amazon-science.github.io/object-centric-learning-framework/configs/evaluation/projects/bridging/)


## License
This project is licensed under the Apache-2.0 License.


## Contributing
We are happy to accept code contributions in the form of pull-requests and
kindly ask contributors to follow the guidance provided below and in
`CONTRIBUTING.md`.

We are using `pre-commit` to manage automatic code formatting and linting. For
someone who has never worked with pre-commit, this can be a bit unusual.
`pre-commit` works by setting up a Git commit hook that runs before each `git
commit`. The hook executes a set of tests and automatic formatting *on all
files that are modified by the commit*:
- If a file does not pass a test, the commit is aborted and you are required to
  fix the problems, `git add` the files and run `git commit` again.
- If a file is automatically formatted, the commit is also aborted. You can
  review the proposed changes using `git diff`, accept them with `git add` and
  run `git commit` again.

It can also make sense to manually run the hooks on all files in the repository
(using `pre-commit run -a`) *before committing*, to make sure the commit
passes. Note that this does not run the hooks on files which are not yet
commited to the repository.

Important: make sure to run `pre-commit` within the environment installed by
`poetry`. Otherwise the checks might fail because the tools are not installed,
or use different versions from the ones specified in `poetry.lock`.
