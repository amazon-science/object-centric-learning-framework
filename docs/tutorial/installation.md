# Installation
You can install OCLF either in a development setup using
[poetry](https://python-poetry.org/docs/#installation) or as a dependency in
your own project via pip.  As OCLF relies heavily on configuration files for
running experiments it is easiest to start with a development setup as this
allows configuration files to be inspected and edited in place for rapid
prototyping.  In contrast, if you want to keep your configurations separate
from OCLF a installation of OCLF as a dependency might be the better way to go
for you.

## Development installation
Installing OCLF requires at least python3.8. Installation can be done using
[poetry](https://python-poetry.org/docs/#installation).  After installing
`poetry`, check out the repo and setup a development environment:

```bash
git clone https://github.com/amazon-science/object-centric-learning-framework.git
cd object-centric-learning-framework
poetry install # Optionally add -E <extra> for each extra that should be installed
```

Valid extras are `timm` for access to timm models for feature extraction and
`clip` for access to OpenAI's clip model.  For instance `poetry install -E timm
-E clip` installs both.

Poetry will create a separate virtual environment where the projects
dependencies are installed.  It can be accessed using `poetry shell` or `poetry
run`.  Please see the [poetry docs](https://python-poetry.org/docs) for further
information on using poetry.


## Installation as a dependency
It is also possible to install OCLF as a dependency for your project via pip
for this simply run

```bash
pip3 install "git+https://github.com/amazon-science/object-centric-learning-framework.git"
```

this might take a while as pip tries to resolve the dependencies specified in
the OCLF `pyproject.toml` file whereas poetry directly uses locked dependencies
which were determined in a previous run and added to the repository.

With OCLF installed as a dependency you cannot directly explore configurations
that are part of OCLF or edit them. Nevertheless, you can access OCLF
components via the python API and run experiments using by adding your own
configurations.  For further information check out [Usage/Separate
codebase](usage.md#separate-codebase).
