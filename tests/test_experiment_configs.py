import glob
import os

import hydra
import pytest

import ocl.cli.eval
import ocl.cli.train

TRAIN_OVERRIDES = [
    # Minimal test run.  Use same parameters as trainer.fast_dev_run=True but keep logging enabled.
    "trainer.devices=1",
    "trainer.max_steps=1",
    "trainer.limit_train_batches=1",
    "trainer.limit_val_batches=1",
    "trainer.limit_test_batches=1",
    "trainer.val_check_interval=1.0",
    "trainer.num_sanity_val_steps=0",  # We run one val batch at the end of the fake training run.
    "++dataset.batch_size=2",
    "++dataset.num_workers=0",
    "++dataset.shuffle_buffer_size=1",  # Disable shuffling when running tests to speed things up.
]


def _remove_filename_components(path):
    """Convert a in the experiment folder to a valid setting for experiment."""
    # Remove file extension.
    path, ext = os.path.splitext(path)
    # Remove `config/experiment` prefix.
    return os.path.join(*path.split(os.path.sep)[2:])


EXPERIMENTS = list(
    map(
        _remove_filename_components,
        filter(
            lambda filename: not os.path.basename(filename).startswith("_"),
            glob.glob("configs/experiment/**/*.yaml", recursive=True),
        ),
    )
)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:LightningDeprecationWarning")
@pytest.mark.slow
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_config(experiment, tmpdir):
    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register TrainingConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="training_config",
            node=ocl.cli.train.TrainingConfig,
        )

        overrides = [f"+experiment={experiment}", f"hydra.run.dir={tmpdir}"]
        config = hydra.compose("training_config", overrides=overrides + TRAIN_OVERRIDES)
        ocl.cli.train.train(config)
