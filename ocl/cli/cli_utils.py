import glob
import os

from hydra.core.hydra_config import HydraConfig


def get_commandline_config_path():
    """Get the path of a config path specified on the command line."""
    hydra_cfg = HydraConfig.get()
    config_sources = hydra_cfg.runtime.config_sources
    config_path = None
    for source in config_sources:
        if source.schema == "file" and source.provider == "command-line":
            config_path = source.path
            break
    return config_path


def find_checkpoint(path):
    """Find checkpoint in output path of previous run."""
    checkpoints = glob.glob(
        os.path.join(path, "lightning_logs", "version_*", "checkpoints", "*.ckpt")
    )
    checkpoints.sort()
    # Return the last checkpoint.
    # TODO (hornmax): If more than one checkpoint is stored this might not lead to the most recent
    # checkpoint being loaded. Generally, I think this is ok as we still allow people to set the
    # checkpoint manually.
    return checkpoints[-1]
