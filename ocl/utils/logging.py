import os
import signal
import traceback
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from mlflow.client import MlflowClient
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.tensorboard._convert_np import make_np
from torch.utils.tensorboard._utils import _prepare_video, convert_to_HWC, figure_to_image
from torch.utils.tensorboard.summary import _calc_scale_factor

from ocl.utils.trees import get_tree_element


def prepare_video_tensor(tensor):
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    return tensor


def write_video_tensor(prefix, tensor, fps):
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print("add_video needs package moviepy")
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print(
            "moviepy is installed, but can't import moviepy.editor.",
            "Some packages could be missing [imageio, requests]",
        )
        return

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
    filename = prefix + ".gif"

    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    return filename


def prepare_image_tensor(tensor, dataformats="NCHW"):
    tensor = make_np(tensor)
    tensor = convert_to_HWC(tensor, dataformats)
    # Do not assume that user passes in values in [0, 255], use data type to detect
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    return tensor


def write_image_tensor(prefix: str, tensor: np.ndarray):
    from PIL import Image

    image = Image.fromarray(tensor)
    filename = prefix + ".png"
    image.save(filename, format="png")
    return filename


class ExtendedMLflowExperiment:
    """MLflow experiment made to mimic tensorboard experiments."""

    def __init__(self, mlflow_client: MlflowClient, run_id: str):
        self._mlflow_client = mlflow_client
        self._run_id = run_id
        self._tempdir = TemporaryDirectory()

    def _get_tmp_prefix_for_step(self, step: int):
        return os.path.join(self._tempdir.name, f"{step:07d}")

    def add_video(self, vid_tensor, fps: int, tag: str, global_step: int):
        path = tag  # TF paths are typically split using "/"
        filename = write_video_tensor(
            self._get_tmp_prefix_for_step(global_step), prepare_video_tensor(vid_tensor), fps
        )
        self._mlflow_client.log_artifact(self._run_id, filename, path)
        os.remove(filename)

    def add_image(self, img_tensor: torch.Tensor, dataformats: str, tag: str, global_step: int):
        path = tag
        filename = write_image_tensor(
            self._get_tmp_prefix_for_step(global_step),
            prepare_image_tensor(img_tensor, dataformats=dataformats),
        )
        self._mlflow_client.log_artifact(self._run_id, filename, path)
        os.remove(filename)

    def add_images(self, img_tensor, dataformats: str, tag: str, global_step: int):
        # Internally works by having an additional N dimension in `dataformats`.
        self.add_image(img_tensor, dataformats, tag, global_step)

    def add_figure(self, figure, close: bool, tag: str, global_step: int):
        if isinstance(figure, list):
            self.add_image(
                figure_to_image(figure, close),
                dataformats="NCHW",
                tag=tag,
                global_step=global_step,
            )
        else:
            self.add_image(
                figure_to_image(figure, close),
                dataformats="CHW",
                tag=tag,
                global_step=global_step,
            )

    def __getattr__(self, name):
        """Fallback to mlflow client for missing attributes.

        Fallback to make the experiment object still behave like the regular MLflow client.  While
        this is suboptimal, it does allow us to save a lot of handcrafted code by relying on
        inheritance and pytorch lightings implementation of the MLflow logger.
        """
        return getattr(self._mlflow_client, name)


class ExtendedMLFlowLogger(MLFlowLogger):
    @property  # type: ignore[misc]
    @rank_zero_experiment
    def experiment(self) -> ExtendedMLflowExperiment:
        return ExtendedMLflowExperiment(super().experiment, self._run_id)

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint):
        self.experiment.log_artifact(
            self._run_id, checkpoint_callback.best_model_path, "checkpoints"
        )

    @rank_zero_only
    def log_artifact(self, local_path, artifact_path=None):
        self.experiment.log_artifact(self._run_id, local_path, artifact_path=artifact_path)

    @rank_zero_only
    def log_artifacts(self, local_path, artifact_path=None):
        self.experiment.log_artifacts(self._run_id, local_path, artifact_path=artifact_path)


class LogHydraConfigCallback(Callback):
    def __init__(self, hydra_output_subdir: str, additional_paths: None, skip_overrides=False):
        self.hydra_output_subdir = hydra_output_subdir
        self.additional_paths = additional_paths
        self.skip_overrides = skip_overrides

    def _parse_overrides(self):
        with open(os.path.join(self.hydra_output_subdir, "overrides.yaml"), "r") as f:
            overrides = yaml.safe_load(f)

        output = {}
        for override in overrides:
            fragments = override.split("=")
            if len(fragments) == 2:
                if override.startswith("+"):
                    fragments[0] = fragments[0][1:]

                output[fragments[0]] = fragments[1]
        return output

    def _parse_additional_paths(self):
        with open(os.path.join(self.hydra_output_subdir, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)

        outputs = {}
        if isinstance(self.additional_paths, dict):
            for output_path, input_path in self.additional_paths.items():
                outputs[output_path] = get_tree_element(config, input_path.split("."))
        elif isinstance(self.additional_paths, list):
            for additional_path in self.additional_paths:
                outputs[additional_path] = get_tree_element(config, additional_path.split("."))
        else:
            raise ValueError("additional_paths of unsupported format")

        return outputs

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        # Log all hydra config files.
        trainer.logger.log_artifacts(self.hydra_output_subdir, "config")
        if not self.skip_overrides:
            trainer.logger.log_hyperparams(self._parse_overrides())
        if self.additional_paths:
            trainer.logger.log_hyperparams(self._parse_additional_paths())

    @rank_zero_only
    def on_exception(self, trainer, pl_module, exception):
        del pl_module
        logger = trainer.logger
        with TemporaryDirectory() as d:
            filename = os.path.join(d, "exception.txt")
            with open(filename, "w") as f:
                traceback.print_exc(file=f)
                f.flush()
            trainer.logger.log_artifact(filename)
            os.remove(filename)
        if logger.experiment.get_run(logger.run_id):
            if isinstance(exception, KeyboardInterrupt):
                logger.experiment.set_terminated(logger.run_id, status="KILLED")
            else:
                logger.experiment.set_terminated(logger.run_id, status="FAILED")

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        # Register our own signal handler to set run to terminated.
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def handler(signum, frame):
            rank_zero_info("Handling SIGTERM")
            logger = trainer.logger
            logger.experiment.set_terminated(logger.run_id, status="KILLED")
            logger.save()
            if previous_sigterm_handler in [None, signal.SIG_DFL]:
                # Not set up by python or default behaviour.
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.raise_signal(signal.SIGTERM)
            elif previous_sigterm_handler != signal.SIG_IGN:
                # If none of the above must be callable.
                previous_sigterm_handler()

        signal.signal(signal.SIGTERM, handler)


class LogModelSummaryCallback(Callback):
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        with TemporaryDirectory() as d:
            filename = os.path.join(d, "model_summary.txt")
            with open(filename, "w") as f:
                f.write(str(ModelSummary(pl_module, max_depth=-1)))
            trainer.logger.log_artifact(filename)
            os.remove(filename)
