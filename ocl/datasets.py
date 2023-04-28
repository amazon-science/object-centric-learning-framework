"""Implementation of datasets."""
import collections
import logging
import os
from distutils.util import strtobool
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import braceexpand
import numpy as np
import pytorch_lightning as pl
import torch
import torchdata
from torch.utils.data import DataLoader
from torch.utils.data._utils import collate as torch_collate
from torchdata.datapipes.iter import IterDataPipe

import ocl.utils.dataset_patches  # noqa: F401
from ocl.data_decoding import default_decoder
from ocl.transforms import Transform

LOGGER = logging.getLogger(__name__)
USE_AWS_SDK = strtobool(os.getenv("USE_AWS_SDK", "True"))


def _filter_keys(d: dict, keys_to_keep: Tuple) -> Dict[str, Any]:
    """Filter dict for keys in keys_to_keep.

    Additionally keeps all keys which start with `_`.

    Args:
        d: Dict to filter
        keys_to_keep: prefixes used to filter keys in dict.
            Keys with a matching prefix in `keys_to_keep` will be kept.

    Returns:
        The filtered dict.
    """
    keys_to_keep = ("_",) + keys_to_keep
    return {
        key: value
        for key, value in d.items()
        if any(key.startswith(prefix) for prefix in keys_to_keep)
    }


def _get_batch_transforms(transforms: Sequence[Transform]) -> Tuple[Transform]:
    return tuple(filter(lambda t: t.is_batch_transform, transforms))


def _get_single_element_transforms(transforms: Sequence[Transform]) -> Tuple[Transform]:
    return tuple(filter(lambda t: not t.is_batch_transform, transforms))


def _collect_fields(transforms: Sequence[Transform]) -> Tuple[str]:
    return tuple(chain.from_iterable(transform.fields for transform in transforms))


def _get_sorted_values(transforms: Dict[str, Transform]) -> Tuple[Transform]:
    return tuple(transforms[key] for key in sorted(transforms.keys()))


class WebdatasetDataModule(pl.LightningDataModule):
    """Webdataset Data Module."""

    def __init__(
        self,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        test_shards: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        train_transforms: Optional[Dict[str, Transform]] = None,
        eval_transforms: Optional[Dict[str, Transform]] = None,
        num_workers: int = 2,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        shuffle_train: bool = True,
        shuffle_buffer_size: Optional[int] = None,
        use_autopadding: bool = False,
    ):
        """Initialize WebdatasetDataModule.

        Args:
            train_shards: Shards associated with training split. Supports braceexpand notation.
            val_shards: Shards associated with validation split. Supports braceexpand notation.
            test_shards: Shards associated with test split. Supports braceexpand notation.
            batch_size: Batch size to use for training.
            eval_batch_size: Batch size to use for evaluation (i.e. on validation and test split).
                If `None` use same value as during training.
            train_transforms: Transforms to apply during training. We use a dict here to make
                composition of configurations with hydra more easy.
            eval_transforms: Transforms to apply during evaluation. We use a dict here to make
                composition of configurations with hydra more easy.
            num_workers: Number of workers to run in parallel.
            train_size: Number of instance in the train split (used for progressbar).
            val_size: Number of instance in the validation split (used for progressbar).
            test_size: Number of instance in the test split (used for progressbar).
            shuffle_train: Shuffle training split. Only used to speed up operations on train split
                unrelated to training. Should typically be left at `False`.
            shuffle_buffer_size: Buffer size to use for shuffling. If `None` uses `4*batch_size`.
            use_autopadding: Enable autopadding of instances with different dimensions.
        """
        super().__init__()
        if shuffle_buffer_size is None:
            # Ensure that data is shuffled umong at least 4 batches.
            # This should give us a good amount of diversity while also
            # ensuring that we don't need to long to start training.
            # TODO: Ideally, this should also take into account that
            # dataset might be smaller that the shuffle buffer size.
            # As this should not typically occur and we cannot know
            # the number of workers ahead of time we ignore this for now.
            shuffle_buffer_size = batch_size * 4

        if train_shards is None and val_shards is None and test_shards is None:
            raise ValueError("No split was specified. Need to specify at least one split.")
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.test_shards = test_shards
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_buffer_size = shuffle_buffer_size
        self.train_transforms = _get_sorted_values(train_transforms) if train_transforms else []
        self.eval_transforms = _get_sorted_values(eval_transforms) if eval_transforms else []

        if use_autopadding:
            self.collate_fn = collate_with_autopadding
        else:
            self.collate_fn = collate_with_batch_size

    def _create_webdataset(
        self,
        uri_expression: Union[str, List[str]],
        shuffle=False,
        n_datapoints: Optional[int] = None,
        keys_to_keep: Tuple[str] = tuple(),
        transforms: Sequence[Callable[[IterDataPipe], IterDataPipe]] = tuple(),
    ):
        if isinstance(uri_expression, str):
            uri_expression = [uri_expression]
        # Get shards for current worker.
        shard_uris = list(
            chain.from_iterable(
                braceexpand.braceexpand(single_expression) for single_expression in uri_expression
            )
        )
        datapipe = torchdata.datapipes.iter.IterableWrapper(shard_uris, deepcopy=False)
        datapipe = datapipe.shuffle(buffer_size=len(shard_uris)).sharding_filter()

        if shard_uris[0].startswith("s3://") and USE_AWS_SDK:
            # S3 specific implementation is much faster than fsspec.
            datapipe = datapipe.load_files_by_s3()
        else:
            datapipe = datapipe.open_files_by_fsspec(mode="rb")

        datapipe = datapipe.load_from_tar().webdataset()
        # Discard unneeded properties of the elements prior to shuffling and decoding.
        datapipe = datapipe.map(partial(_filter_keys, keys_to_keep=keys_to_keep))

        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=self.shuffle_buffer_size)

        # Decode files and remove extensions from input as we already decoded the elements. This
        # makes our pipeline invariant to the exact encoding used in the dataset.
        datapipe = datapipe.map(default_decoder)

        # Apply element wise transforms.
        for transform in transforms:
            datapipe = transform(datapipe)
        return torchdata.datapipes.iter.LengthSetter(datapipe, n_datapoints)

    def _create_dataloader(self, dataset, batch_transforms, size, batch_size, partial_batches):
        # Don't return partial batches during training as these give the partial samples a higher
        # weight in the optimization than the other samples of the dataset.

        # Apply batch transforms.
        dataset = dataset.batch(
            batch_size,
            drop_last=not partial_batches,
        ).collate(collate_fn=self.collate_fn)

        for transform in batch_transforms:
            dataset = transform(dataset)

        dataloader = DataLoader(dataset, num_workers=self.num_workers, batch_size=None)

        return dataloader

    def train_data_iterator(self):
        if self.train_shards is None:
            raise ValueError("Can not create train_data_iterator. No training split was specified.")
        transforms = self.train_transforms
        return self._create_webdataset(
            self.train_shards,
            shuffle=self.shuffle_train,
            n_datapoints=self.train_size,
            keys_to_keep=_collect_fields(transforms),
            transforms=_get_single_element_transforms(transforms),
        )

    def train_dataloader(self):
        return self._create_dataloader(
            dataset=self.train_data_iterator(),
            batch_transforms=_get_batch_transforms(self.train_transforms),
            size=self.train_size,
            batch_size=self.batch_size,
            partial_batches=False,
        )

    def val_data_iterator(self):
        if self.val_shards is None:
            raise ValueError("Can not create val_data_iterator. No val split was specified.")
        transforms = self.eval_transforms
        return self._create_webdataset(
            self.val_shards,
            shuffle=False,
            n_datapoints=self.val_size,
            keys_to_keep=_collect_fields(transforms),
            transforms=_get_single_element_transforms(transforms),
        )

    def val_dataloader(self):
        return self._create_dataloader(
            dataset=self.val_data_iterator(),
            batch_transforms=_get_batch_transforms(self.eval_transforms),
            size=self.val_size,
            batch_size=self.eval_batch_size,
            partial_batches=True,
        )

    def test_data_iterator(self):
        if self.test_shards is None:
            raise ValueError("Can not create test_data_iterator. No test split was specified.")
        return self._create_webdataset(
            self.test_shards,
            shuffle=False,
            n_datapoints=self.test_size,
            keys_to_keep=_collect_fields(self.eval_transforms),
            transforms=_get_single_element_transforms(self.eval_transforms),
        )

    def test_dataloader(self):
        return self._create_dataloader(
            dataset=self.test_data_iterator(),
            batch_transforms=_get_batch_transforms(self.eval_transforms),
            size=self.test_size,
            batch_size=self.eval_batch_size,
            partial_batches=True,
        )


class DummyDataModule(pl.LightningDataModule):
    """Dataset providing dummy data for testing."""

    def __init__(
        self,
        data_shapes: Dict[str, List[int]],
        data_types: Dict[str, str],
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        train_transforms: Optional[Dict[str, Transform]] = None,
        eval_transforms: Optional[Dict[str, Transform]] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        # Remaining args needed for compatibility with other datamodules
        train_shards: Optional[str] = None,
        val_shards: Optional[str] = None,
        test_shards: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        """Initialize DummyDataModule.

        Args:
            data_shapes: Mapping field names to shapes of tensors.
            data_types: Mapping from field names to types of tensors. One of `image`, `binary`,
                `uniform`, `categorical_{n_categories}` or `mask`.
            batch_size: Batch size to use for training.
            eval_batch_size: Batch size to use for evaluation (i.e. on validation and test split).
                If `None` use same value as during training.
            train_transforms: Transforms to apply during training.
            eval_transforms: Transforms to apply during evaluation.
            train_size: Number of instance in the train split (used for progressbar).
            val_size: Number of instance in the validation split (used for progressbar).
            test_size: Number of instance in the test split (used for progressbar).
            train_shards: Kept for compatibility with WebdatasetDataModule. Has no effect.
            val_shards: Kept for compatibility with WebdatasetDataModule. Has no effect.
            test_shards: Kept for compatibility with WebdatasetDataModule. Has no effect.
            num_workers: Kept for compatibility with WebdatasetDataModule. Has no effect.
        """
        super().__init__()
        self.data_shapes = {key: list(shape) for key, shape in data_shapes.items()}
        self.data_types = data_types
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.train_transforms = _get_sorted_values(train_transforms) if train_transforms else []
        self.eval_transforms = _get_sorted_values(eval_transforms) if eval_transforms else []

        self.train_size = train_size
        if self.train_size is None:
            self.train_size = 3 * batch_size + 1
        self.val_size = val_size
        if self.val_size is None:
            self.val_size = 2 * batch_size
        self.test_size = test_size
        if self.test_size is None:
            self.test_size = 2 * batch_size

    @staticmethod
    def _get_random_data_for_dtype(dtype: str, shape: List[int]):
        if dtype == "image":
            return np.random.randint(0, 256, size=shape, dtype=np.uint8)
        elif dtype == "binary":
            return np.random.randint(0, 2, size=shape, dtype=bool)
        elif dtype == "uniform":
            return np.random.rand(*shape).astype(np.float32)
        elif dtype.startswith("categorical_"):
            bounds = [int(b) for b in dtype.split("_")[1:]]
            if len(bounds) == 1:
                lower, upper = 0, bounds[0]
            else:
                lower, upper = bounds
            np_dtype = np.uint8 if upper <= 256 else np.uint64
            return np.random.randint(lower, upper, size=shape, dtype=np_dtype)
        elif dtype.startswith("mask"):
            categories = shape[1]
            np_dtype = np.uint8 if categories <= 256 else np.uint64
            slot_per_pixel = np.random.randint(
                0, categories, size=shape[:1] + shape[2:], dtype=np_dtype
            )
            return (
                np.eye(categories)[slot_per_pixel.reshape(-1)]
                .reshape(shape[:1] + shape[2:] + [categories])
                .transpose([0, 4, 1, 2, 3])
            )
        else:
            raise ValueError(f"Unsupported dtype `{dtype}`")

    def _create_dataset(
        self,
        n_datapoints: int,
        transforms: Sequence[Callable[[Any], Any]],
    ):
        class NumpyDataset(torch.utils.data.IterableDataset):
            def __init__(self, data: Dict[str, np.ndarray], size: int):
                super().__init__()
                self.data = data
                self.size = size

            def __iter__(self):
                for i in range(self.size):
                    elem = {key: value[i] for key, value in self.data.items()}
                    elem["__key__"] = str(i)
                    yield elem

        data = {}
        for key, shape in self.data_shapes.items():
            data[key] = self._get_random_data_for_dtype(self.data_types[key], [n_datapoints] + shape)

        dataset = torchdata.datapipes.iter.IterableWrapper(NumpyDataset(data, n_datapoints))
        for transform in transforms:
            dataset = transform(dataset)

        return dataset

    def _create_dataloader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_with_autopadding
        )

    def train_dataloader(self):
        dataset = self._create_dataset(
            self.train_size, _get_single_element_transforms(self.train_transforms)
        )
        return self._create_dataloader(dataset, self.batch_size)

    def val_dataloader(self):
        dataset = self._create_dataset(
            self.val_size, _get_single_element_transforms(self.eval_transforms)
        )
        return self._create_dataloader(dataset, self.eval_batch_size)

    def test_dataloader(self):
        dataset = self._create_dataset(
            self.test_size, _get_single_element_transforms(self.eval_transforms)
        )
        return self._create_dataloader(dataset, self.eval_batch_size)


def collate_with_batch_size(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Default pytorch collate function with additional `batch_size` output for dict input."""
    if isinstance(batch[0], collections.abc.Mapping):
        out = torch_collate.default_collate(batch)
        out["batch_size"] = len(batch)
        return out
    return torch_collate.default_collate(batch)


def collate_with_autopadding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function that takes a batch of data and stacks it with a batch dimension.

    In contrast to torch's collate function, this function automatically pads tensors of different
    sizes with zeros such that they can be stacked.

    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # As most tensors will not need padding to be stacked, we first try to stack them normally
        # and pad only if normal padding fails. This avoids explicitly checking whether all tensors
        # have the same shape before stacking.
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                if len(batch) * elem.numel() != numel:
                    # Check whether resizing will fail because tensors have unequal sizes to avoid
                    # a memory allocation. This is a sufficient but not necessary condition, so it
                    # can happen that this check will not trigger when padding is necessary.
                    raise RuntimeError()
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *elem.shape)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            # Stacking did not work. Try to pad tensors to the same dimensionality.
            if not all(x.ndim == elem.ndim for x in batch):
                raise ValueError("Tensors in batch have different number of dimensions.")

            shapes = [x.shape for x in batch]
            max_dims = [max(shape[idx] for shape in shapes) for idx in range(elem.ndim)]

            paddings = []
            for shape in shapes:
                padding = []
                # torch.nn.functional.pad wants padding from last to first dim, so go in reverse
                for idx in reversed(range(len(shape))):
                    padding.append(0)
                    padding.append(max_dims[idx] - shape[idx])
                paddings.append(padding)

            batch_padded = [
                torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
                for x, pad in zip(batch, paddings)
            ]

            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch_padded)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch_padded), *batch_padded[0].shape)
            return torch.stack(batch_padded, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if torch_collate.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(torch_collate.default_collate_err_msg_format.format(elem.dtype))

            return collate_with_autopadding([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        out = {key: collate_with_autopadding([d[key] for d in batch]) for key in elem}
        out["batch_size"] = len(batch)
        try:
            return elem_type(out)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return out
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_with_autopadding(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate_with_autopadding(samples) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate_with_autopadding(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate_with_autopadding(samples) for samples in transposed]

    raise TypeError(torch_collate.default_collate_err_msg_format.format(elem_type))
