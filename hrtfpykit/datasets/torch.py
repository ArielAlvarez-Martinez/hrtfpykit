from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np


def collate_samples(batch: Sequence[object]) -> object:
    """Collate map-style dataset samples for PyTorch data loaders.

    ``collate_samples`` is the default batching utility for hrtfpykit dataset
    objects used with ``torch.utils.data.DataLoader``.  A PyTorch data loader
    calls ``collate_fn`` with one argument: a list of samples returned by
    integer dataset indexing.  For hrtfpykit datasets, each sample is usually a
    dictionary with ``sample["inputs"]`` and ``sample["target"]`` entries, and
    those entries may contain nested dictionaries, NumPy arrays, scalar values,
    metadata, paths, meshes, images, or other user-provided resources.

    The function applies a conservative batching policy.  Nested dictionaries
    with the same keys are collated recursively.  NumPy arrays are stacked along
    a new leading batch axis only when every array has the same shape.  Numeric
    scalar values are converted to NumPy arrays.  Ragged arrays, mixed ``None``
    values, strings, paths, metadata objects, meshes, and other heterogeneous
    resources are preserved as Python lists.  This keeps batching safe for
    spec-driven datasets where different ``inputs`` and ``target`` choices can
    produce different shapes or resource types.

    Parameters
    ----------
    batch : sequence
        Sequence of samples returned by a map-style dataset.  In normal PyTorch
        usage, this is the list built internally from calls such as
        ``dataset[index]`` before ``DataLoader`` yields a batch.

    Returns
    -------
    object
        Collated batch with the same nested structure as the samples whenever
        the structure is consistent.  For standard hrtfpykit dataset samples,
        the returned object is a dictionary containing collated ``inputs`` and
        ``target`` entries.

    Raises
    ------
    TypeError
        If ``batch`` is not a sequence of dataset samples.
    ValueError
        If ``batch`` is empty.

    Notes
    -----
    This function does not import torch and does not convert values to torch
    tensors.  It is intended as a safe default ``collate_fn`` for PyTorch data
    loaders while keeping hrtfpykit's dataset API usable without requiring
    PyTorch as a dependency.

    Examples
    --------
    >>> from torch.utils.data import DataLoader
    >>> from hrtfpykit.datasets import HUTUBS, HRTFSpec, collate_samples
    >>> hutubs = HUTUBS(
    ...     root="datasets/hutubs",
    ...     inputs=HRTFSpec(index_by=("subject", "position")),
    ...     target=HRTFSpec(domain="frequency"),
    ... )
    >>> loader = DataLoader(hutubs, batch_size=8, collate_fn=collate_samples)
    >>> batch = next(iter(loader))
    >>> inputs = batch["inputs"]
    >>> target = batch["target"]
    """
    if isinstance(batch, str | bytes) or not isinstance(batch, Sequence):
        raise TypeError("collate_samples expects a sequence of dataset samples")
    if len(batch) == 0:
        raise ValueError("collate_samples expects at least one dataset sample")

    def collate_values(values: list[object]) -> object:
        first_value = values[0]

        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            return list(values)

        if all(isinstance(value, np.ndarray) for value in values):
            arrays = cast(list[np.ndarray], values)
            shapes = {value.shape for value in arrays}
            if len(shapes) == 1:
                return np.stack(arrays, axis=0)
            return list(values)

        if all(isinstance(value, np.generic) for value in values):
            return np.asarray(values)

        if all(isinstance(value, bool | int | float | complex) for value in values):
            return np.asarray(values)

        if all(isinstance(value, Mapping) for value in values):
            mappings = cast(list[Mapping[object, object]], values)
            first_mapping = cast(Mapping[object, object], first_value)
            keys = tuple(first_mapping.keys())
            key_set = set(keys)
            if all(set(value.keys()) == key_set for value in mappings):
                return {
                    key: collate_values([value[key] for value in mappings])
                    for key in keys
                }
            return list(values)

        if all(isinstance(value, Path) for value in values):
            return list(values)

        return list(values)

    return collate_values(list(batch))
