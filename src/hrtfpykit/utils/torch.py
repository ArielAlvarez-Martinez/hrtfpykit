from collections.abc import Mapping, Sequence
import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np


def collate_samples(batch: Sequence[object]) -> object:
    """Collate hrtfpykit dataset samples for PyTorch data loaders.

    ``collate_samples`` is the batching function used with
    ``torch.utils.data.DataLoader``. The data loader receives individual samples
    from ``dataset[index]`` and passes a list of those samples to
    ``collate_samples``. hrtfpykit samples are dictionaries with ``inputs`` and
    ``target`` entries. Each entry is either ``None`` or a dictionary whose keys
    come from the selected specs.

    The values inside ``inputs`` and ``target`` follow the spec that produced
    them. ``HRTFSpec`` returns an IR or TF array extracted from the selected HRTF
    version. ``ITDSpec``, ``ILDSpec``, and ``SHSpec`` return values calculated
    from the selected HRTF version. For these acoustic specs, ``transform`` is an
    HRTF transform applied before extraction or calculation. ``AnthropometrySpec``
    and ``MetadataSpec`` return the selected subject values from their loaded
    resources, commonly a dictionary for CSV resources or a NumPy slice for
    matrix resources. ``MeshSpec`` returns a mesh path string unless its
    ``transform`` loads that path. ``ImageSpec`` and ``VideoSpec`` return a path
    string when one file matches the sample, a list of path strings when several
    files match, or the values produced by their ``transform``. With
    ``ImageSpec(concatenate=True)``, transformed image arrays are concatenated
    along axis zero before collation.

    ``collate_samples`` converts homogeneous numeric values into PyTorch tensors
    so a training loop can use ``batch["inputs"]`` and ``batch["target"]``
    directly. Floating point numeric values are converted to ``torch.float32``,
    which matches the default dtype used by standard PyTorch model parameters.
    Integer indices and boolean values keep their natural tensor dtypes. Arrays
    and tensors with matching shapes are stacked along a leading batch axis.
    Numeric dictionaries with the same keys are converted to ``batch x features``
    tensors. Lists with the same length are collated by position; for example, a
    subject with nine transformed RGB images of shape ``3 x 224 x 224`` becomes
    ``batch x 9 x 3 x 224 x 224``. Strings, paths, ragged values, mixed ``None``
    values, and non numeric objects are kept as Python lists.

    Parameters
    ----------
    batch : sequence
        Sequence of samples returned by a map style dataset. In normal PyTorch
        usage, this is the list built internally from calls such as
        ``dataset[index]`` before ``DataLoader`` yields a batch.

    Returns
    -------
    object
        Collated batch. For standard hrtfpykit dataset samples, the returned
        object is a dictionary containing collated ``inputs`` and ``target``
        entries. Homogeneous numeric values are returned as PyTorch tensors.

    Raises
    ------
    TypeError
        If ``batch`` is not a sequence of dataset samples.
    ValueError
        If ``batch`` is empty.
    ImportError
        If PyTorch is not installed.

    Notes
    -----
    Dataset indexing stays framework neutral. Image, video, and mesh specs return
    paths until their transforms load those paths into arrays or tensors. Tensor
    conversion happens here, at DataLoader collation time. Floating tensors are
    returned as ``torch.float32`` so common training loops do not need to cast
    NumPy ``float64`` values manually.

    Examples
    --------
    This example builds a PyTorch training batch from HRTF magnitudes and
    spherical harmonic targets. ``HRTFSpec`` returns one subject level magnitude
    map, ``SHSpec`` returns the coefficient target, and ``collate_samples``
    stacks both values as tensors. Floating tensors are already returned as
    ``torch.float32``, so the training loop does not need ``.float()`` casts.

    >>> import torch
    >>> from math import prod
    >>> from torch import nn
    >>> from torch.utils.data import DataLoader
    >>> from hrtfpykit.datasets import HUTUBS, HRTFSpec, SHSpec, collate_samples
    >>> train_dataset = HUTUBS(
    ...     root="datasets/hutubs",
    ...     inputs=HRTFSpec(
    ...         domain="frequency",
    ...         signal="tf_magnitude_db",
    ...         ears="left",
    ...         index_by=("subject",),
    ...         name="magnitude",
    ...     ),
    ...     target=SHSpec(
    ...         sh_order=9,
    ...         ears="left",
    ...         index_by=("subject",),
    ...         name="sh",
    ...     ),
    ...     split="train",
    ... )
    >>> train_loader = DataLoader(
    ...     train_dataset,
    ...     batch_size=8,
    ...     collate_fn=collate_samples,
    ... )
    >>> batch = next(iter(train_loader))
    >>> print(batch["inputs"]["magnitude"].shape)
    torch.Size([8, 440, 129])
    >>> print(batch["inputs"]["magnitude"].dtype)
    torch.float32
    >>> print(batch["target"]["sh"].shape)
    torch.Size([8, 100, 129])
    >>> print(batch["target"]["sh"].dtype)
    torch.float32
    >>> class MagnitudeToSHModel(nn.Module):
    ...     def __init__(self, target_shape):
    ...         super().__init__()
    ...         self.target_shape = tuple(target_shape)
    ...         self.encoder = nn.Sequential(
    ...             nn.Conv2d(1, 32, kernel_size=3, padding=1),
    ...             nn.ReLU(),
    ...             nn.AdaptiveAvgPool2d((1, 1)),
    ...             nn.Flatten(),
    ...         )
    ...         self.head = nn.Linear(32, prod(self.target_shape))
    ...     def forward(self, magnitude):
    ...         features = self.encoder(magnitude.unsqueeze(1))
    ...         return self.head(features).reshape(magnitude.shape[0], *self.target_shape)
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> target_shape = batch["target"]["sh"].shape[1:]
    >>> model = MagnitudeToSHModel(target_shape).to(device)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    >>> loss_fn = nn.MSELoss()
    >>> for epoch in range(10):
    ...     total_loss = 0.0
    ...     num_batches = 0
    ...     for batch in train_loader:
    ...         magnitude = batch["inputs"]["magnitude"].to(device)
    ...         target = batch["target"]["sh"].to(device)
    ...         prediction = model(magnitude)
    ...         loss = loss_fn(prediction, target)
    ...         optimizer.zero_grad()
    ...         loss.backward()
    ...         optimizer.step()
    ...         total_loss += float(loss.detach().cpu())
    ...         num_batches += 1
    ...     print(f"epoch {epoch + 1:02d} loss={total_loss / num_batches:.6f}")
    """
    if isinstance(batch, str | bytes) or not isinstance(batch, Sequence):
        raise TypeError("collate_samples expects a sequence of dataset samples")
    if len(batch) == 0:
        raise ValueError("collate_samples expects at least one dataset sample")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("collate_samples requires PyTorch") from exc

    def collate_values(values: list[object]) -> object:
        first_value = values[0]

        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            return list(values)

        if all(
            isinstance(value, np.ndarray)
            or isinstance(value, torch.Tensor)
            for value in values
        ):
            array_values = cast(list[Any], values)
            shapes = {tuple(value.shape) for value in array_values}
            if len(shapes) == 1:
                tensor = torch.stack(
                    [torch.as_tensor(value) for value in array_values],
                    dim=0,
                )
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch.float32)
                return tensor
            return list(values)

        if all(isinstance(value, np.generic) for value in values):
            tensor = torch.as_tensor(np.asarray(values))
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            return tensor

        if all(isinstance(value, bool | int | float | complex) for value in values):
            tensor = torch.as_tensor(np.asarray(values))
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            return tensor

        if all(isinstance(value, Mapping) for value in values):
            mappings = cast(list[Mapping[object, object]], values)
            first_mapping = cast(Mapping[object, object], first_value)
            keys = tuple(first_mapping.keys())
            key_set = set(keys)
            if all(set(value.keys()) == key_set for value in mappings):
                numeric_rows = True
                for value in mappings:
                    for key in keys:
                        if not isinstance(
                            value[key],
                            np.generic | bool | int | float | complex,
                        ):
                            numeric_rows = False
                            break
                    if not numeric_rows:
                        break
                if numeric_rows:
                    array = np.asarray(
                        [[value[key] for key in keys] for value in mappings]
                    )
                    tensor = torch.as_tensor(array)
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch.float32)
                    return tensor
                return {
                    key: collate_values([value[key] for value in mappings])
                    for key in keys
                }
            return list(values)

        if all(isinstance(value, Path) for value in values):
            return list(values)

        if all(
            isinstance(value, Sequence)
            and not isinstance(value, str | bytes | bytearray | np.ndarray | Path)
            and not isinstance(value, torch.Tensor)
            for value in values
        ):
            sequences = cast(list[Sequence[object]], values)
            lengths = {len(value) for value in sequences}
            if len(lengths) == 1 and next(iter(lengths)) > 0:
                row_values = [collate_values(list(value)) for value in sequences]
                if not any(isinstance(value, list) for value in row_values):
                    return collate_values(row_values)
            return list(values)

        return list(values)

    return collate_values(list(batch))
