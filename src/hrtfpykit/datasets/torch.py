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
        If PyTorch is unavailable in the current environment.

    Notes
    -----
    Dataset indexing stays framework neutral. Image, video, and mesh specs return
    paths until their transforms load those paths into arrays or tensors. Tensor
    conversion happens here, at DataLoader collation time. Floating tensors are
    returned as ``torch.float32`` so common training loops do not need to cast
    NumPy ``float64`` values manually.

    Examples
    --------
    This example downloads the first ten measured HUTUBS HRTFs when needed,
    builds one subject-level sample per HRTF, and batches the samples with
    ``torch.utils.data.DataLoader``. ``HRTFSpec`` returns the left-ear HRTF
    magnitude in dB, ``SHSpec`` returns spherical-harmonic coefficients, and
    ``collate_samples`` stacks both values as ``torch.float32`` tensors.

    >>> import torch
    >>> from math import prod
    >>> from torch import nn
    >>> from torch.utils.data import DataLoader
    >>> from hrtfpykit.datasets import HUTUBS, HRTFSpec, SHSpec
    >>> from hrtfpykit.datasets.torch import collate_samples
    >>> selected_subject_ids = tuple(f"pp{i}" for i in range(1, 11))
    >>> train_dataset = HUTUBS(
    ...     root="datasets/hutubs",
    ...     download=True,
    ...     download_resources="hrtf",
    ...     download_hrtf_variant="measured",
    ...     download_server="sofacoustics",
    ...     dataset_hrtf_variant="measured",
    ...     download_subject_ids=selected_subject_ids,
    ...     subject_ids=selected_subject_ids,
    ...     verify_checksum=True,
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



def hrtf_loss(
    prediction: object,
    target: object,
    metric: str = "rmse",
    input_scale: str = "db",
    reduction_method: str = "mean",
    epsilon: float = 1e-12,
) -> object:
    """Compute a scalar PyTorch loss for HRTF and HRIR model outputs.

    ``hrtf_loss`` accepts PyTorch tensors or tensor-convertible values, compares
    ``prediction`` and ``target`` with RMSE, MAE, or LSD, and returns one scalar
    tensor for backpropagation with ``loss.backward()``. The last tensor axis is
    the acoustic axis being scored: HRIR samples for time-domain targets,
    frequency bins for HRTF magnitude targets, and frequency bins for LSD. After
    that axis is reduced, the remaining axes, commonly batch, positions, and
    ears, are reduced with ``reduction_method``.

    The function supports three metrics:

    - ``metric="rmse"`` computes root mean squared error over the final tensor
      axis.
    - ``metric="mae"`` computes mean absolute error over the final tensor axis.
    - ``metric="lsd"`` computes log-spectral distance over the final tensor
      axis.

    ``rmse`` and ``mae`` measure direct tensor error in the representation passed
    to the function. For HRIR targets, the final axis contains time samples. For
    HRTF magnitude or dB-magnitude targets, the final axis contains frequency
    bins and the loss measures magnitude error in that representation.

    ``lsd`` measures spectral magnitude error in decibels. With
    ``input_scale="db"``, ``prediction`` and ``target`` are dB magnitudes. With
    ``input_scale="linear"``, values are linear magnitudes and are converted to
    dB with ``20 * log10(clamp(value, min=epsilon))``. Complex linear tensors are
    converted to magnitudes with ``abs`` before the dB conversion.

    Parameters
    ----------
    prediction : torch.Tensor or tensor-convertible
        Model output tensor. For full HRTF magnitude training this is commonly
        shaped ``(batch, positions, ears, frequency)``. For datasets indexed by
        subject, position, and ear, this may be shaped ``(batch, frequency)``.
        HRIR targets commonly use samples on the final axis.
    target : torch.Tensor or tensor-convertible
        Target tensor with the same shape as ``prediction``.
    metric : {``"rmse"``, ``"mae"``, ``"lsd"``}, default=``"rmse"``
        Loss metric.
    input_scale : {``"db"``, ``"linear"``}, default=``"db"``
        Scale used by ``metric="lsd"`` to interpret ``prediction`` and
        ``target`` before computing the spectral distance.
    reduction_method : {``"mean"``, ``"rms"``}, default=``"mean"``
        Reduction applied after the final tensor axis has been reduced.
        ``"mean"`` averages metric values across remaining axes. ``"rms"``
        computes a root mean square over the remaining metric values.
    epsilon : float, default=1e-12
        Positive numerical floor used by LSD dB conversion and square-root
        stabilization.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.

    Raises
    ------
    ImportError
        If PyTorch is unavailable in the current environment.
    ValueError
        If options are unsupported, input shapes differ, inputs are scalar, or
        ``epsilon`` is not finite and positive.

    Examples
    --------
    This example downloads the first ten measured HUTUBS HRTFs when needed,
    builds a dataset that pairs left-ear HRIR samples with left-ear HRTF
    magnitudes in dB, and trains a small PyTorch model with LSD as the loss.
    The model receives tensors shaped ``batch x positions x samples`` and
    predicts tensors shaped ``batch x positions x frequency``.

    >>> import torch
    >>> from torch import nn
    >>> from torch.utils.data import DataLoader
    >>> from hrtfpykit.datasets import HUTUBS, HRTFSpec
    >>> from hrtfpykit.datasets.torch import collate_samples, hrtf_loss
    >>> selected_subject_ids = tuple(f"pp{i}" for i in range(1, 11))
    >>> train_dataset = HUTUBS(
    ...     root="datasets/hutubs",
    ...     download=True,
    ...     download_resources="hrtf",
    ...     download_hrtf_variant="measured",
    ...     download_server="sofacoustics",
    ...     dataset_hrtf_variant="measured",
    ...     download_subject_ids=selected_subject_ids,
    ...     subject_ids=selected_subject_ids,
    ...     verify_checksum=True,
    ...     inputs=HRTFSpec(
    ...         domain="time",
    ...         signal="ir",
    ...         ears="left",
    ...         index_by=("subject",),
    ...         name="hrir",
    ...     ),
    ...     target=HRTFSpec(
    ...         domain="frequency",
    ...         signal="tf_magnitude_db",
    ...         ears="left",
    ...         index_by=("subject",),
    ...         name="magnitude_db",
    ...     ),
    ...     split="train",
    ... )
    >>> train_loader = DataLoader(
    ...     train_dataset,
    ...     batch_size=8,
    ...     collate_fn=collate_samples,
    ... )
    >>> batch = next(iter(train_loader))
    >>> class HRIRToMagnitudeModel(nn.Module):
    ...     def __init__(self, num_samples, num_frequencies):
    ...         super().__init__()
    ...         self.network = nn.Sequential(
    ...             nn.Linear(num_samples, 256),
    ...             nn.ReLU(),
    ...             nn.Linear(256, num_frequencies),
    ...         )
    ...     def forward(self, hrir):
    ...         return self.network(hrir)
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> num_samples = batch["inputs"]["hrir"].shape[-1]
    >>> num_frequencies = batch["target"]["magnitude_db"].shape[-1]
    >>> model = HRIRToMagnitudeModel(num_samples, num_frequencies).to(device)
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    >>> for epoch in range(10):
    ...     total_loss = 0.0
    ...     num_batches = 0
    ...     for batch in train_loader:
    ...         hrir = batch["inputs"]["hrir"].to(device)
    ...         target = batch["target"]["magnitude_db"].to(device)
    ...         prediction = model(hrir)
    ...         loss = hrtf_loss(
    ...             prediction,
    ...             target,
    ...             metric="lsd",
    ...             input_scale="db",
    ...         )
    ...         optimizer.zero_grad()
    ...         loss.backward()
    ...         optimizer.step()
    ...         total_loss += float(loss.detach().cpu())
    ...         num_batches += 1
    ...     print(f"epoch {epoch + 1:02d} lsd={total_loss / num_batches:.6f}")
    """
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("hrtf_loss requires PyTorch") from exc

    metric_key = str(metric).strip().lower()
    if metric_key not in {"rmse", "mae", "lsd"}:
        raise ValueError("metric must be one of: rmse, mae, lsd")

    input_scale_key = str(input_scale).strip().lower()
    if input_scale_key not in {"db", "linear"}:
        raise ValueError("input_scale must be one of: db, linear")

    reduction_method_key = str(reduction_method).strip().lower()
    if reduction_method_key not in {"mean", "rms"}:
        raise ValueError("reduction_method must be one of: mean, rms")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    prediction_tensor: Any = prediction if torch.is_tensor(prediction) else torch.as_tensor(prediction)
    target_tensor: Any = target if torch.is_tensor(target) else torch.as_tensor(target)
    if target_tensor.device != prediction_tensor.device:
        target_tensor = target_tensor.to(device=prediction_tensor.device)

    if prediction_tensor.shape != target_tensor.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction_tensor.ndim == 0:
        raise ValueError("prediction and target must have at least one dimension")

    if torch.is_complex(prediction_tensor) or torch.is_complex(target_tensor):
        dtype = torch.promote_types(prediction_tensor.dtype, target_tensor.dtype)
        prediction_tensor = prediction_tensor.to(dtype=dtype)
        target_tensor = target_tensor.to(dtype=dtype)
    elif prediction_tensor.is_floating_point() or target_tensor.is_floating_point():
        dtype = prediction_tensor.dtype if prediction_tensor.is_floating_point() else target_tensor.dtype
        prediction_tensor = prediction_tensor.to(dtype=dtype)
        target_tensor = target_tensor.to(dtype=dtype)
    else:
        prediction_tensor = prediction_tensor.to(dtype=torch.float32)
        target_tensor = target_tensor.to(dtype=torch.float32)

    if metric_key == "lsd":
        if input_scale_key == "db":
            if torch.is_complex(prediction_tensor) or torch.is_complex(target_tensor):
                raise ValueError("input_scale='db' does not support complex tensors")
            prediction_db = prediction_tensor
            target_db = target_tensor
        else:
            if torch.is_complex(prediction_tensor):
                prediction_magnitude = torch.abs(prediction_tensor)
            else:
                prediction_magnitude = torch.clamp(prediction_tensor, min=epsilon)
            if torch.is_complex(target_tensor):
                target_magnitude = torch.abs(target_tensor)
            else:
                target_magnitude = torch.clamp(target_tensor, min=epsilon)
            prediction_db = 20.0 * torch.log10(torch.clamp(prediction_magnitude, min=epsilon))
            target_db = 20.0 * torch.log10(torch.clamp(target_magnitude, min=epsilon))
        error = prediction_db - target_db
        metric_values = torch.sqrt(torch.mean(error**2, dim=-1) + epsilon)
    else:
        error = torch.abs(prediction_tensor - target_tensor)
        if metric_key == "rmse":
            metric_values = torch.sqrt(torch.mean(error**2, dim=-1) + epsilon)
        else:
            metric_values = torch.mean(error, dim=-1)

    if metric_values.ndim == 0:
        return metric_values
    if reduction_method_key == "mean":
        return torch.mean(metric_values)
    return torch.sqrt(torch.mean(metric_values**2) + epsilon)
