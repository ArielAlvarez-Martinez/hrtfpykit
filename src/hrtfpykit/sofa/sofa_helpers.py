from typing import TYPE_CHECKING, Any, Union, cast
import pathlib

import netCDF4
import numpy as np

from ..utils.warnings import SOFAShapeWarning, warn_user
from .check import check_sofa_against_conventions

if TYPE_CHECKING:
    from .sofa import SOFA


def require_dataset(sofa: "SOFA") -> netCDF4.Dataset:
    if sofa.netCDF4_dataset is None:
        raise ValueError("Dataset is not loaded")
    return sofa.netCDF4_dataset


def warn_dimension_shape_mismatch(
    variable_name: str,
    dimensions: tuple[str, ...],
    data_shape: tuple[int, ...],
    dataset: netCDF4.Dataset,
) -> None:
    if len(dimensions) != len(data_shape):
        warn_user(
            (
                f"Variable '{variable_name}' dimensions do not coincide with dataset Dimensions "
                f"(dims={dimensions}); got shape {data_shape}."
            ),
            SOFAShapeWarning,
        )
        return

    expected_sizes: list[int] = []
    mismatch = False
    for dim_name, size in zip(dimensions, data_shape):
        dim = dataset.dimensions[dim_name]
        if dim.isunlimited():
            expected_sizes.append(size)
            continue
        expected_sizes.append(dim.size)
        if dim.size != size:
            mismatch = True

    if mismatch:
        dims_desc = ", ".join(
            f"{dim_name}={'unlimited' if dataset.dimensions[dim_name].isunlimited() else dataset.dimensions[dim_name].size}"
            for dim_name in dimensions
        )
        warn_user(
            (
                f"Variable '{variable_name}' dimensions do not coincide with dataset Dimensions "
                f"({dims_desc}); expected shape {tuple(expected_sizes)}, got {data_shape}."
            ),
            SOFAShapeWarning,
        )


def ensure_broadcastable(
    variable_name: str,
    data: np.ndarray,
    target_shape: tuple[int, ...],
) -> None:
    try:
        np.broadcast_to(data, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"Variable '{variable_name}' data shape must match: {target_shape}"
        ) from exc


def check_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    resolved_path = pathlib.Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"SOFA file not found: {resolved_path}")
    if resolved_path.suffix.lower() != ".sofa":
        raise ValueError(f"SOFA file must end with .sofa: {resolved_path}")
    return resolved_path


def open_sofa(
    sofa: "SOFA",
    path: Union[str, pathlib.Path],
    mode: str = "r",
    parallel: bool = False,
    check_sofa: bool = True,
) -> "SOFA":
    resolved_path = check_path(path)
    if check_sofa:
        check_sofa_against_conventions(resolved_path)
    sofa.netCDF4_dataset = netCDF4.Dataset(str(resolved_path), mode=cast(Any, mode), parallel=parallel)
    sofa.path = resolved_path
    return sofa
