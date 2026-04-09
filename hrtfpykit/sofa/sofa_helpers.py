from typing import Any, Dict, Optional, TYPE_CHECKING, Union
import datetime
import importlib.metadata
import pathlib
import platform
import sys

import netCDF4
import numpy as np

from .._warnings import SOFAShapeWarning, warn_user
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
    sofa.netCDF4_dataset = netCDF4.Dataset(resolved_path, mode=mode, parallel=parallel)
    sofa.path = resolved_path
    return sofa


def version_key(value: str) -> tuple:
    parts = value.split(".")
    if all(part.isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return (value,)


def first_dim_option(dim_spec: Optional[str]) -> list[str]:
    if not dim_spec:
        return []
    option = dim_spec.split(",", 1)[0].strip()
    option = option.replace(" ", "")
    return [letter.upper() for letter in option]


def dtype_for(var_type: Optional[str]) -> object:
    if var_type is None:
        return "f8"
    key = var_type.lower()
    if key in ("double", "float64"):
        return "f8"
    if key in ("float", "float32"):
        return "f4"
    if key in ("int", "int32"):
        return "i4"
    if key in ("short", "int16"):
        return "i2"
    if key in ("char", "string"):
        return str
    return "f8"


def reshape_for_broadcast(data: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if data.shape == target_shape:
        return data
    if data.shape == ():
        return data
    non_singleton = tuple(dim for dim in target_shape if dim != 1)
    if data.shape != non_singleton:
        return data
    reshaped: list[int] = []
    data_index = 0
    for dim in target_shape:
        if dim == 1:
            reshaped.append(1)
        else:
            reshaped.append(data.shape[data_index])
            data_index += 1
    return data.reshape(tuple(reshaped))


def complete_global_attributes(
    dataset: netCDF4.Dataset,
    custom_global_attributes: Optional[Dict[str, str]] = None,
    override_default_global_attributes: bool = False,
) -> None:
    def is_missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        return False

    try:
        api_version = importlib.metadata.version("hrtfpykit")
    except importlib.metadata.PackageNotFoundError:
        api_version = "unknown"

    compiler = platform.python_compiler()
    implementation = sys.implementation.name
    if implementation:
        implementation = implementation.capitalize()
    else:
        implementation = "Python"
    python_version = f"{platform.python_version()} [{implementation} - {compiler}]"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    default_custom_attributes: Dict[str, str] = {
        "APIName": "hrtfpykit",
        "APIVersion": api_version,
        "ApplicationName": "Python",
        "ApplicationVersion": python_version,
        "DateCreated": now,
        "DateModified": now,
    }

    resolved = default_custom_attributes
    if custom_global_attributes:
        resolved = {**default_custom_attributes, **custom_global_attributes}

    for attr_name, value in resolved.items():
        if override_default_global_attributes or is_missing(getattr(dataset, attr_name, None)):
            setattr(dataset, attr_name, value)
