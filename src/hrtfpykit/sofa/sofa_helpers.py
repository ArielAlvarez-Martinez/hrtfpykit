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
    is_open = getattr(sofa.netCDF4_dataset, "isopen", None)
    if callable(is_open) and not bool(is_open()):
        raise ValueError("SOFA dataset is closed")
    return sofa.netCDF4_dataset


def get_variable_creation_options(var: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    try:
        filters = var.filters()
    except AttributeError:
        filters = {}

    szip_filter = filters.get("szip")
    blosc_filter = filters.get("blosc")
    compression_name: str | None = None

    if filters.get("zlib"):
        compression_name = "zlib"
    elif filters.get("zstd"):
        compression_name = "zstd"
    elif filters.get("bzip2"):
        compression_name = "bzip2"
    elif isinstance(szip_filter, dict):
        compression_name = "szip"
        szip_coding = szip_filter.get("coding")
        szip_pixels_per_block = szip_filter.get("pixels_per_block")
        if szip_coding is not None:
            options["szip_coding"] = szip_coding
        if szip_pixels_per_block is not None:
            options["szip_pixels_per_block"] = int(szip_pixels_per_block)
    elif szip_filter:
        compression_name = "szip"
        if filters.get("szip_coding") is not None:
            options["szip_coding"] = filters["szip_coding"]
        if filters.get("szip_pixels_per_block") is not None:
            options["szip_pixels_per_block"] = int(filters["szip_pixels_per_block"])
    elif isinstance(blosc_filter, dict):
        blosc_compressor = blosc_filter.get("compressor")
        compression_name = str(blosc_compressor or "blosc_lz")
        if blosc_filter.get("shuffle") is not None:
            options["blosc_shuffle"] = int(blosc_filter["shuffle"])
    elif blosc_filter:
        compression_name = "blosc_lz"
        if filters.get("blosc_shuffle") is not None:
            options["blosc_shuffle"] = int(filters["blosc_shuffle"])

    if compression_name is not None:
        options["compression"] = compression_name
        if compression_name not in {"szip"} and filters.get("complevel") is not None:
            options["complevel"] = int(filters["complevel"])
        if compression_name not in {"szip"} and filters.get("shuffle") is not None:
            options["shuffle"] = bool(filters["shuffle"])

    if "fletcher32" in filters:
        options["fletcher32"] = bool(filters["fletcher32"])
    try:
        chunking = var.chunking()
    except AttributeError:
        chunking = None
    if chunking == "contiguous":
        options["contiguous"] = True
    elif isinstance(chunking, (list, tuple)):
        options["chunksizes"] = tuple(int(size) for size in chunking)
    try:
        endian = var.endian()
    except AttributeError:
        endian = None
    if endian in {"little", "big"}:
        options["endian"] = endian
    for option_name in ("least_significant_digit", "significant_digits", "quantize_mode"):
        if hasattr(var, option_name):
            option_value = getattr(var, option_name)
            if option_value is not None:
                options[option_name] = option_value
    if "_FillValue" in var.ncattrs():
        options["fill_value"] = getattr(var, "_FillValue")
    return options



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
