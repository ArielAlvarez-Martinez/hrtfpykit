from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IndexBy = tuple[str, ...]
PositionSelection = str | tuple[object, ...] | list[int] | np.ndarray | dict[str, object]
AlignBy = str | tuple[str, ...]


@dataclass(frozen=True)
class HRTFSpec:
    variant: str | None = None
    domain: str = "ir"
    signal: str = "ir"
    positions: PositionSelection = "all"
    ears: str | tuple[str, ...] = "both"
    position_encoding: str = "none"
    ear_encoding: str = "none"
    transform: Callable | None = None
    cache: bool = True
    aligned_by: IndexBy | None = None
    variants: tuple[str, ...] | None = None
    default_variant: str | None = None
    filename_pattern: str | None = None
    download_pattern: str | None = None
    download_subject_ids: tuple[str, ...] | None = None
    download_checksums: dict[str, str] | None = None
    supported_domains: tuple[str, ...] = ("ir", "tf")
    supported_signals: tuple[str, ...] = (
        "ir",
        "tf_complex",
        "magnitude",
        "magnitude_db",
        "phase",
        "real",
        "imag",
    )
    ear_labels: tuple[str, ...] = ("left", "right")
    shared_position_grid: bool = True


@dataclass(frozen=True)
class MeshSpec:
    transform: Callable | None = None
    aligned_by: IndexBy | None = None
    filename_pattern: str | None = None
    download_pattern: str | None = None
    download_subject_ids: tuple[str, ...] | None = None
    download_checksums: dict[str, str] | None = None
    extensions: tuple[str, ...] = (".ply", ".stl")


@dataclass(frozen=True)
class AnthropometrySpec:
    columns: str | tuple[str, ...] | list[str] | None = None
    transform: Callable | None = None
    aligned_by: IndexBy | None = None
    filename: str | None = None
    download_filename: str | None = None
    download_checksum: str | None = None
    subject_column_candidates: tuple[str, ...] = (
        "subject_id",
        "subject",
        "id",
        "participant",
        "pp",
    )


@dataclass(frozen=True)
class ImageSpec:
    path: str | Path | None = None
    align_by: AlignBy = ("subject",)
    image_size: int | tuple[int, int] | None = None
    transform: Callable | None = None
    supported_align_by: tuple[IndexBy, ...] | None = None
    extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )


@dataclass(frozen=True)
class VideoSpec:
    path: str | Path | None = None
    align_by: AlignBy = ("subject",)
    transform: Callable | None = None
    supported_align_by: tuple[IndexBy, ...] | None = None
    extensions: tuple[str, ...] = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    )

def get_spec_name(
    spec: HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
) -> str:
    if isinstance(spec, HRTFSpec):
        return "hrtf"
    if isinstance(spec, MeshSpec):
        return "mesh"
    if isinstance(spec, AnthropometrySpec):
        return "anthropometry"
    if isinstance(spec, ImageSpec):
        return "image"
    if isinstance(spec, VideoSpec):
        return "video"
    raise TypeError(f"Unsupported dataset spec: {type(spec)!r}")


def normalize_specs(
    specs: HRTFSpec
    | MeshSpec
    | AnthropometrySpec
    | ImageSpec
    | VideoSpec
    | Sequence[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec]
    | None,
) -> tuple[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]:
    if specs is None:
        return tuple()
    if isinstance(specs, str):
        raise TypeError("inputs and target must use dataset spec objects, not strings")
    if isinstance(specs, (HRTFSpec, MeshSpec, AnthropometrySpec, ImageSpec, VideoSpec)):
        values = (specs,)
    else:
        values = tuple(specs)
    names: set[str] = set()
    normalized: list[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] = []
    for spec in values:
        name = get_spec_name(spec)
        if name in names:
            raise ValueError(f"Duplicate dataset spec {name!r} is not allowed")
        names.add(name)
        normalized.append(spec)
    return tuple(normalized)


def build_spec_map(
    specs: Sequence[HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec],
) -> dict[str, HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec]:
    mapping: dict[str, HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] = {}
    for spec in specs:
        name = get_spec_name(spec)
        if name in mapping:
            raise ValueError(f"Duplicate dataset spec {name!r} is not allowed")
        mapping[name] = spec
    return mapping


def normalize_columns(
    columns: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    if columns is None:
        return None
    if isinstance(columns, str):
        values = (str(columns).strip(),)
    else:
        values = tuple(str(value).strip() for value in columns)
    if len(values) == 0:
        raise ValueError("columns must not be empty")
    if any(value == "" for value in values):
        raise ValueError("columns must not contain empty names")
    if len(set(values)) != len(values):
        raise ValueError("columns must not contain duplicates")
    return values
