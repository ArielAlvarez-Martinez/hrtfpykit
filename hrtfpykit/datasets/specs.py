from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HRTFSpec:
    domain: str = "time"
    signal: str = "ir"
    positions: str | tuple[int, ...] | list[int] | np.ndarray = "all"
    plane: str | tuple[object, ...] | dict[str, object] | None = None
    ears: str | tuple[str, ...] = "both"
    index_by: str | tuple[str, ...] = ("subject",)
    position_encoding: str = "none"
    ear_encoding: str = "none"
    transform: Callable | None = None
    cache: bool = True
    name: str | None = None


@dataclass(frozen=True)
class MeshSpec:
    transform: Callable | None = None
    name: str | None = None


@dataclass(frozen=True)
class AnthropometrySpec:
    select: str | tuple[str, ...] | list[str] | None = "complete"
    ear: str = "both"
    path: str | Path | None = None
    transform: Callable | None = None
    name: str | None = None


@dataclass(frozen=True)
class ImageSpec:
    path: str | Path | None = None
    align_by: str | tuple[str, ...] = ("subject",)
    transform: Callable | None = None
    name: str | None = None


@dataclass(frozen=True)
class VideoSpec:
    path: str | Path | None = None
    align_by: str | tuple[str, ...] = ("subject",)
    transform: Callable | None = None
    name: str | None = None


def get_spec_name(
    spec: HRTFSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
) -> str:
    explicit_name = getattr(spec, "name", None)
    if explicit_name is not None:
        name = str(explicit_name).strip()
        if name == "":
            raise ValueError("Dataset spec name must not be empty")
        return name
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
            raise ValueError(f"Duplicate dataset spec name {name!r} is not allowed")
        names.add(name)
        normalized.append(spec)
    return tuple(normalized)


def normalize_anthropometry_select(
    select: str | tuple[str, ...] | list[str] | None,
) -> str | tuple[str, ...]:
    if select is None:
        return "complete"
    if isinstance(select, str):
        value = str(select).strip()
        if value == "":
            raise ValueError("select must not be empty")
        if value.lower() in {"complete", "all"}:
            return "complete"
        values = (value,)
    else:
        values = tuple(str(value).strip() for value in select)
    if len(values) == 0:
        raise ValueError("select must not be empty")
    if any(value == "" for value in values):
        raise ValueError("select must not contain empty names")
    if len(set(values)) != len(values):
        raise ValueError("select must not contain duplicates")
    return values


def normalize_anthropometry_ear(ear: str) -> str:
    value = str(ear).strip().lower()
    if value not in {"left", "right", "both"}:
        raise ValueError("ear must be 'left', 'right', or 'both'")
    return value
