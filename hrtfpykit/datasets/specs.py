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
    positions_encoding: bool = False
    ear_encoding: bool = False
    frequencies_encoding: bool = False
    samples_encoding: bool = False
    transform: Callable | None = None
    cache: bool = True
    name: str | None = None


@dataclass(frozen=True)
class ITDSpec:
    positions: str | tuple[int, ...] | list[int] | np.ndarray = "all"
    plane: str | tuple[object, ...] | dict[str, object] | None = None
    index_by: str | tuple[str, ...] = ("subject",)
    positions_encoding: bool = False
    method: str = "threshold"
    output: str = "samples"
    thresh_level: float = -10.0
    upper_cut_freq: float = 3000.0
    filter_order: int = 10
    transform: Callable | None = None
    name: str | None = None


@dataclass(frozen=True)
class ILDSpec:
    positions: str | tuple[int, ...] | list[int] | np.ndarray = "all"
    plane: str | tuple[object, ...] | dict[str, object] | None = None
    index_by: str | tuple[str, ...] = ("subject",)
    positions_encoding: bool = False
    frequencies_encoding: bool = False
    mode: str = "broad-band"
    output: str = "db"
    fft_length: int | None = None
    epsilon: float = 1e-12
    transform: Callable | None = None
    name: str | None = None


@dataclass(frozen=True)
class SHSpec:
    sh_order: int
    ears: str | tuple[str, ...] = "both"
    index_by: str | tuple[str, ...] = ("subject",)
    ear_encoding: bool = False
    frequencies_encoding: bool = False
    epsilon: float = 1e-6
    transform: Callable | None = None
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
    concatenate: bool = False
    transform: Callable | None = None
    name: str | None = None


@dataclass(frozen=True)
class VideoSpec:
    path: str | Path | None = None
    align_by: str | tuple[str, ...] = ("subject",)
    transform: Callable | None = None
    name: str | None = None


def get_spec_name(
    spec: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
) -> str:
    explicit_name = getattr(spec, "name", None)
    if explicit_name is not None:
        name = str(explicit_name).strip()
        if name == "":
            raise ValueError("Dataset spec name must not be empty")
        return name
    if isinstance(spec, HRTFSpec):
        return "hrtf"
    if isinstance(spec, ITDSpec):
        return "itd"
    if isinstance(spec, ILDSpec):
        return "ild"
    if isinstance(spec, SHSpec):
        return "sh"
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
    | ITDSpec
    | ILDSpec
    | SHSpec
    | MeshSpec
    | AnthropometrySpec
    | ImageSpec
    | VideoSpec
    | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec]
    | None,
) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]:
    if specs is None:
        return tuple()
    if isinstance(specs, str):
        raise TypeError("inputs and target must use dataset spec objects, not strings")
    if isinstance(specs, (HRTFSpec, ITDSpec, ILDSpec, SHSpec, MeshSpec, AnthropometrySpec, ImageSpec, VideoSpec)):
        values = (specs,)
    else:
        values = tuple(specs)
    names: set[str] = set()
    normalized: list[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] = []
    for spec in values:
        name = get_spec_name(spec)
        if name in names:
            raise ValueError(f"Duplicate dataset spec name {name!r} is not allowed")
        names.add(name)
        normalized.append(spec)
    return tuple(normalized)
