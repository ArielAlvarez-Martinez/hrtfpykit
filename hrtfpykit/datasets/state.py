from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import DatasetConfig
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MetadataSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)


DatasetSpec = HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec


@dataclass
class DatasetState:
    config: type[DatasetConfig] | DatasetConfig | None = None
    name: str = ""
    root: Path = field(default_factory=Path)
    dataset_hrtf_transform: Callable[[object], object] | None = None
    excluded_subjects: tuple[str, ...] = ()
    hrtf_variant: str | None = None

    input_specs: tuple[DatasetSpec, ...] = ()
    target_specs: tuple[DatasetSpec, ...] = ()
    specs: tuple[DatasetSpec, ...] = ()
    input_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    index_by: tuple[str, ...] = ("subject",)
    selected_ears: tuple[tuple[str, int], ...] = ()
    position_one_hot: bool = False
    position_index: bool = False
    frequency_one_hot: bool = False
    frequency_index: bool = False
    sample_one_hot: bool = False
    sample_index: bool = False
    ear_one_hot: bool = False
    ear_index: bool = False

    cache: dict[object, object] = field(default_factory=dict)

    hrtf_paths: dict[str, Path] = field(default_factory=dict)
    mesh_paths: dict[str, Path] = field(default_factory=dict)
    image_path: Path | None = None
    video_path: Path | None = None
    image_index: dict[tuple[str, int | None, str | None], list[str]] = field(default_factory=dict)
    video_index: dict[tuple[str, int | None, str | None], list[str]] = field(default_factory=dict)
    image_counts: dict[str, int] = field(default_factory=dict)
    video_counts: dict[str, int] = field(default_factory=dict)
    anthropometry_path: Path | None = None
    anthropometry_rows: dict[str, object] = field(default_factory=dict)
    metadata_path: Path | None = None
    metadata_rows: dict[str, object] = field(default_factory=dict)
    resource_summary: dict[str, object] = field(default_factory=dict)
    subject_numbers: dict[str, int] = field(default_factory=dict)

    available_subjects: tuple[str, ...] = ()
    selected_subjects: tuple[str, ...] = ()
    split: str = "all"
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0

    sample_rate: float | None = None
    positions: np.ndarray | None = None
    azimuth_angles: np.ndarray | None = None
    elevation_angles: np.ndarray | None = None
    frequency_bins: np.ndarray | None = None
    sample_indices: np.ndarray | None = None
    selected_position_indices: tuple[int, ...] = ()
    selected_azimuth_angles: np.ndarray | None = None
    selected_elevation_angles: np.ndarray | None = None
    selected_frequency_indices: tuple[int, ...] = ()
    selected_sample_indices: tuple[int, ...] = ()
    spec_position_indices: dict[int, tuple[int, ...]] = field(default_factory=dict)

    rows: list[dict[str, str | int | None]] = field(default_factory=list)
    anthropometry_value_selector: Callable[..., object] | None = None
    metadata_value_selector: Callable[..., object] | None = None
    resources_summary: str = ""
    dataset_summary: str = ""
    verbose: bool = False
