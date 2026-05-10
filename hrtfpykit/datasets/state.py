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
    """Mutable construction and runtime state for one dataset instance.

    :class:`~hrtfpykit.datasets.state.DatasetState` is the internal schema owned by
    :class:`~hrtfpykit.datasets.base.BaseDataset`. During construction,
    :class:`~hrtfpykit.datasets.build.DatasetBuilder` replaces the current state
    with a fresh instance and fills it from the dataset configuration, normalized
    specs, scanned resources, split plan, acoustic context, and generated sample
    rows.

    The state object keeps dataset construction explicit. Builder phases can return
    small plans, and the builder copies their final values here instead of spreading
    private attributes across the dataset instance. After construction, dataset
    properties, sample extraction, summaries, and value selectors read from this
    same object.

    Notes
    -----
    Instances are intentionally mutable. They are not validation objects and do not
    enforce construction order by themselves; the builder is responsible for writing
    compatible values before users index the dataset.

    The fields are grouped by the phase that normally writes them. Empty tuples,
    dictionaries, lists, and None values represent state that has not been populated
    yet or resources that were not requested by the active specs.

    Attributes
    ----------
    config : type[DatasetConfig], DatasetConfig, or None
        Dataset configuration class or instance used to define subjects, variants,
        resource templates, and dataset-specific selectors.
    name : str
        Dataset display name copied from the active configuration.
    root : Path
        Expanded local root path used for resource discovery and relative path
        resolution.
    dataset_hrtf_transform : callable or None
        Optional transform applied after loading each subject HRTF through dataset
        loading utilities.
    excluded_subjects : tuple of str
        Canonical subject IDs removed from resource and split selection.
    dataset_hrtf_variant : str, dict, or None
        Selected HRTF resource variant after builder normalization. Mapping values
        may describe dataset-supported type, sample-rate, or version choices.
    dataset_mesh_variant : str, dict, or None
        Selected mesh resource variant after builder normalization.
    input_specs : tuple of DatasetSpec
        Normalized input specs returned by
        :class:`~hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow`.
    target_specs : tuple of DatasetSpec
        Normalized target specs returned by the spec workflow.
    specs : tuple of DatasetSpec
        Combined input and target specs used to decide required resources and sample
        axes.
    input_names : tuple of str
        Public sample keys produced under the "inputs" dictionary.
    target_names : tuple of str
        Public sample keys produced under the "target" dictionary.
    index_by : tuple of str
        Row dimensions requested by the active specs, such as "subject", "position",
        "ear", "frequency", or "sample".
    selected_ears : tuple of tuple[str, int]
        Ear labels and numeric ear indices selected by the active specs.
    position_one_hot, position_index : bool
        Flags controlling whether sample extraction adds position context encodings
        to input dictionaries.
    frequency_one_hot, frequency_index : bool
        Flags controlling whether sample extraction adds frequency context encodings
        to input dictionaries.
    sample_one_hot, sample_index : bool
        Flags controlling whether sample extraction adds time-sample context
        encodings to input dictionaries.
    ear_one_hot, ear_index : bool
        Flags controlling whether sample extraction adds ear context encodings to
        input dictionaries.
    cache : dict
        Shared cache for loaded HRTFs, transformed HRTFs, metrics, and spherical
        harmonic results reused across dataset operations.
    hrtf_paths : dict of str to Path
        Mapping from canonical subject ID to selected HRTF file path.
    mesh_paths : dict of str to Path
        Mapping from canonical subject ID to selected mesh file path.
    image_path : Path or None
        Root path or file path selected for image resources.
    video_path : Path or None
        Root path or file path selected for video resources.
    image_index : dict
        Mapping from media keys to image filenames. Keys contain subject ID,
        optional position index, and optional ear label.
    video_index : dict
        Mapping from media keys to video filenames. Keys contain subject ID,
        optional position index, and optional ear label.
    image_counts : dict of str to int
        Number of selected images available for each subject.
    video_counts : dict of str to int
        Number of selected videos available for each subject.
    anthropometry_path : Path or None
        Selected anthropometry table path, if anthropometry specs are active.
    anthropometry_rows : dict
        Loaded anthropometry rows or table-like data used by anthropometry value
        extraction.
    metadata_path : Path or None
        Selected metadata table path, if metadata specs are active.
    metadata_rows : dict
        Loaded metadata rows or table-like data used by metadata value extraction.
    resource_summary : dict
        Structured resource scanning summary used by error messages and textual
        summaries.
    subject_numbers : dict of str to int
        Numeric subject identifiers derived from canonical subject IDs for resource
        path formatting.
    available_subjects : tuple of str
        Canonical subjects that remain after exclusions and required-resource
        intersection.
    selected_subjects : tuple of str
        Canonical subjects selected by the active split.
    split : str
        Active split name, usually "all", "train", "validation", or "test".
    split_ratio : tuple of float
        Train, validation, and test ratios used by split planning.
    split_seed : int
        Seed used for deterministic split shuffling.
    sample_rate : float or None
        Shared acoustic sample rate inferred from selected HRTF resources.
    positions : ndarray or None
        Source positions from the representative HRTF after acoustic context
        discovery.
    azimuth_angles : ndarray or None
        Azimuth angle values derived from positions.
    elevation_angles : ndarray or None
        Elevation angle values derived from positions.
    frequency_bins : ndarray or None
        Frequency bins available for frequency-domain specs.
    sample_indices : ndarray or None
        Time-domain sample indices available for sample-indexed specs.
    selected_position_indices : tuple of int
        Position indices selected across active specs.
    selected_azimuth_angles : ndarray or None
        Azimuth values corresponding to selected positions.
    selected_elevation_angles : ndarray or None
        Elevation values corresponding to selected positions.
    selected_frequency_indices : tuple of int
        Frequency-bin indices selected across active frequency-domain specs.
    selected_sample_indices : tuple of int
        Time-sample indices selected across active time-domain specs.
    spec_position_indices : dict of int to tuple of int
        Mapping from spec object identity to the position indices selected for that
        specific spec.
    rows : list of dict
        Generated dataset row table. Each row stores the subject and any selected
        indexing dimensions used by sample extraction.
    anthropometry_value_selector : callable or None
        Dataset-specific selector used to adapt loaded anthropometry rows before an
        anthropometry spec transform is applied.
    metadata_value_selector : callable or None
        Dataset-specific selector used to adapt loaded metadata rows before a
        metadata spec transform is applied.
    resources_summary : str
        Human-readable resource summary cached after construction.
    dataset_summary : str
        Human-readable dataset summary cached after construction.
    verbose : bool
        Whether dataset construction and loading helpers should print progress
        information.
    """
    config: type[DatasetConfig] | DatasetConfig | None = None
    name: str = ""
    root: Path = field(default_factory=Path)
    dataset_hrtf_transform: Callable[[object], object] | None = None
    excluded_subjects: tuple[str, ...] = ()
    dataset_hrtf_variant: str | dict[str, object] | None = None
    dataset_mesh_variant: str | dict[str, object] | None = None

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
