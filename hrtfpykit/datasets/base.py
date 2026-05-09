from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Callable, TYPE_CHECKING

import numpy as np

from .build import (
    DatasetBuilder,
)
from .specs_workflow import DatasetSpecWorkflow
from .config import DatasetConfig
from .load import load_hrtf
from .state import DatasetState
from .summary import dataset_summary, resources_summary
from .values import DatasetSampleValueSelector
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

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


class BaseDataset:
    """Base implementation for spec-driven dataset construction.

    ``BaseDataset`` owns the common dataset behavior used by concrete dataset
    classes such as ``HUTUBS`` and ``SONICOM``. It builds an explicit dataset
    state from a dataset configuration, selected specs, resource scans, subject
    exclusions, split settings, and acoustic context. The constructed object is
    a dataset abstraction that can expose HRTF objects, meshes, tabular metadata,
    images, videos, and derived acoustic values through a consistent indexed
    sample interface. Users usually instantiate a concrete dataset class, while
    new dataset integrations can subclass
    ``BaseDataset`` and provide a dataset-specific config and defaults.

    Parameters
    ----------
    root : str or Path
        Local dataset root used to find resources.
    config : DatasetConfig or type[DatasetConfig], optional
        Dataset configuration describing subject IDs, resource paths, and
        downloadable resources.
    dataset_hrtf_transform : callable or None, default=None
        Optional transform applied to loaded subject HRTFs before spec values are
        extracted.
    exclude_subject_ids : str, int, sequence, or None, default=None
        Subject references excluded from resource scanning and split planning.
    inputs : spec, sequence of specs, or None, default=None
        Dataset specs exposed under ``sample['inputs']``.
    target : spec, sequence of specs, or None, default=None
        Dataset specs exposed under ``sample['target']``.
    dataset_hrtf_variant : str, dict, or None, default=None
        HRTF resource variant selected for dataset construction.
    dataset_mesh_variant : str, dict, or None, default=None
        Mesh resource variant selected for dataset construction.
    split : {'all', 'train', 'validation', 'test'}, default='all'
        Subject split used by this dataset instance.
    split_ratio : tuple of float, default=(0.8, 0.1, 0.1)
        Train, validation, and test split ratios.
    split_seed : int, default=0
        Random seed used for deterministic split assignment.
    verbose : bool, default=False
        If ``True``, prints resource and dataset summaries after construction.

    Returns
    -------
    BaseDataset
        Dataset object supporting ``len(dataset)``, ``dataset[index]``,
        summary accessors, and subject-level HRTF loading.

    """

    def __init__(
        self,
        root: str | Path,
        config: type[DatasetConfig] | DatasetConfig | None = None,
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        dataset_hrtf_variant: str | Mapping[str, object] | None = None,
        dataset_mesh_variant: str | Mapping[str, object] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if config is None:
            raise ValueError("BaseDataset requires a dataset config")
        self._state = DatasetState()
        DatasetBuilder(self).build(
            config=config,
            root=root,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            dataset_hrtf_variant=dataset_hrtf_variant,
            dataset_mesh_variant=dataset_mesh_variant,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            verbose=verbose,
        )
        self._state.resources_summary = resources_summary(self)
        self._state.dataset_summary = dataset_summary(self)
        if verbose:
            print(self._state.resources_summary)
            print(self._state.dataset_summary)

    def get_subject_hrtf(self, subject_id: str | int) -> "HRTF":
        """Load one subject HRTF through the dataset resource map.

        This method is the public subject-level access point shared by concrete
        datasets. It applies the same subject mapping, resource lookup, cache, and
        dataset-level HRTF transform used by sample extraction, so direct inspection
        uses the same loading path as ``dataset[index]``.

        Parameters
        ----------
        subject_id : str or int
            Dataset subject reference. Integer values are mapped to the configured
            subject order.

    Returns
    -------
    HRTF
        Loaded HRTF object after applying any dataset-level HRTF transform.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs")
        >>> hrtf = dataset.get_subject_hrtf("pp1")
        """

        return load_hrtf(self, subject_id)

    def resources_summary(self) -> str:
        """Return the resource scan summary created during construction.

        The summary reflects the resources actually relevant to the selected specs,
        not every resource a dataset family might support. It reports missing files,
        partial media resources, and subject removals caused by resource
        intersection.

    Parameters
    ----------
    None
        This method uses the dataset state created during construction.

    Returns
    -------
    str
        Human-readable summary of scanned resources used by the selected specs.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs")
        >>> text = dataset.resources_summary()
        """

        return self._state.resources_summary

    def dataset_summary(self) -> str:
        """Return the dataset summary created during construction.

        The summary captures the final constructed dataset state: root, split, subject
        counts, selected specs, selected resource variants, and acoustic context when
        available.

    Parameters
    ----------
    None
        This method uses the dataset state created during construction.

    Returns
    -------
    str
        Human-readable summary of subjects, split, specs, selected variants,
        and acoustic metadata.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs")
        >>> text = dataset.dataset_summary()
        """

        return self._state.dataset_summary

    @property
    def root(self) -> Path:
        """Return the local dataset root.

        This property exposes the resolved root used by all resource scanners after
        path expansion.

    Returns
    -------
    Path
        Expanded dataset root path.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs")
        >>> root = dataset.root
        """
        return self._state.root

    @property
    def dataset_hrtf_variant(self) -> str | dict[str, object] | None:
        """Return the selected HRTF resource variant.

        This value describes the HRTF resource selected for scanning. Datasets with
        one selector axis return a string such as ``'measured'``. Datasets with
        multiple selector axes return a dictionary containing fields such as
        ``type``, ``sample_rate``, and ``version``.

    Returns
    -------
    str, dict, or None
        Selected HRTF variant.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs", dataset_hrtf_variant="measured")
        >>> dataset.dataset_hrtf_variant
        """
        return self._state.dataset_hrtf_variant

    @property
    def dataset_mesh_variant(self) -> str | dict[str, object] | None:
        """Return the selected mesh resource variant.

        This value describes the geometry resource selected for scanning. Datasets
        with one selector axis return a string. Datasets with multiple selector axes
        return a dictionary containing fields such as ``type`` and ``version``.

    Returns
    -------
    str, dict, or None
        Selected mesh variant.

        Examples
        --------
        >>> from hrtfpykit.datasets import SONICOM
        >>> dataset = SONICOM(root="datasets/sonicom")
        >>> dataset.dataset_mesh_variant
        """
        return self._state.dataset_mesh_variant

    @property
    def split(self) -> str:
        """Return the requested dataset split name.

        The split controls which available subjects become rows in this dataset
        instance. It is stored separately from resource availability so code
        can distinguish usable subjects from selected train/validation/test subjects.

    Returns
    -------
    str
        Split name used by this dataset instance.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs", split="train")
        >>> dataset.split
        """
        return self._state.split

    @property
    def split_ratio(self) -> tuple[float, float, float]:
        """Return train, validation, and test split ratios.

        These ratios are used only when ``split`` is train, validation, or test.
        Exposing them makes split behavior reproducible and visible in constructed
        dataset instances.

    Returns
    -------
    tuple of float
        Three split ratios used during split planning.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs", split_ratio=(0.8, 0.1, 0.1))
        >>> dataset.split_ratio
        """
        return self._state.split_ratio

    @property
    def split_seed(self) -> int:
        """Return the split random seed.

        The seed controls deterministic subject shuffling before train/validation/test
        partitioning. This property exists so a dataset instance fully reports how its
        selected subjects were chosen.

    Returns
    -------
    int
        Seed used for deterministic split planning.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs", split_seed=0)
        >>> dataset.split_seed
        """
        return self._state.split_seed

    @property
    def inputs(self) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec, ...]:
        """Return input specs used by this dataset.

        These are normalized copies of user-provided input specs, so workflow
        normalization can adjust spec fields without mutating caller-owned
        objects. The tuple shows exactly what keys and resources feed
        ``sample["inputs"]``.

    Returns
    -------
    tuple of specs
        Normalized input specs.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec())
        >>> dataset.inputs
        """
        return self._state.input_specs

    @property
    def target(self) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec, ...]:
        """Return target specs used by this dataset.

        These are normalized copies of user-provided target specs. The tuple describes
        what values are produced under ``sample["target"]`` during indexed sample
        extraction.

    Returns
    -------
    tuple of specs
        Normalized target specs.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", target=HRTFSpec())
        >>> dataset.target
        """
        return self._state.target_specs

    @property
    def sample_rate(self) -> float | None:
        """Return dataset-level acoustic sample rate.

        The value is derived from the selected HRTF resources after validation. It
        represents the full dataset acoustic context and is not changed by per-spec
        extraction choices.

    Returns
    -------
    float or None
        Sample rate read from selected HRTF resources.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec())
        >>> dataset.sample_rate
        """
        return self._state.sample_rate

    @property
    def positions(self) -> np.ndarray | None:
        """Return dataset-level source positions.

        These positions describe the full source grid from the selected HRTF resource
        before spec-level row selection. Selected position subsets are exposed
        separately through ``selected_position_indices`` and selected angle
        properties.

    Returns
    -------
    numpy.ndarray or None
        Source positions from selected HRTF resources.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec())
        >>> positions = dataset.positions
        """
        return self._state.positions

    @property
    def azimuth_angles(self) -> np.ndarray | None:
        """Return available dataset azimuth angles.

        The angles are derived from the full dataset source grid. They report spatial
        coverage independently from the subset selected by indexed specs.

    Returns
    -------
    numpy.ndarray or None
        Unique azimuth angles from dataset source positions.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec())
        >>> azimuths = dataset.azimuth_angles
        """
        return self._state.azimuth_angles

    @property
    def elevation_angles(self) -> np.ndarray | None:
        """Return available dataset elevation angles.

        The angles are derived from the full dataset source grid. They help
        show spatial coverage and distinguish full resource context
        from selected row context.

    Returns
    -------
    numpy.ndarray or None
        Unique elevation angles from dataset source positions.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec())
        >>> elevations = dataset.elevation_angles
        """
        return self._state.elevation_angles

    @property
    def frequency_bins(self) -> np.ndarray | None:
        """Return dataset-level frequency bins.

        The bins come from the selected HRTF resource when frequency-domain data is
        available. They are used by frequency-indexed specs and remain part of the
        dataset-level acoustic context.

    Returns
    -------
    numpy.ndarray or None
        Frequency bins from selected HRTF resources.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec(domain="frequency"))
        >>> bins = dataset.frequency_bins
        """
        return self._state.frequency_bins

    @property
    def sample_indices(self) -> np.ndarray | None:
        """Return dataset-level time sample indices.

        The indices describe the full IR sample axis from the selected HRTF resource.
        They support sample-indexed specs while keeping the original acoustic context
        inspectable.

    Returns
    -------
    numpy.ndarray or None
        Time-sample indices from selected HRTF resources.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec(domain="time"))
        >>> indices = dataset.sample_indices
        """
        return self._state.sample_indices

    @property
    def selected_position_indices(self) -> tuple[int, ...]:
        """Return source position indices selected by specs.

        This property exposes the row-generating position subset after explicit
        position or plane selection. It is separate from ``positions`` so selected
        context does not hide the full dataset source grid.

    Returns
    -------
    tuple of int
        Selected source position indices.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(index_by=("subject", "position")),
        ... )
        >>> dataset.selected_position_indices
        """
        return self._state.selected_position_indices

    @property
    def selected_azimuth_angles(self) -> np.ndarray | None:
        """Return azimuth angles selected by position-aware specs.

        The values summarize the selected position subset used for row generation.
        They are ``None`` when no spec selected a position-indexed subset.

    Returns
    -------
    numpy.ndarray or None
        Unique selected azimuth angles.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(index_by=("subject", "position")),
        ... )
        >>> azimuths = dataset.selected_azimuth_angles
        """
        return self._state.selected_azimuth_angles

    @property
    def selected_elevation_angles(self) -> np.ndarray | None:
        """Return elevation angles selected by position-aware specs.

        The values summarize the selected position subset used for row generation.
        They help debug plane selectors and position-indexed datasets.

    Returns
    -------
    numpy.ndarray or None
        Unique selected elevation angles.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(index_by=("subject", "position")),
        ... )
        >>> elevations = dataset.selected_elevation_angles
        """
        return self._state.selected_elevation_angles

    @property
    def selected_frequency_indices(self) -> tuple[int, ...]:
        """Return selected frequency-bin indices.

        These indices are used when frequency appears in the shared dataset
        ``index_by`` axes. They record how many frequency-indexed rows each
        selected subject contributes.

    Returns
    -------
    tuple of int
        Frequency indices used to build indexed rows.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(domain="frequency", index_by=("subject", "frequency")),
        ... )
        >>> dataset.selected_frequency_indices
        """
        return self._state.selected_frequency_indices

    @property
    def selected_sample_indices(self) -> tuple[int, ...]:
        """Return selected time-sample indices.

        These indices are used when samples appear in the shared dataset ``index_by``
        axes. They record how many sample-indexed rows each selected subject
        contributes.

    Returns
    -------
    tuple of int
        Sample indices used to build indexed rows.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(
        ...     root="datasets/hutubs",
        ...     inputs=HRTFSpec(domain="time", index_by=("subject", "samples")),
        ... )
        >>> dataset.selected_sample_indices
        """
        return self._state.selected_sample_indices

    @property
    def excluded_subjects(self) -> list[str]:
        """Return subjects excluded from this dataset instance.

        This list combines config-level exclusions and user-provided exclusions after
        subject-reference normalization. It shows why expected
        subjects do not appear in resource scans or splits.

    Returns
    -------
    list of str
        Excluded canonical subject IDs.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs", exclude_subject_ids=(1, 2))
        >>> dataset.excluded_subjects
        """
        return list(self._state.excluded_subjects)

    @property
    def available_subjects(self) -> list[str]:
        """Return subjects available after resource intersection.

        Available subjects are the subjects that have every resource required by the
        selected specs after exclusions. This is resource availability, not
        necessarily the final train/validation/test split subset.

    Returns
    -------
    list of str
        Canonical subject IDs available for the selected specs.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs")
        >>> dataset.available_subjects
        """
        return list(self._state.available_subjects)

    @property
    def selected_subjects(self) -> list[str]:
        """Return subjects selected for the requested split.

        Selected subjects are the available subjects used to build rows for this
        dataset instance. For ``split="all"`` this usually matches
        ``available_subjects``; for train/validation/test it is a deterministic
        subset.

    Returns
    -------
    list of str
        Canonical subject IDs used to build dataset rows.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs", split="train")
        >>> dataset.selected_subjects
        """
        return list(self._state.selected_subjects)

    def __len__(self) -> int:
        """Return the number of dataset rows.

        Rows are created from selected subjects and any shared indexed axes such as
        position, ear, frequency, or samples. This method exposes the final sample
        count consumed by training loops and indexed access.

    Returns
    -------
    int
        Number of samples addressable by integer indexing.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> dataset = HUTUBS(root="datasets/hutubs")
        >>> len(dataset)
        """
        return len(self._state.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one sample by integer row index.

        This method resolves the row context, dispatches each input and target spec
        through the value selector layer, and adds requested row encodings. It is the
        final runtime path that turns dataset state into model-ready sample
        dictionaries.

        Parameters
        ----------
        index : int
            Dataset row index.

    Returns
    -------
    dict
        Sample dictionary with ``inputs`` and ``target`` entries.

        Examples
        --------
        >>> from hrtfpykit.datasets import HUTUBS
        >>> from hrtfpykit.datasets.specs import HRTFSpec
        >>> dataset = HUTUBS(root="datasets/hutubs", inputs=HRTFSpec())
        >>> sample = dataset[0]
        """
        if not isinstance(index, int):
            raise TypeError("Dataset indexing only supports integer indices")
        state = self._state
        row: dict[str, str | int | None] = state.rows[int(index)]
        subject_id = str(row["subject_id"])
        inputs: dict[str, object] | None = None
        include_context_inputs = any(
            (
                state.position_one_hot,
                state.position_index,
                state.ear_one_hot,
                state.ear_index,
                state.frequency_one_hot,
                state.frequency_index,
                state.sample_one_hot,
                state.sample_index,
            )
        )
        if len(state.input_specs) > 0 or include_context_inputs:
            inputs = {}
            for spec in state.input_specs:
                inputs[DatasetSpecWorkflow.get_spec_name(spec)] = DatasetSampleValueSelector.get_sample_value(
                    self,
                    spec,
                    subject_id,
                    row,
                )
            if row["selected_position_index"] is not None:
                position_index = int(row["selected_position_index"])
                if state.position_one_hot:
                    position_encoding = np.zeros(
                        len(state.selected_position_indices), dtype=float
                    )
                    position_encoding[position_index] = 1.0
                    inputs["position_one_hot"] = position_encoding
                if state.position_index:
                    inputs["position_index"] = position_index
            if row["selected_ear_index"] is not None:
                ear_index = int(row["selected_ear_index"])
                if state.ear_one_hot:
                    ear_encoding = np.zeros(len(state.selected_ears), dtype=float)
                    ear_encoding[ear_index] = 1.0
                    inputs["ear_one_hot"] = ear_encoding
                if state.ear_index:
                    inputs["ear_index"] = ear_index
            if row["selected_frequency_index"] is not None:
                frequency_index = int(row["selected_frequency_index"])
                if state.frequency_one_hot:
                    frequency_encoding = np.zeros(
                        len(state.selected_frequency_indices), dtype=float
                    )
                    frequency_encoding[frequency_index] = 1.0
                    inputs["frequency_one_hot"] = frequency_encoding
                if state.frequency_index:
                    inputs["frequency_index"] = frequency_index
            if row["selected_sample_index"] is not None:
                sample_index = int(row["selected_sample_index"])
                if state.sample_one_hot:
                    sample_encoding = np.zeros(len(state.selected_sample_indices), dtype=float)
                    sample_encoding[sample_index] = 1.0
                    inputs["sample_one_hot"] = sample_encoding
                if state.sample_index:
                    inputs["sample_index"] = sample_index

        sample: dict[str, object] = {
            "inputs": inputs,
            "target": None,
        }
        if len(state.target_specs) > 0:
            target_values: dict[str, object] = {}
            for spec in state.target_specs:
                target_values[DatasetSpecWorkflow.get_spec_name(spec)] = DatasetSampleValueSelector.get_sample_value(
                    self,
                    spec,
                    subject_id,
                    row,
                )
            sample["target"] = target_values
        return sample
