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
        """Construct the shared dataset interface used by dataset integrations.

        :class:`~hrtfpykit.datasets.base.BaseDataset` is the common
        implementation behind concrete datasets such as
        :class:`~hrtfpykit.datasets.HUTUBS` and
        :class:`~hrtfpykit.datasets.SONICOM`. It converts a dataset
        configuration, selected input/target specs, resource variants, subject
        exclusions, split settings, and acoustic metadata into a single
        :class:`~hrtfpykit.datasets.state.DatasetState` object. The resulting
        dataset exposes a stable indexed interface for
        :class:`~hrtfpykit.hrtf.HRTF` objects, derived ITD and ILD values,
        spherical-harmonic coefficients, meshes, anthropometry, metadata,
        images, and videos.

        User code normally instantiates a concrete dataset class. Subclasses use
        :class:`~hrtfpykit.datasets.base.BaseDataset` when they need the shared
        construction pipeline while supplying dataset-specific resource
        configuration, defaults, download behavior, or custom value selectors.

        Parameters
        ----------
        root : str or Path
            Local root directory used to scan dataset resources. The path is
            expanded during construction and stored in
            :attr:`~hrtfpykit.datasets.base.BaseDataset.root`.
        config : DatasetConfig or type[DatasetConfig], optional
            Dataset configuration describing subject identifiers, resource
            templates, supported variants, configured exclusions, and optional
            downloadable resources.
            :class:`~hrtfpykit.datasets.base.BaseDataset` requires this value;
            concrete dataset classes provide it automatically.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to each loaded subject HRTF before spec
            values are extracted. The same transform is used by direct
            :meth:`~hrtfpykit.datasets.base.BaseDataset.get_subject_hrtf` calls
            and by indexed sample extraction.
        exclude_subject_ids : str, int, sequence, or None, default=None
            Additional subject references excluded before resource intersection
            and split planning. Integer references are normalized through the
            dataset configuration when supported.
        inputs : spec, sequence of specs, or None, default=None
            Dataset specs exposed under sample inputs. Specs are normalized by
            the spec workflow before resource scanning.
        target : spec, sequence of specs, or None, default=None
            Dataset specs exposed under sample targets. Multiple target specs
            are returned as a dictionary under the ``target`` key.
        dataset_hrtf_variant : str, dict, or None, default=None
            HRTF resource variant selected for local scanning and loading. Datasets
            with one HRTF selector usually accept a string; datasets with multiple
            selectors may accept a mapping with keys such as ``type``,
            ``sample_rate``, and ``version``.
        dataset_mesh_variant : str, dict, or None, default=None
            Mesh resource variant selected for local scanning and loading. Datasets
            with multiple mesh selectors may accept a mapping with keys such as
            ``type`` and ``version``.
        split : {``all``, ``train``, ``validation``, ``test``}, default=``all``
            Split used to choose which available subjects produce rows in this
            dataset instance.
        split_ratio : tuple of float, default=(0.8, 0.1, 0.1)
            Train, validation, and test ratios used when split is not
            ``all``.
        split_seed : int, default=0
            Seed used for deterministic subject shuffling before split assignment.
        verbose : bool, default=False
            If True, prints the resource and dataset summaries after
            construction.

        Raises
        ------
        ValueError
            If config is None or if construction rejects the requested
            variants, split settings, specs, resources, or acoustic context.

        Attributes
        ----------
        _state : :class:`~hrtfpykit.datasets.state.DatasetState`
            Internal state object containing the resolved configuration, root,
            normalized specs, scanned resources, selected subjects, row contexts,
            acoustic metadata, cache, and generated summaries used by every public
            dataset method.
        """
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

        This method is the subject-level access point shared by concrete
        datasets. It applies the same subject normalization, HRTF path lookup,
        cache, and dataset-level HRTF transform used by indexed sample extraction,
        so direct inspection and indexed sample extraction use the same loading
        path.

        Parameters
        ----------
        subject_id : str or int
            Dataset subject reference. Integer values are mapped to the configured
            subject order.

        Returns
        -------
        HRTF
            Loaded :class:`~hrtfpykit.hrtf.HRTF` object after applying any
            dataset-level HRTF transform.

        Raises
        ------
        ValueError
            If dataset state is incomplete, subject mapping fails, HRTF loading
            fails, or the dataset-level HRTF transform does not return an
            :class:`~hrtfpykit.hrtf.HRTF` object.
        KeyError
            If the mapped subject does not have an available HRTF resource in the
            dataset scan.
        FileNotFoundError
            If the resolved HRTF file is missing.
        """

        return load_hrtf(self, subject_id)

    def resources_summary(self) -> str:
        """Return the resource scan summary created during construction.

        The summary describes resources relevant to the selected specs and
        variants, not every resource a dataset family can support. It reports the
        local resource paths considered during construction, resource counts,
        missing files, partial media resources, and subject removals caused by
        resource intersection.

        Returns
        -------
        str
            Human-readable summary of scanned resources used by the selected specs.
        """

        return self._state.resources_summary

    def dataset_summary(self) -> str:
        """Return the dataset summary created during construction.

        The summary captures the final dataset state after resource intersection
        and split planning: root path, selected split, subject counts, normalized
        input and target specs, selected resource variants, row count, and acoustic
        context when HRTF resources are available.

        Returns
        -------
        str
            Human-readable summary of subjects, split, specs, selected variants,
            row count, and acoustic metadata.
        """

        return self._state.dataset_summary

    @property
    def root(self) -> Path:
        """Return the local dataset root.

        The returned path is the expanded root stored during construction and used
        by every resource scanner. It may point to a directory that contains only
        the resource families required by the selected specs.

        Returns
        -------
        Path
            Expanded local dataset root.
        """
        return self._state.root

    @property
    def dataset_hrtf_variant(self) -> str | dict[str, object] | None:
        """Return the selected HRTF resource variant.

        This value records the HRTF variant used for local resource scanning and
        loading. Datasets with one selector axis return a string such as
        ``measured``. Datasets with multiple selector axes return a dictionary
        containing fields such as ``type``, ``sample_rate``, and
        ``version``. None means no HRTF variant was selected or no HRTF
        resource family is configured.

        Returns
        -------
        str, dict, or None
            Selected HRTF variant stored in the dataset state.
        """
        return self._state.dataset_hrtf_variant

    @property
    def dataset_mesh_variant(self) -> str | dict[str, object] | None:
        """Return the selected mesh resource variant.

        This value records the mesh variant used for local resource scanning and
        loading. Datasets with one selector axis return a string. Datasets with
        multiple selector axes return a dictionary containing fields such as
        ``type`` and ``version``. None means no mesh variant was selected
        or no mesh resource family is configured.

        Returns
        -------
        str, dict, or None
            Selected mesh variant stored in the dataset state.
        """
        return self._state.dataset_mesh_variant

    @property
    def split(self) -> str:
        """Return the requested dataset split name.

        The split controls which available subjects become rows in this dataset
        instance. It is stored separately from resource availability so callers can
        distinguish subjects that have all required resources from the subset chosen
        for train, validation, or test use.

        Returns
        -------
        str
            Split name used by this dataset instance.
        """
        return self._state.split

    @property
    def split_ratio(self) -> tuple[float, float, float]:
        """Return train, validation, and test split ratios.

        These ratios are used by the split planner when split is
        ``train``, ``validation``, or ``test``. They remain visible on the
        dataset object so split behavior can be inspected and reproduced.

        Returns
        -------
        tuple of float
            Three split ratios in train, validation, and test order.
        """
        return self._state.split_ratio

    @property
    def split_seed(self) -> int:
        """Return the split random seed.

        The seed controls deterministic subject shuffling before train,
        validation, and test partitioning. It is part of the dataset state so
        selected subjects can be reproduced from the same resource set.

        Returns
        -------
        int
            Seed used for deterministic split planning.
        """
        return self._state.split_seed

    @property
    def inputs(self) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec, ...]:
        """Return input specs used by this dataset.

        The tuple contains the normalized specs that feed sample inputs.
        It reflects spec workflow decisions such as default names, shared
        ``index_by`` axes, context encodings, and dataset-specific validation.

        Returns
        -------
        tuple of specs
            Normalized input specs in extraction order.
        """
        return self._state.input_specs

    @property
    def target(self) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec, ...]:
        """Return target specs used by this dataset.

        The tuple contains the normalized specs that feed sample targets.
        A dataset with no target specs returns None under the ``target`` key
        during indexed access.

        Returns
        -------
        tuple of specs
            Normalized target specs in extraction order.
        """
        return self._state.target_specs

    @property
    def sample_rate(self) -> float | None:
        """Return dataset-level acoustic sample rate.

        The value is derived from the selected HRTF resources after resource
        validation. It represents the dataset-level acoustic context and is not
        changed by per-spec extraction choices such as position, frequency, or
        sample selection. None means the constructed dataset did not require or
        discover HRTF resources.

        Returns
        -------
        float or None
            Sample rate read from selected HRTF resources.
        """
        return self._state.sample_rate

    @property
    def positions(self) -> np.ndarray | None:
        """Return dataset-level source positions.

        These positions describe the full source grid resolved from the selected
        HRTF resources before spec-level row selection. Position-aware specs may
        use only a subset of this grid; that subset is exposed separately through
        :attr:`~hrtfpykit.datasets.base.BaseDataset.selected_position_indices`,
        :attr:`~hrtfpykit.datasets.base.BaseDataset.selected_azimuth_angles`,
        and
        :attr:`~hrtfpykit.datasets.base.BaseDataset.selected_elevation_angles`.

        Returns
        -------
        numpy.ndarray or None
            Source-position array from selected HRTF resources, or None when
            no acoustic context was built.
        """
        return self._state.positions

    @property
    def azimuth_angles(self) -> np.ndarray | None:
        """Return available dataset azimuth angles.

        The angles are derived from the full dataset source grid. They report
        available spatial coverage independently from the subset selected by
        position-indexed specs.

        Returns
        -------
        numpy.ndarray or None
            Unique azimuth angles from the dataset-level source positions.
        """
        return self._state.azimuth_angles

    @property
    def elevation_angles(self) -> np.ndarray | None:
        """Return available dataset elevation angles.

        The angles are derived from the full dataset source grid. They describe
        the available elevation coverage before any position subset selected by
        specs is applied.

        Returns
        -------
        numpy.ndarray or None
            Unique elevation angles from the dataset-level source positions.
        """
        return self._state.elevation_angles

    @property
    def frequency_bins(self) -> np.ndarray | None:
        """Return dataset-level frequency bins.

        The bins come from the selected HRTF resources when frequency-domain data
        are available or can be derived. They define the dataset-level frequency
        axis used by frequency-indexed specs and remain separate from
        :attr:`~hrtfpykit.datasets.base.BaseDataset.selected_frequency_indices`.

        Returns
        -------
        numpy.ndarray or None
            Frequency bins from selected HRTF resources, or None when no
            frequency-domain acoustic context was built.
        """
        return self._state.frequency_bins

    @property
    def sample_indices(self) -> np.ndarray | None:
        """Return dataset-level time sample indices.

        The indices describe the full HRIR sample axis from the selected HRTF
        resources. They support sample-indexed specs while keeping the complete
        time-domain acoustic context inspectable.

        Returns
        -------
        numpy.ndarray or None
            Time-sample indices from selected HRTF resources, or None when no
            time-domain acoustic context was built.
        """
        return self._state.sample_indices

    @property
    def selected_position_indices(self) -> tuple[int, ...]:
        """Return source position indices selected by specs.

        This property exposes the position subset used to build indexed rows after
        explicit position or plane selection. It is separate from
        :attr:`~hrtfpykit.datasets.base.BaseDataset.positions` so selected row
        context does not hide the full source
        grid.

        Returns
        -------
        tuple of int
            Source-position indices into
            :attr:`~hrtfpykit.datasets.base.BaseDataset.positions`.
        """
        return self._state.selected_position_indices

    @property
    def selected_azimuth_angles(self) -> np.ndarray | None:
        """Return azimuth angles selected by position-aware specs.

        The values summarize the selected position subset used for row generation.
        They are None when no selected spec produced a position-indexed
        acoustic subset.

        Returns
        -------
        numpy.ndarray or None
            Unique azimuth angles for selected positions.
        """
        return self._state.selected_azimuth_angles

    @property
    def selected_elevation_angles(self) -> np.ndarray | None:
        """Return elevation angles selected by position-aware specs.

        The values summarize the selected position subset used for row generation.
        They help inspect plane selectors and position-indexed datasets without
        losing the full elevation coverage available through
        :attr:`~hrtfpykit.datasets.base.BaseDataset.elevation_angles`.

        Returns
        -------
        numpy.ndarray or None
            Unique elevation angles for selected positions.
        """
        return self._state.selected_elevation_angles

    @property
    def selected_frequency_indices(self) -> tuple[int, ...]:
        """Return selected frequency-bin indices.

        These indices are used when ``frequency`` appears in the shared
        dataset index_by axes. They identify the frequency bins that expand
        rows and determine how many frequency-indexed samples each selected
        subject contributes.

        Returns
        -------
        tuple of int
            Frequency-bin indices into
            :attr:`~hrtfpykit.datasets.base.BaseDataset.frequency_bins`.
        """
        return self._state.selected_frequency_indices

    @property
    def selected_sample_indices(self) -> tuple[int, ...]:
        """Return selected time-sample indices.

        These indices are used when ``samples`` appears in the shared dataset
        index_by axes. They identify the HRIR samples that expand rows and
        determine how many sample-indexed samples each selected subject contributes.

        Returns
        -------
        tuple of int
            Time-sample indices into
            :attr:`~hrtfpykit.datasets.base.BaseDataset.sample_indices`.
        """
        return self._state.selected_sample_indices

    @property
    def excluded_subjects(self) -> list[str]:
        """Return subjects excluded from this dataset instance.

        This list combines configuration-level exclusions and user-provided
        exclusions after subject-reference normalization. Excluded subjects are
        removed before resource intersection and split planning, so they never
        contribute rows.

        Returns
        -------
        list of str
            Canonical subject identifiers excluded from this dataset instance.
        """
        return list(self._state.excluded_subjects)

    @property
    def available_subjects(self) -> list[str]:
        """Return subjects available after resource intersection.

        Available subjects are the non-excluded subjects that have every resource
        required by the selected input and target specs. This property describes
        resource availability, not necessarily the final train, validation, or test
        split subset.

        Returns
        -------
        list of str
            Canonical subject identifiers available for the selected specs.
        """
        return list(self._state.available_subjects)

    @property
    def selected_subjects(self) -> list[str]:
        """Return subjects selected for the requested split.

        Selected subjects are the available subjects used to build rows for this
        dataset instance. For split=``all``, this usually matches
        :attr:`~hrtfpykit.datasets.base.BaseDataset.available_subjects`; for
        train, validation, or test splits it is a deterministic subset derived
        from :attr:`~hrtfpykit.datasets.base.BaseDataset.split_ratio` and
        :attr:`~hrtfpykit.datasets.base.BaseDataset.split_seed`.

        Returns
        -------
        list of str
            Canonical subject identifiers used to build dataset rows.
        """
        return list(self._state.selected_subjects)

    def __len__(self) -> int:
        """Return the number of dataset rows.

        Rows are created from selected subjects and any shared indexed axes such
        as position, ear, frequency, or samples. The result is the number of
        integer indices accepted by
        :meth:`~hrtfpykit.datasets.base.BaseDataset.__getitem__` before normal Python list
        bounds checking is applied.

        Returns
        -------
        int
            Number of samples addressable by integer indexing.
        """
        return len(self._state.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one sample by integer row index.

        This method resolves the row context, dispatches each input and target
        spec through the value selector layer, and adds requested context
        encodings. It is the runtime path that turns dataset state into sample
        dictionaries for training, evaluation, or direct inspection.

        Returned samples always contain ``inputs`` and ``target`` keys.
        ``inputs`` is None when no input specs and no context encodings were
        requested. ``target`` is None when no target specs were requested.
        When context encodings are requested by specs, keys such as
        ``position_one_hot``, ``position_index``, ``ear_one_hot``,
        ``frequency_index``, or ``sample_index`` are added to
        sample inputs for rows that carry the corresponding context.

        Parameters
        ----------
        index : int
            Dataset row index. Negative integers follow the underlying row-list
            behavior. Non-integer indices are rejected.

        Returns
        -------
        dict[str, object]
            Sample dictionary with ``inputs`` and ``target`` entries.

        Raises
        ------
        TypeError
            If index is not an integer.
        IndexError
            If index is outside the constructed row table.
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
