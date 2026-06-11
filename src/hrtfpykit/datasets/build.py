from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

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
from .resources import DatasetResources
from .state import DatasetState
from .acoustic_context import DatasetAcousticContext
from .specs_workflow import DatasetSpecWorkflow
from .split import DatasetSplitPlanner

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetBuilder:
    def __init__(self, dataset: "BaseDataset") -> None:
        """Orchestrate construction of one dataset instance state.

        :class:`~hrtfpykit.datasets.build.DatasetBuilder` owns the construction
        sequence for :class:`~hrtfpykit.datasets.base.BaseDataset`. It resets the target
        dataset to a fresh :class:`~hrtfpykit.datasets.state.DatasetState`,
        normalizes HRTF and mesh variants, plans specs, scans resources, applies
        subject exclusions and split selection, derives acoustic context, and
        builds the row table consumed by length queries and integer indexing.

        The builder is an internal orchestration object rather than a user-facing
        dataset. It keeps construction logic outside
        :meth:`~hrtfpykit.datasets.base.BaseDataset.__init__`
        while still writing every value into the shared dataset state explicitly.
        Concrete datasets such as :class:`~hrtfpykit.datasets.HUTUBS`
        and :class:`~hrtfpykit.datasets.SONICOM` call it through
        their base-class initializer.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset instance whose state will be replaced and populated by
            :meth:`~hrtfpykit.datasets.build.DatasetBuilder.build`.
        """
        self._dataset = dataset

    def build(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        dataset_hrtf_transform: Callable[[object], object] | None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None,
        dataset_hrtf_variant: str | Mapping[str, object] | None,
        dataset_mesh_variant: str | Mapping[str, object] | None,
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
        preload_hrtfs: bool,
        check_sofa_against_conventions: bool,
        sofa_open: bool,
        subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        verbose: bool = False,
    ) -> None:
        """Build all explicit state for one dataset instance.

        This method is the central construction pipeline for
        :class:`~hrtfpykit.datasets.base.BaseDataset`. It replaces the current
        dataset state, stores the basic construction arguments, validates and
        normalizes HRTF and mesh variants, asks
        :class:`~hrtfpykit.datasets.specs_workflow.DatasetSpecWorkflow` to
        normalize input and target specs, asks
        :class:`~hrtfpykit.datasets.resources.DatasetResources` to scan local
        resources, asks :class:`~hrtfpykit.datasets.split.DatasetSplitPlanner`
        to select subjects, asks
        :class:`~hrtfpykit.datasets.acoustic_context.DatasetAcousticContext` to
        derive acoustic axes, and finally builds the row table consumed by
        indexed sample extraction.

        Variant normalization happens before resource scanning. HRTF mappings may
        contain ``type``, ``sample_rate``, and ``version`` keys. Mesh
        mappings may contain ``type`` and ``version`` keys. Unsupported keys
        or values are rejected against the dataset configuration before any file
        scan begins.

        Parameters
        ----------
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration used for subject identifiers, supported resource
            variants, resource templates, and configured exclusions.
        root : str or Path
            Local dataset root. The path is expanded with Path(root).expanduser()
            and stored in the dataset state.
        dataset_hrtf_transform : callable or None
            Optional transform stored for later subject HRTF loading. The builder
            does not call the transform.
        inputs : spec, sequence of specs, or None
            Requested input specs exposed under sample inputs.
        target : spec, sequence of specs, or None
            Requested target specs exposed under sample targets.
        dataset_hrtf_variant : str, dict, or None
            HRTF resource variant selected for dataset construction. Mapping
            values are normalized into a stored dictionary when sample-rate or
            version selectors are present; simple type-only variants are stored as
            strings.
        dataset_mesh_variant : str, dict, or None
            Mesh resource variant selected for dataset construction. Mapping
            values are normalized into a stored dictionary when a version selector
            is present; simple type-only variants are stored as strings.
        split : str
            Requested subject split, passed to the split planner.
        split_ratio : tuple of float
            Train, validation, and test ratios used for split planning.
        split_seed : int
            Seed used for deterministic subject shuffling.
        preload_hrtfs : bool
            Whether selected subject HRTFs should be loaded into the dataset
            cache during construction after rows are built.
        check_sofa_against_conventions : bool
            Whether dataset HRTF loading should run SOFA convention checks before
            reading files.
        sofa_open : bool
            Whether HRTFs loaded by the dataset should keep their backing SOFA
            netCDF datasets open.
        subject_ids : str, int, sequence, or None
            Optional subject references used as the initial construction scope
            before exclusions, resource intersection, and split planning. None
            uses every configured subject.
        exclude_subject_ids : str, int, sequence, or None
            Additional subjects excluded from the selected construction scope
            before resource intersection and split planning.
        verbose : bool
            Whether verbose dataset behavior is enabled. The value is stored in
            state for loaders and summaries.

        Returns
        -------
        None
            The method mutates the target dataset state in place.

        Raises
        ------
        ValueError
            If HRTF or mesh variant keys, types, sample rates, or versions are
            unsupported, required variant selectors are missing, or delegated spec,
            resource, split, acoustic-context, or row-building validation fails.
        TypeError
            If delegated workflows receive values with unsupported Python types.

        Notes
        -----
        Specs are normalized before resource scanning so only required resource
        families are scanned. Resource scanning runs before split planning so
        subjects missing required files are removed before train, validation, or
        test selection. Acoustic context runs after split planning because it
        uses one selected subject as the representative HRTF axis source.
        """
        dataset = self._dataset
        state = DatasetState()
        dataset._state = state

        if isinstance(config, type):
            config = cast(DatasetConfig, cast(Any, config)())

        state.config = config
        state.name = str(config.name)
        state.root = Path(root).expanduser()
        state.dataset_hrtf_transform = dataset_hrtf_transform
        state.requested_subjects = None if subject_ids is None else DatasetSplitPlanner.map_subject_ids(
            subject_ids,
            tuple(config.subject_ids),
        )
        state.preload_hrtfs = bool(preload_hrtfs)
        state.check_sofa_against_conventions = bool(check_sofa_against_conventions)
        state.sofa_open = bool(sofa_open)
        state.verbose = bool(verbose)

        state.dataset_hrtf_variant = None
        if config.hrtf is not None:
            if isinstance(dataset_hrtf_variant, Mapping):
                unknown_keys = set(dataset_hrtf_variant) - {"type", "sample_rate", "version"}
                if len(unknown_keys) > 0:
                    raise ValueError(
                        f"Unsupported dataset_hrtf_variant keys {tuple(sorted(unknown_keys))}. "
                        "Expected keys are ('type', 'sample_rate', 'version')"
                    )
                hrtf_type_value = dataset_hrtf_variant.get("type")
                hrtf_sample_rate = dataset_hrtf_variant.get("sample_rate")
                hrtf_version_value = dataset_hrtf_variant.get("version")
            else:
                hrtf_type_value = dataset_hrtf_variant
                hrtf_sample_rate = None
                hrtf_version_value = None

            hrtf_type = None if hrtf_type_value is None else str(hrtf_type_value).strip().lower()
            if hrtf_type is not None:
                if hrtf_type not in config.hrtf.types:
                    raise ValueError(
                        f"Unsupported dataset_hrtf_variant type {hrtf_type!r}. Expected one of {tuple(config.hrtf.types)}"
                    )
                hrtf_type_config = config.hrtf.types[hrtf_type]
                if len(hrtf_type_config.sample_rates) == 0 and hrtf_sample_rate is not None:
                    raise ValueError(
                        f"dataset_hrtf_variant sample_rate is not supported for type={hrtf_type!r}"
                    )
                if len(hrtf_type_config.sample_rates) > 0:
                    if hrtf_sample_rate is None:
                        raise ValueError(
                            f"{config.name} requires dataset_hrtf_variant sample_rate for type={hrtf_type!r}"
                        )
                    if hrtf_sample_rate not in hrtf_type_config.sample_rates:
                        raise ValueError(
                            f"Unsupported dataset_hrtf_variant sample_rate {hrtf_sample_rate!r} for type={hrtf_type!r}. "
                            f"Expected one of {hrtf_type_config.sample_rates}"
                        )
                hrtf_version = None if hrtf_version_value is None else str(hrtf_version_value)
                if len(hrtf_type_config.versions) == 0 and hrtf_version_value is not None:
                    raise ValueError(
                        f"dataset_hrtf_variant version is not supported for type={hrtf_type!r}"
                    )
                if len(hrtf_type_config.versions) > 0 and hrtf_version is not None:
                    if hrtf_version not in hrtf_type_config.versions:
                        raise ValueError(
                            f"Unsupported dataset_hrtf_variant version {hrtf_version!r} for type={hrtf_type!r}. "
                            f"Expected one of {hrtf_type_config.versions}"
                        )
                if hrtf_sample_rate is None and hrtf_version is None:
                    state.dataset_hrtf_variant = hrtf_type
                else:
                    state.dataset_hrtf_variant = {
                        "type": hrtf_type,
                        "sample_rate": hrtf_sample_rate,
                        "version": hrtf_version,
                    }

        state.dataset_mesh_variant = None
        if config.mesh is not None:
            if isinstance(dataset_mesh_variant, Mapping):
                unknown_keys = set(dataset_mesh_variant) - {"type", "version"}
                if len(unknown_keys) > 0:
                    raise ValueError(
                        f"Unsupported dataset_mesh_variant keys {tuple(sorted(unknown_keys))}. "
                        "Expected keys are ('type', 'version')"
                    )
                mesh_type_value = dataset_mesh_variant.get("type")
                mesh_version_value = dataset_mesh_variant.get("version")
            else:
                mesh_type_value = dataset_mesh_variant
                mesh_version_value = None

            mesh_type = None if mesh_type_value is None else str(mesh_type_value).strip().lower()
            if mesh_type is not None:
                if mesh_type not in config.mesh.types:
                    raise ValueError(
                        f"Unsupported dataset_mesh_variant type {mesh_type!r}. Expected one of {tuple(config.mesh.types)}"
                    )
                mesh_type_config = config.mesh.types[mesh_type]
                mesh_version = None if mesh_version_value is None else str(mesh_version_value)
                if len(mesh_type_config.versions) == 0 and mesh_version_value is not None:
                    raise ValueError(
                        f"dataset_mesh_variant version is not supported for type={mesh_type!r}"
                    )
                if len(mesh_type_config.versions) > 0:
                    if mesh_version is None:
                        raise ValueError(
                            f"{config.name} requires dataset_mesh_variant version for type={mesh_type!r}"
                        )
                    if mesh_version not in mesh_type_config.versions:
                        raise ValueError(
                            f"Unsupported dataset_mesh_variant version {mesh_version!r} for type={mesh_type!r}. "
                            f"Expected one of {mesh_type_config.versions}"
                        )
                if mesh_version is None:
                    state.dataset_mesh_variant = mesh_type
                else:
                    state.dataset_mesh_variant = {
                        "type": mesh_type,
                        "version": mesh_version,
                    }

        spec_plan = DatasetSpecWorkflow.build(
            config=config,
            inputs=inputs,
            target=target,
        )
        state.input_specs = spec_plan.input_specs
        state.target_specs = spec_plan.target_specs
        state.specs = spec_plan.specs
        state.input_names = spec_plan.input_names
        state.target_names = spec_plan.target_names
        state.index_by = spec_plan.index_by
        state.selected_ears = spec_plan.selected_ears
        state.position_one_hot = spec_plan.position_one_hot
        state.position_index = spec_plan.position_index
        state.frequency_one_hot = spec_plan.frequency_one_hot
        state.frequency_index = spec_plan.frequency_index
        state.sample_one_hot = spec_plan.sample_one_hot
        state.sample_index = spec_plan.sample_index
        state.ear_one_hot = spec_plan.ear_one_hot
        state.ear_index = spec_plan.ear_index

        resource_plan = DatasetResources.build(
            dataset,
            subject_ids=subject_ids,
            exclude_subject_ids=exclude_subject_ids,
        )
        state.hrtf_paths = resource_plan.hrtf_paths
        state.mesh_paths = resource_plan.mesh_paths
        state.image_path = resource_plan.image_path
        state.video_path = resource_plan.video_path
        state.image_index = resource_plan.image_index
        state.video_index = resource_plan.video_index
        state.image_counts = resource_plan.image_counts
        state.video_counts = resource_plan.video_counts
        state.anthropometry_path = resource_plan.anthropometry_path
        state.anthropometry_rows = resource_plan.anthropometry_rows
        state.metadata_path = resource_plan.metadata_path
        state.metadata_rows = resource_plan.metadata_rows
        state.excluded_subjects = resource_plan.excluded_subjects
        state.resource_summary = resource_plan.resource_summary
        state.subject_numbers = resource_plan.subject_numbers

        split_plan = DatasetSplitPlanner.build(
            dataset,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        state.available_subjects = split_plan.available_subjects
        state.selected_subjects = split_plan.selected_subjects
        state.split = split_plan.split
        state.split_ratio = split_plan.split_ratio
        state.split_seed = split_plan.split_seed

        acoustic_context = DatasetAcousticContext.build(dataset)
        state.sample_rate = acoustic_context.sample_rate
        state.positions = acoustic_context.positions
        state.azimuth_angles = acoustic_context.azimuth_angles
        state.elevation_angles = acoustic_context.elevation_angles
        state.frequency_bins = acoustic_context.frequency_bins
        state.sample_indices = acoustic_context.sample_indices
        state.selected_position_indices = acoustic_context.selected_position_indices
        state.selected_azimuth_angles = acoustic_context.selected_azimuth_angles
        state.selected_elevation_angles = acoustic_context.selected_elevation_angles
        state.selected_frequency_indices = acoustic_context.selected_frequency_indices
        state.selected_sample_indices = acoustic_context.selected_sample_indices
        state.spec_position_indices = {
            spec_id: position_indices
            for spec_id, position_indices in acoustic_context.spec_position_indices
        }
        state.spec_frequency_indices = {
            spec_id: frequency_indices
            for spec_id, frequency_indices in acoustic_context.spec_frequency_indices
        }

        state.rows = self._build_rows(
            subject_ids=state.selected_subjects,
            index_by=state.index_by,
            selected_position_indices=state.selected_position_indices,
            selected_ears=state.selected_ears,
            selected_frequency_indices=state.selected_frequency_indices,
            selected_sample_indices=state.selected_sample_indices,
        )
        dataset.clear_cache()
        if state.preload_hrtfs:
            dataset.preload_hrtfs()

    @staticmethod
    def _build_rows(
        subject_ids: tuple[str, ...],
        index_by: tuple[str, ...],
        selected_position_indices: tuple[int, ...] | list[int],
        selected_ears: tuple[tuple[str, int], ...] | list[tuple[str, int]],
        selected_frequency_indices: tuple[int, ...] | list[int],
        selected_sample_indices: tuple[int, ...] | list[int],
    ) -> list[dict[str, str | int | None]]:
        """Build dataset row dictionaries from selected subjects and row axes.

        Each row records one subject plus optional context for the axes present in
        ``index_by``. The method creates the Cartesian product of selected
        subjects and selected axis values. Axes not present in ``index_by`` are
        represented by None values in the row dictionary, which lets value
        selectors distinguish subject-only specs from position-, ear-,
        frequency-, or sample-indexed specs without changing the row schema.

        Row records are lightweight. They contain indices and labels only; HRTF,
        mesh, table, image, or video resources are loaded later by
        :class:`~hrtfpykit.datasets.values.DatasetSampleValueSelector` when
        :meth:`~hrtfpykit.datasets.base.BaseDataset.__getitem__` requests a sample.

        Parameters
        ----------
        subject_ids : tuple of str
            Canonical subject identifiers included in the selected split.
        index_by : tuple of str
            Dataset row axes. Supported non-subject axes are ``position``,
            ``ear``, ``frequency``, and ``samples``.
        selected_position_indices : sequence
            Source-position indices used when ``position`` is included in
            ``index_by``.
        selected_ears : sequence
            Ear labels and source-ear indices used when ``ear`` is included in
            ``index_by``.
        selected_frequency_indices : sequence
            Frequency-bin indices used when ``frequency`` is included in
            ``index_by``.
        selected_sample_indices : sequence
            Time-sample indices used when ``samples`` is included in
            ``index_by``.

        Returns
        -------
        list of dict
            Row records consumed by
            :meth:`~hrtfpykit.datasets.base.BaseDataset.__getitem__`. Each record
            contains ``subject_id``, ``position_index``,
            ``selected_position_index``, ``ear``, ``ear_index``,
            ``selected_ear_index``, ``frequency_index``,
            ``selected_frequency_index``, ``sample_index``, and
            ``selected_sample_index``.

        Raises
        ------
        ValueError
            If selected axis values cannot be converted to integers or ear entries
            cannot be unpacked as (ear_name, ear_index) pairs.

        Notes
        -----
        ``position_index``, ``frequency_index``, and ``sample_index`` store the
        actual index into the acoustic axis. The corresponding selected
        fields store the ordinal index inside the selected subset and are used for
        one-hot encodings.
        """
        rows: list[dict[str, str | int | None]] = []
        include_position = "position" in index_by
        include_ear = "ear" in index_by
        include_frequency = "frequency" in index_by
        include_samples = "samples" in index_by
        for subject_id in subject_ids:
            position_values = (
                [(None, None)]
                if not include_position
                else [
                    (int(position_index), int(selected_position_index))
                    for selected_position_index, position_index in enumerate(selected_position_indices)
                ]
            )
            ear_values = (
                [(None, None, None)]
                if not include_ear
                else [
                    (ear_name, int(ear_index), int(selected_ear_index))
                    for selected_ear_index, (ear_name, ear_index) in enumerate(selected_ears)
                ]
            )
            frequency_values = (
                [(None, None)]
                if not include_frequency
                else [
                    (int(frequency_index), int(selected_frequency_index))
                    for selected_frequency_index, frequency_index in enumerate(selected_frequency_indices)
                ]
            )
            sample_values = (
                [(None, None)]
                if not include_samples
                else [
                    (int(sample_index), int(selected_sample_index))
                    for selected_sample_index, sample_index in enumerate(selected_sample_indices)
                ]
            )
            for position_index, selected_position_index in position_values:
                for ear_name, ear_index, selected_ear_index in ear_values:
                    for frequency_index, selected_frequency_index in frequency_values:
                        for sample_index, selected_sample_index in sample_values:
                            rows.append(
                                {
                                    "subject_id": subject_id,
                                    "position_index": position_index,
                                    "selected_position_index": selected_position_index,
                                    "ear": ear_name,
                                    "ear_index": ear_index,
                                    "selected_ear_index": selected_ear_index,
                                    "frequency_index": frequency_index,
                                    "selected_frequency_index": selected_frequency_index,
                                    "sample_index": sample_index,
                                    "selected_sample_index": selected_sample_index,
                                }
                            )
        return rows
