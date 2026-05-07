from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable

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
    """Build explicit dataset state from config, specs, resources, and splits.

    ``DatasetBuilder`` owns the construction sequence for ``BaseDataset``. It
    normalizes variants, applies spec planning, scans resources, applies subject
    splits, derives acoustic context, and writes the final ``DatasetState``.
    """
    def __init__(self, dataset: "BaseDataset") -> None:
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
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        verbose: bool = False,
    ) -> None:
        """Build all explicit dataset state for one dataset instance.

        This method is the central construction pipeline for ``BaseDataset``. It
        initializes state, validates resource variants, normalizes specs, scans and
        validates resources, plans the split, derives acoustic context, and builds the
        row table consumed by ``__getitem__``.

        Parameters
        ----------
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration.
        root : str or Path
            Local dataset root.
        dataset_hrtf_transform : callable or None
            Optional loaded-HRTF transform.
        inputs, target : spec, sequence of specs, or None
            Requested input and target specs.
        dataset_hrtf_variant : str, dict, or None
            HRTF resource variant selected for dataset construction.
        dataset_mesh_variant : str, dict, or None
            Mesh resource variant selected for dataset construction.
        split : str
            Requested split.
        split_ratio : tuple of float
            Train, validation, and test ratios.
        split_seed : int
            Deterministic split seed.
        exclude_subject_ids : str, int, sequence, or None
            Additional subjects to exclude.
        verbose : bool
            Whether verbose dataset behavior is enabled.

        Returns
        -------
        None
            Assigns the final dataset state in-place.

        Use Cases
        ---------
        - Execute spec planning, resource scanning, splitting, and acoustic context.
        - Build rows for ``BaseDataset.__getitem__``.
        """
        dataset = self._dataset
        state = DatasetState()
        dataset._state = state

        state.config = config
        state.name = str(config.name)
        state.root = Path(root).expanduser()
        state.dataset_hrtf_transform = dataset_hrtf_transform
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
                if len(hrtf_type_config.versions) > 0:
                    if hrtf_version is None:
                        raise ValueError(
                            f"{config.name} requires dataset_hrtf_variant version for type={hrtf_type!r}"
                        )
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

        state.rows = self._build_rows(
            subject_ids=state.selected_subjects,
            index_by=state.index_by,
            selected_position_indices=state.selected_position_indices,
            selected_ears=state.selected_ears,
            selected_frequency_indices=state.selected_frequency_indices,
            selected_sample_indices=state.selected_sample_indices,
        )

    @staticmethod
    def _build_rows(
        subject_ids: tuple[str, ...],
        index_by: tuple[str, ...],
        selected_position_indices: tuple[int, ...] | list[int],
        selected_ears: tuple[tuple[str, int], ...] | list[tuple[str, int]],
        selected_frequency_indices: tuple[int, ...] | list[int],
        selected_sample_indices: tuple[int, ...] | list[int],
    ) -> list[dict[str, str | int | None]]:
        """Build dataset row dictionaries from selected subjects and indexed axes.

        Rows are records that identify one subject plus optional selected
        position, ear, frequency, and sample indices. Value selectors consume these
        records later, which keeps expensive resource loading out of row construction
        and makes sample extraction deterministic.

        Parameters
        ----------
        subject_ids : tuple of str
            Subjects included in the selected split.
        index_by : tuple of str
            Dataset row axes.
        selected_position_indices : sequence
            Position indices used to expand rows.
        selected_ears : sequence
            Ear names and indices used to expand rows.
        selected_frequency_indices : sequence
            Frequency indices used to expand rows.
        selected_sample_indices : sequence
            Sample indices used to expand rows.

        Returns
        -------
        list of dict
            Row records consumed by ``BaseDataset.__getitem__``.

        Use Cases
        ---------
        - Expand subject-level resources into indexed sample rows.
        - Keep row context explicit for value selectors.
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
