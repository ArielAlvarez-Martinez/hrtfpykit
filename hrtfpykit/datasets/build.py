from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .config import DatasetConfig
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)
from .resources import DatasetResources, DatasetResourcesPlan
from . import acoustic_context as acoustic_context_module
from . import specs_workflow as specs_workflow_module
from . import split as split_module

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetBuilder:
    def __init__(self, dataset: "BaseDataset") -> None:
        self._dataset = dataset

    def build(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        dataset_hrtf_transform: Callable[[object], object] | None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
        variant: str | None,
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
    ) -> None:
        dataset = self._dataset
        self._initialize_dataset_metadata(
            dataset=dataset,
            config=config,
            root=root,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
        )
        self._initialize_variant(
            dataset=dataset,
            config=config,
            variant=variant,
        )
        self._apply_spec_plan(
            dataset=dataset,
            spec_plan=specs_workflow_module.DatasetSpecWorkflow(config).build(
                inputs=inputs,
                target=target,
            ),
        )
        dataset._cache = {}
        self._apply_resource_plan(
            dataset=dataset,
            resource_plan=DatasetResources.build(dataset),
        )
        self._apply_subject_split(
            dataset=dataset,
            split_plan=split_module.DatasetSubjectSplitPlanner.build(
                dataset,
                split=split,
                split_ratio=split_ratio,
                split_seed=split_seed,
            ),
        )
        self._apply_acoustic_context(
            dataset=dataset,
            acoustic_context=acoustic_context_module.DatasetAcousticContext().build(dataset),
        )
        dataset._rows = self._build_rows(
            subject_ids=dataset._subject_ids,
            index_by=dataset._index_by,
            selected_position_indices=dataset._selected_position_indices,
            selected_ears=dataset._selected_ears,
            selected_frequency_indices=dataset._selected_frequency_indices,
            selected_sample_indices=dataset._selected_sample_indices,
        )
        print(dataset._format_load_summary())

    @staticmethod
    def _initialize_dataset_metadata(
        dataset: "BaseDataset",
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        dataset_hrtf_transform: Callable[[object], object] | None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None,
    ) -> None:
        dataset._config = config
        dataset._name = str(config.name)
        dataset._root = Path(root).expanduser()
        dataset._dataset_hrtf_transform = dataset_hrtf_transform
        if exclude_subject_ids is None:
            dataset._exclude_subject_ids = tuple()
        else:
            if isinstance(exclude_subject_ids, tuple):
                raw_exclude_subject_ids = tuple(exclude_subject_ids)
            elif isinstance(exclude_subject_ids, list):
                raw_exclude_subject_ids = tuple(exclude_subject_ids)
            else:
                raw_exclude_subject_ids = (exclude_subject_ids,)
            subject_ids = tuple(config.subject_ids)
            normalized_subject_ids = {str(value) for value in subject_ids}
            all_known = all(str(value) in normalized_subject_ids for value in raw_exclude_subject_ids)
            if all_known:
                dataset._exclude_subject_ids = tuple(dict.fromkeys(str(value) for value in raw_exclude_subject_ids))
            else:
                dataset._exclude_subject_ids = split_module.DatasetSubjectSplitPlanner.map_subject_ids(
                    raw_exclude_subject_ids,
                    subject_ids,
                )

    @staticmethod
    def _initialize_variant(
        dataset: "BaseDataset",
        config: type[DatasetConfig] | DatasetConfig,
        variant: str | None,
    ) -> None:
        preset_variant = variant if variant is not None else getattr(dataset, "variant", None)
        dataset.variant = None
        if config.hrtf is not None:
            dataset.variant = str(config.hrtf.default_variant).strip().lower()
        if preset_variant is not None:
            dataset.variant = preset_variant

    @staticmethod
    def _apply_spec_plan(
        dataset: "BaseDataset",
        spec_plan: specs_workflow_module.DatasetSpecPlan,
    ) -> None:
        dataset._input_specs = spec_plan.input_specs
        dataset._target_specs = spec_plan.target_specs
        dataset._specs = spec_plan.specs
        dataset._input_names = spec_plan.input_names
        dataset._target_names = spec_plan.target_names
        dataset._index_by = spec_plan.index_by
        dataset._selected_ears = list(spec_plan.selected_ears)
        dataset._position_one_hot = spec_plan.position_one_hot
        dataset._position_index = spec_plan.position_index
        dataset._frequency_one_hot = spec_plan.frequency_one_hot
        dataset._frequency_index = spec_plan.frequency_index
        dataset._sample_one_hot = spec_plan.sample_one_hot
        dataset._sample_index = spec_plan.sample_index
        dataset._ear_one_hot = spec_plan.ear_one_hot
        dataset._ear_index = spec_plan.ear_index

    @staticmethod
    def _apply_resource_plan(
        dataset: "BaseDataset",
        resource_plan: DatasetResourcesPlan,
    ) -> None:
        dataset._hrtf_paths = resource_plan.hrtf_paths
        dataset._mesh_paths = resource_plan.mesh_paths
        dataset._image_path = resource_plan.image_path
        dataset._video_path = resource_plan.video_path
        dataset._image_index = resource_plan.image_index
        dataset._video_index = resource_plan.video_index
        dataset._image_counts = resource_plan.image_counts
        dataset._video_counts = resource_plan.video_counts
        dataset._anthropometry_path = resource_plan.anthropometry_path
        dataset._anthropometry_rows = resource_plan.anthropometry_rows
        dataset._resource_summary = resource_plan.resource_summary
        dataset._included_subject_ids = resource_plan.included_subject_ids
        dataset._subject_numbers = resource_plan.subject_numbers

    @staticmethod
    def _apply_subject_split(
        dataset: "BaseDataset",
        split_plan: split_module.DatasetSubjectSplitPlan,
    ) -> None:
        dataset._available_subject_ids = split_plan.available_subject_ids
        dataset._subject_ids = split_plan.subject_ids
        dataset._split = split_plan.split
        dataset._split_ratio = split_plan.split_ratio
        dataset._split_seed = split_plan.split_seed

    @staticmethod
    def _apply_acoustic_context(
        dataset: "BaseDataset",
        acoustic_context: acoustic_context_module.DatasetAcousticContextPlan,
    ) -> None:
        dataset._dataset_sample_rate = acoustic_context.dataset_sample_rate
        dataset._dataset_source_positions = acoustic_context.dataset_source_positions
        dataset._available_azimuth_angles = acoustic_context.available_azimuth_angles
        dataset._available_elevation_angles = acoustic_context.available_elevation_angles
        dataset._azimuth_angles = acoustic_context.azimuth_angles
        dataset._elevation_angles = acoustic_context.elevation_angles
        dataset._frequency_bins = acoustic_context.frequency_bins
        dataset._sample_indices = acoustic_context.sample_indices
        dataset._selected_position_indices = list(acoustic_context.selected_position_indices)
        dataset._selected_frequency_indices = list(acoustic_context.selected_frequency_indices)
        dataset._selected_sample_indices = list(acoustic_context.selected_sample_indices)
        dataset._spec_position_indices = {
            spec_id: list(position_indices)
            for spec_id, position_indices in acoustic_context.spec_position_indices
        }

    @staticmethod
    def _build_rows(
        subject_ids: tuple[str, ...],
        index_by: tuple[str, ...],
        selected_position_indices: tuple[int, ...] | list[int],
        selected_ears: tuple[tuple[str, int], ...] | list[tuple[str, int]],
        selected_frequency_indices: tuple[int, ...] | list[int],
        selected_sample_indices: tuple[int, ...] | list[int],
    ) -> list[dict[str, str | int | None]]:
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
