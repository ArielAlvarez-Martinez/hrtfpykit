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
from .resources import DatasetResources
from .state import DatasetState
from .acoustic_context import DatasetAcousticContext
from .specs_workflow import DatasetSpecWorkflow
from .split import DatasetSubjectSplitPlanner

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
        state = DatasetState()
        dataset._state = state

        state.config = config
        state.name = str(config.name)
        state.root = Path(root).expanduser()
        state.dataset_hrtf_transform = dataset_hrtf_transform

        state.variant = None
        if config.hrtf is not None:
            state.variant = str(config.hrtf.default_variant).strip().lower()
        if variant is not None:
            state.variant = variant

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
        state.excluded_subjects = resource_plan.excluded_subjects
        state.resource_summary = resource_plan.resource_summary
        state.subject_numbers = resource_plan.subject_numbers

        split_plan = DatasetSubjectSplitPlanner.build(
            dataset,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        state.available_subjects = split_plan.available_subjects
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
            subject_ids=state.available_subjects,
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
