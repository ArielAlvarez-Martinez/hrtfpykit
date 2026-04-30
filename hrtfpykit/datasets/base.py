from pathlib import Path
from collections.abc import Sequence
from typing import Callable, TYPE_CHECKING

import numpy as np

from .builder import (
    DatasetAcousticContextBuilder,
    DatasetRowsBuilder,
    DatasetSpecPlanner,
    DatasetSubjectResolver,
    DatasetSubjectSelectionPlanner,
)
from .config import DatasetConfig
from .loader import load_hrtf
from .resources import resolve_dataset_resources
from .sample_values import DatasetSampleValueResolver
from .summary import DatasetSummary
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
    get_spec_name,
)

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


class BaseDataset(DatasetSampleValueResolver):
    _config: type[DatasetConfig] | DatasetConfig | None = None

    _resolve_dataset_subject_id = staticmethod(DatasetSubjectResolver.resolve_subject_id)
    _resolve_dataset_subject_ids = staticmethod(DatasetSubjectResolver.resolve_subject_ids)
    _sort_subject_ids = staticmethod(DatasetSubjectResolver.sort_subject_ids)
    _resolve_positions_selection = staticmethod(
        DatasetAcousticContextBuilder.resolve_positions_selection
    )

    def __init__(
        self,
        root: str | Path,
        config: type[DatasetConfig] | DatasetConfig | None = None,
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None = None,
        variant: str | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        if config is None:
            raise ValueError("BaseDataset requires a dataset config")
        self._initialize_dataset(
            config=config,
            root=root,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            variant=variant,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )

    def _initialize_dataset(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        dataset_hrtf_transform: Callable[[object], object] | None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None,
        variant: str | None,
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
    ) -> None:
        dataset_config = config
        self._config = dataset_config
        self._name = str(self._config.name)
        self._root = Path(root).expanduser()
        self._dataset_hrtf_transform = dataset_hrtf_transform
        self._exclude_subject_ids = self._resolve_dataset_subject_ids(
            exclude_subject_ids,
            tuple(dataset_config.subject_ids),
        )

        preset_variant = variant if variant is not None else getattr(self, "variant", None)
        self.variant = None
        if dataset_config.hrtf is not None:
            self.variant = str(dataset_config.hrtf.default_variant).strip().lower()
        if preset_variant is not None:
            self.variant = preset_variant

        plan = DatasetSpecPlanner(dataset_config).build(inputs=inputs, target=target)
        self._input_specs = plan.input_specs
        self._target_specs = plan.target_specs
        self._specs = plan.specs
        self._input_names = plan.input_names
        self._target_names = plan.target_names
        self._index_by = plan.index_by
        self._selected_ears = list(plan.selected_ears)
        self._position_one_hot = plan.position_one_hot
        self._position_index = plan.position_index
        self._frequency_one_hot = plan.frequency_one_hot
        self._frequency_index = plan.frequency_index
        self._sample_one_hot = plan.sample_one_hot
        self._sample_index = plan.sample_index
        self._ear_one_hot = plan.ear_one_hot
        self._ear_index = plan.ear_index

        self._cache: dict[tuple[object, ...], object] = {}

        resolve_dataset_resources(self)
        subject_split = DatasetSubjectSelectionPlanner().build(
            self,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        self._available_subject_ids = subject_split.available_subject_ids
        self._subject_ids = subject_split.subject_ids
        self._split = subject_split.split
        self._split_ratio = subject_split.split_ratio
        self._split_seed = subject_split.split_seed

        acoustic_context = DatasetAcousticContextBuilder().build(self)
        self._dataset_sample_rate = acoustic_context.dataset_sample_rate
        self._dataset_source_positions = acoustic_context.dataset_source_positions
        self._available_azimuth_angles = acoustic_context.available_azimuth_angles
        self._available_elevation_angles = acoustic_context.available_elevation_angles
        self._azimuth_angles = acoustic_context.azimuth_angles
        self._elevation_angles = acoustic_context.elevation_angles
        self._frequency_bins = acoustic_context.frequency_bins
        self._sample_indices = acoustic_context.sample_indices
        self._selected_position_indices = list(acoustic_context.selected_position_indices)
        self._selected_frequency_indices = list(acoustic_context.selected_frequency_indices)
        self._selected_sample_indices = list(acoustic_context.selected_sample_indices)
        self._spec_position_indices = {
            spec_id: list(position_indices)
            for spec_id, position_indices in acoustic_context.spec_position_indices
        }
        self._rows = DatasetRowsBuilder.build(
            subject_ids=self._subject_ids,
            index_by=self._index_by,
            selected_position_indices=acoustic_context.selected_position_indices,
            selected_ears=self._selected_ears,
            selected_frequency_indices=acoustic_context.selected_frequency_indices,
            selected_sample_indices=acoustic_context.selected_sample_indices,
        )
        print(self._format_load_summary())

    def _get_specs(
        self,
        spec_types: type[object] | tuple[type[object], ...],
        specs: tuple[
            HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
            ...,
        ]
        | None = None,
    ) -> tuple[
        HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec,
        ...,
    ]:
        selected_specs = self._specs if specs is None else specs
        return DatasetSpecPlanner.filter_specs(spec_types, selected_specs)

    def _get_indexed_specs(
        self,
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...]:
        return DatasetSpecPlanner.get_indexed_specs(self._specs)

    def get_subject_hrtf(self, subject_id: str | int) -> "HRTF":
        return load_hrtf(self, subject_id)

    def _format_load_summary(self) -> str:
        lines = [
            f"{self._name} dataset summary",
            f"  root: {self._root}",
            f"  split: {self._split}",
            f"  subjects_loaded: {len(self._subject_ids)}",
            f"  available_subjects: {len(self._available_subject_ids)}",
            f"  samples: {len(self._rows)}",
            f"  inputs: {', '.join(self._input_names) if len(self._input_specs) > 0 else 'none'}",
            f"  target: {', '.join(self._target_names) if len(self._target_specs) > 0 else 'none'}",
        ]
        if len(self._exclude_subject_ids) > 0:
            lines.append(f"  excluded_subjects: {len(self._exclude_subject_ids)}")
        if getattr(self, "variant", None) is not None:
            lines.append(f"  variant: {self.variant}")
        if self._dataset_sample_rate is not None:
            lines.append(f"  dataset_sample_rate: {self._dataset_sample_rate}")
        if self._dataset_source_positions is not None:
            lines.append(f"  dataset_source_positions: {len(self._dataset_source_positions)}")
        lines.append(DatasetSummary.format_resource_summary(self._resource_summary))
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        if not isinstance(index, int):
            raise TypeError("Dataset indexing only supports integer indices")
        row: dict[str, str | int | None] = self._rows[int(index)]
        subject_id = str(row["subject_id"])
        inputs: dict[str, object] | None = None
        include_context_inputs = any(
            (
                self._position_one_hot,
                self._position_index,
                self._ear_one_hot,
                self._ear_index,
                self._frequency_one_hot,
                self._frequency_index,
                self._sample_one_hot,
                self._sample_index,
            )
        )
        if len(self._input_specs) > 0 or include_context_inputs:
            inputs = {}
            for spec in self._input_specs:
                inputs[get_spec_name(spec)] = DatasetSampleValueResolver.get_sample_value(
                    self,
                    spec,
                    subject_id,
                    row,
                )
            if row["selected_position_index"] is not None:
                position_index = int(row["selected_position_index"])
                if self._position_one_hot:
                    position_encoding = np.zeros(
                        len(self._selected_position_indices), dtype=float
                    )
                    position_encoding[position_index] = 1.0
                    inputs["position_one_hot"] = position_encoding
                if self._position_index:
                    inputs["position_index"] = position_index
            if row["selected_ear_index"] is not None:
                ear_index = int(row["selected_ear_index"])
                if self._ear_one_hot:
                    ear_encoding = np.zeros(len(self._selected_ears), dtype=float)
                    ear_encoding[ear_index] = 1.0
                    inputs["ear_one_hot"] = ear_encoding
                if self._ear_index:
                    inputs["ear_index"] = ear_index
            if row["selected_frequency_index"] is not None:
                frequency_index = int(row["selected_frequency_index"])
                if self._frequency_one_hot:
                    frequency_encoding = np.zeros(
                        len(self._selected_frequency_indices), dtype=float
                    )
                    frequency_encoding[frequency_index] = 1.0
                    inputs["frequency_one_hot"] = frequency_encoding
                if self._frequency_index:
                    inputs["frequency_index"] = frequency_index
            if row["selected_sample_index"] is not None:
                sample_index = int(row["selected_sample_index"])
                if self._sample_one_hot:
                    sample_encoding = np.zeros(len(self._selected_sample_indices), dtype=float)
                    sample_encoding[sample_index] = 1.0
                    inputs["sample_one_hot"] = sample_encoding
                if self._sample_index:
                    inputs["sample_index"] = sample_index

        sample: dict[str, object] = {
            "inputs": inputs,
            "target": None,
        }
        if len(self._target_specs) > 0:
            target_values: dict[str, object] = {}
            for spec in self._target_specs:
                target_values[get_spec_name(spec)] = DatasetSampleValueResolver.get_sample_value(
                    self,
                    spec,
                    subject_id,
                    row,
                )
            sample["target"] = target_values
        return sample
