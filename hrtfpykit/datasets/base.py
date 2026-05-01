from pathlib import Path
from collections.abc import Sequence
from typing import Callable, TYPE_CHECKING

import numpy as np

from .build import (
    DatasetBuilder,
)
from .specs_workflow import DatasetSpecWorkflow
from .config import DatasetConfig
from .load import load_hrtf
from .values import DatasetSampleValueSelector
from . import summary as dataset_summary_module
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

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


class BaseDataset(DatasetSampleValueSelector):
    _config: type[DatasetConfig] | DatasetConfig | None = None

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
        DatasetBuilder(self).build(
            config=config,
            root=root,
            dataset_hrtf_transform=dataset_hrtf_transform,
            inputs=inputs,
            target=target,
            variant=variant,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            exclude_subject_ids=exclude_subject_ids,
        )

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
        return DatasetSpecWorkflow.filter_specs(spec_types, selected_specs)

    def _get_indexed_specs(
        self,
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...]:
        return DatasetSpecWorkflow.get_indexed_specs(self._specs)

    def get_subject_hrtf(self, subject_id: str | int) -> "HRTF":
        return load_hrtf(self, subject_id)

    def _format_load_summary(self) -> str:
        return self.dataset_summary()

    def dataset_summary(self) -> str:
        return dataset_summary_module.dataset_summary(self)

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
                inputs[DatasetSpecWorkflow.get_spec_name(spec)] = DatasetSampleValueSelector.get_sample_value(
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
                target_values[DatasetSpecWorkflow.get_spec_name(spec)] = DatasetSampleValueSelector.get_sample_value(
                    self,
                    spec,
                    subject_id,
                    row,
                )
            sample["target"] = target_values
        return sample
