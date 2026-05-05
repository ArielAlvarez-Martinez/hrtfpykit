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
        dataset_hrtf_type: str | None = None,
        dataset_hrtf_sample_rate: int | str | None = None,
        dataset_hrtf_version: str | None = None,
        dataset_mesh_type: str | None = None,
        dataset_mesh_version: str | None = None,
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
            dataset_hrtf_type=dataset_hrtf_type,
            dataset_hrtf_sample_rate=dataset_hrtf_sample_rate,
            dataset_hrtf_version=dataset_hrtf_version,
            dataset_mesh_type=dataset_mesh_type,
            dataset_mesh_version=dataset_mesh_version,
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
        return load_hrtf(self, subject_id)

    def resources_summary(self) -> str:
        return self._state.resources_summary

    def dataset_summary(self) -> str:
        return self._state.dataset_summary

    @property
    def root(self) -> Path:
        return self._state.root

    @property
    def dataset_hrtf_type(self) -> str | None:
        return self._state.dataset_hrtf_type

    @property
    def dataset_hrtf_sample_rate(self) -> int | str | None:
        return self._state.dataset_hrtf_sample_rate

    @property
    def dataset_hrtf_version(self) -> str | None:
        return self._state.dataset_hrtf_version

    @property
    def dataset_mesh_type(self) -> str | None:
        return self._state.dataset_mesh_type

    @property
    def dataset_mesh_version(self) -> str | None:
        return self._state.dataset_mesh_version

    @property
    def split(self) -> str:
        return self._state.split

    @property
    def split_ratio(self) -> tuple[float, float, float]:
        return self._state.split_ratio

    @property
    def split_seed(self) -> int:
        return self._state.split_seed

    @property
    def inputs(self) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec, ...]:
        return self._state.input_specs

    @property
    def target(self) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec, ...]:
        return self._state.target_specs

    @property
    def sample_rate(self) -> float | None:
        return self._state.sample_rate

    @property
    def positions(self) -> np.ndarray | None:
        return self._state.positions

    @property
    def azimuth_angles(self) -> np.ndarray | None:
        return self._state.azimuth_angles

    @property
    def elevation_angles(self) -> np.ndarray | None:
        return self._state.elevation_angles

    @property
    def frequency_bins(self) -> np.ndarray | None:
        return self._state.frequency_bins

    @property
    def sample_indices(self) -> np.ndarray | None:
        return self._state.sample_indices

    @property
    def selected_position_indices(self) -> tuple[int, ...]:
        return self._state.selected_position_indices

    @property
    def selected_azimuth_angles(self) -> np.ndarray | None:
        return self._state.selected_azimuth_angles

    @property
    def selected_elevation_angles(self) -> np.ndarray | None:
        return self._state.selected_elevation_angles

    @property
    def selected_frequency_indices(self) -> tuple[int, ...]:
        return self._state.selected_frequency_indices

    @property
    def selected_sample_indices(self) -> tuple[int, ...]:
        return self._state.selected_sample_indices

    @property
    def excluded_subjects(self) -> list[str]:
        return list(self._state.excluded_subjects)

    @property
    def available_subjects(self) -> list[str]:
        return list(self._state.available_subjects)

    @property
    def selected_subjects(self) -> list[str]:
        return list(self._state.selected_subjects)

    def __len__(self) -> int:
        return len(self._state.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
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
