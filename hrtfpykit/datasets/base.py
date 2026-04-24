from pathlib import Path
import re
from typing import Callable

import numpy as np

from .acoustic import DatasetAcousticContext
from .config import DatasetConfig
from .index import build_rows
from .planner import DatasetSpecPlanner
from .resolver import DatasetResourceResolver
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
from .values import DatasetValueResolver


class BaseDataset(
    DatasetValueResolver,
    DatasetAcousticContext,
    DatasetResourceResolver,
    DatasetSpecPlanner,
):
    config: DatasetConfig | None = None

    def __init__(
        self,
        root: str | Path,
        hrtf_transform: Callable | None = None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec
        | ITDSpec
        | ILDSpec
        | SHSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        target: HRTFSpec
        | ITDSpec
        | ILDSpec
        | SHSpec
        | MeshSpec
        | AnthropometrySpec
        | ImageSpec
        | VideoSpec
        | tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec, ...]
        | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        if type(self).config is None:
            raise ValueError(f"{type(self).__name__} must define a dataset config")
        self.config = type(self).config
        self.name = str(self.config.name)
        self.root = Path(root).expanduser()
        self.hrtf_transform = hrtf_transform
        self.exclude_subject_ids = self.resolve_dataset_subject_ids(
            exclude_subject_ids,
            tuple(self.config.subject_ids),
        )

        preset_variant = getattr(self, "variant", None)
        self.variant = None
        if self.config.hrtf is not None:
            self.variant = str(self.config.hrtf.default_variant).strip().lower()
        if preset_variant is not None:
            self.variant = preset_variant

        self.plan_dataset_specs(inputs, target)

        self._cache_hrtf = True if len(self.hrtf_specs) == 0 else any(bool(spec.cache) for spec in self.hrtf_specs)
        self._hrtf_cache: dict[str, object] = {}
        self._dataset_transformed_hrtf_cache: dict[str, object] = {}
        self._transformed_hrtf_cache: dict[tuple[str, int], object] = {}
        self._metric_cache: dict[tuple[str, int], np.ndarray] = {}
        self._sh_cache: dict[tuple[str, int], np.ndarray] = {}

        self.resolve_dataset_resources()
        self.resolve_dataset_subjects(split, split_ratio, split_seed)
        self.prepare_acoustic_context()
        self._rows = build_rows(
            subject_ids=self.subject_ids,
            index_by=self.index_by,
            position_indices=self._selected_position_indices,
            ears=self._selected_ears,
            frequency_indices=self._selected_frequency_indices,
            sample_indices=self._selected_sample_indices,
        )
        print(self.format_load_summary())

    @staticmethod
    def normalize_subject_id(value: str) -> str:
        return str(value).strip().lower()

    @classmethod
    def resolve_dataset_subject_id(
        cls,
        value: str | int,
        subject_ids: tuple[str, ...],
    ) -> str:
        if len(subject_ids) == 0:
            raise ValueError("subject_ids must not be empty")
        if isinstance(value, int):
            index = int(value)
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        text = str(value).strip()
        if text == "":
            raise ValueError("Subject reference must not be empty")
        subject_map = {subject_id.lower(): subject_id for subject_id in subject_ids}
        if text.lower() in subject_map:
            return subject_map[text.lower()]
        subject_index_match = re.fullmatch(r"subject[_-]?(\d+)", text.strip().lower())
        if subject_index_match is not None:
            index = int(subject_index_match.group(1))
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        if text.isdigit():
            index = int(text)
            if index < 1 or index > len(subject_ids):
                raise ValueError(f"Subject index {index} is out of range for {len(subject_ids)} subjects")
            return subject_ids[index - 1]
        normalized = cls.normalize_subject_id(text)
        if normalized.lower() in subject_map:
            return subject_map[normalized.lower()]
        raise ValueError(f"Unknown subject reference {value!r}")

    @classmethod
    def resolve_dataset_subject_ids(
        cls,
        values: str | int | tuple[str | int, ...] | list[str | int] | None,
        subject_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        if values is None:
            return tuple()
        if isinstance(values, (str, int)):
            return (cls.resolve_dataset_subject_id(values, subject_ids),)
        return tuple(
            dict.fromkeys(
                cls.resolve_dataset_subject_id(value, subject_ids)
                for value in values
            )
        )

    def __len__(self) -> int:
        return len(self._rows)

    def add_input_encodings(
        self,
        inputs: dict[str, object],
        row: dict[str, str | int | None],
    ) -> None:
        if self._positions_encoding == "one-hot" and row["selected_position_index"] is not None:
            position_encoding = np.zeros(len(self._selected_position_indices), dtype=float)
            position_encoding[int(row["selected_position_index"])] = 1.0
            inputs["position"] = position_encoding
        if self._frequencies_encoding == "one-hot" and row["selected_frequency_index"] is not None:
            frequency_encoding = np.zeros(len(self._selected_frequency_indices), dtype=float)
            frequency_encoding[int(row["selected_frequency_index"])] = 1.0
            inputs["frequency"] = frequency_encoding
        if self._samples_encoding == "one-hot" and row["selected_sample_index"] is not None:
            sample_encoding = np.zeros(len(self._selected_sample_indices), dtype=float)
            sample_encoding[int(row["selected_sample_index"])] = 1.0
            inputs["sample"] = sample_encoding
        if self._ear_encoding == "one-hot" and row["selected_ear_index"] is not None:
            ear_encoding = np.zeros(len(self._selected_ears), dtype=float)
            ear_encoding[int(row["selected_ear_index"])] = 1.0
            inputs["ear"] = ear_encoding

    def __getitem__(self, index: int) -> dict[str, object]:
        if not isinstance(index, int):
            raise TypeError("Dataset indexing only supports integer indices")
        row = self._rows[int(index)]
        subject_id = str(row["subject_id"])
        inputs: dict[str, object] | None = None
        if len(self._input_specs) > 0:
            inputs = {}
            for spec in self._input_specs:
                inputs[get_spec_name(spec)] = self.get_spec_value(spec, subject_id, row)
            self.add_input_encodings(inputs, row)

        sample: dict[str, object] = {
            "inputs": inputs,
            "target": None,
        }
        if len(self._target_specs) > 0:
            target_values: dict[str, object] = {}
            for spec in self._target_specs:
                target_values[get_spec_name(spec)] = self.get_spec_value(spec, subject_id, row)
            sample["target"] = target_values
        return sample
