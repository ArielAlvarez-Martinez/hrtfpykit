from pathlib import Path
from collections.abc import Sequence
from typing import Callable
import numpy as np

from .base import BaseDataset
from .config import HUTUBSConfig
from .download import BaseDownload
from . import summary as summary_module
from .sanitize import sanitize_grouped_by
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


class HUTUBS:
    config = HUTUBSConfig

    def __init__(
        self,
        root: str | Path,
        dataset_hrtf_variant: str = "measured",
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_variant: str = "all",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        self.dataset_hrtf_variant = str(dataset_hrtf_variant).strip().lower()
        if self.config.hrtf is None:
            raise ValueError("HUTUBS config does not define HRTF metadata")
        if self.dataset_hrtf_variant not in self.config.hrtf.variants:
            raise ValueError(
                f"Unsupported dataset_hrtf_variant {self.dataset_hrtf_variant!r}. Expected one of {self.config.hrtf.variants}"
            )
        if download:
            downloaded, download_report = BaseDownload(
                config=self.config,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=download_hrtf_variant,
            )
            if downloaded:
                print(download_report)
        self._dataset = BaseDataset(
            root=root,
            config=self.config,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            variant=self.dataset_hrtf_variant,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        self._dataset._anthropometry_value_selector = self._select_anthropometry_value
        print(summary_module.resources_summary(self._dataset))
        print(summary_module.dataset_summary(self._dataset))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._dataset[index]

    @property
    def dataset_sample_rate(self) -> float | None:
        return self._dataset._dataset_sample_rate

    @property
    def dataset_positions(self) -> np.ndarray | None:
        return self._dataset._dataset_source_positions

    @property
    def dataset_azimuth_angles(self) -> np.ndarray | None:
        return (
            self._dataset._azimuth_angles
            if self._dataset._azimuth_angles is not None
            else self._dataset._available_azimuth_angles
        )

    @property
    def dataset_elevation_angles(self) -> np.ndarray | None:
        return (
            self._dataset._elevation_angles
            if self._dataset._elevation_angles is not None
            else self._dataset._available_elevation_angles
        )

    @property
    def dataset_excluded_subjects(self) -> list[str]:
        return list(self._dataset._exclude_subject_ids)

    @property
    def dataset_available_subjects(self) -> list[str]:
        return list(self._dataset._available_subject_ids)

    def get_subject_hrtf(self, subject_id: str | int) -> object:
        return self._dataset.get_subject_hrtf(subject_id)

    def _select_anthropometry_value(
        self,
        spec: AnthropometrySpec,
        row: dict[str, str | int | None],
        value: object,
    ) -> object:
        if not isinstance(value, dict):
            return value
        grouped_by = sanitize_grouped_by(spec.grouped_by)

        selected_ear = str(spec.ear).strip().lower() if spec.ear else None
        if selected_ear == "both" or selected_ear == "":
            selected_ear = None

        if "ear" in grouped_by:
            if selected_ear not in {"left", "right"}:
                row_ear = row.get("ear")
                if row_ear is not None and str(row_ear).strip().lower() in {"left", "right"}:
                    selected_ear = str(row_ear).strip().lower()

        if selected_ear is None:
            return value

        config = self._dataset._config.anthropometry if self._dataset._config is not None else None
        left_prefix = "L_"
        right_prefix = "R_"
        if config is not None:
            left_prefix = str(config.left_prefix)
            right_prefix = str(config.right_prefix)
        target_prefix = left_prefix if selected_ear == "left" else right_prefix

        return {
            key: feature
            for key, feature in value.items()
            if str(key).startswith(target_prefix)
            or (
                not str(key).startswith(left_prefix)
                and not str(key).startswith(right_prefix)
            )
        }
