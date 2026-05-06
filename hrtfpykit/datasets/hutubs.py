from pathlib import Path
from collections.abc import Sequence
from typing import Callable

from .base import BaseDataset
from .config import HUTUBSConfig
from .download import BaseDownload
from .sanitize import sanitize_grouped_by
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


class HUTUBS(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        dataset_hrtf_type: str = "measured",
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_type: str = "all",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        hrtf_type = str(dataset_hrtf_type).strip().lower()
        if HUTUBSConfig.hrtf is None:
            raise ValueError("HUTUBS config does not define HRTF metadata")
        if hrtf_type not in HUTUBSConfig.hrtf.types:
            raise ValueError(
                f"Unsupported dataset_hrtf_type {hrtf_type!r}. Expected one of {tuple(HUTUBSConfig.hrtf.types)}"
            )
        if download:
            downloaded, download_report = BaseDownload(
                config=HUTUBSConfig,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
            ).download(
                download_resources=download_resources,
                download_hrtf_type=download_hrtf_type,
                download_hrtf_sample_rate=None,
                download_hrtf_version=None,
                download_mesh_type=None,
                download_mesh_version=None,
            )
            if downloaded:
                if verbose:
                    print(download_report)
        super().__init__(
            root=root,
            config=HUTUBSConfig,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            dataset_hrtf_type=hrtf_type,
            dataset_hrtf_sample_rate=None,
            dataset_hrtf_version=None,
            dataset_mesh_type=None,
            dataset_mesh_version=None,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            verbose=verbose,
        )
        self._state.anthropometry_value_selector = self._select_anthropometry_value

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

        config = self._state.config.anthropometry if self._state.config is not None else None
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
