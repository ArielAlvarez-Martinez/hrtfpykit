from pathlib import Path
from collections.abc import Sequence
from typing import Callable

from .base import BaseDataset
from .config import SONICOMConfig
from .download import BaseDownload
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


class SONICOM(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        dataset_hrtf_type: str = "measured",
        dataset_hrtf_sample_rate: int | str = 44100,
        dataset_hrtf_version: str = "Windowed",
        dataset_mesh_type: str = "scanned",
        dataset_mesh_version: str = "watertight",
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "metadata",
        download_hrtf_type: str | tuple[str, ...] | list[str] | None = "measured",
        download_hrtf_sample_rate: int | str | tuple[int | str, ...] | list[int | str] | None = 44100,
        download_hrtf_version: str | tuple[str, ...] | list[str] | None = "Windowed",
        download_mesh_type: str | tuple[str, ...] | list[str] | None = "scanned",
        download_mesh_version: str | tuple[str, ...] | list[str] | None = "watertight",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if download:
            downloaded, download_report = BaseDownload(
                config=SONICOMConfig,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
            ).download(
                download_resources=download_resources,
                download_hrtf_type=download_hrtf_type,
                download_hrtf_sample_rate=download_hrtf_sample_rate,
                download_hrtf_version=download_hrtf_version,
                download_mesh_type=download_mesh_type,
                download_mesh_version=download_mesh_version,
            )
            if downloaded:
                if verbose:
                    print(download_report)

        super().__init__(
            root=root,
            config=SONICOMConfig,
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
