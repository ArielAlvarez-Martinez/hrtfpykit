import warnings
from typing import TYPE_CHECKING

from .loader import load_hrtf
from .summary import DatasetSummary
from .specs import (
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)

if TYPE_CHECKING:
    from .base import BaseDataset


def validate_hrtf_resources(dataset: "BaseDataset") -> None:
    if len(dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))) == 0:
        return
    missing_hrtf_subject_ids = list(
        dataset._resource_summary.get("hrtf", {}).get("missing_subject_ids", tuple())
    )
    if len(missing_hrtf_subject_ids) > 0:
        preview = ", ".join(missing_hrtf_subject_ids[:5])
        suffix = "" if len(missing_hrtf_subject_ids) <= 5 else ", ..."
        warnings.warn(
            f"{dataset._name}: {len(missing_hrtf_subject_ids)} subjects do not have a matching HRTF file under "
            f"{dataset._root} and will be excluded ({preview}{suffix})",
            stacklevel=2,
        )
    validated_hrtf_paths = {}
    validated_sample_rate = None
    for subject_id, path in dataset._hrtf_paths.items():
        if not path.exists():
            warnings.warn(
                f"{dataset._name}: subject {subject_id} HRTF path is missing and will be excluded: {path}",
                stacklevel=2,
            )
            continue
        try:
            hrtf = load_hrtf(
                dataset,
                subject_id,
                subject_ids=tuple(dataset._config.subject_ids),
            )
        except Exception:
            continue
        current_sample_rate = (
            None if hrtf.IR.sample_rate is None else float(hrtf.IR.sample_rate)
        )
        if validated_sample_rate is None:
            validated_sample_rate = current_sample_rate
        elif current_sample_rate != validated_sample_rate:
            raise ValueError(
                f"{dataset._name} requires a consistent sample_rate across loaded HRTFs, "
                f"but subject {subject_id!r} has sample_rate={current_sample_rate} "
                f"and previous subjects use sample_rate={validated_sample_rate}"
            )
        validated_hrtf_paths[subject_id] = path
    invalid_hrtf_subject_ids = tuple(
        subject_id
        for subject_id in dataset._hrtf_paths
        if subject_id not in validated_hrtf_paths
    )
    dataset._hrtf_paths = validated_hrtf_paths
    dataset._resource_summary["hrtf"]["valid"] = len(dataset._hrtf_paths)
    dataset._resource_summary["hrtf"]["invalid"] = len(invalid_hrtf_subject_ids)
    dataset._resource_summary["hrtf"]["invalid_subject_ids"] = invalid_hrtf_subject_ids


def validate_mesh_resources(dataset: "BaseDataset") -> None:
    if len(dataset._get_specs(MeshSpec)) == 0:
        return
    missing_mesh_subject_ids = tuple(
        dataset._resource_summary.get("mesh", {}).get("missing_subject_ids", tuple())
    )
    if len(missing_mesh_subject_ids) > 0:
        warnings.warn(
            f"{dataset._name}: {len(missing_mesh_subject_ids)} subjects do not have a matching mesh file under "
            f"{dataset._root} and will be excluded when mesh is required "
            f"({DatasetSummary.preview_values(missing_mesh_subject_ids)})",
            stacklevel=2,
        )


def validate_image_resources(dataset: "BaseDataset", summary: dict[str, object]) -> None:
    if len(dataset._get_specs(ImageSpec)) == 0:
        return
    missing_subject_ids = tuple(summary["missing_subject_ids"])
    if len(missing_subject_ids) > 0:
        raise ValueError(
            f"{dataset._name} image path is incompatible with the selected dataset subjects. "
            f"Missing subject folders under {dataset._image_path}: "
            f"{DatasetSummary.preview_values(missing_subject_ids)}"
        )
    if len(set(dataset._image_counts.values())) > 1:
        warnings.warn(
            f"{dataset._name}: subjects do not all have the same number of images under {dataset._image_path} "
            f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(dataset._image_counts.items())[:5])}"
            f"{'' if len(dataset._image_counts) <= 5 else ', ...'})",
            stacklevel=2,
        )


def validate_video_resources(dataset: "BaseDataset", summary: dict[str, object]) -> None:
    if len(dataset._get_specs(VideoSpec)) == 0:
        return
    missing_subject_ids = tuple(summary["missing_subject_ids"])
    if len(missing_subject_ids) > 0:
        raise ValueError(
            f"{dataset._name} video path is incompatible with the selected dataset subjects. "
            f"Missing subject folders under {dataset._video_path}: "
            f"{DatasetSummary.preview_values(missing_subject_ids)}"
        )
    if len(set(dataset._video_counts.values())) > 1:
        warnings.warn(
            f"{dataset._name}: subjects do not all have the same number of videos under {dataset._video_path} "
            f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(dataset._video_counts.items())[:5])}"
            f"{'' if len(dataset._video_counts) <= 5 else ', ...'})",
            stacklevel=2,
        )
