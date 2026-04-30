from pathlib import Path
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .config import DatasetConfig
from .loader import load_anthropometry
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
from .validation import (
    validate_hrtf_resources,
    validate_image_resources,
    validate_mesh_resources,
    validate_video_resources,
)

if TYPE_CHECKING:
    from .base import BaseDataset


def resolve_optional_path(
    path: str | Path | None,
    root: Path,
) -> Path | None:
    if path is None:
        return None
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = root / resolved_path
    return resolved_path


def scan_hrtf_paths(
    config: type[DatasetConfig] | DatasetConfig,
    root: Path,
    variant: str | None,
    excluded_subject_ids: set[str],
    required: bool,
) -> tuple[dict[str, Path], dict[str, object] | None]:
    hrtf_paths: dict[str, Path] = {}
    if config.hrtf is None or not required:
        return hrtf_paths, None
    hrtf_subject_ids = (
        tuple(config.subject_ids)
        if config.hrtf.subject_ids is None
        else tuple(config.hrtf.subject_ids)
    )
    checked_hrtf_subject_ids = tuple(
        subject_id
        for subject_id in hrtf_subject_ids
        if subject_id not in excluded_subject_ids
    )
    for subject_id in checked_hrtf_subject_ids:
        relative_path = config.hrtf.path_pattern.format(
            subject_id=subject_id,
            variant=variant,
        )
        candidate = (root / relative_path).expanduser()
        if candidate.is_file():
            hrtf_paths[subject_id] = candidate
    missing_hrtf_subject_ids = tuple(
        subject_id
        for subject_id in checked_hrtf_subject_ids
        if subject_id not in hrtf_paths
    )
    return hrtf_paths, {
        "pattern": config.hrtf.path_pattern,
        "variant": variant,
        "checked": len(checked_hrtf_subject_ids),
        "found": len(hrtf_paths),
        "missing": len(missing_hrtf_subject_ids),
        "missing_subject_ids": missing_hrtf_subject_ids,
    }


def scan_mesh_paths(
    config: type[DatasetConfig] | DatasetConfig,
    root: Path,
    excluded_subject_ids: set[str],
    required: bool,
) -> tuple[dict[str, Path], dict[str, object] | None]:
    mesh_paths: dict[str, Path] = {}
    if config.mesh is None or not required:
        return mesh_paths, None
    mesh_subject_ids = (
        tuple(config.subject_ids)
        if config.mesh.subject_ids is None
        else tuple(config.mesh.subject_ids)
    )
    checked_mesh_subject_ids = tuple(
        subject_id
        for subject_id in mesh_subject_ids
        if subject_id not in excluded_subject_ids
    )
    for subject_id in checked_mesh_subject_ids:
        for extension in config.mesh.extensions:
            relative_path = config.mesh.path_pattern.format(
                subject_id=subject_id,
                extension=extension,
            )
            candidate = (root / relative_path).expanduser()
            if candidate.is_file():
                mesh_paths[subject_id] = candidate
                break
    missing_mesh_subject_ids = tuple(
        subject_id
        for subject_id in checked_mesh_subject_ids
        if subject_id not in mesh_paths
    )
    return mesh_paths, {
        "pattern": config.mesh.path_pattern,
        "extensions": tuple(config.mesh.extensions),
        "checked": len(checked_mesh_subject_ids),
        "found": len(mesh_paths),
        "missing": len(missing_mesh_subject_ids),
        "missing_subject_ids": missing_mesh_subject_ids,
    }


def scan_anthropometry_path(
    root: Path,
    configured_path: str | Path | None,
    requested_path: Path | None,
) -> Path | None:
    if requested_path is not None:
        return requested_path
    if configured_path is None:
        return None
    candidate = (root / configured_path).expanduser()
    if candidate.is_file():
        return candidate
    return requested_path


def resolve_subject_resource_folder(
    root: Path,
    subject_id: str,
    subject_number: int,
    resource_name: str,
) -> Path | None:
    candidate_names = (
        str(subject_id).strip().lower(),
        f"subject{subject_number}",
        f"subject_{subject_number}",
    )
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.strip().lower() in candidate_names
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{resource_name} path {root} contains multiple folders for subject {subject_id!r}: "
            + ", ".join(str(path.name) for path in matches)
        )
    if len(matches) == 0:
        return None
    return matches[0]


def collect_ordered_resource_files(
    path: Path,
    extensions: tuple[str, ...],
) -> list[str]:
    normalized_extensions = {extension.lower() for extension in extensions}

    def sort_key(file: Path) -> tuple[int, str, int | float, str]:
        stem = file.stem.strip().lower()
        match = re.fullmatch(r"([a-z_ -]*?)(\d+)", stem)
        if match is None:
            return (1, stem, float("inf"), file.name.lower())
        prefix = match.group(1).strip()
        return (0, prefix, int(match.group(2)), file.name.lower())

    return sorted(
        (
            str(file)
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in normalized_extensions
        ),
        key=lambda file: sort_key(Path(file)),
    )


def scan_media_paths(
    path: Path,
    subject_ids: tuple[str, ...],
    subject_numbers: dict[str, int],
    extensions: tuple[str, ...],
    grouped_by: tuple[str, ...],
    resource_name: str,
) -> tuple[
    dict[tuple[str, int | None, str | None], list[str]],
    dict[str, int],
    tuple[str, ...],
]:
    grouped_paths: dict[tuple[str, int | None, str | None], list[str]] = {}
    if not path.exists():
        raise ValueError(f"{resource_name} path does not exist: {path}")
    subject_counts: dict[str, int] = {}
    missing_subject_ids: list[str] = []
    for subject_id in subject_ids:
        subject_folder = resolve_subject_resource_folder(
            path,
            subject_id,
            int(subject_numbers[subject_id]),
            resource_name,
        )
        if subject_folder is None:
            missing_subject_ids.append(subject_id)
            continue
        subject_files = collect_ordered_resource_files(subject_folder, extensions)
        grouped_paths[(subject_id, None, None)] = subject_files
        subject_count = len(subject_files)
        if "ear" in grouped_by:
            for ear in ("left", "right"):
                ear_folder = subject_folder / ear
                files = collect_ordered_resource_files(ear_folder, extensions)
                grouped_paths[(subject_id, None, ear)] = files
        else:
            subject_count = len(subject_files)
        subject_counts[subject_id] = subject_count
    return grouped_paths, subject_counts, tuple(missing_subject_ids)


def scan_image_paths(
    path: Path,
    subject_ids: tuple[str, ...],
    subject_numbers: dict[str, int],
    extensions: tuple[str, ...],
    grouped_by: tuple[str, ...],
) -> tuple[
    dict[tuple[str, int | None, str | None], list[str]],
    dict[str, int],
    tuple[str, ...],
]:
    return scan_media_paths(
        path,
        subject_ids,
        subject_numbers,
        extensions,
        grouped_by,
        "Image",
    )


def scan_video_paths(
    path: Path,
    subject_ids: tuple[str, ...],
    subject_numbers: dict[str, int],
    extensions: tuple[str, ...],
    grouped_by: tuple[str, ...],
) -> tuple[
    dict[tuple[str, int | None, str | None], list[str]],
    dict[str, int],
    tuple[str, ...],
]:
    return scan_media_paths(
        path,
        subject_ids,
        subject_numbers,
        extensions,
        grouped_by,
        "Video",
    )


def resolve_dataset_resources(dataset: "BaseDataset") -> None:
    dataset._resource_summary = {}

    image_path = None
    for spec in dataset._get_specs(ImageSpec):
        if spec.path is not None:
            image_path = spec.path
            break
    video_path = None
    for spec in dataset._get_specs(VideoSpec):
        if spec.path is not None:
            video_path = spec.path
            break
    anthropometry_path = None
    for spec in dataset._get_specs(AnthropometrySpec):
        if spec.path is not None:
            anthropometry_path = spec.path
            break

    dataset._image_path = resolve_optional_path(image_path, dataset._root)
    dataset._video_path = resolve_optional_path(video_path, dataset._root)
    dataset._anthropometry_path = resolve_optional_path(
        anthropometry_path,
        dataset._root,
    )

    dataset._image_grouped_by = None
    image_specs = dataset._get_specs(ImageSpec)
    if len(image_specs) > 0:
        dataset._image_grouped_by = ("subject",)
        for spec in image_specs:
            if isinstance(spec.grouped_by, str):
                grouped_by = (str(spec.grouped_by).strip().lower(),)
            else:
                grouped_by = tuple(str(value).strip().lower() for value in spec.grouped_by)
            if "ear" in grouped_by:
                dataset._image_grouped_by = ("subject", "ear")
                break

    dataset._video_grouped_by = None
    video_specs = dataset._get_specs(VideoSpec)
    if len(video_specs) > 0:
        dataset._video_grouped_by = ("subject",)
        for spec in video_specs:
            if isinstance(spec.grouped_by, str):
                grouped_by = (str(spec.grouped_by).strip().lower(),)
            else:
                grouped_by = tuple(str(value).strip().lower() for value in spec.grouped_by)
            if "ear" in grouped_by:
                dataset._video_grouped_by = ("subject", "ear")
                break

    excluded_subject_ids = set(dataset._exclude_subject_ids)
    included_subject_ids = tuple(
        subject_id
        for subject_id in dataset._config.subject_ids
        if subject_id not in excluded_subject_ids
    )
    dataset._included_subject_ids = included_subject_ids
    subject_numbers = {
        subject_id: index
        for index, subject_id in enumerate(tuple(dataset._config.subject_ids), start=1)
    }

    dataset._hrtf_paths, hrtf_summary = scan_hrtf_paths(
        dataset._config,
        dataset._root,
        dataset.variant,
        excluded_subject_ids,
        len(dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))) > 0,
    )
    if hrtf_summary is not None:
        dataset._resource_summary["hrtf"] = hrtf_summary

    dataset._mesh_paths, mesh_summary = scan_mesh_paths(
        dataset._config,
        dataset._root,
        excluded_subject_ids,
        len(dataset._get_specs(MeshSpec)) > 0,
    )
    if mesh_summary is not None:
        dataset._resource_summary["mesh"] = mesh_summary

    dataset._anthropometry_path = scan_anthropometry_path(
        dataset._root,
        None if dataset._config.anthropometry is None else dataset._config.anthropometry.path,
        dataset._anthropometry_path,
    )
    dataset._anthropometry_rows = {}
    if len(dataset._get_specs(AnthropometrySpec)) > 0:
        anthropometry_summary: dict[str, object] = {
            "path": None if dataset._anthropometry_path is None else str(dataset._anthropometry_path),
            "found": dataset._anthropometry_path is not None and dataset._anthropometry_path.is_file(),
        }
        if dataset._anthropometry_path is None:
            dataset._anthropometry_rows = {}
        else:
            anthropometry_row_exclusions: Sequence[int] | None
            anthropometry_column_exclusions: Sequence[int] | None
            anthropometry_rows = []
            anthropometry_columns = []
            for spec in dataset._get_specs(AnthropometrySpec):
                if spec.exclude_row is not None:
                    if isinstance(spec.exclude_row, int):
                        anthropometry_rows.append(int(spec.exclude_row))
                    else:
                        anthropometry_rows.extend(int(row) for row in spec.exclude_row)
                if spec.exclude_column is not None:
                    if isinstance(spec.exclude_column, int):
                        anthropometry_columns.append(int(spec.exclude_column))
                    else:
                        anthropometry_columns.extend(int(column) for column in spec.exclude_column)
            anthropometry_row_exclusions = (
                tuple(dict.fromkeys(anthropometry_rows))
                if len(anthropometry_rows) > 0
                else None
            )
            anthropometry_column_exclusions = (
                tuple(dict.fromkeys(anthropometry_columns))
                if len(anthropometry_columns) > 0
                else None
            )
            dataset._anthropometry_rows = load_anthropometry(
                dataset,
                dataset._anthropometry_path,
                exclude_row=anthropometry_row_exclusions,
                exclude_column=anthropometry_column_exclusions,
            )
            anthropometry_summary["subjects"] = len(dataset._anthropometry_rows)
            anthropometry_summary["rows"] = len(dataset._anthropometry_rows)
        dataset._resource_summary["anthropometry"] = anthropometry_summary

    validate_hrtf_resources(dataset)
    validate_mesh_resources(dataset)

    if dataset._image_path is None or dataset._config.image is None or dataset._image_grouped_by is None:
        dataset._image_index = {}
        dataset._image_counts = {}
        image_summary = {
            "path": None if dataset._image_path is None else str(dataset._image_path),
            "found": 0,
            "missing": 0,
            "missing_subject_ids": tuple(),
        }
    else:
        dataset._image_index, dataset._image_counts, missing_subject_ids = scan_image_paths(
            dataset._image_path,
            included_subject_ids,
            subject_numbers,
            tuple(dataset._config.image.extensions),
            dataset._image_grouped_by,
        )
        image_summary = {
            "path": str(dataset._image_path),
            "found": len({key[0] for key in dataset._image_index}),
            "missing": len(missing_subject_ids),
            "missing_subject_ids": tuple(missing_subject_ids),
        }
    if len(dataset._get_specs(ImageSpec)) > 0:
        dataset._resource_summary["image"] = image_summary
        validate_image_resources(dataset, image_summary)

    if dataset._video_path is None or dataset._config.video is None or dataset._video_grouped_by is None:
        dataset._video_index = {}
        dataset._video_counts = {}
        video_summary = {
            "path": None if dataset._video_path is None else str(dataset._video_path),
            "found": 0,
            "missing": 0,
            "missing_subject_ids": tuple(),
        }
    else:
        dataset._video_index, dataset._video_counts, missing_subject_ids = scan_video_paths(
            dataset._video_path,
            included_subject_ids,
            subject_numbers,
            tuple(dataset._config.video.extensions),
            dataset._video_grouped_by,
        )
        video_summary = {
            "path": str(dataset._video_path),
            "found": len({key[0] for key in dataset._video_index}),
            "missing": len(missing_subject_ids),
            "missing_subject_ids": tuple(missing_subject_ids),
        }
    if len(dataset._get_specs(VideoSpec)) > 0:
        dataset._resource_summary["video"] = video_summary
        validate_video_resources(dataset, video_summary)
