from pathlib import Path
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import DatasetConfig
from .load import load_hrtf
from .load import load_anthropometry
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
from .sanitize import sanitize_extensions
from .sanitize import sanitize_grouped_by
from .split import DatasetSubjectSelectionPlanner
from .summary import resources_summary

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetResourcesValidator:
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def validate_hrtf_resources(self) -> None:
        if len(self._dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))) == 0:
            return
        missing_hrtf_subject_ids = list(
            self._dataset._resource_summary.get("hrtf", {}).get(
                "missing_subject_ids",
                tuple(),
            )
        )
        if len(missing_hrtf_subject_ids) > 0:
            preview = ", ".join(missing_hrtf_subject_ids[:5])
            suffix = "" if len(missing_hrtf_subject_ids) <= 5 else ", ..."
            warnings.warn(
                f"{self._dataset._name}: {len(missing_hrtf_subject_ids)} subjects do not have a matching HRTF file under "
                f"{self._dataset._root} and will be excluded ({preview}{suffix})",
                stacklevel=2,
            )
        validated_hrtf_paths = {}
        validated_sample_rate = None
        for subject_id, path in self._dataset._hrtf_paths.items():
            if not path.exists():
                warnings.warn(
                    f"{self._dataset._name}: subject {subject_id} HRTF path is missing and will be excluded: {path}",
                    stacklevel=2,
                )
                continue
            try:
                hrtf = load_hrtf(
                    self._dataset,
                    subject_id,
                    subject_ids=tuple(self._dataset._config.subject_ids),
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
                    f"{self._dataset._name} requires a consistent sample_rate across loaded HRTFs, "
                    f"but subject {subject_id!r} has sample_rate={current_sample_rate} "
                    f"and previous subjects use sample_rate={validated_sample_rate}"
                )
            validated_hrtf_paths[subject_id] = path
        self._dataset._hrtf_paths = validated_hrtf_paths

    def validate_mesh_resources(self) -> None:
        if len(self._dataset._get_specs(MeshSpec)) == 0:
            return
        missing_mesh_subject_ids = tuple(
            self._dataset._resource_summary.get("mesh", {}).get(
                "missing_subject_ids",
                tuple(),
            )
        )
        if len(missing_mesh_subject_ids) > 0:
            warnings.warn(
                f"{self._dataset._name}: {len(missing_mesh_subject_ids)} subjects do not have a matching mesh file under "
                f"{self._dataset._root} and will be excluded when mesh is required "
                f"({', '.join(str(value) for value in missing_mesh_subject_ids[:5])}"
                f"{', ...' if len(missing_mesh_subject_ids) > 5 else ''})",
                stacklevel=2,
            )

    def validate_image_resources(self, summary: dict[str, object]) -> None:
        if len(self._dataset._get_specs(ImageSpec)) == 0:
            return
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            raise ValueError(
                f"{self._dataset._name} image path is incompatible with the selected dataset subjects. "
                f"Missing subject folders under {self._dataset._image_path}: "
                f"{', '.join(str(value) for value in missing_subject_ids[:5])}"
                f"{', ...' if len(missing_subject_ids) > 5 else ''}"
            )
        if len(set(self._dataset._image_counts.values())) > 1:
            warnings.warn(
                f"{self._dataset._name}: subjects do not all have the same number of images under {self._dataset._image_path} "
                f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(self._dataset._image_counts.items())[:5])}"
                f"{'' if len(self._dataset._image_counts) <= 5 else ', ...'})",
                stacklevel=2,
            )

    def validate_video_resources(self, summary: dict[str, object]) -> None:
        if len(self._dataset._get_specs(VideoSpec)) == 0:
            return
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            raise ValueError(
                f"{self._dataset._name} video path is incompatible with the selected dataset subjects. "
                f"Missing subject folders under {self._dataset._video_path}: "
                f"{', '.join(str(value) for value in missing_subject_ids[:5])}"
                f"{', ...' if len(missing_subject_ids) > 5 else ''}"
            )
        if len(set(self._dataset._video_counts.values())) > 1:
            warnings.warn(
                f"{self._dataset._name}: subjects do not all have the same number of videos under {self._dataset._video_path} "
                f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(self._dataset._video_counts.items())[:5])}"
                f"{'' if len(self._dataset._video_counts) <= 5 else ', ...'})",
                stacklevel=2,
            )

    def validate_anthropometry_resources(self) -> None:
        if len(self._dataset._get_specs(AnthropometrySpec)) == 0:
            return
        if self._dataset._anthropometry_path is None:
            raise ValueError(
                f"{self._dataset._name} requires an anthropometry file but none was selected"
            )
        if not self._dataset._anthropometry_path.is_file():
            raise ValueError(
                f"{self._dataset._name} anthropometry path is invalid: {self._dataset._anthropometry_path}"
            )
        if not isinstance(self._dataset._anthropometry_rows, dict):
            raise ValueError(
                f"{self._dataset._name} anthropometry data is invalid: expected a mapping but got {type(self._dataset._anthropometry_rows)!r}"
            )


class DatasetResourcesScanner:
    @staticmethod
    def scan_anthropometry_paths(
        config: type[DatasetConfig] | DatasetConfig,
        root: Path,
        requested_path: Path | None,
        required: bool,
    ) -> tuple[Path | None, dict[str, object]]:
        if config.anthropometry is None or not required:
            return None, {
                "path": None,
                "found": False,
                "subjects": 0,
                "rows": 0,
            }
        anthropometry_path = requested_path
        if anthropometry_path is None and config.anthropometry is not None:
            anthropometry_path = (root / config.anthropometry.path).expanduser()
        if anthropometry_path is None:
            return None, {
                "path": None,
                "found": False,
                "subjects": 0,
                "rows": 0,
            }
        return anthropometry_path, {
            "path": str(anthropometry_path),
            "found": anthropometry_path.is_file(),
            "subjects": 0,
            "rows": 0,
            "extensions": tuple(config.anthropometry.extensions)
            if config.anthropometry is not None and config.anthropometry.extensions is not None
            else tuple(),
        }

    @staticmethod
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

    @staticmethod
    def scan_mesh_paths(
        config: type[DatasetConfig] | DatasetConfig,
        root: Path,
        excluded_subject_ids: set[str],
        required: bool,
        extensions: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Path], dict[str, object] | None]:
        mesh_paths: dict[str, Path] = {}
        if config.mesh is None or not required:
            return mesh_paths, None
        normalized_extensions = [extension.lower() for extension in tuple(extensions or tuple())]
        normalized_extensions = [
            extension if extension.startswith(".") else f".{extension}"
            for extension in normalized_extensions
            if str(extension).strip() != ""
        ]
        normalized_extensions = list(dict.fromkeys(normalized_extensions))
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
            relative_path = config.mesh.path_pattern.format(
                subject_id=subject_id,
            )
            pattern_path = Path(relative_path)
            candidate_paths: list[Path] = []
            if len(normalized_extensions) == 0:
                candidate_paths = [pattern_path]
            elif pattern_path.suffix == "":
                candidate_paths = [
                    pattern_path.with_name(f"{pattern_path.name}{extension}")
                    for extension in normalized_extensions
                ]
            else:
                base_path = pattern_path.with_suffix("")
                candidate_paths = [
                    base_path.with_suffix(extension)
                    for extension in normalized_extensions
                ]
            for candidate in dict.fromkeys(candidate_paths):
                resolved_candidate = (root / candidate).expanduser()
                if resolved_candidate.is_file():
                    mesh_paths[subject_id] = resolved_candidate
                    break
        missing_mesh_subject_ids = tuple(
            subject_id
            for subject_id in checked_mesh_subject_ids
            if subject_id not in mesh_paths
        )
        return mesh_paths, {
            "pattern": config.mesh.path_pattern,
            "extensions": tuple(normalized_extensions),
            "checked": len(checked_mesh_subject_ids),
            "found": len(mesh_paths),
            "missing": len(missing_mesh_subject_ids),
            "missing_subject_ids": missing_mesh_subject_ids,
        }

    @staticmethod
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
        normalized_extensions = {extension.lower() for extension in extensions}

        def sort_key(file: Path) -> tuple[int, str, int | float, str]:
            stem = file.stem.strip().lower()
            match = re.fullmatch(r"([a-z_ -]*?)(\d+)", stem)
            if match is None:
                return (1, stem, float("inf"), file.name.lower())
            prefix = match.group(1).strip()
            return (0, prefix, int(match.group(2)), file.name.lower())

        for subject_id in subject_ids:
            candidate_names = (
                str(subject_id).strip().lower(),
                f"subject{subject_numbers[subject_id]}",
                f"subject_{subject_numbers[subject_id]}",
            )
            matches = [
                subject_path
                for subject_path in path.iterdir()
                if subject_path.is_dir()
                and subject_path.name.strip().lower() in candidate_names
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"{resource_name} path {path} contains multiple folders for subject {subject_id!r}: "
                    + ", ".join(str(path_item.name) for path_item in matches)
                )
            if len(matches) == 0:
                missing_subject_ids.append(subject_id)
                continue
            subject_folder = matches[0]

            subject_files = sorted(
                (
                    str(file)
                    for file in subject_folder.rglob("*")
                    if file.is_file() and file.suffix.lower() in normalized_extensions
                ),
                key=lambda file: sort_key(Path(file)),
            )
            grouped_paths[(subject_id, None, None)] = subject_files
            subject_count = len(subject_files)
            if "ear" in grouped_by:
                for ear in ("left", "right"):
                    ear_folder = subject_folder / ear
                    files = sorted(
                        (
                            str(file)
                            for file in ear_folder.rglob("*")
                            if file.is_file() and file.suffix.lower() in normalized_extensions
                        ),
                        key=lambda file: sort_key(Path(file)),
                    )
                    grouped_paths[(subject_id, None, ear)] = files
            else:
                subject_count = len(subject_files)
            subject_counts[subject_id] = subject_count
        return grouped_paths, subject_counts, tuple(missing_subject_ids)

    @staticmethod
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
        return DatasetResourcesScanner.scan_media_paths(
            path,
            subject_ids,
            subject_numbers,
            extensions,
            grouped_by,
            "Image",
        )

    @staticmethod
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
        return DatasetResourcesScanner.scan_media_paths(
            path,
            subject_ids,
            subject_numbers,
            extensions,
            grouped_by,
            "Video",
        )


@dataclass(frozen=True)
class DatasetResourcesPlan:
    hrtf_paths: dict[str, Path]
    mesh_paths: dict[str, Path]
    image_path: Path | None
    video_path: Path | None
    image_index: dict[tuple[str, int | None, str | None], list[str]]
    video_index: dict[tuple[str, int | None, str | None], list[str]]
    image_counts: dict[str, int]
    video_counts: dict[str, int]
    anthropometry_path: Path | None
    anthropometry_rows: dict[str, object]
    included_subject_ids: tuple[str, ...]
    subject_numbers: dict[str, int]
    resource_summary: dict[str, object]

class DatasetResources:
    @staticmethod
    def _resolve_optional_path(
        path: str | Path | None,
        root: Path,
    ) -> Path | None:
        if path is None:
            return None
        resolved_path = Path(path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = root / resolved_path
        return resolved_path

    @staticmethod
    def build(dataset: "BaseDataset") -> DatasetResourcesPlan:
        if dataset._config is None:
            raise ValueError("Dataset config is not initialized")
        config = dataset._config
        root = dataset._root
        excluded_subject_ids = set(getattr(dataset, "_exclude_subject_ids", tuple()))
        available_subject_ids = tuple(getattr(dataset, "_included_subject_ids", tuple()))
        if len(available_subject_ids) == 0:
            sorted_subject_ids = DatasetSubjectSelectionPlanner.sort_subject_ids(
                tuple(config.subject_ids)
            )
            available_subject_ids = tuple(
                subject_id
                for subject_id in sorted_subject_ids
                if subject_id not in excluded_subject_ids
            )
            dataset._included_subject_ids = available_subject_ids
        subject_numbers = getattr(dataset, "_subject_numbers", None)
        if subject_numbers is None:
            subject_numbers = DatasetSubjectSelectionPlanner.build_subject_number_map(
                DatasetSubjectSelectionPlanner.sort_subject_ids(tuple(config.subject_ids))
            )
        dataset._subject_numbers = subject_numbers

        dataset._hrtf_paths = {}
        dataset._mesh_paths = {}
        dataset._image_path = None
        dataset._video_path = None
        dataset._image_index = {}
        dataset._video_index = {}
        dataset._image_counts = {}
        dataset._video_counts = {}
        dataset._anthropometry_path = None
        dataset._anthropometry_rows = {}
        dataset._resource_summary = {}

        validator = DatasetResourcesValidator(dataset)
        scanner = DatasetResourcesScanner()

        has_acoustic_specs = len(
            dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))
        ) > 0
        has_mesh_specs = len(dataset._get_specs(MeshSpec)) > 0
        has_anthro_specs = len(dataset._get_specs(AnthropometrySpec)) > 0
        has_image_specs = len(dataset._get_specs(ImageSpec)) > 0
        has_video_specs = len(dataset._get_specs(VideoSpec)) > 0

        hrtf_paths, hrtf_summary = scanner.scan_hrtf_paths(
            config=config,
            root=root,
            variant=dataset.variant if getattr(dataset, "variant", None) is not None else None,
            excluded_subject_ids=excluded_subject_ids,
            required=has_acoustic_specs,
        )
        if hrtf_summary is None:
            hrtf_summary = resources_summary(
                checked=0,
                found=0,
                missing=0,
                missing_subject_ids=tuple(),
            )
        dataset._hrtf_paths = hrtf_paths
        dataset._resource_summary["hrtf"] = hrtf_summary
        validator.validate_hrtf_resources()

        if has_mesh_specs:
            mesh_root_path = root
            mesh_specs = dataset._get_specs(MeshSpec)
            first_mesh_spec = mesh_specs[0]
            requested_mesh_path = None if first_mesh_spec.path is None else first_mesh_spec.path
            resolved_mesh_path = DatasetResources._resolve_optional_path(requested_mesh_path, root)
            if resolved_mesh_path is not None:
                mesh_root_path = resolved_mesh_path

            mesh_extensions = sanitize_extensions(
                resource_name="MeshSpec",
                extensions=first_mesh_spec.extensions,
            )
            if len(mesh_extensions) == 0 and config.mesh is not None:
                mesh_extensions = sanitize_extensions(
                    resource_name="MeshConfig",
                    extensions=config.mesh.extensions,
                )
            mesh_paths, mesh_summary = scanner.scan_mesh_paths(
                config=config,
                root=mesh_root_path,
                excluded_subject_ids=excluded_subject_ids,
                required=has_mesh_specs,
                extensions=mesh_extensions,
            )
            mesh_summary = resources_summary(
                checked=int(mesh_summary.get("checked", 0)),
                found=int(mesh_summary.get("found", 0)),
                missing=int(mesh_summary.get("missing", 0)),
                missing_subject_ids=tuple(mesh_summary.get("missing_subject_ids", tuple())),
            )
            dataset._mesh_paths = mesh_paths
            validator.validate_mesh_resources()
            dataset._resource_summary["mesh"] = mesh_summary
        else:
            dataset._mesh_paths = {}
            dataset._resource_summary["mesh"] = resources_summary()

        anthropometry_path, anthropometry_summary = scanner.scan_anthropometry_paths(
            config=config,
            root=root,
            requested_path=None,
            required=has_anthro_specs,
        )
        if anthropometry_summary is None:
            anthropometry_summary = resources_summary(
                checked=0,
                found=0,
                missing=0,
            )

        if has_anthro_specs:
            anthropometry_specs = dataset._get_specs(AnthropometrySpec)
            first_anthro_spec = anthropometry_specs[0]
            requested_anthro_path = None if first_anthro_spec.path is None else first_anthro_spec.path
            resolved_anthro_path = DatasetResources._resolve_optional_path(requested_anthro_path, root)
            if resolved_anthro_path is not None:
                anthropometry_path = resolved_anthro_path
            anthropometry_extensions = sanitize_extensions(
                resource_name="AnthropometrySpec",
                extensions=first_anthro_spec.extensions,
            )
            if len(anthropometry_extensions) == 0 and config.anthropometry is not None:
                anthropometry_extensions = sanitize_extensions(
                    resource_name="AnthropometryConfig",
                    extensions=config.anthropometry.extensions,
                )
            anthropometry_extension = (
                anthropometry_extensions[0] if len(anthropometry_extensions) > 0 else None
            )
            dataset._anthropometry_path = anthropometry_path
            if anthropometry_path is None or not anthropometry_path.is_file():
                dataset._anthropometry_rows = {}
            else:
                dataset._anthropometry_rows = load_anthropometry(
                    dataset,
                    path=anthropometry_path,
                    exclude_row=first_anthro_spec.exclude_row,
                    exclude_column=first_anthro_spec.exclude_column,
                    accessed_by=first_anthro_spec.accessed_by,
                    extension=anthropometry_extension,
                )
                anthropometry_summary = resources_summary(
                    checked=1,
                    found=1,
                    missing=0,
                )
        else:
            dataset._anthropometry_path = None
            dataset._anthropometry_rows = {}

        dataset._resource_summary["anthropometry"] = anthropometry_summary
        validator.validate_anthropometry_resources()

        if has_image_specs:
            image_specs = dataset._get_specs(ImageSpec)
            first_image_spec = image_specs[0]
            if first_image_spec.path is None:
                raise ValueError("ImageSpec requires a path")
            requested_image_path = DatasetResources._resolve_optional_path(first_image_spec.path, root)
            dataset._image_path = requested_image_path
            grouped_by = ("subject",)
            if any("ear" in sanitize_grouped_by(spec.grouped_by) for spec in image_specs):
                grouped_by = ("subject", "ear")
            image_extensions = sanitize_extensions(
                resource_name="ImageSpec",
                extensions=first_image_spec.extensions,
            )
            if len(image_extensions) == 0 and config.image is not None:
                image_extensions = sanitize_extensions(
                    resource_name="ImageConfig",
                    extensions=config.image.extensions,
                )
            image_index, image_counts, missing_subject_ids = scanner.scan_image_paths(
                path=requested_image_path,
                subject_ids=available_subject_ids,
                subject_numbers=subject_numbers,
                extensions=image_extensions,
                grouped_by=grouped_by,
            )
            dataset._image_index = image_index
            dataset._image_counts = image_counts
            image_summary = resources_summary(
                checked=len(available_subject_ids),
                found=len(image_counts),
                missing=len(missing_subject_ids),
                missing_subject_ids=tuple(missing_subject_ids),
            )
            dataset._resource_summary["image"] = image_summary
            validator.validate_image_resources(image_summary)
        else:
            dataset._resource_summary["image"] = resources_summary()

        if has_video_specs:
            video_specs = dataset._get_specs(VideoSpec)
            first_video_spec = video_specs[0]
            if first_video_spec.path is None:
                raise ValueError("VideoSpec requires a path")
            requested_video_path = DatasetResources._resolve_optional_path(first_video_spec.path, root)
            dataset._video_path = requested_video_path
            grouped_by = ("subject",)
            if any("ear" in sanitize_grouped_by(spec.grouped_by) for spec in video_specs):
                grouped_by = ("subject", "ear")
            video_extensions = sanitize_extensions(
                resource_name="VideoSpec",
                extensions=first_video_spec.extensions,
            )
            if len(video_extensions) == 0 and config.video is not None:
                video_extensions = sanitize_extensions(
                    resource_name="VideoConfig",
                    extensions=config.video.extensions,
                )
            video_index, video_counts, missing_subject_ids = scanner.scan_video_paths(
                path=requested_video_path,
                subject_ids=available_subject_ids,
                subject_numbers=subject_numbers,
                extensions=video_extensions,
                grouped_by=grouped_by,
            )
            dataset._video_index = video_index
            dataset._video_counts = video_counts
            video_summary = resources_summary(
                checked=len(available_subject_ids),
                found=len(video_counts),
                missing=len(missing_subject_ids),
                missing_subject_ids=tuple(missing_subject_ids),
            )
            dataset._resource_summary["video"] = video_summary
            validator.validate_video_resources(video_summary)
        else:
            dataset._resource_summary["video"] = resources_summary()
        return DatasetResourcesPlan(
            hrtf_paths=dict(dataset._hrtf_paths),
            mesh_paths=dict(dataset._mesh_paths),
            image_path=dataset._image_path,
            video_path=dataset._video_path,
            image_index=dict(dataset._image_index),
            video_index=dict(dataset._video_index),
            image_counts=dict(dataset._image_counts),
            video_counts=dict(dataset._video_counts),
            anthropometry_path=dataset._anthropometry_path,
            anthropometry_rows=dict(dataset._anthropometry_rows),
            included_subject_ids=tuple(dataset._included_subject_ids),
            subject_numbers=dict(dataset._subject_numbers),
            resource_summary=dict(dataset._resource_summary),
        )
