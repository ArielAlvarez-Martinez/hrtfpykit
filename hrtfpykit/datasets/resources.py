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
from .split import DatasetSubjectSplitPlanner
from .summary import resources_summary

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetResourcesValidator:
    def __init__(self, dataset: "BaseDataset") -> None:
        self._dataset = dataset

    def validate_hrtf_resources(
        self,
        hrtf_paths: dict[str, Path],
        hrtf_summary: dict[str, object],
    ) -> dict[str, Path]:
        state = self._dataset._state
        if not any(isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec, SHSpec)) for spec in state.specs):
            return hrtf_paths
        missing_hrtf_subject_ids = list(
            hrtf_summary.get(
                "missing_subject_ids",
                tuple(),
            )
        )
        if len(missing_hrtf_subject_ids) > 0:
            preview = ", ".join(missing_hrtf_subject_ids[:5])
            suffix = "" if len(missing_hrtf_subject_ids) <= 5 else ", ..."
            warnings.warn(
                f"{state.name}: {len(missing_hrtf_subject_ids)} subjects do not have a matching HRTF file under "
                f"{state.root} and will be excluded ({preview}{suffix})",
                stacklevel=2,
            )
        validated_hrtf_paths = {}
        validated_sample_rate = None
        for subject_id, path in hrtf_paths.items():
            if not path.exists():
                warnings.warn(
                    f"{state.name}: subject {subject_id} HRTF path is missing and will be excluded: {path}",
                    stacklevel=2,
                )
                continue
            try:
                hrtf = load_hrtf(
                    self._dataset,
                    subject_id,
                    subject_ids=tuple(state.config.subject_ids),
                    hrtf_paths=hrtf_paths,
                    cache=state.cache,
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
                    f"{state.name} requires a consistent sample_rate across loaded HRTFs, "
                    f"but subject {subject_id!r} has sample_rate={current_sample_rate} "
                    f"and previous subjects use sample_rate={validated_sample_rate}"
                )
            validated_hrtf_paths[subject_id] = path
        return validated_hrtf_paths

    def validate_mesh_resources(self, mesh_summary: dict[str, object]) -> None:
        state = self._dataset._state
        if not any(isinstance(spec, MeshSpec) for spec in state.specs):
            return
        missing_mesh_subject_ids = tuple(
            mesh_summary.get(
                "missing_subject_ids",
                tuple(),
            )
        )
        if len(missing_mesh_subject_ids) > 0:
            warnings.warn(
                f"{state.name}: {len(missing_mesh_subject_ids)} subjects do not have a matching mesh file under "
                f"{state.root} and will be excluded when mesh is required "
                f"({', '.join(str(value) for value in missing_mesh_subject_ids[:5])}"
                f"{', ...' if len(missing_mesh_subject_ids) > 5 else ''})",
                stacklevel=2,
            )

    def validate_image_resources(
        self,
        summary: dict[str, object],
        image_path: Path | None,
        image_counts: dict[str, int],
    ) -> None:
        state = self._dataset._state
        if not any(isinstance(spec, ImageSpec) for spec in state.specs):
            return
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            raise ValueError(
                f"{state.name} image path is incompatible with the selected dataset subjects. "
                f"Missing subject folders under {image_path}: "
                f"{', '.join(str(value) for value in missing_subject_ids[:5])}"
                f"{', ...' if len(missing_subject_ids) > 5 else ''}"
            )
        if len(set(image_counts.values())) > 1:
            warnings.warn(
                f"{state.name}: subjects do not all have the same number of images under {image_path} "
                f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(image_counts.items())[:5])}"
                f"{'' if len(image_counts) <= 5 else ', ...'})",
                stacklevel=2,
            )

    def validate_video_resources(
        self,
        summary: dict[str, object],
        video_path: Path | None,
        video_counts: dict[str, int],
    ) -> None:
        state = self._dataset._state
        if not any(isinstance(spec, VideoSpec) for spec in state.specs):
            return
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            raise ValueError(
                f"{state.name} video path is incompatible with the selected dataset subjects. "
                f"Missing subject folders under {video_path}: "
                f"{', '.join(str(value) for value in missing_subject_ids[:5])}"
                f"{', ...' if len(missing_subject_ids) > 5 else ''}"
            )
        if len(set(video_counts.values())) > 1:
            warnings.warn(
                f"{state.name}: subjects do not all have the same number of videos under {video_path} "
                f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(video_counts.items())[:5])}"
                f"{'' if len(video_counts) <= 5 else ', ...'})",
                stacklevel=2,
            )

    def validate_anthropometry_resources(
        self,
        anthropometry_path: Path | None,
        anthropometry_rows: dict[str, object],
    ) -> None:
        state = self._dataset._state
        if not any(isinstance(spec, AnthropometrySpec) for spec in state.specs):
            return
        if anthropometry_path is None:
            raise ValueError(
                f"{state.name} requires an anthropometry file but none was selected"
            )
        if not anthropometry_path.is_file():
            raise ValueError(
                f"{state.name} anthropometry path is invalid: {anthropometry_path}"
            )
        if not isinstance(anthropometry_rows, dict):
            raise ValueError(
                f"{state.name} anthropometry data is invalid: expected a mapping but got {type(anthropometry_rows)!r}"
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
    excluded_subjects: tuple[str, ...]
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
    def build(
        dataset: "BaseDataset",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
    ) -> DatasetResourcesPlan:
        state = dataset._state
        if state.config is None:
            raise ValueError("Dataset config is not initialized")
        config = state.config
        root = state.root
        excluded_subjects = DatasetSubjectSplitPlanner.map_subject_ids(
            exclude_subject_ids,
            tuple(config.subject_ids),
        )
        excluded_subject_set = set(excluded_subjects)
        resource_subjects = tuple()
        if len(resource_subjects) == 0:
            sorted_subjects = DatasetSubjectSplitPlanner.sort_subject_ids(
                tuple(config.subject_ids)
            )
            resource_subjects = tuple(
                subject_id
                for subject_id in sorted_subjects
                if subject_id not in excluded_subject_set
            )
        subject_numbers = state.subject_numbers
        if len(subject_numbers) == 0:
            subject_numbers = DatasetSubjectSplitPlanner.build_subject_number_map(
                DatasetSubjectSplitPlanner.sort_subject_ids(tuple(config.subject_ids))
            )
        resource_summary = {}
        mesh_paths = {}
        image_path = None
        video_path = None
        image_index = {}
        video_index = {}
        image_counts = {}
        video_counts = {}
        anthropometry_path = None
        anthropometry_rows = {}

        validator = DatasetResourcesValidator(dataset)
        scanner = DatasetResourcesScanner()

        has_acoustic_specs = any(isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec, SHSpec)) for spec in state.specs)
        has_mesh_specs = any(isinstance(spec, MeshSpec) for spec in state.specs)
        has_anthro_specs = any(isinstance(spec, AnthropometrySpec) for spec in state.specs)
        has_image_specs = any(isinstance(spec, ImageSpec) for spec in state.specs)
        has_video_specs = any(isinstance(spec, VideoSpec) for spec in state.specs)

        hrtf_paths, hrtf_summary = scanner.scan_hrtf_paths(
            config=config,
            root=root,
            variant=state.variant if state.variant is not None else None,
            excluded_subject_ids=excluded_subject_set,
            required=has_acoustic_specs,
        )
        if hrtf_summary is None:
            hrtf_summary = resources_summary(
                checked=0,
                found=0,
                missing=0,
                missing_subject_ids=tuple(),
            )
        resource_summary["hrtf"] = hrtf_summary
        hrtf_paths = validator.validate_hrtf_resources(hrtf_paths, hrtf_summary)

        if has_mesh_specs:
            mesh_root_path = root
            mesh_specs = tuple(spec for spec in state.specs if isinstance(spec, MeshSpec))
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
                excluded_subject_ids=excluded_subject_set,
                required=has_mesh_specs,
                extensions=mesh_extensions,
            )
            mesh_summary = resources_summary(
                checked=int(mesh_summary.get("checked", 0)),
                found=int(mesh_summary.get("found", 0)),
                missing=int(mesh_summary.get("missing", 0)),
                missing_subject_ids=tuple(mesh_summary.get("missing_subject_ids", tuple())),
            )
            validator.validate_mesh_resources(mesh_summary)
            resource_summary["mesh"] = mesh_summary
        else:
            resource_summary["mesh"] = resources_summary()

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
            anthropometry_specs = tuple(spec for spec in state.specs if isinstance(spec, AnthropometrySpec))
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
            if anthropometry_path is None or not anthropometry_path.is_file():
                anthropometry_rows = {}
            else:
                anthropometry_rows = load_anthropometry(
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
            anthropometry_path = None
            anthropometry_rows = {}

        resource_summary["anthropometry"] = anthropometry_summary
        validator.validate_anthropometry_resources(
            anthropometry_path,
            anthropometry_rows,
        )

        if has_image_specs:
            image_specs = tuple(spec for spec in state.specs if isinstance(spec, ImageSpec))
            first_image_spec = image_specs[0]
            if first_image_spec.path is None:
                raise ValueError("ImageSpec requires a path")
            requested_image_path = DatasetResources._resolve_optional_path(first_image_spec.path, root)
            image_path = requested_image_path
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
                subject_ids=resource_subjects,
                subject_numbers=subject_numbers,
                extensions=image_extensions,
                grouped_by=grouped_by,
            )
            image_summary = resources_summary(
                checked=len(resource_subjects),
                found=len(image_counts),
                missing=len(missing_subject_ids),
                missing_subject_ids=tuple(missing_subject_ids),
            )
            resource_summary["image"] = image_summary
            validator.validate_image_resources(
                image_summary,
                image_path,
                image_counts,
            )
        else:
            resource_summary["image"] = resources_summary()

        if has_video_specs:
            video_specs = tuple(spec for spec in state.specs if isinstance(spec, VideoSpec))
            first_video_spec = video_specs[0]
            if first_video_spec.path is None:
                raise ValueError("VideoSpec requires a path")
            requested_video_path = DatasetResources._resolve_optional_path(first_video_spec.path, root)
            video_path = requested_video_path
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
                subject_ids=resource_subjects,
                subject_numbers=subject_numbers,
                extensions=video_extensions,
                grouped_by=grouped_by,
            )
            video_summary = resources_summary(
                checked=len(resource_subjects),
                found=len(video_counts),
                missing=len(missing_subject_ids),
                missing_subject_ids=tuple(missing_subject_ids),
            )
            resource_summary["video"] = video_summary
            validator.validate_video_resources(
                video_summary,
                video_path,
                video_counts,
            )
        else:
            resource_summary["video"] = resources_summary()
        return DatasetResourcesPlan(
            hrtf_paths=dict(hrtf_paths),
            mesh_paths=dict(mesh_paths),
            image_path=image_path,
            video_path=video_path,
            image_index=dict(image_index),
            video_index=dict(video_index),
            image_counts=dict(image_counts),
            video_counts=dict(video_counts),
            anthropometry_path=anthropometry_path,
            anthropometry_rows=dict(anthropometry_rows),
            excluded_subjects=tuple(excluded_subjects),
            subject_numbers=dict(subject_numbers),
            resource_summary=dict(resource_summary),
        )
