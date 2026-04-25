from pathlib import Path
import csv
import re
import warnings

import numpy as np

from .index import normalize_index_by, normalize_positions, split_subject_ids
from ..hrtf.coordinates import get_spherical_positions
from ..hrtf.hrtf import load_hrtf
from ..hrtf.planes import get_frontal_plane, get_horizontal_plane, get_median_plane

class DatasetResourceResolver:
    @staticmethod
    def resolve_positions_selection(
        positions: str | tuple[int, ...] | list[int] | np.ndarray,
        plane: str | tuple[object, ...] | dict[str, object] | None,
        hrtf,
    ) -> list[int]:
        position_count = int(hrtf.Sources.get_positions().shape[0])
        if plane is None:
            return normalize_positions(positions, position_count)
        if not isinstance(positions, str) or str(positions).strip().lower() != "all":
            raise ValueError("plane selection cannot be combined with custom positions")
        if isinstance(plane, str):
            plane_key = str(plane).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = default_angle
            angle_unit = "degrees"
        elif isinstance(plane, tuple):
            if len(plane) not in {2, 3} or not isinstance(plane[0], str):
                raise ValueError(
                    "Plane selection must be ('horizontal'|'median'|'frontal', angle[, angle_unit])"
                )
            plane_key = str(plane[0]).strip().lower()
            angle = plane[1]
            angle_unit = "degrees" if len(plane) == 2 else str(plane[2]).strip().lower()
        else:
            plane_key = str(plane.get("plane")).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = plane.get("angle", plane.get("plane_angle", default_angle))
            angle_unit = str(plane.get("angle_unit", "degrees")).strip().lower()
        if plane_key not in {"horizontal", "median", "frontal"}:
            raise ValueError("plane must be horizontal, median, or frontal")
        if plane_key == "horizontal":
            indices, _ = get_horizontal_plane(
                hrtf=hrtf,
                elevation=float(angle),
                angle_unit=angle_unit,
            )
        elif plane_key == "median":
            indices, _ = get_median_plane(
                hrtf=hrtf,
                azimuth=float(angle),
                angle_unit=angle_unit,
            )
        else:
            indices, _ = get_frontal_plane(
                hrtf=hrtf,
                azimuth=float(angle),
                angle_unit=angle_unit,
            )
        return [int(index) for index in np.asarray(indices, dtype=int).reshape(-1)]

    @staticmethod
    def is_hrtf_object(value: object) -> bool:
        return (
            hasattr(value, "IR")
            and hasattr(value, "TF")
            and hasattr(value, "Sources")
            and hasattr(value, "transform")
        )

    @staticmethod
    def is_explicit_hrtf_transform(transform) -> bool:
        return bool(getattr(transform, "__hrtf_transform__", False))

    @staticmethod
    def is_raw_hrtf_transform_method(transform) -> bool:
        transform_module = str(getattr(transform, "__module__", ""))
        transform_qualname = str(getattr(transform, "__qualname__", ""))
        return transform_module.endswith(".transforms") and transform_qualname.startswith("Transform.")

    @staticmethod
    def normalize_anthropometry_select(
        select: str | tuple[str, ...] | list[str] | None,
    ) -> str | tuple[str, ...]:
        if select is None:
            return "complete"
        if isinstance(select, str):
            value = str(select).strip()
            if value == "":
                raise ValueError("select must not be empty")
            if value.lower() in {"complete", "all"}:
                return "complete"
            values = (value,)
        else:
            values = tuple(str(value).strip() for value in select)
        if len(values) == 0:
            raise ValueError("select must not be empty")
        if any(value == "" for value in values):
            raise ValueError("select must not contain empty names")
        if len(set(values)) != len(values):
            raise ValueError("select must not contain duplicates")
        return values

    @staticmethod
    def normalize_anthropometry_ear(ear: str) -> str:
        value = str(ear).strip().lower()
        if value not in {"left", "right", "both"}:
            raise ValueError("ear must be 'left', 'right', or 'both'")
        return value

    @staticmethod
    def convert_table_value(value: str) -> float | str | None:
        text = str(value).strip()
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return text

    @classmethod
    def load_anthropometry_rows(
        cls,
        path: Path,
        subject_column_candidates: tuple[str, ...],
        subject_ids: tuple[str, ...],
        resolve_subject_id,
    ) -> dict[str, dict[str, float | str | None]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or len(reader.fieldnames) == 0:
                raise ValueError(f"Anthropometry file {path} does not contain headers")
            fieldnames = {fieldname.lower(): fieldname for fieldname in reader.fieldnames}
            subject_column = reader.fieldnames[0]
            for candidate in subject_column_candidates:
                if candidate.lower() in fieldnames:
                    subject_column = fieldnames[candidate.lower()]
                    break
            rows: dict[str, dict[str, float | str | None]] = {}
            for row in reader:
                raw_subject_id = row.get(subject_column)
                if raw_subject_id is None or str(raw_subject_id).strip() == "":
                    continue
                try:
                    subject_id = resolve_subject_id(raw_subject_id, subject_ids)
                except ValueError:
                    continue
                converted: dict[str, float | str | None] = {}
                for key, value in row.items():
                    if key is None or key == subject_column:
                        continue
                    converted[key] = cls.convert_table_value("" if value is None else value)
                rows[subject_id] = converted
        return rows

    @staticmethod
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

    @staticmethod
    def collect_ordered_media_files(
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

    @classmethod
    def scan_aligned_media_paths(
        cls,
        path: Path,
        subject_ids: tuple[str, ...],
        subject_numbers: dict[str, int],
        extensions: tuple[str, ...],
        align_by: tuple[str, ...],
        resource_name: str,
    ) -> tuple[
        dict[tuple[str, int | None, str | None], list[str]],
        dict[str, int],
        tuple[str, ...],
    ]:
        index: dict[tuple[str, int | None, str | None], list[str]] = {}
        if not path.exists():
            raise ValueError(f"{resource_name} path does not exist: {path}")
        subject_counts: dict[str, int] = {}
        missing_subject_ids: list[str] = []
        for subject_id in subject_ids:
            subject_folder = cls.resolve_subject_resource_folder(
                path,
                subject_id,
                int(subject_numbers[subject_id]),
                resource_name,
            )
            if subject_folder is None:
                missing_subject_ids.append(subject_id)
                continue
            subject_count = 0
            if "ear" in align_by:
                for ear in ("left", "right"):
                    ear_folder = subject_folder / ear
                    if not ear_folder.is_dir():
                        raise ValueError(
                            f"{resource_name} path {path} is incompatible with align_by={align_by}: "
                            f"subject {subject_id!r} is missing the {ear!r} folder"
                        )
                    files = cls.collect_ordered_media_files(ear_folder, extensions)
                    if len(files) == 0:
                        raise ValueError(
                            f"{resource_name} path {path} is incompatible with align_by={align_by}: "
                            f"subject {subject_id!r} has no {resource_name.lower()}s in {ear_folder}"
                        )
                    subject_count += len(files)
                    index[(subject_id, None, ear)] = files
            else:
                files = cls.collect_ordered_media_files(subject_folder, extensions)
                if len(files) == 0:
                    raise ValueError(
                        f"{resource_name} path {path} is incompatible with align_by={align_by}: "
                        f"subject {subject_id!r} has no {resource_name.lower()}s in {subject_folder}"
                    )
                subject_count = len(files)
                index[(subject_id, None, None)] = files
            subject_counts[subject_id] = subject_count
        return index, subject_counts, tuple(missing_subject_ids)

    @staticmethod
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

    @staticmethod
    def validate_aligned_asset_spec(
        dataset_name: str,
        asset_name: str,
        spec,
        supported_align_by: tuple[tuple[str, ...], ...] | None,
        asset_path: Path | None,
        asset_align_by: tuple[str, ...] | None,
        index_by: tuple[str, ...],
    ) -> None:
        if spec is None:
            return
        if supported_align_by is None:
            raise ValueError(f"{dataset_name} does not define a {asset_name} pipeline")
        if asset_path is None:
            raise ValueError(f"{type(spec).__name__}.path is required when {asset_name} is selected")
        if asset_align_by not in supported_align_by:
            raise ValueError(f"{asset_name} align_by must be one of {supported_align_by}")
        if asset_align_by is not None and "position" in asset_align_by and "position" not in index_by:
            raise ValueError(
                f"{asset_name} align_by including 'position' requires index_by to include 'position'"
            )
        if asset_align_by is not None and "ear" in asset_align_by and "ear" not in index_by:
            raise ValueError(
                f"{asset_name} align_by including 'ear' requires index_by to include 'ear'"
            )

    @staticmethod
    def preview_values(values: tuple[str, ...] | list[str], limit: int = 5) -> str:
        if len(values) == 0:
            return "none"
        preview = ", ".join(str(value) for value in values[:limit])
        if len(values) > limit:
            preview = f"{preview}, ..."
        return preview

    @classmethod
    def format_resource_summary(cls, resource_summary: dict[str, dict[str, object]]) -> str:
        if len(resource_summary) == 0:
            return "Resource summary: none"
        lines = ["Resource summary:"]
        for resource_name, summary in resource_summary.items():
            parts = [str(resource_name)]
            for key in (
                "pattern",
                "path",
                "variant",
                "extensions",
                "checked",
                "found",
                "valid",
                "invalid",
                "missing",
                "subjects",
                "rows",
            ):
                if key in summary:
                    parts.append(f"{key}={summary[key]!r}")
            if "missing_subject_ids" in summary:
                missing_subject_ids = tuple(summary["missing_subject_ids"])
                if len(missing_subject_ids) > 0:
                    parts.append(
                        f"missing_subject_ids={cls.preview_values(missing_subject_ids)}"
                    )
            if "invalid_subject_ids" in summary:
                invalid_subject_ids = tuple(summary["invalid_subject_ids"])
                if len(invalid_subject_ids) > 0:
                    parts.append(
                        f"invalid_subject_ids={cls.preview_values(invalid_subject_ids)}"
                    )
            lines.append("  " + parts[0] + ": " + ", ".join(parts[1:]))
        return "\n".join(lines)

    def format_load_summary(self) -> str:
        lines = [
            f"{self.name} dataset summary",
            f"  root: {self.root}",
            f"  split: {self.split}",
            f"  subjects_loaded: {len(self.subject_ids)}",
            f"  available_subjects: {len(self.available_subject_ids)}",
            f"  samples: {len(self._rows)}",
            f"  inputs: {', '.join(self.input_names) if len(self._input_specs) > 0 else 'none'}",
            f"  target: {', '.join(self.target_names) if len(self._target_specs) > 0 else 'none'}",
        ]
        if len(self.exclude_subject_ids) > 0:
            lines.append(f"  excluded_subjects: {len(self.exclude_subject_ids)}")
        if getattr(self, "variant", None) is not None:
            lines.append(f"  variant: {self.variant}")
        if self.sample_rate is not None:
            lines.append(f"  sample_rate: {self.sample_rate}")
        if self.selected_positions is not None:
            lines.append(f"  selected_positions: {len(self.selected_positions)}")
        lines.append(self.format_resource_summary(self.resource_summary))
        return "\n".join(lines)

    @staticmethod
    def sort_subject_ids(subject_ids: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        def subject_sort_key(value: str) -> tuple[int, str]:
            match = re.search(r"(\d+)$", str(value))
            if match is None:
                return (0, str(value).lower())
            return (int(match.group(1)), str(value).lower())

        return sorted(subject_ids, key=subject_sort_key)

    def resolve_dataset_paths(self) -> None:
        self._image_path = self.resolve_optional_path(
            None if self.primary_image_spec is None else self.primary_image_spec.path,
            self.root,
        )
        self._video_path = self.resolve_optional_path(
            None if self.primary_video_spec is None else self.primary_video_spec.path,
            self.root,
        )
        self._anthropometry_path = self.resolve_optional_path(
            None if self.primary_anthropometry_spec is None else self.primary_anthropometry_spec.path,
            self.root,
        )
        self._image_align_by = (
            None if self.primary_image_spec is None else normalize_index_by(self.primary_image_spec.align_by)
        )
        self._video_align_by = (
            None if self.primary_video_spec is None else normalize_index_by(self.primary_video_spec.align_by)
        )

    def validate_dataset_assets(self) -> None:
        image_supported_align_by = (
            None if self.config.image is None else tuple(self.config.image.supported_align_by)
        )
        video_supported_align_by = (
            None if self.config.video is None else tuple(self.config.video.supported_align_by)
        )
        self.validate_aligned_asset_spec(
            dataset_name=self.name,
            asset_name="image",
            spec=self.primary_image_spec,
            supported_align_by=image_supported_align_by,
            asset_path=self._image_path,
            asset_align_by=self._image_align_by,
            index_by=self.index_by,
        )
        self.validate_aligned_asset_spec(
            dataset_name=self.name,
            asset_name="video",
            spec=self.primary_video_spec,
            supported_align_by=video_supported_align_by,
            asset_path=self._video_path,
            asset_align_by=self._video_align_by,
            index_by=self.index_by,
        )
        if self.primary_mesh_spec is not None and self.config.mesh is None:
            raise ValueError(f"{self.name} does not provide mesh data")
        if (
            self.primary_anthropometry_spec is not None
            and self._anthropometry_path is None
            and self.config.anthropometry is None
        ):
            raise ValueError(f"{self.name} does not provide anthropometry")
        if self._anthropometry_path is not None:
            if not self._anthropometry_path.exists():
                raise ValueError(f"AnthropometrySpec.path does not exist: {self._anthropometry_path}")
            if not self._anthropometry_path.is_file():
                raise ValueError(f"AnthropometrySpec.path is not a file: {self._anthropometry_path}")

    def initialize_dataset_subjects(self) -> tuple[set[str], tuple[str, ...], dict[str, int]]:
        excluded_subject_ids = set(self.exclude_subject_ids)
        included_subject_ids = tuple(
            subject_id
            for subject_id in self.config.subject_ids
            if subject_id not in excluded_subject_ids
        )
        self.included_subject_ids = included_subject_ids
        subject_numbers = {
            subject_id: index
            for index, subject_id in enumerate(tuple(self.config.subject_ids), start=1)
        }
        return excluded_subject_ids, included_subject_ids, subject_numbers

    def resolve_hrtf_resources(self, excluded_subject_ids: set[str]) -> None:
        self._hrtf_paths = {}
        if self.config.hrtf is None or self.primary_hrtf_backed_spec is None:
            return

        hrtf_subject_ids = (
            tuple(self.config.subject_ids)
            if self.config.hrtf.subject_ids is None
            else tuple(self.config.hrtf.subject_ids)
        )
        checked_hrtf_subject_ids = tuple(
            subject_id for subject_id in hrtf_subject_ids if subject_id not in excluded_subject_ids
        )
        for subject_id in checked_hrtf_subject_ids:
            relative_path = self.config.hrtf.path_pattern.format(
                subject_id=subject_id,
                variant=self.variant,
            )
            candidate = (self.root / relative_path).expanduser()
            if candidate.is_file():
                self._hrtf_paths[subject_id] = candidate
        missing_hrtf_subject_ids = tuple(
            subject_id
            for subject_id in checked_hrtf_subject_ids
            if subject_id not in self._hrtf_paths
        )
        self.resource_summary["hrtf"] = {
            "pattern": self.config.hrtf.path_pattern,
            "variant": self.variant,
            "checked": len(checked_hrtf_subject_ids),
            "found": len(self._hrtf_paths),
            "missing": len(missing_hrtf_subject_ids),
            "missing_subject_ids": missing_hrtf_subject_ids,
        }

    def resolve_mesh_resources(self, excluded_subject_ids: set[str]) -> None:
        self._mesh_paths = {}
        if self.config.mesh is None or self.primary_mesh_spec is None:
            return
        mesh_subject_ids = (
            tuple(self.config.subject_ids)
            if self.config.mesh.subject_ids is None
            else tuple(self.config.mesh.subject_ids)
        )
        checked_mesh_subject_ids = tuple(
            subject_id for subject_id in mesh_subject_ids if subject_id not in excluded_subject_ids
        )
        for subject_id in checked_mesh_subject_ids:
            for extension in self.config.mesh.extensions:
                relative_path = self.config.mesh.path_pattern.format(
                    subject_id=subject_id,
                    extension=extension,
                )
                candidate = (self.root / relative_path).expanduser()
                if candidate.is_file():
                    self._mesh_paths[subject_id] = candidate
                    break
        missing_mesh_subject_ids = tuple(
            subject_id
            for subject_id in checked_mesh_subject_ids
            if subject_id not in self._mesh_paths
        )
        self.resource_summary["mesh"] = {
            "pattern": self.config.mesh.path_pattern,
            "extensions": tuple(self.config.mesh.extensions),
            "checked": len(checked_mesh_subject_ids),
            "found": len(self._mesh_paths),
            "missing": len(missing_mesh_subject_ids),
            "missing_subject_ids": missing_mesh_subject_ids,
        }

    def resolve_anthropometry_resources(self) -> None:
        if self._anthropometry_path is None and self.config.anthropometry is not None:
            candidate = (self.root / self.config.anthropometry.path).expanduser()
            if candidate.is_file():
                self._anthropometry_path = candidate
        self._anthropometry_rows = {}
        if self.primary_anthropometry_spec is None:
            return
        subject_column_candidates = (
            (
                "subject_id",
                "subject",
                "id",
                "participant",
                "pp",
            )
            if self.config.anthropometry is None
            else tuple(self.config.anthropometry.subject_column_candidates)
        )
        anthropometry_summary: dict[str, object] = {
            "path": None if self._anthropometry_path is None else str(self._anthropometry_path),
            "found": self._anthropometry_path is not None and self._anthropometry_path.is_file(),
        }
        if self._anthropometry_path is None:
            self._anthropometry_rows = {}
        else:
            self._anthropometry_rows = self.load_anthropometry_rows(
                self._anthropometry_path,
                subject_column_candidates,
                tuple(self.config.subject_ids),
                self.resolve_dataset_subject_id,
            )
            anthropometry_summary["subjects"] = len(self._anthropometry_rows)
            anthropometry_summary["rows"] = len(self._anthropometry_rows)
        self.resource_summary["anthropometry"] = anthropometry_summary

    def validate_hrtf_resources(self) -> None:
        if self.primary_hrtf_backed_spec is None:
            return
        missing_hrtf_subject_ids = list(
            self.resource_summary.get("hrtf", {}).get("missing_subject_ids", tuple())
        )
        if len(missing_hrtf_subject_ids) > 0:
            preview = ", ".join(missing_hrtf_subject_ids[:5])
            suffix = "" if len(missing_hrtf_subject_ids) <= 5 else ", ..."
            warnings.warn(
                f"{self.name}: {len(missing_hrtf_subject_ids)} subjects do not have a matching HRTF file under "
                f"{self.root} and will be excluded ({preview}{suffix})",
                stacklevel=2,
            )
        validated_hrtf_paths = {}
        validated_sample_rate = None
        for subject_id, path in self._hrtf_paths.items():
            if not path.exists():
                warnings.warn(
                    f"{self.name}: subject {subject_id} HRTF path is missing and will be excluded: {path}",
                    stacklevel=2,
                )
                continue
            try:
                hrtf = load_hrtf(path)
            except Exception as exc:
                warnings.warn(
                    f"{self.name}: subject {subject_id} HRTF file could not be loaded and will be excluded: "
                    f"{path} ({exc})",
                    stacklevel=2,
                )
                continue
            hrtf = self.resolve_dataset_hrtf(subject_id, hrtf)
            current_sample_rate = (
                None if hrtf.IR.sample_rate is None else float(hrtf.IR.sample_rate)
            )
            if validated_sample_rate is None:
                validated_sample_rate = current_sample_rate
            elif current_sample_rate != validated_sample_rate:
                raise ValueError(
                    f"{self.name} requires a consistent sample_rate across loaded HRTFs, "
                    f"but subject {subject_id!r} has sample_rate={current_sample_rate} "
                    f"and previous subjects use sample_rate={validated_sample_rate}"
                )
            validated_hrtf_paths[subject_id] = path
            if self._cache_hrtf:
                self._hrtf_cache[subject_id] = hrtf
        invalid_hrtf_subject_ids = tuple(
            subject_id
            for subject_id in self._hrtf_paths
            if subject_id not in validated_hrtf_paths
        )
        self._hrtf_paths = validated_hrtf_paths
        self.resource_summary["hrtf"]["valid"] = len(self._hrtf_paths)
        self.resource_summary["hrtf"]["invalid"] = len(invalid_hrtf_subject_ids)
        self.resource_summary["hrtf"]["invalid_subject_ids"] = invalid_hrtf_subject_ids

    def warn_missing_mesh_resources(self) -> None:
        if self.primary_mesh_spec is None:
            return
        missing_mesh_subject_ids = tuple(
            self.resource_summary.get("mesh", {}).get("missing_subject_ids", tuple())
        )
        if len(missing_mesh_subject_ids) > 0:
            warnings.warn(
                f"{self.name}: {len(missing_mesh_subject_ids)} subjects do not have a matching mesh file under "
                f"{self.root} and will be excluded when mesh is required "
                f"({self.preview_values(missing_mesh_subject_ids)})",
                stacklevel=2,
            )

    def resolve_media_resources(
        self,
        resource_name: str,
        included_subject_ids: tuple[str, ...],
        subject_numbers: dict[str, int],
    ) -> None:
        if resource_name == "image":
            path = self._image_path
            align_by = self._image_align_by
            config = self.config.image
            spec = self.primary_image_spec
            index_name = "_image_index"
            counts_name = "_image_counts"
        elif resource_name == "video":
            path = self._video_path
            align_by = self._video_align_by
            config = self.config.video
            spec = self.primary_video_spec
            index_name = "_video_index"
            counts_name = "_video_counts"
        else:
            raise ValueError(f"Unsupported media resource {resource_name!r}")

        if path is None or config is None or align_by is None:
            index = {}
            counts = {}
            summary = {
                "path": None if path is None else str(path),
                "found": 0,
                "missing": 0,
                "missing_subject_ids": tuple(),
            }
        else:
            index, counts, missing_subject_ids = self.scan_aligned_media_paths(
                path,
                included_subject_ids,
                subject_numbers,
                tuple(config.extensions),
                align_by,
                resource_name.capitalize(),
            )
            found_subjects = len({key[0] for key in index})
            summary = {
                "path": str(path),
                "found": found_subjects,
                "missing": len(missing_subject_ids),
                "missing_subject_ids": tuple(missing_subject_ids),
            }
        setattr(self, index_name, index)
        setattr(self, counts_name, counts)
        if spec is None:
            return
        self.resource_summary[resource_name] = summary
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            raise ValueError(
                f"{self.name} {resource_name} path is incompatible with the selected dataset subjects. "
                f"Missing subject folders under {path}: "
                f"{self.preview_values(missing_subject_ids)}"
            )
        if len(set(counts.values())) > 1:
            warnings.warn(
                f"{self.name}: subjects do not all have the same number of {resource_name}s under {path} "
                f"({', '.join(f'{subject_id}={count}' for subject_id, count in sorted(counts.items())[:5])}"
                f"{'' if len(counts) <= 5 else ', ...'})",
                stacklevel=2,
            )

    def resolve_dataset_resources(self) -> None:
        self.resource_summary = {}
        self.resolve_dataset_paths()
        self.validate_dataset_assets()
        excluded_subject_ids, included_subject_ids, subject_numbers = self.initialize_dataset_subjects()
        self.resolve_hrtf_resources(excluded_subject_ids)
        self.resolve_mesh_resources(excluded_subject_ids)
        self.resolve_anthropometry_resources()
        self.validate_hrtf_resources()
        self.warn_missing_mesh_resources()
        self.resolve_media_resources("image", included_subject_ids, subject_numbers)
        self.resolve_media_resources("video", included_subject_ids, subject_numbers)

    def collect_required_subject_sets(self) -> list[set[str]]:
        required_subject_sets: list[set[str]] = []
        if self.primary_hrtf_backed_spec is not None:
            required_subject_sets.append(set(self._hrtf_paths))
        if self.primary_mesh_spec is not None:
            required_subject_sets.append(set(self._mesh_paths))
        if self.primary_anthropometry_spec is not None:
            required_subject_sets.append(set(self._anthropometry_rows))
        if self.primary_image_spec is not None:
            required_subject_sets.append({key[0] for key in self._image_index})
        if self.primary_video_spec is not None:
            required_subject_sets.append({key[0] for key in self._video_index})
        return required_subject_sets

    def resolve_dataset_subjects(
        self,
        split: str,
        split_ratio: tuple[float, float, float],
        split_seed: int,
    ) -> None:
        required_subject_sets = self.collect_required_subject_sets()
        if len(required_subject_sets) == 0:
            subject_ids = self.sort_subject_ids(list(self.included_subject_ids))
        else:
            subject_ids = self.sort_subject_ids(set.intersection(*required_subject_sets))
        if len(subject_ids) == 0 and len(required_subject_sets) > 0:
            available_counts = []
            if len(self.hrtf_backed_specs) > 0:
                available_counts.append(f"hrtf={len(self._hrtf_paths)}")
            if self.primary_mesh_spec is not None:
                available_counts.append(f"mesh={len(self._mesh_paths)}")
            if self.primary_anthropometry_spec is not None:
                available_counts.append(f"anthropometry={len(self._anthropometry_rows)}")
            if self.primary_image_spec is not None:
                available_counts.append(f"image={len({key[0] for key in self._image_index})}")
            if self.primary_video_spec is not None:
                available_counts.append(f"video={len({key[0] for key in self._video_index})}")
            raise ValueError(
                "No subjects match the selected dataset configuration. "
                f"Selected specs: {', '.join(sorted(set(self.input_names + self.target_names)))}. "
                f"Available subject counts by spec: {', '.join(available_counts)}. "
                f"Root: {self.root}\n"
                f"{self.format_resource_summary(self.resource_summary)}"
            )
        self.available_subject_ids = tuple(subject_ids)
        split_subjects = split_subject_ids(subject_ids, split, split_ratio, split_seed)
        if len(split_subjects) == 0:
            raise ValueError(f"Split {split!r} produced an empty dataset")
        self.subject_ids = tuple(split_subjects)
        self.split = split
        self.split_ratio = split_ratio
        self.split_seed = split_seed

    def resolve_dataset_hrtf(self, subject_id: str, hrtf):
        if self.hrtf_transform is None:
            return hrtf
        transformed_hrtf = self._dataset_transformed_hrtf_cache.get(subject_id)
        if transformed_hrtf is not None:
            return transformed_hrtf
        transformed_hrtf = self.hrtf_transform(hrtf)
        if not self.is_hrtf_object(transformed_hrtf):
            raise ValueError("hrtf_transform must return an HRTF object")
        if self._cache_hrtf:
            self._dataset_transformed_hrtf_cache[subject_id] = transformed_hrtf
        return transformed_hrtf

    def get_subject_hrtf(self, subject_id: str | int):
        resolved_subject_id = self.resolve_dataset_subject_id(subject_id, self.subject_ids)
        if resolved_subject_id not in self._hrtf_paths:
            raise KeyError(
                f"Subject {subject_id!r} resolved to {resolved_subject_id!r} but does not have an available HRTF file"
            )
        path = self._hrtf_paths[resolved_subject_id]
        if not path.exists():
            warnings.warn(
                f"{self.name}: subject {resolved_subject_id} HRTF path is missing: {path}",
                stacklevel=2,
            )
            raise FileNotFoundError(
                f"HRTF path is missing for subject {resolved_subject_id}: {path}"
            )
        hrtf = self._hrtf_cache.get(resolved_subject_id)
        if hrtf is None:
            try:
                hrtf = load_hrtf(path)
            except Exception as exc:
                warnings.warn(
                    f"{self.name}: subject {resolved_subject_id} HRTF file could not be loaded: {path} ({exc})",
                    stacklevel=2,
                )
                raise
            if self._cache_hrtf:
                self._hrtf_cache[resolved_subject_id] = hrtf
        return self.resolve_dataset_hrtf(resolved_subject_id, hrtf)

    def reset_acoustic_context(self) -> None:
        self.sample_rate = None
        self.available_positions = None
        self.selected_positions = None
        self.available_azimuth_angles = None
        self.available_elevation_angles = None
        self.azimuth_angles = None
        self.elevation_angles = None
        self.frequency_bins = None
        self.sample_indices = None
        self._selected_position_indices = []
        self._selected_frequency_indices = []
        self._selected_sample_indices = []

    def configure_reference_hrtf(self, reference_hrtf) -> None:
        self.sample_rate = (
            None if reference_hrtf.IR.sample_rate is None else float(reference_hrtf.IR.sample_rate)
        )
        self.available_positions = np.asarray(
            reference_hrtf.Sources.get_positions(angle_unit="degrees"),
            dtype=float,
        )

    def configure_frequency_and_sample_axes(self, reference_hrtf) -> None:
        if reference_hrtf.TF.frequency_bins is not None:
            self.frequency_bins = np.asarray(reference_hrtf.TF.frequency_bins, dtype=float)
            self._selected_frequency_indices = list(range(int(self.frequency_bins.shape[0])))
        self.sample_indices = np.arange(reference_hrtf.IR.values.shape[-1], dtype=int)
        self._selected_sample_indices = list(range(int(self.sample_indices.shape[0])))

    def configure_spatial_context(self, reference_hrtf) -> None:
        if self.primary_spatial_spec is not None:
            self._selected_position_indices = self.resolve_positions_selection(
                self.primary_spatial_spec.positions,
                self.primary_spatial_spec.plane,
                reference_hrtf,
            )
            self.selected_positions = np.asarray(
                self.available_positions[self._selected_position_indices],
                dtype=float,
            )

    def configure_angle_context(self, reference_hrtf) -> None:
        spherical_positions = np.asarray(
            get_spherical_positions(reference_hrtf.Sources, angle_unit="degrees"),
            dtype=float,
        )
        self.available_azimuth_angles = np.unique(np.round(spherical_positions[:, 0], 2))
        self.available_elevation_angles = np.unique(np.round(spherical_positions[:, 1], 2))
        if self.primary_spatial_spec is not None:
            selected_spherical_positions = np.asarray(
                spherical_positions[self._selected_position_indices],
                dtype=float,
            )
            self.azimuth_angles = np.unique(np.round(selected_spherical_positions[:, 0], 2))
            self.elevation_angles = np.unique(np.round(selected_spherical_positions[:, 1], 2))

    def prepare_acoustic_context(self) -> None:
        self.reset_acoustic_context()
        if self.primary_hrtf_backed_spec is None:
            return
        reference_subject_id = self.subject_ids[0]
        reference_hrtf = self.get_subject_hrtf(reference_subject_id)
        self.configure_reference_hrtf(reference_hrtf)
        self.configure_frequency_and_sample_axes(reference_hrtf)
        self.configure_spatial_context(reference_hrtf)
        self.configure_angle_context(reference_hrtf)
