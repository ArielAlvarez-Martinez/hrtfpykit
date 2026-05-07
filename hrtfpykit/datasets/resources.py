from pathlib import Path
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import DatasetConfig
from .load import load_hrtf
from .load import load_table
from .specs_registry import get_specs, has_specs
from .sanitize import sanitize_extensions
from .sanitize import sanitize_grouped_by
from .split import DatasetSplitPlanner
from .summary import resources_summary

if TYPE_CHECKING:
    from .base import BaseDataset


class DatasetResourcesValidator:
    """Validate scanned resources against the selected specs.

    ``DatasetResourcesValidator`` checks the resource plans produced by scanners
    before subject intersection and split planning continue.
    """
    def __init__(self, dataset: "BaseDataset") -> None:
        self._dataset = dataset

    def validate_hrtf_resources(
        self,
        hrtf_paths: dict[str, Path],
        hrtf_summary: dict[str, object],
    ) -> dict[str, Path]:
        """Validate HRTF paths and loadability before split planning.

        The scanner can find files by path pattern, but construction should fail if
        those files are corrupt or incompatible with the HRTF loader. This validator
        loads each candidate, enforces consistent sample rate, collects failures, and
        returns only usable subject paths.

        Parameters
        ----------
        hrtf_paths : dict
            Subject-to-path HRTF resources.
        hrtf_summary : dict
            Scanner summary for HRTF resources.

        Returns
        -------
        dict Validated subject-to-path HRTF resources.

        Use Cases
        ---------
        - Fail on corrupt HRTF files.
        - Enforce consistent sample rates across subjects.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="hrtf"):
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
        failed_hrtf_loads: list[tuple[str, Path, Exception]] = []
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
            except Exception as exc:
                failed_hrtf_loads.append((subject_id, path, exc))
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
        if len(failed_hrtf_loads) > 0:
            failure_lines = []
            for subject_id, path, exc in failed_hrtf_loads[:10]:
                failure_lines.append(
                    f"{subject_id}: {path} ({type(exc).__name__}: {exc})"
                )
            if len(failed_hrtf_loads) > 10:
                failure_lines.append(f"... {len(failed_hrtf_loads) - 10} more")
            raise ValueError(
                f"{state.name} failed to load {len(failed_hrtf_loads)} HRTF file(s): "
                + "; ".join(failure_lines)
            )
        return validated_hrtf_paths

    def validate_mesh_resources(self, mesh_summary: dict[str, object]) -> None:
        """Validate scanned mesh resource summary.

        Mesh files participate in subject intersection like other required resources,
        but missing subjects are reported through warnings rather than hidden. This
        validator keeps that reporting behavior separate from path scanning.

        Parameters
        ----------
        mesh_summary : dict
            Scanner summary for mesh resources.

        Returns
        -------
        None Emits warnings for missing mesh subjects.

        Use Cases
        ---------
        - Report mesh resources excluded by intersection.
        - Keep mesh validation separate from scanning.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="mesh"):
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
        """Validate scanned image resources for selected specs.

        The scanner indexes subject or subject-ear image files, while this validator
        reports missing subjects and uneven media counts. Missing media subjects are
        removed later during resource intersection.

        Parameters
        ----------
        summary : dict
            Scanner summary for image resources.
        image_path : Path or None
            Image root path.
        image_counts : dict
            Per-subject image counts.

        Returns
        -------
        None Emits warnings for missing or uneven image resources.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="image"):
            return
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            warnings.warn(
                f"{state.name}: {len(missing_subject_ids)} subjects do not have matching image folders/files under "
                f"{image_path} and will be excluded when image is required "
                f"("
                f"{', '.join(str(value) for value in missing_subject_ids[:5])}"
                f"{', ...' if len(missing_subject_ids) > 5 else ''})",
                stacklevel=2,
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
        """Validate scanned video resources for selected specs.

        The scanner indexes subject or subject-ear video files, while this validator
        reports missing subjects and uneven media counts. It mirrors image validation
        so media resource behavior uses the same validation path.

        Parameters
        ----------
        summary : dict
            Scanner summary for video resources.
        video_path : Path or None
            Video root path.
        video_counts : dict
            Per-subject video counts.

        Returns
        -------
        None Emits warnings for missing or uneven video resources.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="video"):
            return
        missing_subject_ids = tuple(summary["missing_subject_ids"])
        if len(missing_subject_ids) > 0:
            warnings.warn(
                f"{state.name}: {len(missing_subject_ids)} subjects do not have matching video folders/files under "
                f"{video_path} and will be excluded when video is required "
                f"("
                f"{', '.join(str(value) for value in missing_subject_ids[:5])}"
                f"{', ...' if len(missing_subject_ids) > 5 else ''})",
                stacklevel=2,
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
        """Validate selected anthropometry resources after table loading.

        Anthropometry specs require a real table path and loaded mapping data. This
        method turns missing files or invalid loaded values into explicit construction
        errors before split intersection uses the resource.

        Parameters
        ----------
        anthropometry_path : Path or None
            Selected anthropometry table path.
        anthropometry_rows : dict
            Loaded anthropometry rows.

        Returns
        -------
        None Raises when required anthropometry resources are invalid.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="anthropometry"):
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

    def validate_metadata_resources(
        self,
        metadata_path: Path | None,
        metadata_rows: dict[str, object],
    ) -> None:
        """Validate selected metadata resources after table loading.

        Metadata specs require a real table path and loaded mapping data. This method
        keeps metadata validation separate from anthropometry while preserving the
        same table-resource contract.

        Parameters
        ----------
        metadata_path : Path or None
            Selected metadata table path.
        metadata_rows : dict
            Loaded metadata rows.

        Returns
        -------
        None Raises when required metadata resources are invalid.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="metadata"):
            return
        if metadata_path is None:
            raise ValueError(
                f"{state.name} requires a metadata file but none was selected"
            )
        if not metadata_path.is_file():
            metadata_specs = get_specs(state.specs, resource_name="metadata")
            metadata_spec_path = None
            if len(metadata_specs) > 0:
                metadata_spec_path = metadata_specs[0].path
            config_metadata_path = None
            if state.config is not None and state.config.metadata is not None:
                config_metadata_path = state.root / state.config.metadata.path
            raise ValueError(
                f"{state.name} metadata resource is required because MetadataSpec is requested, "
                f"but the selected metadata file does not exist or is not a file. "
                f"selected_path={metadata_path}; "
                f"root={state.root}; "
                f"metadata_spec_path={metadata_spec_path}; "
                f"config_metadata_path={config_metadata_path}; "
                f"fix: place the metadata file at selected_path, pass MetadataSpec(path=...), "
                f"or download it with download=True and download_resources='metadata' "
                f"or download_resources=('hrtf', 'metadata')."
            )
        if not isinstance(metadata_rows, dict):
            raise ValueError(
                f"{state.name} metadata data is invalid: expected a mapping but got {type(metadata_rows)!r}"
            )


class DatasetResourcesScanner:
    """Scan dataset roots for resources requested by specs.

    This utility locates HRTF, mesh, table, image, and video resources from config
    paths, spec path overrides, and per-subject path patterns.
    """

    @staticmethod
    def scan_anthropometry_paths(
        config: type[DatasetConfig] | DatasetConfig,
        root: Path,
        requested_path: Path | None,
        required: bool,
    ) -> tuple[Path | None, dict[str, object]]:
        """Scan for the anthropometry table required by selected specs.

        This scanner resolves either an explicit spec path or the dataset-configured
        anthropometry path and returns a summary without loading the table. Loading
        happens later so path scanning, validation, and table parsing stay separate.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate anthropometry table without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
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
    def scan_metadata_paths(
        config: type[DatasetConfig] | DatasetConfig,
        root: Path,
        requested_path: Path | None,
        required: bool,
    ) -> tuple[Path | None, dict[str, object]]:
        """Scan for the metadata table required by selected specs.

        This scanner resolves either an explicit spec path or the dataset-configured
        metadata path and returns a summary without loading the table. Keeping
        metadata separate from anthropometry prevents path and state collisions.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate metadata table without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
        if config.metadata is None and requested_path is None and not required:
            return None, {
                "path": None,
                "found": False,
                "subjects": 0,
                "rows": 0,
            }
        metadata_path = requested_path
        if metadata_path is None and config.metadata is not None:
            metadata_path = (root / config.metadata.path).expanduser()
        if metadata_path is None:
            return None, {
                "path": None,
                "found": False,
                "subjects": 0,
                "rows": 0,
            }
        return metadata_path, {
            "path": str(metadata_path),
            "found": metadata_path.is_file(),
            "subjects": 0,
            "rows": 0,
            "extensions": tuple(config.metadata.extensions)
            if config.metadata is not None and config.metadata.extensions is not None
            else tuple(),
        }

    @staticmethod
    def scan_hrtf_paths(
        config: type[DatasetConfig] | DatasetConfig,
        root: Path,
        dataset_hrtf_variant: str | dict[str, object] | None,
        excluded_subject_ids: set[str],
        required: bool,
    ) -> tuple[dict[str, Path], dict[str, object] | None]:
        """Scan for HRTF files required by selected acoustic specs.

        The scanner formats the dataset HRTF path pattern with subject IDs, subject
        numbers, type, sample-rate, and version selectors. It returns the paths that
        exist plus a summary of checked, found, and missing subjects for validation
        and error reporting.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate HRTF files without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
        hrtf_paths: dict[str, Path] = {}
        if config.hrtf is None or not required:
            return hrtf_paths, None
        if isinstance(dataset_hrtf_variant, dict):
            hrtf_type = str(dataset_hrtf_variant["type"])
            hrtf_sample_rate = dataset_hrtf_variant.get("sample_rate")
            hrtf_version = dataset_hrtf_variant.get("version")
        else:
            hrtf_type = dataset_hrtf_variant
            hrtf_sample_rate = None
            hrtf_version = None
        if hrtf_type is None:
            raise ValueError(f"{config.name} requires dataset_hrtf_variant for HRTF resources")
        hrtf_type_config = config.hrtf.types[hrtf_type]
        sample_rate_label = None
        if hrtf_sample_rate is not None:
            sample_rate_label = str(hrtf_sample_rate)
            if hrtf_type_config.sample_rate_labels is not None:
                sample_rate_label = hrtf_type_config.sample_rate_labels.get(
                    hrtf_sample_rate,
                    sample_rate_label,
                )
        hrtf_subject_ids = (
            tuple(config.subject_ids)
            if config.hrtf.subject_ids is None
            else tuple(config.hrtf.subject_ids)
        )
        subject_numbers = DatasetSplitPlanner.build_subject_number_map(
            DatasetSplitPlanner.sort_subject_ids(tuple(config.subject_ids))
        )
        checked_hrtf_subject_ids = tuple(
            subject_id
            for subject_id in hrtf_subject_ids
            if subject_id not in excluded_subject_ids
        )
        for subject_id in checked_hrtf_subject_ids:
            version_label = None
            if hrtf_type_config.version_labels is not None and hrtf_version is not None:
                version_label = hrtf_type_config.version_labels.get(
                    hrtf_version,
                    str(hrtf_version),
                )
            relative_path = hrtf_type_config.path_pattern.format(
                subject_id=subject_id,
                subject_number=subject_numbers[subject_id],
                type=hrtf_type,
                hrtf_type=hrtf_type,
                sample_rate=hrtf_sample_rate,
                hrtf_sample_rate=hrtf_sample_rate,
                sample_rate_label=sample_rate_label,
                version=hrtf_version,
                hrtf_version=hrtf_version,
                version_label=version_label,
                hrtf_version_label=version_label,
                variant=hrtf_type,
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
            "pattern": hrtf_type_config.path_pattern,
            "hrtf_variant": {
                "type": hrtf_type,
                "sample_rate": hrtf_sample_rate,
                "version": hrtf_version,
            }
            if hrtf_sample_rate is not None or hrtf_version is not None
            else hrtf_type,
            "checked": len(checked_hrtf_subject_ids),
            "found": len(hrtf_paths),
            "missing": len(missing_hrtf_subject_ids),
            "missing_subject_ids": missing_hrtf_subject_ids,
        }

    @staticmethod
    def scan_mesh_paths(
        config: type[DatasetConfig] | DatasetConfig,
        root: Path,
        dataset_mesh_variant: str | dict[str, object] | None,
        excluded_subject_ids: set[str],
        required: bool,
        extensions: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Path], dict[str, object] | None]:
        """Scan for mesh files required by selected mesh specs.

        The scanner resolves the selected mesh variant, applies extension candidates,
        formats per-subject path patterns, and records which subjects have usable
        mesh files. It does not load mesh geometry.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate mesh files without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
        mesh_paths: dict[str, Path] = {}
        if config.mesh is None or not required:
            return mesh_paths, None
        if isinstance(dataset_mesh_variant, dict):
            mesh_type = str(dataset_mesh_variant["type"])
            mesh_version = dataset_mesh_variant.get("version")
        else:
            mesh_type = dataset_mesh_variant
            mesh_version = None
        if mesh_type is None:
            mesh_type = "default" if "default" in config.mesh.types else None
        if mesh_type is None:
            raise ValueError(f"{config.name} requires dataset_mesh_variant for mesh resources")
        mesh_type_config = config.mesh.types[mesh_type]
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
        subject_numbers = DatasetSplitPlanner.build_subject_number_map(
            DatasetSplitPlanner.sort_subject_ids(tuple(config.subject_ids))
        )
        for subject_id in checked_mesh_subject_ids:
            version_label = None
            if mesh_type_config.version_labels is not None and mesh_version is not None:
                version_label = mesh_type_config.version_labels.get(
                    mesh_version,
                    str(mesh_version),
                )
            relative_path = mesh_type_config.path_pattern.format(
                subject_id=subject_id,
                subject_number=subject_numbers[subject_id],
                type=mesh_type,
                mesh_type=mesh_type,
                version=mesh_version,
                mesh_version=mesh_version,
                version_label=version_label,
                mesh_version_label=version_label,
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
            "pattern": mesh_type_config.path_pattern,
            "mesh_variant": {
                "type": mesh_type,
                "version": mesh_version,
            }
            if mesh_version is not None
            else mesh_type,
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
        ears: tuple[str, ...],
        resource_name: str,
    ) -> tuple[
        dict[tuple[str, int | None, str | None], list[str]],
        dict[str, int],
        tuple[str, ...],
    ]:
        """Scan subject-grouped media folders for image or video resources.

        This shared media scanner supports subject folders named by canonical ID,
        ``subjectN``, or ``subject_N`` and can enforce subject-ear grouping. It
        returns a media index keyed by subject and optional ear so value selection can
        be row-context aware.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate media files without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
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
            if "ear" in grouped_by:
                ear_files_by_name: dict[str, list[str]] = {}
                for ear in ears:
                    ear_folder = subject_folder / ear
                    ear_files_by_name[ear] = sorted(
                        (
                            str(file)
                            for file in ear_folder.rglob("*")
                            if file.is_file() and file.suffix.lower() in normalized_extensions
                        ),
                        key=lambda file: sort_key(Path(file)),
                    )
                if any(len(files) == 0 for files in ear_files_by_name.values()):
                    missing_subject_ids.append(subject_id)
                    continue
                grouped_paths[(subject_id, None, None)] = subject_files
                for ear, files in ear_files_by_name.items():
                    grouped_paths[(subject_id, None, ear)] = files
                subject_count = sum(len(files) for files in ear_files_by_name.values())
            else:
                grouped_paths[(subject_id, None, None)] = subject_files
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
        ears: tuple[str, ...],
    ) -> tuple[
        dict[tuple[str, int | None, str | None], list[str]],
        dict[str, int],
        tuple[str, ...],
    ]:
        """Scan image files by delegating to the shared media scanner.

        The method preserves image-specific resource naming while reusing the same
        subject and ear grouping behavior as videos. This keeps media resource
        policies shared without duplicating scan logic.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate image files without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
        return DatasetResourcesScanner.scan_media_paths(
            path,
            subject_ids,
            subject_numbers,
            extensions,
            grouped_by,
            ears,
            "Image",
        )

    @staticmethod
    def scan_video_paths(
        path: Path,
        subject_ids: tuple[str, ...],
        subject_numbers: dict[str, int],
        extensions: tuple[str, ...],
        grouped_by: tuple[str, ...],
        ears: tuple[str, ...],
    ) -> tuple[
        dict[tuple[str, int | None, str | None], list[str]],
        dict[str, int],
        tuple[str, ...],
    ]:
        """Scan video files by delegating to the shared media scanner.

        The method preserves video-specific resource naming while reusing the same
        subject and ear grouping behavior as images. This keeps media resource
        policies shared without duplicating scan logic.

        Parameters
        ----------
        *args, **kwargs Scanner arguments describing config, root, subject scope,
        extensions, grouping, and required state.

        Returns
        -------
        tuple Resource paths or indexes plus scanner summary data.

        Use Cases
        ---------
        - Locate video files without mutating dataset state.
        - Produce summary data for validation and split intersection.
        """
        return DatasetResourcesScanner.scan_media_paths(
            path,
            subject_ids,
            subject_numbers,
            extensions,
            grouped_by,
            ears,
            "Video",
        )


@dataclass(frozen=True)
class DatasetResourcesPlan:
    """Store resource scan output for dataset state assignment.

    Parameters
    ----------
    hrtf_paths, mesh_paths : dict
        Subject resource maps.
    image_path, video_path : Path or None
        Media root paths.
    image_index, video_index : dict
        Media indexes keyed by subject and optional ear.
    anthropometry_path, metadata_path : Path or None
        Table paths.
    anthropometry_rows, metadata_rows : dict
        Loaded table rows.
    excluded_subjects : tuple of str
        Combined config and user exclusions.
    subject_numbers : dict
        Subject numeric identifiers.
    resource_summary : dict
        Resource summary by resource name.

    Returns
    -------
    DatasetResourcesPlan Immutable plan consumed by ``DatasetBuilder``.
    """
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
    metadata_path: Path | None
    metadata_rows: dict[str, object]
    excluded_subjects: tuple[str, ...]
    subject_numbers: dict[str, int]
    resource_summary: dict[str, object]

class DatasetResources:
    """Build resource plans from dataset specs and configuration.

    This utility scans only the resource families required by selected specs, loads
    table resources, indexes media resources, and returns the resource plan consumed
    by ``DatasetBuilder``.
    """

    @staticmethod
    def _resolve_optional_path(
        path: str | Path | None,
        root: Path,
    ) -> Path | None:
        """Resolve an optional user-provided path against the dataset root.

        Specs may override configured resource locations with either absolute or
        relative paths. This helper normalizes that override once so scanner code can
        work with concrete paths.

        Parameters
        ----------
        path : str, Path, or None
            Optional path to resolve.
        root : Path
            Dataset root for relative paths.

        Returns
        -------
        Path or None Absolute path or ``None`` when no path was provided.
        """
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
        """Build the complete resource plan for a dataset instance.

        This method decides which resource families are required by the selected
        specs, applies subject exclusions, scans resource paths, loads tables, indexes
        media, validates results, and packages everything into an explicit plan. It is
        the only place resource discovery is assigned into dataset state.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset whose state contains config, root, specs, and selectors.
        exclude_subject_ids : str, int, sequence, or None
            Additional subject exclusions.

        Returns
        -------
        DatasetResourcesPlan Resource plan assigned into dataset state.

        Use Cases
        ---------
        - Intersect selected specs with available resources.
        - Load anthropometry and metadata tables.
        - Index image and video media paths.
        """
        state = dataset._state
        if state.config is None:
            raise ValueError("Dataset config is not initialized")
        config = state.config
        root = state.root
        config_excluded_subjects = DatasetSplitPlanner.map_subject_ids(
            tuple(config.excluded_subject_ids),
            tuple(config.subject_ids),
        )
        requested_excluded_subjects = DatasetSplitPlanner.map_subject_ids(
            exclude_subject_ids,
            tuple(config.subject_ids),
        )
        excluded_subjects = tuple(
            dict.fromkeys(config_excluded_subjects + requested_excluded_subjects)
        )
        excluded_subject_set = set(excluded_subjects)
        resource_subjects = tuple()
        if len(resource_subjects) == 0:
            sorted_subjects = DatasetSplitPlanner.sort_subject_ids(
                tuple(config.subject_ids)
            )
            resource_subjects = tuple(
                subject_id
                for subject_id in sorted_subjects
                if subject_id not in excluded_subject_set
            )
        subject_numbers = state.subject_numbers
        if len(subject_numbers) == 0:
            subject_numbers = DatasetSplitPlanner.build_subject_number_map(
                DatasetSplitPlanner.sort_subject_ids(tuple(config.subject_ids))
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
        metadata_path = None
        metadata_rows = {}

        validator = DatasetResourcesValidator(dataset)
        scanner = DatasetResourcesScanner()

        has_acoustic_specs = has_specs(state.specs, resource_name="hrtf")
        has_mesh_specs = has_specs(state.specs, resource_name="mesh")
        has_anthro_specs = has_specs(state.specs, resource_name="anthropometry")
        has_metadata_specs = has_specs(state.specs, resource_name="metadata")
        has_image_specs = has_specs(state.specs, resource_name="image")
        has_video_specs = has_specs(state.specs, resource_name="video")

        hrtf_paths, hrtf_summary = scanner.scan_hrtf_paths(
            config=config,
            root=root,
            dataset_hrtf_variant=state.dataset_hrtf_variant,
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
            mesh_specs = get_specs(state.specs, resource_name="mesh")
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
                dataset_mesh_variant=state.dataset_mesh_variant,
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
            anthropometry_specs = get_specs(state.specs, resource_name="anthropometry")
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
                anthropometry_rows = load_table(
                    dataset,
                    path=anthropometry_path,
                    exclude_row=first_anthro_spec.exclude_row,
                    exclude_column=first_anthro_spec.exclude_column,
                    accessed_by=first_anthro_spec.accessed_by,
                    subject_id=first_anthro_spec.subject_id,
                    extension=anthropometry_extension,
                    resource_name="Anthropometry",
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

        metadata_path, metadata_summary = scanner.scan_metadata_paths(
            config=config,
            root=root,
            requested_path=None,
            required=has_metadata_specs,
        )
        if metadata_summary is None:
            metadata_summary = resources_summary(
                checked=0,
                found=0,
                missing=0,
            )

        if has_metadata_specs:
            metadata_specs = get_specs(state.specs, resource_name="metadata")
            first_metadata_spec = metadata_specs[0]
            requested_metadata_path = None if first_metadata_spec.path is None else first_metadata_spec.path
            resolved_metadata_path = DatasetResources._resolve_optional_path(requested_metadata_path, root)
            if resolved_metadata_path is not None:
                metadata_path = resolved_metadata_path
            metadata_extensions = sanitize_extensions(
                resource_name="MetadataSpec",
                extensions=first_metadata_spec.extensions,
            )
            if len(metadata_extensions) == 0 and config.metadata is not None:
                metadata_extensions = sanitize_extensions(
                    resource_name="MetadataConfig",
                    extensions=config.metadata.extensions,
                )
            metadata_extension = (
                metadata_extensions[0] if len(metadata_extensions) > 0 else None
            )
            if metadata_path is None or not metadata_path.is_file():
                metadata_rows = {}
            else:
                metadata_rows = load_table(
                    dataset,
                    path=metadata_path,
                    exclude_row=first_metadata_spec.exclude_row,
                    exclude_column=first_metadata_spec.exclude_column,
                    accessed_by=first_metadata_spec.accessed_by,
                    subject_id=first_metadata_spec.subject_id,
                    extension=metadata_extension,
                    resource_name="Metadata",
                )
                metadata_summary = resources_summary(
                    checked=1,
                    found=1,
                    missing=0,
                )
        else:
            metadata_path = None
            metadata_rows = {}

        resource_summary["metadata"] = metadata_summary
        validator.validate_metadata_resources(
            metadata_path,
            metadata_rows,
        )

        if has_image_specs:
            image_specs = get_specs(state.specs, resource_name="image")
            first_image_spec = image_specs[0]
            if first_image_spec.path is None:
                raise ValueError("ImageSpec requires a path")
            requested_image_path = DatasetResources._resolve_optional_path(first_image_spec.path, root)
            image_path = requested_image_path
            grouped_by = ("subject",)
            ears = tuple(ear for ear, _ in state.selected_ears) if len(state.selected_ears) > 0 else ("left", "right")
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
                ears=ears,
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
            video_specs = get_specs(state.specs, resource_name="video")
            first_video_spec = video_specs[0]
            if first_video_spec.path is None:
                raise ValueError("VideoSpec requires a path")
            requested_video_path = DatasetResources._resolve_optional_path(first_video_spec.path, root)
            video_path = requested_video_path
            grouped_by = ("subject",)
            ears = tuple(ear for ear, _ in state.selected_ears) if len(state.selected_ears) > 0 else ("left", "right")
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
                ears=ears,
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
            metadata_path=metadata_path,
            metadata_rows=dict(metadata_rows),
            excluded_subjects=tuple(excluded_subjects),
            subject_numbers=dict(subject_numbers),
            resource_summary=dict(resource_summary),
        )
