from pathlib import Path
import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

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
    def __init__(self, dataset: "BaseDataset") -> None:
        """Validate scanned resources for the dataset construction pipeline.

        :class:`~hrtfpykit.datasets.resources.DatasetResourcesValidator` checks
        resource plans produced by
        :class:`~hrtfpykit.datasets.resources.DatasetResourcesScanner` before
        subject intersection and split planning continue. It verifies that HRTF
        files can be loaded consistently, that table resources satisfy metadata or
        anthropometry specs, and that media or mesh resources report missing
        subjects clearly enough for construction summaries.

        The validator is bound to a
        :class:`~hrtfpykit.datasets.base.BaseDataset` instance so it can read the
        active :class:`~hrtfpykit.datasets.state.DatasetState`, selected specs,
        resource summaries, and HRTF cache while validating resources found under
        the dataset root. It does not scan paths itself; it validates the scanner
        output that will later be assigned into dataset state.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset object whose current construction state provides specs,
            configuration, resource summaries, HRTF paths, and cache objects used
            during validation.
        """
        self._dataset = dataset

    def validate_hrtf_resources(
        self,
        hrtf_paths: dict[str, Path],
        hrtf_summary: dict[str, object],
    ) -> dict[str, Path]:
        """Validate HRTF paths and loadability before split planning.

        The scanner can find files by path pattern, but construction should reject
        corrupt, missing, or incompatible acoustic resources before samples are
        indexed. This validator loads each candidate through
        :func:`~hrtfpykit.datasets.load.load_hrtf`, reuses the dataset HRTF cache,
        enforces a consistent sample rate across loaded subjects, records load
        failures, and returns only paths that are still usable.

        If no selected spec requires HRTF resources, the input mapping is returned
        unchanged. When HRTF specs are active, missing subjects are warned about so
        users can see why the later subject intersection may remove them.

        Parameters
        ----------
        hrtf_paths : dict[str, Path]
            Subject-to-path HRTF resources produced by
            :meth:`~hrtfpykit.datasets.resources.DatasetResourcesScanner.scan_hrtf_paths`.
        hrtf_summary : dict
            Scanner summary containing checked, found, missing, and
            ``missing_subject_ids`` entries for HRTF resources.

        Returns
        -------
        dict[str, Path]
            Validated subject-to-path HRTF resources.

        Raises
        ------
        ValueError
            If any candidate HRTF cannot be loaded or if loaded subjects do not
            share the same sample rate.

        Warns
        -----
        UserWarning
            If subjects are missing a matching HRTF path or a path disappears
            between scanning and validation.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="hrtf"):
            return hrtf_paths
        if state.config is None:
            raise ValueError("Dataset config is not initialized")
        missing_hrtf_subject_ids = list(
            cast(
                tuple[str, ...],
                hrtf_summary.get(
                "missing_subject_ids",
                tuple(),
                ),
            ),
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
            except Exception as error:
                failed_hrtf_loads.append((subject_id, path, error))
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
            for subject_id, path, failed_error in failed_hrtf_loads[:10]:
                failure_lines.append(
                    f"{subject_id}: {path} ({type(failed_error).__name__}: {failed_error})"
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
        validator keeps that reporting behavior separate from path scanning and only
        runs when at least one selected spec requires mesh resources.

        Parameters
        ----------
        mesh_summary : dict
            Resource summary for mesh resources, including
            ``missing_subject_ids`` when some subjects do not have matching mesh
            files.

        Returns
        -------
        None
            Emits warnings for missing mesh subjects when mesh specs are active.

        Warns
        -----
        UserWarning
            If required mesh resources are missing for one or more subjects.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="mesh"):
            return
        missing_mesh_subject_ids = tuple(
            cast(
                tuple[str, ...],
                mesh_summary.get(
                    "missing_subject_ids",
                    tuple(),
                ),
            )
        )
        if len(missing_mesh_subject_ids) > 0:
            mesh_root = mesh_summary.get("root", state.root)
            warnings.warn(
                f"{state.name}: {len(missing_mesh_subject_ids)} subjects do not have a matching mesh file under "
                f"{mesh_root} and will be excluded when mesh is required "
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
        removed later during resource intersection. Uneven image counts are allowed
        but warned about because transforms or batching code may expect a stable
        number of files per subject.

        Parameters
        ----------
        summary : dict
            Resource summary for image resources, including
            ``missing_subject_ids``.
        image_path : Path or None
            Root folder used by the image scanner.
        image_counts : dict[str, int]
            Number of indexed image files per subject.

        Returns
        -------
        None
            Emits warnings for missing or uneven image resources.

        Warns
        -----
        UserWarning
            If required image resources are missing for subjects or if subjects do
            not all expose the same number of image files.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="image"):
            return
        missing_subject_ids = tuple(cast(tuple[str, ...], summary["missing_subject_ids"]))
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
        so media resource behavior uses the same validation path. Uneven video
        counts are allowed but warned about for the same reason as image counts:
        downstream transforms may assume a fixed number of files per subject.

        Parameters
        ----------
        summary : dict
            Resource summary for video resources, including
            ``missing_subject_ids``.
        video_path : Path or None
            Root folder used by the video scanner.
        video_counts : dict[str, int]
            Number of indexed video files per subject.

        Returns
        -------
        None
            Emits warnings for missing or uneven video resources.

        Warns
        -----
        UserWarning
            If required video resources are missing for subjects or if subjects do
            not all expose the same number of video files.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="video"):
            return
        missing_subject_ids = tuple(cast(tuple[str, ...], summary["missing_subject_ids"]))
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
        errors before subject intersection uses the resource. It does not validate
        the semantic content of individual anthropometry fields; value selection is
        handled later by dataset sample selectors.

        Parameters
        ----------
        anthropometry_path : Path or None
            Selected anthropometry table path.
        anthropometry_rows : dict[str, object]
            Loaded anthropometry table mapping produced by
            :func:`~hrtfpykit.datasets.load.load_table`.

        Returns
        -------
        None
            Raises when required anthropometry resources are invalid.

        Raises
        ------
        ValueError
            If anthropometry specs are active and no table path is selected, the
            selected path is not a file, or loaded anthropometry data is not a
            mapping.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="anthropometry"):
            return
        if anthropometry_path is None:
            raise ValueError(
                f"AnthropometrySpec was requested, but no anthropometry file was selected for "
                f"{state.name}. Pass AnthropometrySpec(path=...) or use a dataset "
                "configuration that declares anthropometry."
            )
        if not anthropometry_path.is_file():
            selected_source = "AnthropometrySpec(path=...)"
            if state.config is not None and state.config.anthropometry is not None:
                configured_path = (
                    state.root / state.config.anthropometry.path
                ).expanduser()
                if anthropometry_path == configured_path:
                    selected_source = (
                        f"{state.config.__class__.__name__}.anthropometry.path="
                        f"{state.config.anthropometry.path!r}"
                    )
            download_hint = ""
            if (
                state.config is not None
                and state.config.download_servers is not None
                and any(
                    "anthropometry" in download_server.available_resources
                    for download_server in state.config.download_servers.values()
                )
            ):
                download_hint = (
                    f" {state.name} can download this resource; use download=True "
                    "with download_resources='anthropometry' or "
                    "download_resources='all'. download=True only downloads "
                    "the resources named in download_resources."
                )
            raise ValueError(
                "AnthropometrySpec requires an anthropometry file, but the "
                f"selected path is missing: {anthropometry_path}. "
                f"Source: {selected_source}. Dataset root: {state.root}."
                f"{download_hint} For a custom file, pass "
                "AnthropometrySpec(path=...)."
            )
        if not isinstance(anthropometry_rows, dict):
            raise ValueError(
                f"{state.name} anthropometry data is invalid: expected a mapping, "
                f"got {type(anthropometry_rows)!r}"
            )

    def validate_metadata_resources(
        self,
        metadata_path: Path | None,
        metadata_rows: dict[str, object],
    ) -> None:
        """Validate selected metadata resources after table loading.

        Metadata specs require a real table path and loaded mapping data. This method
        keeps metadata validation separate from anthropometry while preserving the
        same table-resource contract. When the selected path is missing, the error
        includes the spec path, configured path, root, and download hint because
        metadata files are often optional official downloads.

        Parameters
        ----------
        metadata_path : Path or None
            Selected metadata table path.
        metadata_rows : dict[str, object]
            Loaded metadata table mapping produced by
            :func:`~hrtfpykit.datasets.load.load_table`.

        Returns
        -------
        None
            Raises when required metadata resources are invalid.

        Raises
        ------
        ValueError
            If metadata specs are active and no table path is selected, the
            selected path is not a file, or loaded metadata data is not a mapping.
        """
        state = self._dataset._state
        if not has_specs(state.specs, resource_name="metadata"):
            return
        if metadata_path is None:
            raise ValueError(
                f"MetadataSpec was requested, but no metadata file was selected for "
                f"{state.name}. Pass MetadataSpec(path=...) or use a dataset "
                "configuration that declares metadata."
            )
        if not metadata_path.is_file():
            selected_source = "MetadataSpec(path=...)"
            if state.config is not None and state.config.metadata is not None:
                configured_path = (state.root / state.config.metadata.path).expanduser()
                if metadata_path == configured_path:
                    selected_source = (
                        f"{state.config.__class__.__name__}.metadata.path="
                        f"{state.config.metadata.path!r}"
                    )
            download_hint = ""
            if (
                state.config is not None
                and state.config.download_servers is not None
                and any(
                    "metadata" in download_server.available_resources
                    for download_server in state.config.download_servers.values()
                )
            ):
                download_hint = (
                    f" {state.name} can download this resource; use download=True "
                    "with download_resources='metadata' or "
                    "download_resources='all'. download=True only downloads "
                    "the resources named in download_resources."
                )
            raise ValueError(
                "MetadataSpec requires a metadata file, but the selected path "
                f"is missing: {metadata_path}. Source: {selected_source}. "
                f"Dataset root: {state.root}.{download_hint} For a custom "
                "file, pass MetadataSpec(path=...)."
            )
        if not isinstance(metadata_rows, dict):
            raise ValueError(
                f"{state.name} metadata data is invalid: expected a mapping, "
                f"got {type(metadata_rows)!r}"
            )


class DatasetResourcesScanner:
    """Scan dataset roots for resources requested by selected specs.

    :class:`~hrtfpykit.datasets.resources.DatasetResourcesScanner` contains the
    stateless path-discovery operations used by
    :class:`~hrtfpykit.datasets.resources.DatasetResources`. It resolves HRTF,
    mesh, metadata, anthropometry, image, and video resources from
    :class:`~hrtfpykit.datasets.config.DatasetConfig` declarations, spec path
    overrides, subject path templates, allowed extensions, and media grouping
    settings.

    Scanner methods do not mutate dataset state and generally do not load resource
    contents. Table loading is handled by
    :func:`~hrtfpykit.datasets.load.load_table`, while HRTF loadability is checked
    by :class:`~hrtfpykit.datasets.resources.DatasetResourcesValidator`.
    """

    @staticmethod
    def resolve_resource_patterns(
        root: Path,
        patterns: tuple[str | Path, ...],
        resource_name: str,
    ) -> Path:
        fallback = (root / patterns[0]).expanduser()
        for pattern in patterns:
            pattern_path = Path(pattern).expanduser()
            pattern_text = str(pattern_path)
            has_glob = any(character in pattern_text for character in "*?[")
            if has_glob:
                if pattern_path.is_absolute():
                    matches = sorted(
                        path
                        for path in pattern_path.parent.glob(pattern_path.name)
                        if path.is_file()
                    )
                else:
                    matches = sorted(
                        path
                        for path in root.glob(pattern_text)
                        if path.is_file()
                    )
                if len(matches) > 1:
                    raise ValueError(
                        f"{resource_name} local path pattern {pattern!r} matched "
                        f"{len(matches)} files under {root}. Make the pattern more specific."
                    )
                if len(matches) == 1:
                    return matches[0]
                continue

            candidate = pattern_path if pattern_path.is_absolute() else root / pattern_path
            candidate = candidate.expanduser()
            if candidate.is_file():
                return candidate
        return fallback

    @staticmethod
    def resolve_resource_path(
        root: Path,
        relative_path: str | Path,
        subject_id: str | None = None,
    ) -> Path:
        return DatasetResourcesScanner.resolve_resource_patterns(
            root,
            (relative_path,),
            "Resource",
        )

    @staticmethod
    def scan_anthropometry_paths(
        config: DatasetConfig,
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
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration that may define an official anthropometry table.
        root : Path
            Dataset root used to resolve configured relative paths.
        requested_path : Path or None
            Explicit spec path override. When None, the configured
            anthropometry path is used if available.
        required : bool
            Whether selected specs require anthropometry resources.

        Returns
        -------
        tuple[Path or None, dict]
            Selected anthropometry path and a summary containing ``path``,
            ``found``, ``subjects``, ``rows``, and available
            ``extensions`` when the configuration declares them.

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
            anthropometry_path = DatasetResourcesScanner.resolve_resource_patterns(
                root,
                (config.anthropometry.path,) + config.anthropometry.local_path_patterns,
                "Anthropometry",
            )
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
        config: DatasetConfig,
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
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration that may define an official metadata table.
        root : Path
            Dataset root used to resolve configured relative paths.
        requested_path : Path or None
            Explicit spec path override. When None, the configured metadata
            path is used if available.
        required : bool
            Whether selected specs require metadata resources.

        Returns
        -------
        tuple[Path or None, dict]
            Selected metadata path and a summary containing ``path``,
            ``found``, ``subjects``, ``rows``, and available
            ``extensions`` when the configuration declares them.

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
            metadata_path = DatasetResourcesScanner.resolve_resource_patterns(
                root,
                (config.metadata.path,) + config.metadata.local_path_patterns,
                "Metadata",
            )
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
        config: DatasetConfig,
        root: Path,
        dataset_hrtf_variant: str | dict[str, object] | None,
        excluded_subject_ids: set[str],
        required: bool,
    ) -> tuple[dict[str, Path], dict[str, object] | None]:
        """Scan for HRTF files required by selected acoustic specs.

        The scanner resolves the dataset HRTF path rule for each subject. A rule
        can be one format template shared by all subjects or a mapping from
        subject ID to relative path. Template rules are formatted with subject
        IDs, subject numbers, type, sample rate, and version selectors. The
        method returns the paths that exist plus a summary of checked, found, and
        missing subjects for validation and error reporting. It does not load
        SOFA files; loadability and sample rate consistency are handled by
        :meth:`~hrtfpykit.datasets.resources.DatasetResourcesValidator.validate_hrtf_resources`.

        Parameters
        ----------
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration containing HRTF path templates and subject IDs.
        root : Path
            Dataset root used to resolve subject HRTF paths.
        dataset_hrtf_variant : str, dict, or None
            Selected HRTF resource variant. A string selects the HRTF type; a
            mapping can provide ``type``, ``sample_rate``, and ``version``
            entries. When None and the dataset exposes exactly one HRTF type,
            that type is selected automatically.
        excluded_subject_ids : set of str
            Canonical subject IDs removed before scanning.
        required : bool
            Whether selected specs require HRTF resources.

        Returns
        -------
        tuple[dict[str, Path], dict or None]
            Subject-to-path HRTF map and a summary dictionary. If HRTF resources
            are not configured or not required, the path map is empty and the
            summary is None.

        Raises
        ------
        ValueError
            If HRTF resources are required but no HRTF type can be selected.

        """
        hrtf_paths: dict[str, Path] = {}
        if config.hrtf is None or not required:
            return hrtf_paths, None
        if isinstance(dataset_hrtf_variant, dict):
            hrtf_type: str | None = str(dataset_hrtf_variant["type"]).strip().lower()
            hrtf_sample_rate = dataset_hrtf_variant.get("sample_rate")
            hrtf_version = dataset_hrtf_variant.get("version")
        else:
            hrtf_type = None if dataset_hrtf_variant is None else str(dataset_hrtf_variant).strip().lower()
            hrtf_sample_rate = None
            hrtf_version = None
        if hrtf_type is None:
            if len(config.hrtf.types) == 1:
                hrtf_type = next(iter(config.hrtf.types))
            else:
                raise ValueError(f"{config.name} requires dataset_hrtf_variant for HRTF resources")
        hrtf_type_config = config.hrtf.types[hrtf_type]
        sample_rate_label = None
        if hrtf_sample_rate is not None:
            sample_rate_label = str(hrtf_sample_rate)
            if hrtf_type_config.sample_rate_labels is not None:
                sample_rate_label = hrtf_type_config.sample_rate_labels.get(
                    cast(int | str, hrtf_sample_rate),
                    sample_rate_label,
                )
        hrtf_subject_ids = (
            tuple(config.subject_ids)
            if config.hrtf.subject_ids is None
            else tuple(config.hrtf.subject_ids)
        )
        subject_numbers = DatasetSplitPlanner.build_subject_number_map(
            tuple(DatasetSplitPlanner.sort_subject_ids(tuple(config.subject_ids)))
        )
        checked_hrtf_subject_ids = tuple(
            subject_id
            for subject_id in hrtf_subject_ids
            if subject_id not in excluded_subject_ids
        )
        hrtf_versions: tuple[object | None, ...]
        if hrtf_version is not None:
            hrtf_versions = (hrtf_version,)
        elif len(hrtf_type_config.versions) > 0:
            hrtf_versions = tuple(hrtf_type_config.versions)
        else:
            hrtf_versions = (None,)
        for subject_id in checked_hrtf_subject_ids:
            for selected_hrtf_version in hrtf_versions:
                version_label = None
                if hrtf_type_config.version_labels is not None and selected_hrtf_version is not None:
                    version_label = hrtf_type_config.version_labels.get(
                        cast(str, selected_hrtf_version),
                        str(selected_hrtf_version),
                    )
                path_pattern = hrtf_type_config.path_pattern
                if isinstance(path_pattern, dict):
                    selected_path_pattern = path_pattern.get(subject_id)
                    if selected_path_pattern is None:
                        continue
                else:
                    selected_path_pattern = path_pattern
                format_values = {
                    "subject_id": subject_id,
                    "subject_number": subject_numbers[subject_id],
                    "type": hrtf_type,
                    "hrtf_type": hrtf_type,
                    "sample_rate": hrtf_sample_rate,
                    "hrtf_sample_rate": hrtf_sample_rate,
                    "sample_rate_label": sample_rate_label,
                    "version": selected_hrtf_version,
                    "hrtf_version": selected_hrtf_version,
                    "version_label": version_label,
                    "hrtf_version_label": version_label,
                    "variant": hrtf_type,
                }
                relative_path = selected_path_pattern.format(**format_values)
                format_values["filename"] = Path(relative_path).name
                local_patterns = tuple(
                    local_path_pattern.format(**format_values)
                    for local_path_pattern in hrtf_type_config.local_path_patterns
                )
                candidate = DatasetResourcesScanner.resolve_resource_patterns(
                    root,
                    (relative_path,) + local_patterns,
                    "HRTF",
                )
                if candidate.is_file():
                    hrtf_paths[subject_id] = candidate
                    break
        missing_hrtf_subject_ids = tuple(
            subject_id
            for subject_id in checked_hrtf_subject_ids
            if subject_id not in hrtf_paths
        )
        return hrtf_paths, {
            "pattern": hrtf_type_config.path_pattern
            if isinstance(hrtf_type_config.path_pattern, str)
            else f"{len(hrtf_type_config.path_pattern)} subject paths",
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
        config: DatasetConfig,
        root: Path,
        dataset_mesh_variant: str | dict[str, object] | None,
        excluded_subject_ids: set[str],
        required: bool,
        extensions: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Path], dict[str, object] | None]:
        """Scan for mesh files required by selected mesh specs.

        The scanner resolves the selected mesh variant, applies extension
        candidates, resolves one path rule for each subject, and records which
        subjects have usable mesh files. The path rule can be one format template
        shared by all subjects or a mapping from subject ID to relative path. It
        does not load mesh geometry; sample selection returns paths or transformed
        path values later.

        Parameters
        ----------
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration containing mesh path templates and subject IDs.
        root : Path
            Root used to resolve mesh paths. This may be the dataset root or a
            spec-level mesh path override.
        dataset_mesh_variant : str, dict, or None
            Selected mesh resource variant. A string selects the mesh type; a
            mapping can provide ``type`` and ``version`` entries.
        excluded_subject_ids : set of str
            Canonical subject IDs removed before scanning.
        required : bool
            Whether selected specs require mesh resources.
        extensions : tuple of str or None, default=None
            Candidate mesh extensions. If the selected mesh path has no suffix,
            each extension is appended; otherwise the suffix is replaced by each
            extension.

        Returns
        -------
        tuple[dict[str, Path], dict or None]
            Subject-to-path mesh map and a summary dictionary. If mesh resources
            are not configured or not required, the path map is empty and the
            summary is None.

        Raises
        ------
        ValueError
            If mesh resources are required but no mesh type can be selected.

        """
        mesh_paths: dict[str, Path] = {}
        if config.mesh is None or not required:
            return mesh_paths, None
        if isinstance(dataset_mesh_variant, dict):
            mesh_type: str | None = str(dataset_mesh_variant["type"])
            mesh_version = dataset_mesh_variant.get("version")
        else:
            mesh_type = dataset_mesh_variant
            mesh_version = None
        if mesh_type is None and "default" in config.mesh.types:
            mesh_type = "default"
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
            tuple(DatasetSplitPlanner.sort_subject_ids(tuple(config.subject_ids)))
        )
        for subject_id in checked_mesh_subject_ids:
            version_label = None
            if mesh_type_config.version_labels is not None and mesh_version is not None:
                version_label = mesh_type_config.version_labels.get(
                    cast(str, mesh_version),
                    str(mesh_version),
                )
            path_pattern = mesh_type_config.path_pattern
            if isinstance(path_pattern, dict):
                selected_path_pattern = path_pattern.get(subject_id)
                if selected_path_pattern is None:
                    continue
            else:
                selected_path_pattern = path_pattern
            format_values = {
                "subject_id": subject_id,
                "subject_number": subject_numbers[subject_id],
                "type": mesh_type,
                "mesh_type": mesh_type,
                "version": mesh_version,
                "mesh_version": mesh_version,
                "version_label": version_label,
                "mesh_version_label": version_label,
            }
            relative_path = selected_path_pattern.format(**format_values)
            format_values["filename"] = Path(relative_path).name
            formatted_patterns = (relative_path,) + tuple(
                local_path_pattern.format(**format_values)
                for local_path_pattern in mesh_type_config.local_path_patterns
            )
            candidate_paths: list[Path] = []
            for formatted_pattern in formatted_patterns:
                pattern_path = Path(formatted_pattern)
                if len(normalized_extensions) == 0 or any(
                    character in str(pattern_path) for character in "*?["
                ):
                    candidate_paths.append(pattern_path)
                elif pattern_path.suffix == "":
                    candidate_paths.extend(
                        pattern_path.with_name(f"{pattern_path.name}{extension}")
                        for extension in normalized_extensions
                    )
                else:
                    base_path = pattern_path.with_suffix("")
                    candidate_paths.extend(
                        base_path.with_suffix(extension)
                        for extension in normalized_extensions
                    )
            for candidate in dict.fromkeys(candidate_paths):
                resolved_candidate = DatasetResourcesScanner.resolve_resource_patterns(
                    root,
                    (candidate,),
                    "Mesh",
                )
                if resolved_candidate.is_file():
                    mesh_paths[subject_id] = resolved_candidate
                    break
        missing_mesh_subject_ids = tuple(
            subject_id
            for subject_id in checked_mesh_subject_ids
            if subject_id not in mesh_paths
        )
        return mesh_paths, {
            "pattern": mesh_type_config.path_pattern
            if isinstance(mesh_type_config.path_pattern, str)
            else f"{len(mesh_type_config.path_pattern)} subject paths",
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
        returns a media index keyed by subject and optional ear so value selection
        can be row-context aware. Files are discovered recursively below each
        matched subject folder and sorted with numeric filename stems in natural
        order.

        When ear grouping is active, the scanner records both subject-level media
        files and ear-specific files. A subject is treated as missing if any
        requested ear folder has no matching files, because sample rows that carry
        ear context must be able to resolve every selected ear consistently.

        Parameters
        ----------
        path : Path
            Media root containing one folder per subject.
        subject_ids : tuple of str
            Canonical subject IDs to scan.
        subject_numbers : dict[str, int]
            Numeric subject identifiers used to recognize ``subjectN`` and
            ``subject_N`` folder names.
        extensions : tuple of str
            Accepted file extensions, including the leading dot.
        grouped_by : tuple of str
            Resource grouping. When it contains ``ear``, files are also scanned
            in ear-named subfolders.
        ears : tuple of str
            Ear labels expected when ear grouping is active.
        resource_name : str
            Resource label used in validation errors.

        Returns
        -------
        tuple[dict, dict, tuple]
            Media index keyed by (subject_id, None, ear_or_none), per-subject
            file counts, and subject IDs missing required media folders or ear
            files.

        Raises
        ------
        ValueError
            If path does not exist or if multiple subject folders match the
            same canonical subject.

        """
        grouped_paths: dict[tuple[str, int | None, str | None], list[str]] = {}
        if not path.is_dir():
            spec_name = f"{resource_name}Spec"
            raise ValueError(
                f"{spec_name} selected root does not exist or is not a directory: "
                f"{path}. Expected one folder per dataset subject, named with "
                "the subject ID, subjectN, or subject_N. If this is a custom "
                f"{resource_name.lower()} resource, pass "
                f"{spec_name}(path=...) to an existing folder."
            )
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
                    f"{resource_name} root {path} has multiple folders for subject "
                    f"{subject_id!r}: "
                    + ", ".join(str(path_item.name) for path_item in matches)
                    + ". Keep one folder using one accepted name: "
                    + ", ".join(candidate_names)
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
        path : Path
            Image root containing one folder per subject.
        subject_ids : tuple of str
            Canonical subject IDs to scan.
        subject_numbers : dict[str, int]
            Numeric subject identifiers used for subject-folder aliases.
        extensions : tuple of str
            Accepted image extensions.
        grouped_by : tuple of str
            Image grouping, optionally including ``ear``.
        ears : tuple of str
            Ear labels expected when ear grouping is active.

        Returns
        -------
        tuple[dict, dict, tuple]
            Image index, per-subject image counts, and missing subject IDs.

        Raises
        ------
        ValueError
            If the image root does not exist or if multiple folders match the same
            canonical subject. These errors are raised by
            :meth:`~hrtfpykit.datasets.resources.DatasetResourcesScanner.scan_media_paths`.

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
        path : Path
            Video root containing one folder per subject.
        subject_ids : tuple of str
            Canonical subject IDs to scan.
        subject_numbers : dict[str, int]
            Numeric subject identifiers used for subject-folder aliases.
        extensions : tuple of str
            Accepted video extensions.
        grouped_by : tuple of str
            Video grouping, optionally including ``ear``.
        ears : tuple of str
            Ear labels expected when ear grouping is active.

        Returns
        -------
        tuple[dict, dict, tuple]
            Video index, per-subject video counts, and missing subject IDs.

        Raises
        ------
        ValueError
            If the video root does not exist or if multiple folders match the same
            canonical subject. These errors are raised by
            :meth:`~hrtfpykit.datasets.resources.DatasetResourcesScanner.scan_media_paths`.

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
    """Store scanned resources before they are assigned into dataset state.

    :class:`~hrtfpykit.datasets.resources.DatasetResourcesPlan` is the immutable
    handoff object returned by
    :meth:`~hrtfpykit.datasets.resources.DatasetResources.build`. The dataset
    builder copies these fields into
    :class:`~hrtfpykit.datasets.state.DatasetState` after scanning, loading table
    resources, validating required resources, and resolving subject exclusions.

    Notes
    -----
    The plan stores paths and already-loaded table mappings, not fully materialized
    media, mesh, or HRTF data. HRTF files are loaded through the dataset cache when
    subjects or samples are requested.

    Attributes
    ----------
    hrtf_paths, mesh_paths : dict[str, Path]
        Validated subject resource maps for acoustic and mesh files.
    image_path, video_path : Path or None
        Media root paths selected by image and video specs.
    image_index, video_index : dict
        Media indexes keyed by subject, optional position placeholder, and optional
        ear label.
    image_counts, video_counts : dict[str, int]
        Number of indexed media files per subject.
    anthropometry_path, metadata_path : Path or None
        Selected table paths for anthropometry and metadata resources.
    anthropometry_rows, metadata_rows : dict[str, object]
        Loaded table mappings used by sample value selectors.
    excluded_subjects : tuple of str
        Combined configuration-level and user-requested subject exclusions.
    subject_numbers : dict[str, int]
        Numeric subject identifiers derived from canonical subject IDs.
    resource_summary : dict[str, object]
        Resource summary entries by resource family.
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

    :class:`~hrtfpykit.datasets.resources.DatasetResources` coordinates resource
    discovery for :class:`~hrtfpykit.datasets.build.DatasetBuilder`. It examines
    the selected specs stored in
    :class:`~hrtfpykit.datasets.state.DatasetState`, scans only the required
    resource families, applies configuration-level and user-level subject
    exclusions, loads table resources, indexes media resources, validates scanned
    resources, and returns a
    :class:`~hrtfpykit.datasets.resources.DatasetResourcesPlan`.

    The class is intentionally stateless. All durable results are returned in the
    resource plan and later copied into dataset state by the builder.
    """

    @staticmethod
    def _resolve_optional_path(
        path: str | Path | None,
        root: Path,
    ) -> Path | None:
        """Resolve an optional user-provided path against the dataset root.

        Specs may override configured resource locations with either absolute or
        relative paths. This helper normalizes that override once so scanner code can
        work with concrete paths. Absolute paths are expanded and returned as-is;
        relative paths are resolved below the dataset root.

        Parameters
        ----------
        path : str, Path, or None
            Optional path to resolve.
        root : Path
            Dataset root for relative paths.

        Returns
        -------
        Path or None
            Resolved path, or None when no path was provided.
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
        subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
    ) -> DatasetResourcesPlan:
        """Build the complete resource plan for a dataset instance.

        This method decides which resource families are required by the selected
        specs, applies subject exclusions, scans resource paths, loads tables, indexes
        media, validates results, and packages everything into an explicit plan. It is
        the resource-discovery boundary used during dataset construction.

        Resource families are activated by the selected specs: acoustic specs scan
        HRTF paths, mesh specs scan mesh files, table specs resolve and load
        metadata or anthropometry tables, and media specs index subject-grouped
        image or video files. Resource summaries are produced for every family so
        dataset summaries can report both active and inactive resources.

        The build step never downloads official resources. It only evaluates local
        files that already exist below the dataset root or below paths supplied by
        specs. Download planning and transfer are handled separately by
        :class:`~hrtfpykit.datasets.download.BaseDownload`.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset whose state contains configuration, root, selected specs,
            resource variants, selected ears, subject numbers, and cache.
        subject_ids : str, int, sequence, or None
            Optional subject references used as the initial resource scanning scope.
            None uses every configured subject.
        exclude_subject_ids : str, int, sequence, or None
            Additional subject references excluded from the selected scanning scope
            before subject intersection. Integer values are one-based subject
            positions.

        Returns
        -------
        DatasetResourcesPlan
            Resource plan assigned into dataset state by
            :class:`~hrtfpykit.datasets.build.DatasetBuilder`.

        Raises
        ------
        ValueError
            If dataset configuration is not initialized, selected variants are
            incomplete, required table or media paths are missing, table loading
            fails validation, HRTF resources cannot be loaded consistently, or
            requested image/video specs do not provide a path.
        """
        state = dataset._state
        if state.config is None:
            raise ValueError("Dataset config is not initialized")
        config = state.config
        root = state.root
        requested_subjects = None if subject_ids is None else DatasetSplitPlanner.map_subject_ids(
            subject_ids,
            tuple(config.subject_ids),
        )
        excluded_subjects = DatasetSplitPlanner.map_subject_ids(
            exclude_subject_ids,
            tuple(config.subject_ids),
        )
        requested_subject_set = None if requested_subjects is None else set(requested_subjects)
        excluded_subject_set = set(excluded_subjects)
        sorted_subjects = DatasetSplitPlanner.sort_subject_ids(
            tuple(config.subject_ids)
        )
        resource_subjects = tuple(
            subject_id
            for subject_id in sorted_subjects
            if (requested_subject_set is None or subject_id in requested_subject_set)
            and subject_id not in excluded_subject_set
        )
        scanner_excluded_subject_set = {
            subject_id
            for subject_id in config.subject_ids
            if subject_id not in set(resource_subjects)
        }
        subject_numbers = state.subject_numbers
        if len(subject_numbers) == 0:
            subject_numbers = DatasetSplitPlanner.build_subject_number_map(
                tuple(DatasetSplitPlanner.sort_subject_ids(tuple(config.subject_ids)))
            )
        resource_summary: dict[str, object] = {}
        mesh_paths: dict[str, Path] = {}
        image_path: Path | None = None
        video_path: Path | None = None
        image_index: dict[tuple[str, int | None, str | None], list[str]] = {}
        video_index: dict[tuple[str, int | None, str | None], list[str]] = {}
        image_counts: dict[str, int] = {}
        video_counts: dict[str, int] = {}
        anthropometry_path: Path | None = None
        anthropometry_rows: dict[str, object] = {}
        metadata_path: Path | None = None
        metadata_rows: dict[str, object] = {}

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
            excluded_subject_ids=scanner_excluded_subject_set,
            required=has_acoustic_specs,
        )
        if hrtf_summary is None:
            hrtf_summary = cast(dict[str, object], resources_summary(
                checked=0,
                found=0,
                missing=0,
                missing_subject_ids=tuple(),
            ))
        resource_summary["hrtf"] = hrtf_summary
        hrtf_paths = validator.validate_hrtf_resources(hrtf_paths, hrtf_summary)
        hrtf_summary["subjects_checked"] = int(cast(Any, hrtf_summary.get("checked", 0)))
        hrtf_summary["subjects_available"] = len(hrtf_paths)
        hrtf_summary["subjects_missing"] = int(cast(Any, hrtf_summary.get("missing", 0)))
        hrtf_summary["files"] = len(hrtf_paths)

        if has_mesh_specs:
            mesh_root_path = root
            mesh_specs = cast(tuple[Any, ...], get_specs(state.specs, resource_name="mesh"))
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
                excluded_subject_ids=scanner_excluded_subject_set,
                required=has_mesh_specs,
                extensions=mesh_extensions,
            )
            if mesh_summary is None:
                mesh_summary = cast(dict[str, object], resources_summary())
            mesh_summary = cast(dict[str, object], resources_summary(
                checked=int(cast(Any, mesh_summary.get("checked", 0))),
                found=int(cast(Any, mesh_summary.get("found", 0))),
                missing=int(cast(Any, mesh_summary.get("missing", 0))),
                missing_subject_ids=tuple(cast(tuple[str, ...], mesh_summary.get("missing_subject_ids", tuple()))),
            ))
            mesh_summary["subjects_checked"] = int(cast(Any, mesh_summary.get("checked", 0)))
            mesh_summary["subjects_available"] = len(mesh_paths)
            mesh_summary["subjects_missing"] = int(cast(Any, mesh_summary.get("missing", 0)))
            mesh_summary["files"] = len(mesh_paths)
            mesh_summary["root"] = str(mesh_root_path)
            validator.validate_mesh_resources(mesh_summary)
            resource_summary["mesh"] = mesh_summary
        else:
            resource_summary["mesh"] = cast(dict[str, object], resources_summary())

        anthropometry_path, anthropometry_summary = scanner.scan_anthropometry_paths(
            config=config,
            root=root,
            requested_path=None,
            required=has_anthro_specs,
        )
        if has_anthro_specs:
            anthropometry_specs = cast(tuple[Any, ...], get_specs(state.specs, resource_name="anthropometry"))
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
                anthropometry_rows = cast(dict[str, object], load_table(
                    dataset,
                    path=anthropometry_path,
                    exclude_row=first_anthro_spec.exclude_row,
                    exclude_column=first_anthro_spec.exclude_column,
                    accessed_by=first_anthro_spec.accessed_by,
                    subject_id=first_anthro_spec.subject_id,
                    extension=anthropometry_extension,
                    resource_name="Anthropometry",
                ))
                anthropometry_summary = cast(dict[str, object], resources_summary(
                    checked=1,
                    found=1,
                    missing=0,
                ))
            available_subject_ids = tuple(
                subject_id
                for subject_id in resource_subjects
                if subject_id in anthropometry_rows
            )
            missing_subject_ids = tuple(
                subject_id
                for subject_id in resource_subjects
                if subject_id not in anthropometry_rows
            )
            anthropometry_summary["subjects_checked"] = len(resource_subjects)
            anthropometry_summary["subjects_available"] = len(available_subject_ids)
            anthropometry_summary["subjects_missing"] = len(missing_subject_ids)
            anthropometry_summary["files"] = (
                1 if anthropometry_path is not None and anthropometry_path.is_file() else 0
            )
            anthropometry_summary["missing_subject_ids"] = missing_subject_ids
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
        if has_metadata_specs:
            metadata_specs = cast(tuple[Any, ...], get_specs(state.specs, resource_name="metadata"))
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
                metadata_rows = cast(dict[str, object], load_table(
                    dataset,
                    path=metadata_path,
                    exclude_row=first_metadata_spec.exclude_row,
                    exclude_column=first_metadata_spec.exclude_column,
                    accessed_by=first_metadata_spec.accessed_by,
                    subject_id=first_metadata_spec.subject_id,
                    extension=metadata_extension,
                    resource_name="Metadata",
                ))
                metadata_summary = cast(dict[str, object], resources_summary(
                    checked=1,
                    found=1,
                    missing=0,
                ))
            available_subject_ids = tuple(
                subject_id
                for subject_id in resource_subjects
                if subject_id in metadata_rows
            )
            missing_subject_ids = tuple(
                subject_id
                for subject_id in resource_subjects
                if subject_id not in metadata_rows
            )
            metadata_summary["subjects_checked"] = len(resource_subjects)
            metadata_summary["subjects_available"] = len(available_subject_ids)
            metadata_summary["subjects_missing"] = len(missing_subject_ids)
            metadata_summary["files"] = (
                1 if metadata_path is not None and metadata_path.is_file() else 0
            )
            metadata_summary["missing_subject_ids"] = missing_subject_ids
        else:
            metadata_path = None
            metadata_rows = {}

        resource_summary["metadata"] = metadata_summary
        validator.validate_metadata_resources(
            metadata_path,
            metadata_rows,
        )

        if has_image_specs:
            image_specs = cast(tuple[Any, ...], get_specs(state.specs, resource_name="image"))
            first_image_spec = image_specs[0]
            requested_image_path = DatasetResources._resolve_optional_path(
                first_image_spec.path,
                root,
            )
            if requested_image_path is None and config.image is not None and config.image.path is not None:
                requested_image_path = (root / config.image.path).expanduser()
            if requested_image_path is None:
                raise ValueError(
                    f"ImageSpec was requested for {state.name}, but no image "
                    f"root was selected. Dataset root: {root}. Pass "
                    "ImageSpec(path=...) to a folder containing one folder per "
                    "subject, or use a dataset configuration that declares an "
                    "image path."
                )
            image_path = requested_image_path
            image_grouped_by: tuple[str, ...] = ("subject",)
            ears = tuple(ear for ear, _ in state.selected_ears) if len(state.selected_ears) > 0 else ("left", "right")
            if any("ear" in sanitize_grouped_by(spec.grouped_by) for spec in image_specs):
                image_grouped_by = ("subject", "ear")
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
                grouped_by=image_grouped_by,
                ears=ears,
            )
            image_summary = cast(dict[str, object], resources_summary(
                checked=len(resource_subjects),
                found=len(image_counts),
                missing=len(missing_subject_ids),
                missing_subject_ids=tuple(missing_subject_ids),
            ))
            image_summary["subjects_checked"] = len(resource_subjects)
            image_summary["subjects_available"] = len(image_counts)
            image_summary["subjects_missing"] = len(missing_subject_ids)
            image_summary["files"] = sum(image_counts.values())
            resource_summary["image"] = image_summary
            validator.validate_image_resources(
                image_summary,
                image_path,
                image_counts,
            )
        else:
            resource_summary["image"] = cast(dict[str, object], resources_summary())

        if has_video_specs:
            video_specs = cast(tuple[Any, ...], get_specs(state.specs, resource_name="video"))
            first_video_spec = video_specs[0]
            requested_video_path = DatasetResources._resolve_optional_path(
                first_video_spec.path,
                root,
            )
            if requested_video_path is None and config.video is not None and config.video.path is not None:
                requested_video_path = (root / config.video.path).expanduser()
            if requested_video_path is None:
                raise ValueError(
                    f"VideoSpec was requested for {state.name}, but no video "
                    f"root was selected. Dataset root: {root}. Pass "
                    "VideoSpec(path=...) to a folder containing one folder per "
                    "subject, or use a dataset configuration that declares a "
                    "video path."
                )
            video_path = requested_video_path
            video_grouped_by: tuple[str, ...] = ("subject",)
            ears = tuple(ear for ear, _ in state.selected_ears) if len(state.selected_ears) > 0 else ("left", "right")
            if any("ear" in sanitize_grouped_by(spec.grouped_by) for spec in video_specs):
                video_grouped_by = ("subject", "ear")
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
                grouped_by=video_grouped_by,
                ears=ears,
            )
            video_summary = cast(dict[str, object], resources_summary(
                checked=len(resource_subjects),
                found=len(video_counts),
                missing=len(missing_subject_ids),
                missing_subject_ids=tuple(missing_subject_ids),
            ))
            video_summary["subjects_checked"] = len(resource_subjects)
            video_summary["subjects_available"] = len(video_counts)
            video_summary["subjects_missing"] = len(missing_subject_ids)
            video_summary["files"] = sum(video_counts.values())
            resource_summary["video"] = video_summary
            validator.validate_video_resources(
                video_summary,
                video_path,
                video_counts,
            )
        else:
            resource_summary["video"] = cast(dict[str, object], resources_summary())
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
