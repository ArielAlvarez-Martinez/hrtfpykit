from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Mapping
import gzip
import json
import re
import shutil
import hashlib
import tarfile
import warnings
from typing import Any, cast
import urllib.error
import urllib.request
import zipfile
from urllib.parse import quote, urlparse, urlunparse

from .config import DatasetConfig, DownloadServerConfig
from .resources import DatasetResourcesScanner
from .summary import download_summary
from .split import DatasetSplitPlanner

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class MissingChecksumError(ValueError):
    """Raised when an official download checksum is missing."""

    pass


class UnsupportedDownloadServerError(ValueError):
    """Raised when a dataset is asked to use an unknown download server."""

    pass


class BaseDownload(ABC):
    def __init__(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        excluded_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        verify_checksum: bool = True,
        download_server: str | None = None,
    ) -> None:
        """Manage secure official downloads for one dataset configuration.

        :class:`~hrtfpykit.datasets.download.BaseDownload` is the shared
        downloader used by dataset classes before construction. It converts
        dataset download metadata from
        :class:`~hrtfpykit.datasets.config.DatasetConfig` into file-level jobs,
        composes HTTPS URLs, resolves local destinations below the dataset root,
        verifies existing files, downloads missing or invalid files, checks
        archive integrity, and validates accepted files with SHA-256 when checksum
        verification is enabled.

        The downloader separates download selection from dataset construction. Dataset
        classes pass explicit download variants and resource groups to this object,
        while :class:`~hrtfpykit.datasets.base.BaseDataset` later decides which
        already-local resources are scanned for samples. Download subject
        exclusions are normalized once during initialization so every
        subject-specific download family uses the same filtered subject list.

        Parameters
        ----------
        config : DatasetConfig or type[DatasetConfig]
            Dataset configuration instance or built-in configuration class with
            official download metadata.
        root : str or Path
            Dataset root where downloaded resources are stored. The path is
            expanded and resolved but not created until
            :meth:`~hrtfpykit.datasets.download.BaseDownload.validate_download_root`
            or :meth:`~hrtfpykit.datasets.download.BaseDownload.download` is
            called.
        subject_ids : str, int, sequence, or None, default=None
            Optional subject identifiers or one-based subject numbers used as the
            initial subject-specific download scope. None uses every configured
            subject supported by the selected download server.
        excluded_subject_ids : str, int, sequence, or None, default=None
            Subject identifiers or one-based subject numbers to exclude from the
            selected subject-specific download scope. These exclusions are combined
            with download_exclude_subject_ids declared by the selected download
            server.
        verify_checksum : bool, default=True
            Whether downloaded and existing files are verified against the official
            SHA-256 checksums declared by the dataset configuration. The recommended
            behavior is to keep this enabled. Set it to False only when you
            intentionally want to skip checksum verification; file existence,
            non-empty checks, and archive integrity checks still run.

        Attributes
        ----------
        root : Path
            Resolved dataset root used to compose local download destinations.
        requested_subject_ids : tuple of str or None
            Normalized user-requested download subject scope. None means no
            user-level inclusion filter was requested.
        excluded_subject_ids : tuple of str
            Normalized union of selected-server and user-requested download
            subject exclusions.
        verify_checksum_enabled : bool
            Whether official checksum verification is applied to planned downloads.

        Raises
        ------
        ValueError
            If root points to an existing file or if a requested subject
            exclusion cannot be mapped by
            :class:`~hrtfpykit.datasets.split.DatasetSplitPlanner`.
        """
        if isinstance(config, type):
            config = cast(DatasetConfig, cast(Any, config)())

        self.config: DatasetConfig = config
        self.download_server, self.download_config = self.resolve_download_config(config, download_server)
        self.root: Path = self.sanitize_root(Path(root))
        self.verify_checksum_enabled = verify_checksum
        server_excluded_subject_ids = DatasetSplitPlanner.map_subject_ids(
            tuple(self.download_config.download_exclude_subject_ids),
            tuple(config.subject_ids),
        )
        requested_subject_ids = None if subject_ids is None else DatasetSplitPlanner.map_subject_ids(
            subject_ids,
            tuple(config.subject_ids),
        )
        requested_excluded_subject_ids = DatasetSplitPlanner.map_subject_ids(
            excluded_subject_ids,
            tuple(config.subject_ids),
        )
        self.requested_subject_ids = None if requested_subject_ids is None else tuple(requested_subject_ids)
        self.requested_excluded_subject_ids = tuple(requested_excluded_subject_ids)
        self.excluded_subject_ids = tuple(
            dict.fromkeys(server_excluded_subject_ids + requested_excluded_subject_ids)
        )
        self.last_download_jobs: list[dict[str, object]] = []
        self.last_downloaded_count = 0
        self.last_verified_count = 0
        self.last_failures: list[str] = []

    def validate_supported_download_filters(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None,
        download_hrtf_variant: str | Mapping[str, object] | None,
        download_mesh_variant: str | Mapping[str, object] | None,
    ) -> None:
        """Validate request axes supported by the selected download server.

        Some servers expose individual files and can filter by resource,
        subject, HRTF variant, or mesh variant. Other servers expose complete
        archives and cannot honor those selectors. This method checks the
        selected :class:`DownloadServerConfig.supports_filter` declaration before
        planning starts, so unsupported selectors fail with an explicit message
        instead of being silently ignored.

        Parameters
        ----------
        download_resources : str, sequence of str, or None
            Requested resource groups. None means the server default, normally
            all available resources.
        download_hrtf_variant : str, mapping, or None
            Requested HRTF selector. Unsupported when the selected server cannot
            filter HRTF variants.
        download_mesh_variant : str, mapping, or None
            Requested mesh selector. Unsupported when the selected server cannot
            filter mesh variants.

        Returns
        -------
        None
            Raises when the selected server cannot honor one of the requested
            filters.

        Raises
        ------
        ValueError
            If resource, subject, HRTF variant, or mesh variant filtering is
            requested for a server that does not support it.
        """
        supports_filter = self.download_config.supports_filter or {}
        server_name = self.download_server or self.__class__.__name__
        requested_resources = ("all",) if download_resources is None else (download_resources,) if isinstance(download_resources, str) else tuple(download_resources)
        normalized_resources = tuple(str(resource).strip().lower() for resource in requested_resources)
        if supports_filter.get("resource", True) is False and "all" not in normalized_resources:
            raise ValueError(
                f"{self.config.name} download_server {server_name!r} does not support download_resources filtering. "
                "Set download_resources=None or choose a download server that supports resource filtering."
            )
        if supports_filter.get("subject", True) is False and (
            self.requested_subject_ids is not None or len(self.requested_excluded_subject_ids) > 0
        ):
            raise ValueError(
                f"{self.config.name} download_server {server_name!r} does not support download_subject_ids or download_exclude_subject_ids because it cannot filter subjects. "
                "Set download_exclude_subject_ids=None and download_subject_ids=None, or choose a download server that supports subject filtering."
            )
        if supports_filter.get("hrtf_variant", True) is False and download_hrtf_variant not in (None, "all"):
            raise ValueError(
                f"{self.config.name} download_server {server_name!r} does not support download_hrtf_variant filtering. "
                f"Got download_hrtf_variant={download_hrtf_variant!r}. Set download_hrtf_variant=None or choose a download server that supports HRTF variant filtering."
            )
        if supports_filter.get("mesh_variant", True) is False and download_mesh_variant not in (None, "all"):
            raise ValueError(
                f"{self.config.name} download_server {server_name!r} does not support download_mesh_variant filtering. "
                f"Got download_mesh_variant={download_mesh_variant!r}. Set download_mesh_variant=None or choose a download server that supports mesh variant filtering."
            )

    @staticmethod
    def resolve_download_config(
        config: DatasetConfig,
        download_server: str | None,
    ) -> tuple[str | None, DownloadServerConfig]:
        """Select the download server config used by one downloader instance.

        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration declaring one or more official download
            sources.
        download_server : str or None
            Requested server name. Dataset classes resolve their own defaults before constructing a downloader.

        Returns
        -------
        tuple
            Selected server name and its :class:`DownloadServerConfig`.
        """

        if config.download_servers is None or len(config.download_servers) == 0:
            raise ValueError(f"{config.name} does not define downloadable resources")

        selected_server = download_server
        if selected_server is None:
            if len(config.download_servers) == 1:
                selected_server = next(iter(config.download_servers))
            else:
                raise ValueError(
                    f"{config.name} defines multiple download servers; pass download_server. "
                    f"Available servers: {tuple(config.download_servers)}"
                )
        if selected_server not in config.download_servers:
            raise UnsupportedDownloadServerError(
                f"{config.name} download_server accepts {tuple(config.download_servers)}; got {selected_server!r}"
            )
        return selected_server, config.download_servers[selected_server]

    @staticmethod
    def sanitize_root(root: Path) -> Path:
        """Normalize a dataset root before download paths are composed.

        The downloader accepts user-provided roots but must guarantee that later
        writes target a directory tree, not an existing regular file. This helper
        expands ~ and resolves the path early so later checks in
        :meth:`~hrtfpykit.datasets.download.BaseDownload.compose_download_path`
        compare absolute paths against a stable root.

        Parameters
        ----------
        root : Path
            Candidate root path.

        Returns
        -------
        Path
            Resolved root path.

        Raises
        ------
        ValueError
            If the normalized path already exists and is not a directory.
        """
        normalized = Path(root).expanduser()
        if normalized.exists() and not normalized.is_dir():
            raise ValueError(f"Dataset root must be a directory, got file: {normalized}")
        return normalized.resolve()

    @staticmethod
    def validate_download_url(url: str) -> str:
        """Validate an official download URL before opening a connection.

        Dataset downloads should not silently accept insecure or malformed URLs. This
        helper enforces the HTTPS scheme and a non-empty host before file transfer
        starts, and is used by both
        :meth:`~hrtfpykit.datasets.download.BaseDownload.build_download_url` and
        :meth:`~hrtfpykit.datasets.download.BaseDownload.download_file`.

        Parameters
        ----------
        url : str
            URL to validate.

        Returns
        -------
        str
            Original URL when valid.

        Raises
        ------
        ValueError
            If url is not HTTPS or does not contain a host.
        """
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"Only https downloads are allowed, got: {url}")
        if parsed.netloc.strip() == "":
            raise ValueError(f"Download URL is missing a host: {url}")
        path = quote(parsed.path, safe="/%")
        query = quote(parsed.query, safe="=&;%")
        return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, parsed.fragment))

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Compute the SHA-256 digest for a local resource file.

        The downloader reads the file in chunks so large SOFA, mesh, or archive
        resources can be verified without loading the entire file into memory. The
        result is compared with official checksums by
        :meth:`~hrtfpykit.datasets.download.BaseDownload.verify_checksum`.

        Parameters
        ----------
        path : Path
            File path to hash.

        Returns
        -------
        str
            SHA-256 hex digest.

        Raises
        ------
        OSError
            If path cannot be opened or read.
        """
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if len(chunk) == 0:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def sanitize_checksum(checksum: str) -> str:
        """Normalize and validate a SHA-256 checksum value.

        Checksums can be written with or without a sha256: prefix, but the
        downloader stores and compares plain lowercase hex digests. This helper
        rejects malformed values before they reach
        :meth:`~hrtfpykit.datasets.download.BaseDownload.verify_checksum`.

        Parameters
        ----------
        checksum : str
            Checksum string, optionally prefixed by sha256:.

        Returns
        -------
        str
            Lowercase SHA-256 hex digest.

        Raises
        ------
        ValueError
            If checksum is not a 64-character SHA-256 hexadecimal digest.
        """
        value = str(checksum).strip().lower()
        if value.startswith("sha256:"):
            value = value.split(":", 1)[1]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Checksums must be SHA-256 hex digests")
        return value

    def sanitize_download_resources(
        self,
        requested: str | tuple[str, ...] | list[str] | None,
    ) -> tuple[str, ...]:
        """Normalize requested official resource groups.

        This helper validates the resource groups requested by the user against the
        dataset download configuration and expands the pseudo-resource ``all``.
        It keeps dataset constructors small while making unsupported download
        requests fail before URL construction or network access.

        Parameters
        ----------
        requested : str, sequence of str, or None
            Requested resource names. None and ``all`` select every official
            resource. Accepted names are declared by the configuration's
            :class:`~hrtfpykit.datasets.config.DownloadServerConfig`.

        Returns
        -------
        tuple of str
            Lowercase resource names selected for planning.

        Raises
        ------
        ValueError
            If the dataset has no official download configuration or if any
            requested resource is unsupported.
        """
        if requested is None:
            requested_values: tuple[str, ...] = ("all",)
        elif isinstance(requested, str):
            requested_values = (requested,)
        else:
            requested_values = tuple(requested)
        available = tuple(self.download_config.available_resources)
        normalized = tuple(str(value).strip().lower() for value in requested_values)
        if "all" in normalized:
            return tuple(value for value in available if value != "all")
        invalid = [value for value in normalized if value not in available]
        if invalid:
            server_name = self.download_server or self.__class__.__name__
            raise ValueError(
                f"Unsupported download_resources for {self.config.name} download_server {server_name!r}: {invalid}. "
                f"Expected one of {available + ('all',)}"
            )
        return normalized

    def sanitize_download_values(
        self,
        requested: str | int | tuple[str | int, ...] | list[str | int] | None,
        available: tuple[str | int, ...],
        default: str | int | None,
        label: str,
    ) -> tuple[str, ...]:
        """Normalize a variant selector used by download planning.

        HRTF and mesh downloads can be selected by type, version, and sample rate.
        This helper handles None, scalar values, sequences, all, and
        validation against available values through one selector rule set. Returned
        values are strings so path-template expansion and dictionary lookup can use
        one normalized representation.

        Parameters
        ----------
        requested : str, int, sequence, or None
            Requested selector values. None selects default when it is
            provided, otherwise all available values.
        available : tuple of str or int
            Values supported by the selected resource type.
        default : str, int, or None
            Optional fallback value used when requested is None or when a
            selector axis is absent but the dataset still exposes a default value.
        label : str
            Error label for validation messages.

        Returns
        -------
        tuple of str
            Normalized selector values.

        Raises
        ------
        ValueError
            If the selector axis is unsupported or if any requested value is not
            available.
        """
        if len(available) == 0:
            if requested in (None, "all"):
                return tuple()
            if default is None:
                raise ValueError(f"{label} is not supported by this dataset")
            return (str(default),)
        if requested is None:
            if default is None:
                return tuple(str(value) for value in available)
            return (str(default),)
        if isinstance(requested, (str, int)):
            requested_values: tuple[str | int, ...] = (requested,)
        else:
            requested_values = tuple(requested)
        normalized_available = {str(value).strip().lower(): value for value in available}
        normalized_requested = tuple(str(value).strip().lower() for value in requested_values)
        if "all" in normalized_requested:
            return tuple(str(value) for value in available)
        invalid = [value for value in normalized_requested if value not in normalized_available]
        if invalid:
            raise ValueError(f"Unsupported {label}: {invalid}. Expected one of {available + ('all',)}")
        return tuple(str(normalized_available[value]) for value in normalized_requested)

    def validate_download_variants(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None,
        download_hrtf_variant: str | Mapping[str, object] | None,
        download_mesh_variant: str | Mapping[str, object] | None,
    ) -> None:
        """Validate HRTF and mesh variant selectors for requested resources.

        The validator checks selector keys and values against the dataset config
        before a server-specific planner creates jobs. It is shared by all
        downloaders so unsupported types, sample rates, versions, and mesh
        variants produce consistent error messages across SOFAcoustics, Imperial,
        TU Berlin, and SONICOM ecosystem downloads.

        Parameters
        ----------
        download_resources : str, sequence of str, or None
            Requested resource groups. Variant selectors are validated only for
            resources included by this request.
        download_hrtf_variant : str, mapping, or None
            HRTF selector. A string selects the HRTF type. A mapping may contain
            ``type``, ``sample_rate``, and ``version``.
        download_mesh_variant : str, mapping, or None
            Mesh selector. A string selects the mesh type. A mapping may contain
            ``type`` and ``version``.

        Returns
        -------
        None
            Raises when any requested variant axis is unsupported.

        Raises
        ------
        ValueError
            If a requested resource family is unavailable, a selector mapping
            contains unsupported keys, or a selector value is outside the
            configured dataset variants.
        """
        resources = self.sanitize_download_resources(download_resources)
        if "hrtf" in resources:
            if self.config.hrtf is None:
                raise ValueError(f"{self.config.name} does not provide official hrtf files")
            if isinstance(download_hrtf_variant, Mapping):
                unknown_keys = set(download_hrtf_variant) - {"type", "sample_rate", "version"}
                if unknown_keys:
                    raise ValueError(
                        f"Unsupported download_hrtf_variant keys {tuple(sorted(unknown_keys))}. "
                        "Expected keys are ('type', 'sample_rate', 'version')"
                    )
                hrtf_type = download_hrtf_variant.get("type")
                hrtf_sample_rate = download_hrtf_variant.get("sample_rate")
                hrtf_version = download_hrtf_variant.get("version")
            else:
                hrtf_type = download_hrtf_variant
                hrtf_sample_rate = None
                hrtf_version = None
            hrtf_types = self.sanitize_download_values(
                cast(Any, hrtf_type),
                tuple(self.config.hrtf.types),
                None,
                "download_hrtf_variant['type']",
            )
            requested_hrtf_types = (
                (hrtf_type,)
                if isinstance(hrtf_type, (str, int)) or hrtf_type is None
                else tuple(cast(Any, hrtf_type))
            )
            hrtf_type_all = any(str(value).strip().lower() == "all" for value in requested_hrtf_types)
            sample_rate_error: ValueError | None = None
            version_error: ValueError | None = None
            sample_rate_valid = hrtf_sample_rate is None
            version_valid = hrtf_version is None
            for selected_hrtf_type in hrtf_types:
                hrtf_type_config = self.config.hrtf.types[selected_hrtf_type]
                try:
                    self.sanitize_download_values(
                        cast(Any, hrtf_sample_rate),
                        hrtf_type_config.sample_rates,
                        None,
                        "download_hrtf_variant['sample_rate']",
                    )
                except ValueError as exc:
                    if sample_rate_error is None:
                        sample_rate_error = exc
                else:
                    sample_rate_valid = True
                try:
                    self.sanitize_download_values(
                        cast(Any, hrtf_version),
                        hrtf_type_config.versions,
                        None,
                        "download_hrtf_variant['version']",
                    )
                except ValueError as exc:
                    if version_error is None:
                        version_error = exc
                else:
                    version_valid = True
            if not sample_rate_valid and sample_rate_error is not None:
                raise sample_rate_error
            if not version_valid and version_error is not None:
                raise version_error
            if not hrtf_type_all and len(hrtf_types) == 0:
                raise ValueError(f"Unsupported download_hrtf_variant['type']: {download_hrtf_variant!r}")

        if "mesh" in resources:
            if self.config.mesh is None:
                raise ValueError(f"{self.config.name} does not provide official mesh data")
            if isinstance(download_mesh_variant, Mapping):
                unknown_keys = set(download_mesh_variant) - {"type", "version"}
                if unknown_keys:
                    raise ValueError(
                        f"Unsupported download_mesh_variant keys {tuple(sorted(unknown_keys))}. "
                        "Expected keys are ('type', 'version')"
                    )
                mesh_type = download_mesh_variant.get("type")
                mesh_version = download_mesh_variant.get("version")
            else:
                mesh_type = download_mesh_variant
                mesh_version = None
            mesh_types = self.sanitize_download_values(
                cast(Any, mesh_type),
                tuple(self.config.mesh.types),
                None,
                "download_mesh_variant['type']",
            )
            requested_mesh_types = (
                (mesh_type,)
                if isinstance(mesh_type, (str, int)) or mesh_type is None
                else tuple(cast(Any, mesh_type))
            )
            mesh_type_all = any(str(value).strip().lower() == "all" for value in requested_mesh_types)
            version_error = None
            version_valid = mesh_version is None
            for selected_mesh_type in mesh_types:
                mesh_type_config = self.config.mesh.types[selected_mesh_type]
                try:
                    self.sanitize_download_values(
                        cast(Any, mesh_version),
                        mesh_type_config.versions,
                        None,
                        "download_mesh_variant['version']",
                    )
                except ValueError as exc:
                    if version_error is None:
                        version_error = exc
                else:
                    version_valid = True
            if not version_valid and version_error is not None:
                raise version_error
            if not mesh_type_all and len(mesh_types) == 0:
                raise ValueError(f"Unsupported download_mesh_variant['type']: {download_mesh_variant!r}")

    def validate_download_root(self) -> Path:
        """Create and validate the root used for downloaded resources.

        This method is called immediately before downloads so path checks and
        directory creation happen at the boundary where files may be written. It
        updates the stored root with the resolved directory used by subsequent
        download jobs.

        Returns
        -------
        Path
            Resolved root directory.

        Raises
        ------
        ValueError
            If the configured root exists and is not a directory.
        OSError
            If the directory cannot be created.
        """
        self.root = self.sanitize_root(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def compose_download_path(self, filename: str) -> Path:
        """Resolve one planned destination below the dataset root.

        Download configs provide relative resource paths, and this method turns them
        into absolute local paths while rejecting absolute paths or .. escapes. It
        is the downloader boundary that prevents resource path templates from writing
        outside the dataset root.

        Parameters
        ----------
        filename : str
            Relative resource path.

        Returns
        -------
        Path
            Absolute destination path under the root.

        Raises
        ------
        ValueError
            If filename is absolute, contains a parent-directory escape, or
            resolves outside the dataset root.
        """
        candidate = Path(filename)
        if candidate.is_absolute():
            raise ValueError(f"Download filename must be relative: {filename}")
        if any(part == ".." for part in candidate.parts):
            raise ValueError(f"Download filename must not escape root: {filename}")
        destination = (self.root / candidate).resolve()
        try:
            destination.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Resolved download path escapes root: {destination}") from exc
        return destination

    def build_download_url(self, resource: str, filename: str) -> str:
        """Compose the official HTTPS URL for one relative resource path.

        The method combines the selected resource base URL with one planned
        relative path, then validates the resulting URL through the shared HTTPS
        rules. Dataset-level
        :class:`~hrtfpykit.datasets.config.DownloadServerConfig` metadata may provide
        resource-specific base URLs for datasets whose HRTF, mesh, anthropometry,
        or metadata resources are hosted on different servers. When no
        resource-specific base URL is configured, the dataset default base URL is
        used. The relative resource path is used for the download URL and local
        destination. Checksum lookup uses the explicit checksum key selected by
        the download planner.

        URL unsafe characters in the relative path, such as spaces in official
        filenames, are percent-encoded only in the returned URL. Local
        destination paths keep the original relative path.

        Parameters
        ----------
        resource : str
            Resource group name for the planned download, such as ``hrtf``,
            ``mesh``, ``anthropometry``, or ``metadata``.
        filename : str
            Relative resource path.

        Returns
        -------
        str
            Validated HTTPS download URL.

        Raises
        ------
        ValueError
            If the dataset has no official download base URL or if the composed
            URL is not valid HTTPS.
        """
        resource_name = str(resource).strip().lower()
        resource_base_urls = self.download_config.resource_base_urls or {}
        resource_base_url = resource_base_urls.get(
            resource_name,
            self.download_config.base_url,
        )
        validated_base_url = self.validate_download_url(str(resource_base_url).rstrip("/"))
        return f"{validated_base_url}/{quote(filename, safe='/')}"

    def get_checksum(
        self,
        resource: str,
        checksum_key: str,
        hrtf_type: str | None = None,
        hrtf_version: str | None = None,
        hrtf_sample_rate: str | int | None = None,
        mesh_type: str | None = None,
        mesh_version: str | None = None,
    ) -> str | None:
        """Return the official checksum for one planned download job.

        Checksum maps can be flat or hierarchical by HRTF type/version/sample-rate or
        mesh type/version. This method hides that structure from plan builders and
        returns the checksum relevant to one exact checksum key and variant
        context. Missing checksum metadata is treated as a hard error whenever
        checksum verification is enabled.

        Parameters
        ----------
        resource : str
            Resource group name, such as ``hrtf``, ``mesh``,
            ``anthropometry``, or ``metadata``.
        checksum_key : str
            Exact key used inside the selected checksum map. This is independent
            from local scanner path patterns and may differ from the planned
            download destination path.
        hrtf_type, hrtf_version, hrtf_sample_rate : str, int, or None
            HRTF selector context used when the resource checksum map is grouped
            by acoustic type, processing version, or sample rate.
        mesh_type, mesh_version : str or None
            Mesh selector context used when the resource checksum map is grouped
            by geometry type or version.

        Returns
        -------
        str
            SHA-256 checksum for the planned resource path.

        Raises
        ------
        ValueError
            If download checksums are missing, if the checksum map has an
            unexpected shape, if required variant context is absent, or if the
            selected checksum key has no string checksum.
        """
        if self.download_config.checksums is None:
            raise ValueError(
                f"{self.config.name} cannot download {resource!r} resources because no checksum map is configured"
            )
        checksums = self.download_config.checksums
        resource_checksums = checksums.get(resource)
        if resource_checksums is None:
            raise ValueError(
                f"{self.config.name} cannot download {resource!r} resources because no checksums are configured for that resource"
            )
        if isinstance(resource_checksums, dict) and len(resource_checksums) == 0:
            raise ValueError(
                f"{self.config.name} cannot download {resource!r} resources because the checksum map for that resource is missing or empty"
            )
        checksum: object | None = None
        if resource == "hrtf":
            if not isinstance(resource_checksums, dict):
                raise ValueError("HRTF checksums must be grouped by filename or type")
            direct_checksum = resource_checksums.get(checksum_key)
            if isinstance(direct_checksum, str):
                checksum = direct_checksum
            elif hrtf_type is None:
                raise ValueError("HRTF checksum lookup requires a type when checksums are grouped by type")
            else:
                type_checksums = resource_checksums.get(hrtf_type)
                if type_checksums is None:
                    if any(isinstance(value, str) for value in resource_checksums.values()):
                        raise MissingChecksumError(
                            f"{self.config.name} is missing a checksum for {resource!r} resource {checksum_key!r}"
                        )
                    raise ValueError(
                        f"{self.config.name} is missing HRTF checksums for type={hrtf_type!r}"
                    )
                if isinstance(type_checksums, dict) and hrtf_version is not None and hrtf_version in type_checksums:
                    version_checksums = type_checksums.get(hrtf_version)
                    if not isinstance(version_checksums, dict):
                        raise ValueError("HRTF version checksums must be a filename dictionary")
                    sample_rate_checksums = None
                    if hrtf_sample_rate is not None:
                        sample_rate_checksums = version_checksums.get(hrtf_sample_rate)
                        if sample_rate_checksums is None:
                            sample_rate_checksums = version_checksums.get(str(hrtf_sample_rate))
                    if sample_rate_checksums is not None:
                        if not isinstance(sample_rate_checksums, dict):
                            raise ValueError("HRTF sample-rate checksums must be a filename dictionary")
                        checksum = sample_rate_checksums.get(checksum_key)
                    else:
                        checksum = version_checksums.get(checksum_key)
                else:
                    if not isinstance(type_checksums, dict):
                        raise ValueError("HRTF type checksums must be a filename dictionary")
                    checksum = type_checksums.get(checksum_key)
        elif resource == "mesh":
            if not isinstance(resource_checksums, dict):
                raise ValueError("Mesh checksums must be grouped by type or filename")
            type_checksums = None if mesh_type is None else resource_checksums.get(mesh_type)
            if isinstance(type_checksums, dict):
                version_checksums = None
                if mesh_version is not None:
                    version_checksums = type_checksums.get(mesh_version)
                if version_checksums is not None:
                    if not isinstance(version_checksums, dict):
                        raise ValueError("Mesh version checksums must be a filename dictionary")
                    checksum = version_checksums.get(checksum_key)
                else:
                    checksum = type_checksums.get(checksum_key)
            else:
                checksum = resource_checksums.get(checksum_key)
        elif isinstance(resource_checksums, dict):
            checksum = resource_checksums.get(checksum_key)
        elif isinstance(resource_checksums, str):
            checksum = resource_checksums
        else:
            raise ValueError(f"{resource} checksums must be a string or filename dictionary")
        if checksum is None:
            raise MissingChecksumError(
                f"{self.config.name} is missing a checksum for {resource!r} resource {checksum_key!r}"
            )
        if not isinstance(checksum, str):
            raise ValueError(f"{resource} checksum for {checksum_key} must be a string")
        return checksum

    def get_included_subject_ids(self, subject_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Remove excluded subjects from one resource subject list.

        The downloader combines config-level and user-level exclusions once during
        initialization. This helper applies that normalized exclusion set to each
        subject-specific resource family before path templates are expanded.

        Parameters
        ----------
        subject_ids : tuple of str
            Candidate subject IDs.

        Returns
        -------
        tuple of str
            Subject IDs not excluded from downloads.
        """
        requested_subject_ids_set = None if self.requested_subject_ids is None else set(self.requested_subject_ids)
        excluded_subject_ids_set = set(self.excluded_subject_ids)
        return tuple(
            subject_id
            for subject_id in subject_ids
            if (requested_subject_ids_set is None or subject_id in requested_subject_ids_set)
            and subject_id not in excluded_subject_ids_set
        )

    def download_file(
        self,
        url: str,
        destination: Path,
        checksum: str | None,
    ) -> str:
        """Download one planned file or verify an existing destination.

        This method validates the URL, verifies an existing destination when one is
        present, downloads to a temporary .part file when needed, checks byte
        count when the server reports a content length, verifies file integrity,
        and atomically moves the verified temporary file into place. Invalid
        existing destinations are removed and replaced by a fresh download.

        Parameters
        ----------
        url : str
            HTTPS source URL.
        destination : Path
            Local destination path.
        checksum : str or None
            Expected SHA-256 checksum. When None, checksum verification is skipped
            but file existence, non-empty checks, and archive integrity checks still
            run.

        Returns
        -------
        str
            ``downloaded`` when fetched or ``verified`` when existing file passed
        validation.

        Raises
        ------
        ValueError
            If the URL is invalid, the transfer fails, the response is incomplete,
            the downloaded file is empty or corrupt, or checksum verification fails
            when a checksum is provided.
        """
        validated_url = self.validate_download_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            try:
                self.verify_downloaded_file(destination, checksum)
                return "verified"
            except ValueError:
                destination.unlink()

        temporary_path = destination.with_suffix(destination.suffix + ".part")
        if temporary_path.exists():
            temporary_path.unlink()
        bytes_written = 0
        progress_bar = None
        try:
            with urllib.request.urlopen(validated_url, timeout=60) as response, temporary_path.open("wb") as file:
                expected_length_header = response.headers.get("Content-Length")
                expected_length = (
                    None if expected_length_header is None else int(expected_length_header)
                )
                if tqdm is not None:
                    progress_bar = tqdm(
                        total=expected_length,
                        desc=destination.name,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    )
                while True:
                    chunk = response.read(1024 * 1024)
                    if len(chunk) == 0:
                        break
                    file.write(chunk)
                    bytes_written += len(chunk)
                    if progress_bar is not None:
                        progress_bar.update(len(chunk))
            if expected_length is not None and bytes_written != expected_length:
                raise ValueError(
                    f"Incomplete download for {destination.name}: expected {expected_length} bytes, got {bytes_written}"
                )
            self.verify_downloaded_file(temporary_path, checksum)
            temporary_path.replace(destination)
            return "downloaded"
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            if isinstance(exc, urllib.error.HTTPError):
                reason = f"HTTP {exc.code} {exc.reason}"
            elif isinstance(exc, urllib.error.URLError):
                reason = f"URL error: {exc.reason}"
            else:
                reason = str(exc)
            raise ValueError(f"Could not securely download {validated_url} ({reason})") from exc
        finally:
            if progress_bar is not None:
                progress_bar.close()

    @abstractmethod
    def build_download_plan(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None = None,
        download_hrtf_variant: str | Mapping[str, object] | None = None,
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Build the concrete file jobs required by one download request.

        Subclasses implement this method because each server describes its
        downloadable files differently. Direct file servers expand configured
        resource path templates, archive servers create archive jobs, and catalog
        servers read remote listings before creating jobs. The returned jobs are
        then executed by :meth:`download`, which owns the shared verification,
        transfer, accounting, and summary workflow.

        Parameters
        ----------
        download_resources : str, sequence of str, or None, default=None
            Resource groups requested by the caller. None selects every official
            resource.
        download_hrtf_variant : str, mapping, or None, default=None
            HRTF variant selector passed through from the dataset constructor. None
            selects every available HRTF variant.
        download_mesh_variant : str, mapping, or None, default=None
            Mesh variant selector passed through from the dataset constructor.

        Returns
        -------
        list of dict
            Download jobs. Each job must include at least ``resource``,
            ``relative_path``, ``url``, ``destination``, and ``checksum``.
        """

    def download(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None = None,
        download_hrtf_variant: str | Mapping[str, object] | None = None,
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> tuple[bool, str]:
        """Execute secure downloads for the selected official resources.

        This method validates the root, builds the plan, executes each job, tracks
        downloaded and verified files, and returns a human-readable summary. Partial
        failures stay in the summary so dataset construction can continue with the
        resources that were downloaded or already verified. If every planned job fails
        and no usable file is produced, the download stops before dataset construction.

        Download execution follows the explicit download plan. It does not fall back
        to specs, dataset HRTF variants, or dataset mesh variants when a resource or
        variant is missing from the download arguments. Existing files are accepted
        only after the same integrity checks used for newly downloaded files. Checksum
        checks are applied when checksum verification is enabled.

        Parameters
        ----------
        download_resources : str, sequence of str, or None, default=None
            Resource groups to download or verify. None and ``all`` expand to every
            official resource declared by the dataset configuration.
        download_hrtf_variant : str, dict, or None, default=None
            HRTF variant requested for download. None selects every available
            variant. This value is passed directly to
            :meth:`~hrtfpykit.datasets.download.BaseDownload.build_download_plan`.
        download_mesh_variant : str, dict, or None, default=None
            Mesh variant requested for download. This value is passed directly to
            :meth:`~hrtfpykit.datasets.download.BaseDownload.build_download_plan`.

        Returns
        -------
        tuple[bool, str]
            True and a summary when at least one file was downloaded;
            False and a summary when all planned files already existed or no jobs
            were needed. The summary text is generated by
            :func:`~hrtfpykit.datasets.summary.download_summary`.

        Warns
        -----
        UserWarning
            If the request produces no planned files.

        Raises
        ------
        ValueError
            If planning fails, if the root cannot be used, or if every planned
            download job fails without producing or verifying any usable file.
        """

        self.validate_supported_download_filters(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        self.validate_download_variants(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        self.validate_download_root()
        download_jobs = self.build_download_plan(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        if len(download_jobs) == 0:
            self.last_download_jobs = download_jobs
            self.last_downloaded_count = 0
            self.last_verified_count = 0
            self.last_failures = []
            summary = download_summary(
                self.config,
                self.root,
                download_jobs,
                downloaded_count=0,
                verified_count=0,
                failures=[],
            )
            requested_resources = self.sanitize_download_resources(download_resources)
            request_parts = [
                f"Requested resources={download_resources!r}",
            ]
            if "hrtf" in requested_resources:
                request_parts.append(f"download_hrtf_variant={download_hrtf_variant!r}")
            if "mesh" in requested_resources:
                request_parts.append(f"download_mesh_variant={download_mesh_variant!r}")
            warnings.warn(
                f"{self.config.name}: download request produced no planned files. "
                f"{', '.join(request_parts)}. "
                "The selected resources or variant may not be available from the configured server.",
                stacklevel=2,
            )
            return False, summary
        downloaded_count = 0
        verified_count = 0
        failures: list[str] = []
        progress_bar = None
        try:
            for index, job in enumerate(download_jobs, start=1):
                try:
                    destination = Path(cast(Any, job["destination"]))
                    checksum = cast(str | None, job["checksum"])
                    relative_path = cast(str, job["relative_path"])
                    local_path_patterns = tuple(cast(tuple[str, ...], job.get("local_path_patterns", tuple())))
                    existing_path = DatasetResourcesScanner.resolve_resource_patterns(
                        self.root,
                        (relative_path,) + local_path_patterns,
                        cast(str, job["resource"]),
                    )
                    if existing_path.is_file():
                        try:
                            self.verify_downloaded_file(existing_path, checksum)
                        except ValueError:
                            status = self.download_file(
                                str(job["url"]),
                                destination,
                                checksum=checksum,
                            )
                        else:
                            status = "verified"
                    else:
                        status = self.download_file(
                            str(job["url"]),
                            destination,
                            checksum=checksum,
                        )
                except ValueError as exc:
                    failures.append(f"{job['relative_path']}: {exc}")
                    status = "failed"
                if status == "downloaded":
                    downloaded_count += 1
                elif status == "verified":
                    verified_count += 1
                if (
                    progress_bar is None
                    and status == "downloaded"
                    and tqdm is not None
                ):
                    progress_bar = tqdm(
                        total=len(download_jobs),
                        initial=index - 1,
                        desc=f"{self.config.name} download",
                        unit="file",
                    )
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        self.last_download_jobs = download_jobs
        self.last_downloaded_count = downloaded_count
        self.last_verified_count = verified_count
        self.last_failures = failures
        summary = download_summary(
            self.config,
            self.root,
            download_jobs,
            downloaded_count,
            verified_count,
            failures,
        )
        if (
            len(failures) == len(download_jobs)
            and downloaded_count == 0
            and verified_count == 0
        ):
            raise ValueError(
                f"{self.config.name} download failed because none of the planned files "
                "could be downloaded or verified. The selected download server may be "
                "unavailable, rejecting access, or missing every requested resource. "
                "Retry later, choose another download_server if the dataset supports "
                "one, or place the required files under the dataset root before "
                f"constructing the dataset.\n{summary}"
            )
        return self.finalize_download(downloaded_count > 0, summary)

    def finalize_download(self, downloaded: bool, summary: str) -> tuple[bool, str]:
        """Finalize a completed shared download workflow.

        This hook lets subclasses add server-specific post-processing after the
        common download loop has finished. The base implementation is a no-op
        because most servers expose final files directly and need no extra work.
        Archive-based (zip mostly) downloaders can override it to extract, normalize, or
        rewrite the final summary without duplicating :meth:`download`.

        Parameters
        ----------
        downloaded : bool
            Whether the shared workflow downloaded at least one file. Verified
            existing files do not set this flag.
        summary : str
            Summary generated from the executed download jobs.

        Returns
        -------
        tuple[bool, str]
            Final downloaded flag and summary returned to the dataset
            constructor.
        """

        return downloaded, summary

    def verify_checksum(self, path: Path, checksum: str) -> None:
        """Verify one file against its required SHA-256 checksum.

        The method computes the current digest and raises on mismatch. Missing
        checksums are not valid for downloader-managed resources.

        Parameters
        ----------
        path : Path
            File path to verify.
        checksum : str
            Expected SHA-256 checksum.

        Returns
        -------
        None
            Raises when checksum validation fails.

        Raises
        ------
        ValueError
            If checksum is malformed or if the computed digest does not match.
        OSError
            If path cannot be read.
        """
        expected = self.sanitize_checksum(checksum)
        current = self.compute_sha256(path)
        if current != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected}, got {current}"
            )

    def verify_archive_integrity(self, path: Path) -> None:
        """Verify archive containers before accepting a downloaded file.

        Archive downloads can be non-empty and checksum-valid while still being
        structurally corrupt. This method performs format-specific integrity checks for
        ZIP, TAR, and GZIP files before a download is accepted. Non-archive files
        return without additional structural checks and are still covered by size and
        optional checksum validation in
        :meth:`~hrtfpykit.datasets.download.BaseDownload.verify_downloaded_file`.

        Parameters
        ----------
        path : Path
            Downloaded file path.

        Returns
        -------
        None
            Raises when archive integrity checks fail.

        Raises
        ------
        ValueError
            If a ZIP member is reported corrupt.
        OSError
            If the archive cannot be opened or traversed.
        tarfile.TarError
            If a TAR archive is structurally invalid.
        gzip.BadGzipFile
            If a GZIP stream is structurally invalid.
        """
        suffix = path.suffix.lower()
        lower_name = path.name.lower()
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"ZIP archive is corrupt: {path.name} member {bad_member}")
            return
        if lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
            with tarfile.open(path, "r:*") as archive:
                archive.getmembers()
            return
        if suffix == ".gz":
            with gzip.open(path, "rb") as archive:
                while True:
                    chunk = archive.read(1024 * 1024)
                    if len(chunk) == 0:
                        break

    def build_local_path_patterns(
        self,
        resource: str,
        subject_id: str | None,
        filename: str,
        resource_type: str | None = None,
        sample_rate: str | int | None = None,
        sample_rate_label: str | None = None,
        version: str | None = None,
    ) -> tuple[str, ...]:
        """Build scanner-compatible local alternatives for one planned job.

        Catalog and archive servers may download files from URLs that do not
        match every local layout accepted by the resource scanner. This method
        expands the configured ``local_path_patterns`` for HRTF and mesh
        resources so :meth:`download` can detect already-local files before
        transferring them again.

        Parameters
        ----------
        resource : str
            Resource family, currently ``hrtf`` or ``mesh`` for subject-specific
            local path alternatives.
        subject_id : str or None
            Subject identifier for subject-specific resources. None returns no
            local alternatives.
        filename : str
            File name used by ``{filename}`` placeholders in local path patterns.
        resource_type : str or None, default=None
            HRTF or mesh type used by type placeholders.
        sample_rate : str, int, or None, default=None
            HRTF sample rate used by sample-rate placeholders.
        sample_rate_label : str or None, default=None
            Human-readable sample-rate label when the server provides one.
        version : str or None, default=None
            HRTF or mesh version used by version placeholders.

        Returns
        -------
        tuple of str
            Relative local path patterns that the scanner can resolve under the
            dataset root.
        """
        if subject_id is None:
            return tuple()
        if resource == "hrtf":
            if self.config.hrtf is None or resource_type is None:
                return tuple()
            type_config = self.config.hrtf.types.get(resource_type)
            if type_config is None:
                return tuple()
            selected_sample_rate_label = sample_rate_label
            if selected_sample_rate_label is None and sample_rate is not None:
                selected_sample_rate_label = str(sample_rate)
                if type_config.sample_rate_labels is not None:
                    selected_sample_rate_label = type_config.sample_rate_labels.get(
                        sample_rate,
                        selected_sample_rate_label,
                    )
            version_label = None
            if version is not None:
                version_label = version
                if type_config.version_labels is not None:
                    version_label = type_config.version_labels.get(version, version_label)
            format_values = {
                "subject_id": subject_id,
                "type": resource_type,
                "hrtf_type": resource_type,
                "sample_rate": sample_rate,
                "hrtf_sample_rate": sample_rate,
                "sample_rate_label": selected_sample_rate_label,
                "version": version,
                "hrtf_version": version,
                "version_label": version_label,
                "hrtf_version_label": version_label,
                "variant": resource_type,
                "filename": filename,
            }
            return tuple(
                local_path_pattern.format(**format_values)
                for local_path_pattern in type_config.local_path_patterns
            )
        if resource == "mesh":
            if self.config.mesh is None or resource_type is None:
                return tuple()
            type_config = self.config.mesh.types.get(resource_type)
            if type_config is None:
                return tuple()
            version_label = None
            if version is not None:
                version_label = version
                if type_config.version_labels is not None:
                    version_label = type_config.version_labels.get(version, version_label)
            format_values = {
                "subject_id": subject_id,
                "type": resource_type,
                "mesh_type": resource_type,
                "version": version,
                "mesh_version": version,
                "version_label": version_label,
                "mesh_version_label": version_label,
                "variant": resource_type,
                "filename": filename,
            }
            return tuple(
                local_path_pattern.format(**format_values)
                for local_path_pattern in type_config.local_path_patterns
            )
        return tuple()

    def verify_downloaded_file(self, path: Path, checksum: str | None) -> None:
        """Verify that a local download candidate is complete and trusted.

        This method combines basic file checks, archive integrity checks, and optional
        SHA-256 validation. It is used for both existing files and temporary downloads
        before dataset construction uses them.

        Parameters
        ----------
        path : Path
            Downloaded file path.
        checksum : str or None
            Expected SHA-256 checksum. When None, checksum verification is skipped.

        Returns
        -------
        None
            Raises when the file is missing, empty, corrupt, or checksum-invalid
            when a checksum is provided.

        Raises
        ------
        ValueError
            If path is missing, not a file, empty, archive-corrupt, or has a
            checksum mismatch when a checksum is provided.
        OSError
            If the file cannot be inspected or read.
        """
        if not path.exists():
            raise ValueError(f"Downloaded file is missing: {path}")
        if not path.is_file():
            raise ValueError(f"Downloaded path is not a file: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Downloaded file is empty: {path}")
        self.verify_archive_integrity(path)
        if checksum is not None:
            self.verify_checksum(path, checksum)


class PathPatternDownload(BaseDownload):
    """Downloader for servers whose files are addressed by path templates.

    This class implements the shared direct-file planning used by servers such
    as SOFAcoustics and the original SONICOM Imperial transfer server. It expands
    dataset resource path patterns into concrete URLs and destinations, then the
    base class handles local-file checks, checksum verification, and transfer.
    """

    def build_download_plan(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None = None,
        download_hrtf_variant: str | Mapping[str, object] | None = None,
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Build file-level jobs for the selected official resources.

        The plan expands resource groups, subject scopes, HRTF variants, mesh
        variants, resource path rules, URLs, destinations, and checksums into
        concrete jobs. Resource path rules can be one format template shared by
        all subjects or a mapping from subject ID to relative path. Checksum
        fields are None when checksum verification is disabled. It performs
        planning without writing files, which lets constructors and tests inspect
        download intent separately from execution.

        This planner only uses download arguments. It does not inspect dataset specs,
        dataset construction variants, or any future
        :class:`~hrtfpykit.datasets.base.BaseDataset` state. Passing ``all`` in a
        download variant expands that download axis across the available values
        declared by the dataset config. When a resource type has no sample rate or
        version axis, the planner still creates jobs with the corresponding selector
        omitted.

        Parameters
        ----------
        download_resources : str, sequence of str, or None, default=None
            Resource groups to include in the plan. None selects every official
            resource. Supported names are declared by
            the configuration's
            :class:`~hrtfpykit.datasets.config.DownloadServerConfig`; ``all`` expands
            to every declared official resource.
        download_hrtf_variant : str, dict, or None, default=None
            HRTF variant requested for download. None selects every available HRTF
            variant. A string selects a type. A mapping
            can contain ``type``, ``sample_rate``, and ``version`` keys to
            select one or more axes explicitly.
        download_mesh_variant : str, dict, or None, default=None
            Mesh variant requested for download. A string selects a mesh type. A
            mapping can contain ``type`` and ``version`` keys.

        Returns
        -------
        list of dict
            Planned download jobs. Each job contains ``resource``,
            ``relative_path``, ``url``, ``destination``, and
            ``checksum``. ``checksum`` is None when checksum verification is
            disabled. Subject-specific jobs also contain ``subject_id``; HRTF and
            mesh jobs additionally include ``hrtf_variant`` or ``mesh_variant``.

        Raises
        ------
        ValueError
            If a requested resource or variant is unsupported, if the dataset does
            not provide a requested official resource, if variant mappings contain
            unsupported keys, or if checksum lookup fails.
        """

        self.validate_supported_download_filters(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        self.validate_download_variants(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        resources = self.sanitize_download_resources(download_resources)
        download_jobs: list[dict[str, object]] = []
        if isinstance(download_hrtf_variant, Mapping):
            unknown_keys = set(download_hrtf_variant) - {"type", "sample_rate", "version"}
            if len(unknown_keys) > 0:
                raise ValueError(
                    f"Unsupported download_hrtf_variant keys {tuple(sorted(unknown_keys))}. "
                    "Expected keys are ('type', 'sample_rate', 'version')"
                )
            hrtf_variant_type = download_hrtf_variant.get("type")
            hrtf_variant_sample_rate = download_hrtf_variant.get("sample_rate")
            hrtf_variant_version = download_hrtf_variant.get("version")
        else:
            hrtf_variant_type = download_hrtf_variant
            hrtf_variant_sample_rate = None
            hrtf_variant_version = None

        if isinstance(download_mesh_variant, Mapping):
            unknown_keys = set(download_mesh_variant) - {"type", "version"}
            if len(unknown_keys) > 0:
                raise ValueError(
                    f"Unsupported download_mesh_variant keys {tuple(sorted(unknown_keys))}. "
                    "Expected keys are ('type', 'version')"
                )
            mesh_variant_type = download_mesh_variant.get("type")
            mesh_variant_version = download_mesh_variant.get("version")
        else:
            mesh_variant_type = download_mesh_variant
            mesh_variant_version = None

        if "hrtf" in resources:
            if self.config.hrtf is None:
                raise ValueError(f"{self.config.name} does not provide official hrtf files")
            hrtf_subject_ids = (
                tuple(self.config.subject_ids)
                if self.config.hrtf.subject_ids is None
                else tuple(self.config.hrtf.subject_ids)
            )
            subject_ids = self.get_included_subject_ids(
                hrtf_subject_ids
            )
            subject_numbers = DatasetSplitPlanner.build_subject_number_map(
                tuple(DatasetSplitPlanner.sort_subject_ids(tuple(self.config.subject_ids)))
            )
            hrtf_types = self.sanitize_download_values(
                cast(Any, hrtf_variant_type),
                tuple(self.config.hrtf.types),
                None,
                "download_hrtf_variant['type']",
            )
            requested_hrtf_types = (
                (hrtf_variant_type,)
                if isinstance(hrtf_variant_type, (str, int)) or hrtf_variant_type is None
                else tuple(cast(Any, hrtf_variant_type))
            )
            hrtf_type_all = any(str(value).strip().lower() == "all" for value in requested_hrtf_types)
            for hrtf_type in hrtf_types:
                hrtf_type_config = self.config.hrtf.types[hrtf_type]
                try:
                    hrtf_sample_rates: tuple[str | None, ...] = self.sanitize_download_values(
                        cast(Any, hrtf_variant_sample_rate),
                        hrtf_type_config.sample_rates,
                        None,
                        "download_hrtf_variant['sample_rate']",
                    )
                except ValueError:
                    if hrtf_type_all:
                        continue
                    raise
                if len(hrtf_sample_rates) == 0:
                    hrtf_sample_rates = (None,)
                try:
                    hrtf_versions: tuple[str | None, ...] = self.sanitize_download_values(
                        cast(Any, hrtf_variant_version),
                        hrtf_type_config.versions,
                        None,
                        "download_hrtf_variant['version']",
                    )
                except ValueError:
                    if hrtf_type_all:
                        continue
                    raise
                if len(hrtf_versions) == 0:
                    hrtf_versions = (None,)
                for hrtf_sample_rate in hrtf_sample_rates:
                    sample_rate_value: str | int | None = hrtf_sample_rate
                    for available_sample_rate in hrtf_type_config.sample_rates:
                        if str(available_sample_rate) == str(hrtf_sample_rate):
                            sample_rate_value = available_sample_rate
                            break
                    sample_rate_label = None if sample_rate_value is None else str(sample_rate_value)
                    if hrtf_type_config.sample_rate_labels is not None and sample_rate_value is not None:
                        sample_rate_label = hrtf_type_config.sample_rate_labels.get(
                            sample_rate_value,
                            sample_rate_label,
                        )
                    for hrtf_version in hrtf_versions:
                        hrtf_version_label = None
                        if hrtf_type_config.version_labels is not None and hrtf_version is not None:
                            hrtf_version_label = hrtf_type_config.version_labels.get(
                                str(hrtf_version),
                                str(hrtf_version),
                            )
                        for subject_id in subject_ids:
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
                                "sample_rate": sample_rate_value,
                                "hrtf_sample_rate": sample_rate_value,
                                "sample_rate_label": sample_rate_label,
                                "version": hrtf_version,
                                "hrtf_version": hrtf_version,
                                "version_label": hrtf_version_label,
                                "hrtf_version_label": hrtf_version_label,
                                "variant": hrtf_type,
                            }
                            relative_path = selected_path_pattern.format(**format_values)
                            format_values["filename"] = Path(relative_path).name
                            local_path_patterns = tuple(
                                local_path_pattern.format(**format_values)
                                for local_path_pattern in hrtf_type_config.local_path_patterns
                            )
                            destination = self.compose_download_path(relative_path)
                            checksum_key = Path(relative_path).name if self.config.name == "SONICOM" else relative_path
                            if self.verify_checksum_enabled:
                                try:
                                    checksum = self.get_checksum(
                                        "hrtf",
                                        checksum_key,
                                        hrtf_type=hrtf_type,
                                        hrtf_version=None if hrtf_version is None else str(hrtf_version),
                                        hrtf_sample_rate=sample_rate_value,
                                    )
                                except MissingChecksumError:
                                    continue
                            else:
                                checksum = None
                            download_jobs.append(
                                {
                                    "resource": "hrtf",
                                    "subject_id": subject_id,
                                    "hrtf_variant": {
                                        "type": hrtf_type,
                                        "sample_rate": sample_rate_value,
                                        "version": hrtf_version,
                                    }
                                    if sample_rate_value is not None or hrtf_version is not None
                                    else hrtf_type,
                                    "relative_path": relative_path,
                                    "checksum_key": checksum_key,
                                    "local_path_patterns": local_path_patterns,
                                    "url": self.build_download_url("hrtf", relative_path),
                                    "destination": destination,
                                    "checksum": checksum,
                                }
                            )

        if "mesh" in resources:
            if self.config.mesh is None:
                raise ValueError(f"{self.config.name} does not provide official mesh data")
            mesh_subject_ids = (
                tuple(self.config.subject_ids)
                if self.config.mesh.subject_ids is None
                else tuple(self.config.mesh.subject_ids)
            )
            subject_ids = self.get_included_subject_ids(
                mesh_subject_ids
            )
            subject_numbers = DatasetSplitPlanner.build_subject_number_map(
                tuple(DatasetSplitPlanner.sort_subject_ids(tuple(self.config.subject_ids)))
            )
            mesh_types = self.sanitize_download_values(
                cast(Any, mesh_variant_type),
                tuple(self.config.mesh.types),
                None,
                "download_mesh_variant['type']",
            )
            requested_mesh_types = (
                (mesh_variant_type,)
                if isinstance(mesh_variant_type, (str, int)) or mesh_variant_type is None
                else tuple(cast(Any, mesh_variant_type))
            )
            mesh_type_all = any(str(value).strip().lower() == "all" for value in requested_mesh_types)
            if len(mesh_types) == 0 and "default" in self.config.mesh.types:
                mesh_types = ("default",)
            for mesh_type in mesh_types:
                mesh_type_config = self.config.mesh.types[mesh_type]
                try:
                    mesh_versions: tuple[str | None, ...] = self.sanitize_download_values(
                        cast(Any, mesh_variant_version),
                        mesh_type_config.versions,
                        None,
                        "download_mesh_variant['version']",
                    )
                except ValueError:
                    if mesh_type_all:
                        continue
                    raise
                if len(mesh_versions) == 0:
                    mesh_versions = (None,)
                for mesh_version in mesh_versions:
                    mesh_version_label = None
                    if mesh_type_config.version_labels is not None and mesh_version is not None:
                        mesh_version_label = mesh_type_config.version_labels.get(
                            str(mesh_version),
                            str(mesh_version),
                        )
                    for subject_id in subject_ids:
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
                            "version_label": mesh_version_label,
                            "mesh_version_label": mesh_version_label,
                        }
                        relative_path = selected_path_pattern.format(**format_values)
                        format_values["filename"] = Path(relative_path).name
                        local_path_patterns = tuple(
                            local_path_pattern.format(**format_values)
                            for local_path_pattern in mesh_type_config.local_path_patterns
                        )
                        destination = self.compose_download_path(relative_path)
                        checksum_key = Path(relative_path).name if self.config.name == "SONICOM" else relative_path
                        if self.verify_checksum_enabled:
                            try:
                                checksum = self.get_checksum(
                                    "mesh",
                                    checksum_key,
                                    mesh_type=mesh_type,
                                    mesh_version=None if mesh_version is None else str(mesh_version),
                                )
                            except MissingChecksumError:
                                continue
                        else:
                            checksum = None
                        download_jobs.append(
                            {
                                "resource": "mesh",
                                "subject_id": subject_id,
                                "mesh_variant": {
                                    "type": mesh_type,
                                    "version": mesh_version,
                                }
                                if mesh_version is not None
                                else mesh_type,
                                "relative_path": relative_path,
                                "checksum_key": checksum_key,
                                "local_path_patterns": local_path_patterns,
                                "url": self.build_download_url("mesh", relative_path),
                                "destination": destination,
                                "checksum": checksum,
                            }
                        )

        if "anthropometry" in resources:
            if self.config.anthropometry is None:
                raise ValueError(f"{self.config.name} does not provide official anthropometry")
            relative_path = self.config.anthropometry.path
            destination = self.compose_download_path(relative_path)
            if self.verify_checksum_enabled:
                try:
                    checksum = self.get_checksum("anthropometry", relative_path)
                except MissingChecksumError:
                    skip_job = True
                else:
                    skip_job = False
            else:
                checksum = None
                skip_job = False
            if not skip_job:
                download_jobs.append(
                    {
                        "resource": "anthropometry",
                        "subject_id": None,
                        "relative_path": relative_path,
                        "checksum_key": relative_path,
                        "local_path_patterns": self.config.anthropometry.local_path_patterns,
                        "url": self.build_download_url("anthropometry", relative_path),
                        "destination": destination,
                        "checksum": checksum,
                    }
                )

        if "metadata" in resources:
            if self.config.metadata is None:
                raise ValueError(f"{self.config.name} does not provide official metadata")
            relative_path = self.config.metadata.path
            checksum_key = Path(relative_path).name if self.config.name == "SONICOM" else relative_path
            destination = self.compose_download_path(relative_path)
            if self.verify_checksum_enabled:
                try:
                    checksum = self.get_checksum("metadata", checksum_key)
                except MissingChecksumError:
                    skip_job = True
                else:
                    skip_job = False
            else:
                checksum = None
                skip_job = False
            if not skip_job:
                download_jobs.append(
                    {
                        "resource": "metadata",
                        "subject_id": None,
                        "relative_path": relative_path,
                        "checksum_key": checksum_key,
                        "local_path_patterns": self.config.metadata.local_path_patterns,
                        "url": self.build_download_url("metadata", relative_path),
                        "destination": destination,
                        "checksum": checksum,
                    }
                )

        return download_jobs


class SOFAcousticsDownload(PathPatternDownload):
    def __init__(self, *args: Any, download_server: str | None = None, **kwargs: Any) -> None:
        """Create a SOFAcoustics downloader bound to its server key.

        SOFAcoustics exposes stable dataset-relative file paths. This
        constructor fixes the selected download server to ``sofacoustics`` and
        delegates path expansion, local-file checks, checksum verification, and
        transfer execution to :class:`PathPatternDownload` and
        :class:`BaseDownload`.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to :class:`PathPatternDownload`.
        download_server : str or None, default=None
            Optional server name. When provided it must be ``sofacoustics``.
        **kwargs : Any
            Keyword arguments forwarded to :class:`PathPatternDownload`.

        Raises
        ------
        UnsupportedDownloadServerError
            If download_server is not ``sofacoustics``.
        """
        if download_server is not None and download_server != "sofacoustics":
            raise UnsupportedDownloadServerError(
                f"SOFAcousticsDownload download_server accepts ('sofacoustics',); got {download_server!r}"
            )
        kwargs["download_server"] = "sofacoustics"
        super().__init__(*args, **kwargs)


class ImperialDownload(PathPatternDownload):
    def __init__(self, *args: Any, download_server: str | None = None, **kwargs: Any) -> None:
        """Create an Imperial downloader bound to its server key.

        The Imperial transfer server exposes SONICOM files through configured
        path patterns, including subject, HRTF type, sample-rate, version, and
        mesh variant axes. This constructor fixes the selected download server
        to ``imperial`` and delegates path expansion to
        :class:`PathPatternDownload`.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to :class:`PathPatternDownload`.
        download_server : str or None, default=None
            Optional server name. When provided it must be ``imperial``.
        **kwargs : Any
            Keyword arguments forwarded to :class:`PathPatternDownload`.

        Raises
        ------
        UnsupportedDownloadServerError
            If download_server is not ``imperial``.
        """
        if download_server is not None and download_server != "imperial":
            raise UnsupportedDownloadServerError(
                f"ImperialDownload download_server accepts ('imperial',); got {download_server!r}"
            )
        kwargs["download_server"] = "imperial"
        super().__init__(*args, **kwargs)


class TUBerlinDownload(BaseDownload):
    def __init__(self, *args: Any, download_server: str | None = None, **kwargs: Any) -> None:
        """Create a TU Berlin archive downloader bound to its server key.

        TU Berlin distributes HUTUBS as complete archive files instead of
        individual subject resources. This constructor fixes the selected
        download server to ``tu-berlin``. Planning reads archive metadata from
        :attr:`DownloadServerConfig.archives`, while the finalization step
        extracts verified ZIP files, flattens usable HUTUBS files into the
        dataset root, and removes temporary extracted folders. Subject and
        variant filtering are not supported by this server; use
        ``download_resources`` to select archive families.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to :class:`BaseDownload`.
        download_server : str or None, default=None
            Optional server name. When provided it must be ``tu-berlin``.
        **kwargs : Any
            Keyword arguments forwarded to :class:`BaseDownload`.

        Raises
        ------
        UnsupportedDownloadServerError
            If download_server is not ``tu-berlin``.
        """
        if download_server is not None and download_server != "tu-berlin":
            raise UnsupportedDownloadServerError(
                f"TUBerlinDownload download_server accepts ('tu-berlin',); got {download_server!r}"
            )
        kwargs["download_server"] = "tu-berlin"
        super().__init__(*args, **kwargs)

    def build_download_plan(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None = None,
        download_hrtf_variant: str | Mapping[str, object] | None = None,
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Build archive download jobs for TU Berlin HUTUBS resources.

        Parameters
        ----------
        download_resources : str, sequence of str, or None, default=None
            Archive resource families to download, such as ``hrtf``, ``mesh``,
            or ``anthropometry``.
        download_hrtf_variant : str, mapping, or None, default=None
            Must be None because TU Berlin archives contain complete HUTUBS
            resource families.
        download_mesh_variant : str, mapping, or None, default=None
            Must be None because TU Berlin archives contain complete mesh
            resource families.

        Returns
        -------
        list of dict
            Archive download jobs with resource, URL, destination, relative path,
            and checksum entries.
        """

        self.validate_supported_download_filters(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        self.validate_download_variants(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        resources = self.sanitize_download_resources(download_resources)
        download_jobs: list[dict[str, object]] = []
        for resource in resources:
            if resource == "hrtf" and self.config.hrtf is not None:
                expected_paths = []
                hrtf_subject_ids = (
                    tuple(self.config.subject_ids)
                    if self.config.hrtf.subject_ids is None
                    else tuple(self.config.hrtf.subject_ids)
                )
                for hrtf_type, hrtf_type_config in self.config.hrtf.types.items():
                    for subject_id in hrtf_subject_ids:
                        expected_paths.append(
                            self.root / str(hrtf_type_config.path_pattern).format(
                                subject_id=subject_id,
                                subject_number=0,
                                type=hrtf_type,
                                hrtf_type=hrtf_type,
                                sample_rate=None,
                                hrtf_sample_rate=None,
                                sample_rate_label=None,
                                version=None,
                                hrtf_version=None,
                                version_label=None,
                                hrtf_version_label=None,
                                variant=hrtf_type,
                            )
                        )
                if len(expected_paths) > 0 and all(path.is_file() for path in expected_paths):
                    continue
            elif resource == "mesh" and self.config.mesh is not None:
                expected_paths = []
                mesh_subject_ids = (
                    tuple(self.config.subject_ids)
                    if self.config.mesh.subject_ids is None
                    else tuple(self.config.mesh.subject_ids)
                )
                for mesh_type, mesh_type_config in self.config.mesh.types.items():
                    for subject_id in mesh_subject_ids:
                        expected_paths.append(
                            self.root / str(mesh_type_config.path_pattern).format(
                                subject_id=subject_id,
                                subject_number=0,
                                type=mesh_type,
                                mesh_type=mesh_type,
                                version=None,
                                mesh_version=None,
                                version_label=None,
                                mesh_version_label=None,
                            )
                        )
                if len(expected_paths) > 0 and all(path.is_file() for path in expected_paths):
                    continue
            elif resource == "anthropometry" and self.config.anthropometry is not None:
                if (self.root / self.config.anthropometry.path).is_file():
                    continue
            for archive in self.download_config.archives.get(resource, tuple()):
                archive_name = str(archive["name"])
                relative_path = f"archives/{archive_name}"
                checksum_key = relative_path
                checksum = self.get_checksum(resource, checksum_key) if self.verify_checksum_enabled else None
                download_jobs.append(
                    {
                        "resource": resource,
                        "subject_id": None,
                        "relative_path": relative_path,
                        "checksum_key": checksum_key,
                        "local_path_patterns": tuple(),
                        "url": self.validate_download_url(str(archive["url"])),
                        "destination": self.compose_download_path(relative_path),
                        "checksum": checksum,
                    }
                )
        return download_jobs

    def finalize_download(self, downloaded: bool, summary: str) -> tuple[bool, str]:
        """Extract TU Berlin archives and report normalized resource counts.

        TU Berlin downloads complete ZIP archives. After the shared download loop
        has verified those archives, this hook extracts them, moves usable HUTUBS
        SOFA, mesh, and anthropometry files into the dataset root, removes the
        temporary extracted folders, and rebuilds the summary using the normalized
        file counts.

        Parameters
        ----------
        downloaded : bool
            Whether the shared workflow downloaded at least one archive.
        summary : str
            Summary produced before archive extraction. It is accepted for API
            compatibility with the base hook and replaced with a normalized
            summary.

        Returns
        -------
        tuple[bool, str]
            Downloaded flag from the shared workflow and a summary that reflects
            normalized HUTUBS files.
        """

        for job in self.last_download_jobs:
            destination = Path(cast(Any, job["destination"]))
            if not destination.exists() or destination.suffix.lower() != ".zip":
                continue
            with zipfile.ZipFile(destination, "r") as archive:
                for member in archive.infolist():
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or any(part == ".." for part in member_path.parts):
                        raise ValueError(f"Archive member escapes dataset root: {member.filename}")
                    member_destination = (self.root / member_path).resolve()
                    try:
                        member_destination.relative_to(self.root)
                    except ValueError as exc:
                        raise ValueError(f"Archive member escapes dataset root: {member.filename}") from exc
                archive.extractall(self.root)

        hrir_dir = self.root / "HRIRs"
        mesh_dir = self.root / "3D head meshes"
        anthropometry_dir = self.root / "Antrhopometric measures"
        for source, pattern in (
            (hrir_dir, "pp*_HRIRs_*.sofa"),
            (mesh_dir, "pp*_3DheadMesh.ply"),
            (anthropometry_dir, "AntrhopometricMeasures.csv"),
        ):
            if not source.exists():
                continue
            for path in source.glob(pattern):
                if not path.is_file():
                    continue
                destination = self.root / path.name
                if path.resolve() == destination.resolve():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)

        for extracted_dir in (hrir_dir, mesh_dir, anthropometry_dir):
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)

        hrtf_subject_ids = {path.name.split("_", 1)[0] for path in self.root.glob("pp*_HRIRs_*.sofa")}
        mesh_subject_ids = {path.name.split("_", 1)[0] for path in self.root.glob("pp*_3DheadMesh.ply")}
        normalized_resources: dict[str, dict[str, object]] = {
            "hrtf": {
                "resource_count": len(tuple(self.root.glob("pp*_HRIRs_*.sofa"))),
                "subject_ids": tuple(sorted(hrtf_subject_ids)),
            },
            "mesh": {
                "resource_count": len(tuple(self.root.glob("pp*_3DheadMesh.ply"))),
                "subject_ids": tuple(sorted(mesh_subject_ids)),
            },
            "anthropometry": {
                "resource_count": 1 if (self.root / "AntrhopometricMeasures.csv").is_file() else 0,
                "subject_ids": tuple(),
            },
        }
        counted_jobs: list[dict[str, object]] = []
        for job in self.last_download_jobs:
            counted_job = dict(job)
            resource = str(counted_job["resource"])
            normalized_resource = normalized_resources.get(resource)
            if normalized_resource is not None:
                counted_job["resource_count"] = normalized_resource["resource_count"]
                counted_job["subject_ids"] = normalized_resource["subject_ids"]
            counted_jobs.append(counted_job)

        summary = download_summary(
            self.config,
            self.root,
            counted_jobs,
            self.last_downloaded_count,
            self.last_verified_count,
            self.last_failures,
        )
        return downloaded, summary


class SONICOMEcosystemDownload(BaseDownload):
    def __init__(self, *args: Any, download_server: str | None = None, **kwargs: Any) -> None:
        """Create a SONICOM ecosystem downloader bound to its server key.

        The SONICOM ecosystem publishes database JSON endpoints containing
        concrete file names and URLs. This constructor fixes the selected
        download server to ``sonicom-ecosystem``. Planning reads the configured
        database URLs, filters rows by dataset, subject, resource, and requested
        variants, maps selected files into the local dataset resource layout,
        and lets :class:`BaseDownload` handle transfer and verification.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to :class:`BaseDownload`.
        download_server : str or None, default=None
            Optional server name. When provided it must be ``sonicom-ecosystem``.
        **kwargs : Any
            Keyword arguments forwarded to :class:`BaseDownload`.

        Raises
        ------
        UnsupportedDownloadServerError
            If download_server is not ``sonicom-ecosystem``.
        """
        if download_server is not None and download_server != "sonicom-ecosystem":
            raise UnsupportedDownloadServerError(
                f"SONICOMEcosystemDownload download_server accepts ('sonicom-ecosystem',); got {download_server!r}"
            )
        kwargs["download_server"] = "sonicom-ecosystem"
        super().__init__(*args, **kwargs)

    def build_download_plan(
        self,
        download_resources: str | tuple[str, ...] | list[str] | None = None,
        download_hrtf_variant: str | Mapping[str, object] | None = None,
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        """Build file-level jobs from SONICOM ecosystem JSON catalogs.

        The planner reads the configured ecosystem database URLs, validates the
        requested resources and variants, filters catalog rows by subject and
        file metadata, maps each selected row to the dataset's configured local
        resource layout, resolves checksum keys, and returns jobs for the shared
        :meth:`BaseDownload.download` execution loop.

        Parameters
        ----------
        download_resources : str, sequence of str, or None, default=None
            Resource groups to plan from the ecosystem catalogs. None selects all
            resources supported by the selected server config.
        download_hrtf_variant : str, mapping, or None, default=None
            HRTF selector. A mapping may contain ``type``, ``sample_rate``, and
            ``version``.
        download_mesh_variant : str, mapping, or None, default=None
            Mesh selector. A mapping may contain ``type`` and ``version``.

        Returns
        -------
        list of dict
            Planned download jobs with resource, subject, variant, URL,
            destination, checksum key, checksum, and scanner-compatible local
            path alternatives.

        Raises
        ------
        ValueError
            If requested resources or variants are unsupported, an ecosystem
            database cannot be read, a catalog row cannot be mapped to the
            configured dataset layout, or required checksum metadata is missing.
        """
        self.validate_supported_download_filters(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        self.validate_download_variants(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        resources = self.sanitize_download_resources(download_resources)
        if self.config.hrtf is None:
            hrtf_types: set[str] = set()
            hrtf_sample_rates: set[int] = set()
            hrtf_versions: set[str] = set()
        else:
            available_hrtf_types = set(self.config.hrtf.types)
            available_hrtf_sample_rates = {
                int(sample_rate)
                for hrtf_type_config in self.config.hrtf.types.values()
                for sample_rate in hrtf_type_config.sample_rates
            }
            available_hrtf_versions = {
                str(version)
                for hrtf_type_config in self.config.hrtf.types.values()
                for version in hrtf_type_config.versions
            }
            if isinstance(download_hrtf_variant, Mapping):
                unknown_keys = set(download_hrtf_variant) - {"type", "sample_rate", "version"}
                if unknown_keys:
                    raise ValueError(
                        f"Unsupported download_hrtf_variant keys {tuple(sorted(unknown_keys))}. "
                        "Expected keys are ('type', 'sample_rate', 'version')"
                    )
                hrtf_type = download_hrtf_variant.get("type", "all")
                hrtf_sample_rate = download_hrtf_variant.get("sample_rate", "all")
                hrtf_version = download_hrtf_variant.get("version", "all")
            else:
                hrtf_type = "all" if download_hrtf_variant is None else download_hrtf_variant
                hrtf_sample_rate = "all"
                hrtf_version = "all"
            hrtf_types = available_hrtf_types if str(hrtf_type).lower() == "all" else {str(hrtf_type).lower()}
            hrtf_sample_rates = (
                available_hrtf_sample_rates
                if str(hrtf_sample_rate).lower() == "all"
                else {int(cast(Any, hrtf_sample_rate))}
            )
            hrtf_versions = available_hrtf_versions if str(hrtf_version).lower() == "all" else {str(hrtf_version)}

        if self.config.mesh is None:
            mesh_types: set[str] = set()
            mesh_versions: set[str] = set()
        else:
            available_mesh_types = set(self.config.mesh.types)
            available_mesh_versions = {
                str(version)
                for mesh_type_config in self.config.mesh.types.values()
                for version in mesh_type_config.versions
            }
            if isinstance(download_mesh_variant, Mapping):
                unknown_keys = set(download_mesh_variant) - {"type", "version"}
                if unknown_keys:
                    raise ValueError(
                        f"Unsupported download_mesh_variant keys {tuple(sorted(unknown_keys))}. "
                        "Expected keys are ('type', 'version')"
                    )
                mesh_type = download_mesh_variant.get("type", "all")
                mesh_version = download_mesh_variant.get("version", "all")
            else:
                mesh_type = "all" if download_mesh_variant is None else download_mesh_variant
                mesh_version = "all"
            mesh_types = available_mesh_types if str(mesh_type).lower() == "all" else {str(mesh_type).lower()}
            mesh_versions = available_mesh_versions if str(mesh_version).lower() == "all" else {str(mesh_version)}

        requested_subject_ids = None if self.requested_subject_ids is None else set(self.requested_subject_ids)
        excluded_subject_ids = set(self.excluded_subject_ids)
        download_jobs: list[dict[str, object]] = []
        for resource in resources:
            rules = self.download_config.catalog_rules.get(resource, tuple())
            for rule in rules:
                database_urls = self.download_config.database_urls.get(rule.database_key, tuple())
                selected_database_urls: tuple[str, ...]
                if isinstance(database_urls, str):
                    selected_database_urls = (database_urls,)
                else:
                    selected_database_urls = tuple(database_urls)
                for database_url in selected_database_urls:
                    validated_url = self.validate_download_url(database_url)
                    try:
                        with urllib.request.urlopen(validated_url, timeout=60) as response:
                            payload = json.loads(response.read().decode("utf-8"))
                    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
                        if isinstance(exc, urllib.error.HTTPError):
                            reason = f"HTTP {exc.code} {exc.reason}"
                        elif isinstance(exc, urllib.error.URLError):
                            reason = f"URL error: {exc.reason}"
                        else:
                            reason = str(exc)
                        raise ValueError(f"Could not load SONICOM ecosystem database {validated_url} ({reason})") from exc
                    data = payload.get("data") if isinstance(payload, dict) else None
                    if not isinstance(data, list):
                        raise ValueError(f"SONICOM ecosystem database {validated_url} did not return a data list")
                    for row in (dict(item) for item in data if isinstance(item, dict)):
                        filename = str(row.get("Datafile Name", ""))
                        file_url = str(row.get("Datafile URL", ""))
                        match = re.match(rule.filename_regex, filename)
                        if match is None:
                            continue
                        values: dict[str, object] = dict(match.groupdict())
                        values["filename"] = filename
                        if rule.subject_id_field is None:
                            subject_id = str(values.get("subject_id", ""))
                        else:
                            subject_id = str(row.get(rule.subject_id_field, ""))
                            values["subject_id"] = subject_id
                        if subject_id not in self.config.subject_ids or subject_id in excluded_subject_ids:
                            continue
                        if requested_subject_ids is not None and subject_id not in requested_subject_ids:
                            continue
                        if resource == "hrtf":
                            hrtf_type = str(values.get("type", rule.hrtf_type or ""))
                            if hrtf_type not in hrtf_types:
                                continue
                            hrtf_type_config = None if self.config.hrtf is None else self.config.hrtf.types.get(hrtf_type)
                            if hrtf_type_config is None:
                                continue
                            version = str(values.get("version", rule.version or ""))
                            if version == "":
                                version = ""
                            sample_rate: int | None = None
                            sample_rate_label = values.get("sample_rate_label")
                            if sample_rate_label is not None and hrtf_type_config.sample_rate_labels is not None:
                                for rate, label in hrtf_type_config.sample_rate_labels.items():
                                    if str(label) == str(sample_rate_label):
                                        sample_rate = int(rate)
                                        break
                            elif values.get("sample_rate") is not None:
                                sample_rate = int(str(values["sample_rate"]))
                            if len(hrtf_type_config.sample_rates) > 0 and sample_rate not in {int(rate) for rate in hrtf_type_config.sample_rates}:
                                continue
                            if len(hrtf_sample_rates) > 0 and sample_rate not in hrtf_sample_rates:
                                continue
                            if len(hrtf_type_config.versions) > 0 and version not in hrtf_type_config.versions:
                                continue
                            if len(hrtf_versions) > 0 and version not in hrtf_versions:
                                continue
                            values["sample_rate"] = sample_rate
                            values["version"] = version
                            values["type"] = hrtf_type
                            variant: object = {"type": hrtf_type, "sample_rate": sample_rate, "version": version}
                        elif resource == "mesh":
                            mesh_type = str(values.get("type", rule.mesh_type or ""))
                            if mesh_type not in mesh_types:
                                continue
                            mesh_type_config = None if self.config.mesh is None else self.config.mesh.types.get(mesh_type)
                            if mesh_type_config is None:
                                continue
                            version = str(values.get("version", rule.version or ""))
                            if len(mesh_type_config.versions) > 0 and version not in mesh_type_config.versions:
                                continue
                            if len(mesh_versions) > 0 and version not in mesh_versions:
                                continue
                            values["version"] = version
                            values["type"] = mesh_type
                            variant = {"type": mesh_type, "version": version}
                        else:
                            continue
                        relative_path = rule.relative_path_pattern.format(**values)
                        if resource == "hrtf" and str(values.get("type")) == "synthetic":
                            checksum_key = f"{subject_id}/{filename}"
                        else:
                            checksum_key = filename if rule.checksum_key == "filename" else relative_path
                        if self.verify_checksum_enabled:
                            try:
                                if resource == "hrtf":
                                    checksum = self.get_checksum(
                                        resource,
                                        checksum_key,
                                        hrtf_type=str(values["type"]),
                                        hrtf_version=str(values["version"]),
                                        hrtf_sample_rate=cast(int | None, values.get("sample_rate")),
                                    )
                                else:
                                    checksum = self.get_checksum(
                                        resource,
                                        checksum_key,
                                        mesh_type=str(values["type"]),
                                        mesh_version=str(values["version"]),
                                    )
                            except MissingChecksumError:
                                continue
                        else:
                            checksum = None
                        local_path_patterns = self.build_local_path_patterns(
                            resource,
                            subject_id,
                            filename,
                            resource_type=str(values["type"]),
                            sample_rate=cast(str | int | None, values.get("sample_rate")),
                            sample_rate_label=None if values.get("sample_rate_label") is None else str(values["sample_rate_label"]),
                            version=str(values["version"]),
                        )
                        job = {
                            "resource": resource,
                            "subject_id": subject_id,
                            "relative_path": relative_path,
                            "checksum_key": checksum_key,
                            "local_path_patterns": local_path_patterns,
                            "url": self.validate_download_url(file_url),
                            "destination": self.compose_download_path(relative_path),
                            "checksum": checksum,
                        }
                        if resource == "hrtf":
                            job["hrtf_variant"] = variant
                        elif resource == "mesh":
                            job["mesh_variant"] = variant
                        download_jobs.append(job)
        return download_jobs
