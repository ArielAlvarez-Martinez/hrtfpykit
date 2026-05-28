from pathlib import Path
from collections.abc import Mapping
import gzip
import hashlib
import tarfile
from typing import Any, cast
import urllib.error
import urllib.request
import zipfile
from urllib.parse import quote, urlparse

from .config import DatasetConfig
from .summary import download_summary
from .split import DatasetSplitPlanner

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class MissingChecksumError(ValueError):
    pass


class BaseDownload:
    def __init__(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        excluded_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        verify_checksum: bool = True,
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
        already-local resources are scanned for samples. Subject exclusions are
        normalized once during initialization so every subject-specific resource
        family uses the same filtered subject list.

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
        excluded_subject_ids : str, int, sequence, or None, default=None
            Subject identifiers or one-based subject numbers to exclude from
            subject-specific download jobs. These exclusions are combined with
            exclusions declared by the dataset configuration.
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
        excluded_subject_ids : tuple of str
            Normalized union of configuration-level and user-requested subject
            exclusions.
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
        self.root: Path = self.sanitize_root(Path(root))
        self.verify_checksum_enabled = verify_checksum
        config_excluded_subject_ids = DatasetSplitPlanner.map_subject_ids(
            tuple(config.excluded_subject_ids),
            tuple(config.subject_ids),
        )
        requested_excluded_subject_ids = DatasetSplitPlanner.map_subject_ids(
            excluded_subject_ids,
            tuple(config.subject_ids),
        )
        self.excluded_subject_ids = tuple(
            dict.fromkeys(config_excluded_subject_ids + requested_excluded_subject_ids)
        )

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
        return url

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
        requested: str | tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Normalize requested official resource groups.

        This helper validates the resource groups requested by the user against the
        dataset download configuration and expands the pseudo-resource ``all``.
        It keeps dataset constructors small while making unsupported download
        requests fail before URL construction or network access.

        Parameters
        ----------
        requested : str or sequence of str
            Requested resource names. Accepted names are declared by the
            configuration's
            :class:`~hrtfpykit.datasets.config.DownloadConfig`.

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
        if self.config.download is None:
            raise ValueError(f"{self.config.name} does not define downloadable resources")
        if isinstance(requested, str):
            requested_values: tuple[str, ...] = (requested,)
        else:
            requested_values = tuple(requested)
        available = tuple(self.config.download.available_resources)
        normalized = tuple(str(value).strip().lower() for value in requested_values)
        if "all" in normalized:
            return tuple(value for value in available if value != "all")
        invalid = [value for value in normalized if value not in available]
        if invalid:
            raise ValueError(f"Unsupported download_resources: {invalid}")
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
        :class:`~hrtfpykit.datasets.config.DownloadConfig` metadata may provide
        resource-specific base URLs for datasets whose HRTF, mesh, anthropometry,
        or metadata resources are hosted on different servers. When no
        resource-specific base URL is configured, the dataset default base URL is
        used. The relative resource path remains the same path used for local
        destinations and checksum lookup.

        URL unsafe characters in the relative path, such as spaces in official
        filenames, are percent-encoded only in the returned URL. Local
        destination paths and checksum keys keep the original relative path.

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
        if self.config.download is None:
            raise ValueError(f"{self.config.name} does not define an official download base URL")
        resource_name = str(resource).strip().lower()
        resource_base_urls = self.config.download.resource_base_urls or {}
        resource_base_url = resource_base_urls.get(
            resource_name,
            self.config.download.base_url,
        )
        validated_base_url = self.validate_download_url(str(resource_base_url).rstrip("/"))
        return f"{validated_base_url}/{quote(filename, safe='/')}"

    def get_checksum(
        self,
        resource: str,
        relative_path: str,
        hrtf_type: str | None = None,
        hrtf_version: str | None = None,
        hrtf_sample_rate: str | int | None = None,
        mesh_type: str | None = None,
        mesh_version: str | None = None,
    ) -> str | None:
        """Return the official checksum for one planned download job.

        Checksum maps can be flat or hierarchical by HRTF type/version/sample-rate or
        mesh type/version. This method hides that structure from plan builders and
        returns the checksum relevant to one concrete relative path and variant
        context. Missing checksum metadata is treated as a hard error whenever
        checksum verification is enabled.

        Parameters
        ----------
        resource : str
            Resource group name, such as ``hrtf``, ``mesh``,
            ``anthropometry``, or ``metadata``.
        relative_path : str
            Relative resource path used as the final key in checksum maps.
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
            selected resource path has no string checksum.
        """
        if self.config.download is None or self.config.download.checksums is None:
            raise ValueError(
                f"{self.config.name} cannot download {resource!r} resources because no checksum map is configured"
            )
        checksums = self.config.download.checksums
        resource_checksums = checksums.get(resource)
        if resource_checksums is None:
            raise ValueError(
                f"{self.config.name} cannot download {resource!r} resources because no checksums are configured for that resource"
            )
        checksum: object | None = None
        if resource == "hrtf":
            if not isinstance(resource_checksums, dict):
                raise ValueError("HRTF checksums must be grouped by filename or type")
            direct_checksum = resource_checksums.get(relative_path)
            if isinstance(direct_checksum, str):
                checksum = direct_checksum
            elif hrtf_type is None:
                raise ValueError("HRTF checksum lookup requires a type when checksums are grouped by type")
            else:
                type_checksums = resource_checksums.get(hrtf_type)
                if type_checksums is None:
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
                        checksum = sample_rate_checksums.get(relative_path)
                    else:
                        checksum = version_checksums.get(relative_path)
                else:
                    if not isinstance(type_checksums, dict):
                        raise ValueError("HRTF type checksums must be a filename dictionary")
                    checksum = type_checksums.get(relative_path)
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
                    checksum = version_checksums.get(relative_path)
                else:
                    checksum = type_checksums.get(relative_path)
            else:
                checksum = resource_checksums.get(relative_path)
        elif isinstance(resource_checksums, dict):
            checksum = resource_checksums.get(relative_path)
        elif isinstance(resource_checksums, str):
            checksum = resource_checksums
        else:
            raise ValueError(f"{resource} checksums must be a string or filename dictionary")
        if checksum is None:
            raise MissingChecksumError(
                f"{self.config.name} is missing a checksum for {resource!r} resource {relative_path!r}"
            )
        if not isinstance(checksum, str):
            raise ValueError(f"{resource} checksum for {relative_path} must be a string")
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
        excluded_subject_ids_set = set(self.excluded_subject_ids)
        return tuple(
            subject_id for subject_id in subject_ids if subject_id not in excluded_subject_ids_set
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
            raise ValueError(f"Could not securely download {validated_url}") from exc
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def build_download_plan(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_variant: str | Mapping[str, object] | None = "all",
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
        download_resources : str or sequence of str, default=``all``
            Resource groups to include in the plan. Supported names are declared by
            the configuration's
            :class:`~hrtfpykit.datasets.config.DownloadConfig`; ``all`` expands
            to every declared official resource.
        download_hrtf_variant : str, dict, or None, default=``all``
            HRTF variant requested for download. A string selects a type. A mapping
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
                            relative_path = selected_path_pattern.format(
                                subject_id=subject_id,
                                subject_number=subject_numbers[subject_id],
                                type=hrtf_type,
                                hrtf_type=hrtf_type,
                                sample_rate=sample_rate_value,
                                hrtf_sample_rate=sample_rate_value,
                                sample_rate_label=sample_rate_label,
                                version=hrtf_version,
                                hrtf_version=hrtf_version,
                                version_label=hrtf_version_label,
                                hrtf_version_label=hrtf_version_label,
                                variant=hrtf_type,
                            )
                            destination = self.compose_download_path(relative_path)
                            if self.verify_checksum_enabled:
                                try:
                                    checksum = self.get_checksum(
                                        "hrtf",
                                        relative_path,
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
                        relative_path = selected_path_pattern.format(
                            subject_id=subject_id,
                            subject_number=subject_numbers[subject_id],
                            type=mesh_type,
                            mesh_type=mesh_type,
                            version=mesh_version,
                            mesh_version=mesh_version,
                            version_label=mesh_version_label,
                            mesh_version_label=mesh_version_label,
                        )
                        destination = self.compose_download_path(relative_path)
                        if self.verify_checksum_enabled:
                            try:
                                checksum = self.get_checksum(
                                    "mesh",
                                    relative_path,
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
                        "url": self.build_download_url("anthropometry", relative_path),
                        "destination": destination,
                        "checksum": checksum,
                    }
                )

        if "metadata" in resources:
            if self.config.metadata is None:
                raise ValueError(f"{self.config.name} does not provide official metadata")
            relative_path = self.config.metadata.path
            destination = self.compose_download_path(relative_path)
            if self.verify_checksum_enabled:
                try:
                    checksum = self.get_checksum("metadata", relative_path)
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
                        "url": self.build_download_url("metadata", relative_path),
                        "destination": destination,
                        "checksum": checksum,
                    }
                )

        return download_jobs

    def download(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_variant: str | Mapping[str, object] | None = "all",
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> tuple[bool, str]:
        """Execute secure downloads for the selected official resources.

        This method validates the root, builds the plan, executes each job, tracks
        downloaded and verified files, and returns a human-readable summary. It raises
        one combined error if any planned job fails so callers get complete context
        instead of a silent partial dataset.

        Download execution follows the explicit download plan. It does not fall back
        to specs, dataset HRTF variants, or dataset mesh variants when a resource or
        variant is missing from the download arguments. Existing files are accepted
        only after the same integrity checks used for newly downloaded files. Checksum
        checks are applied when checksum verification is enabled.

        Parameters
        ----------
        download_resources : str or sequence of str, default=``all``
            Resource groups to download or verify. ``all`` expands to every
            official resource declared by the dataset configuration.
        download_hrtf_variant : str, dict, or None, default=``all``
            HRTF variant requested for download. This value is passed directly to
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

        Raises
        ------
        ValueError
            If planning fails, if the root cannot be used, or if one or more
            planned jobs fails. When job execution fails, the raised message is the
            complete download summary with failure examples.
        """

        self.validate_download_root()
        download_jobs = self.build_download_plan(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
            download_mesh_variant=download_mesh_variant,
        )
        if len(download_jobs) == 0:
            summary = download_summary(
                self.config,
                self.root,
                download_jobs,
                downloaded_count=0,
                verified_count=0,
                failures=[],
            )
            return False, summary
        downloaded_count = 0
        verified_count = 0
        failures: list[str] = []
        progress_bar = None
        try:
            for index, job in enumerate(download_jobs, start=1):
                try:
                    status = self.download_file(
                        str(job["url"]),
                        Path(cast(Any, job["destination"])),
                        checksum=cast(str | None, job["checksum"]),
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
        summary = download_summary(
            self.config,
            self.root,
            download_jobs,
            downloaded_count,
            verified_count,
            failures,
        )
        if len(failures) > 0:
            raise ValueError(summary)
        return downloaded_count > 0, summary

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
