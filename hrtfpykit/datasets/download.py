from pathlib import Path
from collections.abc import Mapping
import gzip
import hashlib
import tarfile
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urlparse

from .config import DatasetConfig
from .summary import download_summary
from .split import DatasetSplitPlanner

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class BaseDownload:
    """Download and verify official dataset resources.

    ``BaseDownload`` is the shared downloader used by dataset classes before
    construction. It converts a dataset config and selector arguments into a
    concrete download plan, writes files under the dataset root, verifies archive
    integrity, and verifies every planned file against a SHA-256 checksum.

    Parameters
    ----------
    config : DatasetConfig or type[DatasetConfig]
        Dataset configuration with official download metadata.
    root : str or Path
        Local root where downloaded resources are stored.
    excluded_subject_ids : str, int, sequence, or None, default=None
        Subject references excluded from subject-specific download jobs.

    Returns
    -------
    BaseDownload Downloader object used to build plans and download selected
    resources.

    """

    def __init__(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        excluded_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
    ) -> None:
        self.config: type[DatasetConfig] | DatasetConfig = config
        self.root: Path = self.sanitize_root(Path(root))
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
        """Normalize and validate a dataset download root.

        The downloader accepts user-provided roots but must guarantee that later
        writes target a directory, not an existing file. This helper expands and
        resolves the path before download planning or file writes occur.

        Parameters
        ----------
        root : Path
            Candidate root path.

        Returns
        -------
        Path Resolved root path.

        """
        normalized = Path(root).expanduser()
        if normalized.exists() and not normalized.is_dir():
            raise ValueError(f"Dataset root must be a directory, got file: {normalized}")
        return normalized.resolve()

    @staticmethod
    def validate_download_url(url: str) -> str:
        """Validate an HTTPS download URL before use.

        Dataset downloads should not silently accept insecure or malformed URLs. This
        helper enforces the URL scheme and host early so download failures are
        reported before file transfer.

        Parameters
        ----------
        url : str
            URL to validate.

        Returns
        -------
        str Original URL when valid.

        """
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"Only https downloads are allowed, got: {url}")
        if parsed.netloc.strip() == "":
            raise ValueError(f"Download URL is missing a host: {url}")
        return url

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Compute the SHA-256 digest for a local file.

        The downloader reads the file in chunks so large SOFA, mesh, or archive
        resources can be verified without loading the entire file into memory. The
        result is used both for verification and checksum generation workflows.

        Parameters
        ----------
        path : Path
            File path to hash.

        Returns
        -------
        str SHA-256 hex digest.

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
        """Normalize and validate a SHA-256 checksum string.

        Checksums can be written with or without a ``sha256:`` prefix, but the
        downloader stores and compares plain lowercase hex digests. This helper
        rejects malformed values before they are used in verification.

        Parameters
        ----------
        checksum : str
            Checksum string, optionally prefixed by ``sha256:``.

        Returns
        -------
        str Lowercase SHA-256 hex digest.

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
        """Normalize requested download resource names.

        This helper validates the resource groups requested by the user against the
        dataset config and expands the pseudo-resource ``all``. It keeps dataset
        constructors small while making unsupported download requests fail before any
        network operation.

        Parameters
        ----------
        requested : str or sequence of str
            Requested resource names.

        Returns
        -------
        tuple of str Normalized resource names.

        """
        if self.config.download is None:
            raise ValueError(f"{self.config.name} does not define downloadable resources")
        if isinstance(requested, str):
            requested_values = (requested,)
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
        """Normalize selector values used by download planning.

        HRTF and mesh downloads can be selected by type, version, and sample rate.
        This helper handles ``None``, scalar values, sequences, ``all``, and
        validation against available values through one selector rule set.

        Parameters
        ----------
        requested : str, int, sequence, or None
            Requested selector values.
        available : tuple
            Available selector values.
        default : str, int, or None
            Optional fallback value.
        label : str
            Error label for validation messages.

        Returns
        -------
        tuple of str Normalized selector values.

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
            requested_values = (requested,)
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
        """Create and validate the downloader root directory.

        This method is called immediately before downloads so path checks and
        directory creation happen at the boundary where files may be written. It
        updates ``self.root`` with the resolved path used by subsequent jobs.

        Returns
        -------
        Path Resolved root directory.

        """
        self.root = self.sanitize_root(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def compose_download_path(self, filename: str) -> Path:
        """Resolve a destination path under the dataset root.

        Download configs provide relative resource paths, and this method turns them
        into absolute local paths while rejecting absolute paths or ``..`` escapes. It
        is the downloader boundary that prevents resource path templates from writing
        outside the dataset root.

        Parameters
        ----------
        filename : str
            Relative resource path.

        Returns
        -------
        Path Absolute destination path under the root.

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

    def build_download_url(self, filename: str) -> str:
        """Compose a full download URL from a relative resource path.

        The method combines the dataset download base URL with one planned relative
        path, then validates the resulting URL through the shared HTTPS rules. This
        keeps URL generation deterministic and centralized.

        Parameters
        ----------
        filename : str
            Relative resource path.

        Returns
        -------
        str Validated HTTPS download URL.

        """
        if self.config.download is None:
            raise ValueError(f"{self.config.name} does not define an official download base URL")
        validated_base_url = self.validate_download_url(self.config.download.base_url.rstrip("/"))
        return f"{validated_base_url}/{filename}"

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
        """Look up the checksum for one planned resource path.

        Checksum maps can be flat or hierarchical by HRTF type/version/sample-rate or
        mesh type/version. This method hides that structure from plan builders and
        returns the exact checksum relevant to one concrete resource path.

        Parameters
        ----------
        resource : str
            Resource group name.
        relative_path : str
            Relative resource path.
        hrtf_type, hrtf_version, hrtf_sample_rate : str, int, or None
            HRTF selector context for hierarchical checksum maps.
        mesh_type, mesh_version : str or None
            Mesh selector context for hierarchical checksum maps.

        Returns
        -------
        str
            SHA-256 checksum for the planned resource path.

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
        if resource == "hrtf":
            if hrtf_type is None:
                raise ValueError("HRTF checksum lookup requires a type")
            if not isinstance(resource_checksums, dict):
                raise ValueError("HRTF checksums must be grouped by type")
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
            raise ValueError(
                f"{self.config.name} is missing a checksum for {resource!r} resource {relative_path!r}"
            )
        if not isinstance(checksum, str):
            raise ValueError(f"{resource} checksum for {relative_path} must be a string")
        return checksum

    def get_included_subject_ids(self, subject_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Filter subject IDs by downloader exclusions.

        The downloader combines config-level and user-level exclusions once during
        initialization. This helper applies that normalized exclusion set to each
        subject-specific resource family.

        Parameters
        ----------
        subject_ids : tuple of str
            Candidate subject IDs.

        Returns
        -------
        tuple of str Subject IDs not excluded from downloads.

        """
        excluded_subject_ids_set = set(self.excluded_subject_ids)
        return tuple(
            subject_id for subject_id in subject_ids if subject_id not in excluded_subject_ids_set
        )

    def download_file(
        self,
        url: str,
        destination: Path,
        checksum: str,
    ) -> str:
        """Download or verify one planned file.

        This method validates the URL, verifies existing destinations, downloads to a
        temporary ``.part`` file when needed, checks size and integrity, then
        atomically moves the verified file into place. It is the file-write path for the downloader.

        Parameters
        ----------
        url : str
            HTTPS source URL.
        destination : Path
            Local destination path.
        checksum : str
            Required SHA-256 checksum.

        Returns
        -------
        str ``'downloaded'`` when fetched or ``'verified'`` when existing file passed
        validation.

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
        """Build the file-level download plan for selected resources.

        The plan expands resource groups, subject scopes, HRTF variants, mesh
        variants, path templates, URLs, destinations, and checksums into concrete
        jobs. It performs planning without writing files, which lets constructors and
        tests inspect download intent separately from execution.

        This planner only uses download arguments. It does not inspect dataset specs,
        dataset construction variants, or any future ``BaseDataset`` state. Passing
        ``"all"`` in a download variant expands that download axis across the
        available values declared by the dataset config.

        Parameters
        ----------
        download_resources : str or sequence of str, default='all'
            Resource groups to include in the plan. This value is not inferred from
            dataset specs.
        download_hrtf_variant : str, dict, or None, default='all'
            HRTF variant requested for download. This value is independent from any
            dataset construction HRTF variant.
        download_mesh_variant : str, dict, or None, default=None
            Mesh variant requested for download. This value is independent from any
            dataset construction mesh variant.

        Returns
        -------
        list of dict Planned download jobs containing resource names, selectors, URLs,
        destinations, subjects, relative paths, and checksums.

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
                DatasetSplitPlanner.sort_subject_ids(tuple(self.config.subject_ids))
            )
            hrtf_types = self.sanitize_download_values(
                hrtf_variant_type,
                tuple(self.config.hrtf.types),
                None,
                "download_hrtf_variant['type']",
            )
            requested_hrtf_types = (
                (hrtf_variant_type,)
                if isinstance(hrtf_variant_type, (str, int)) or hrtf_variant_type is None
                else tuple(hrtf_variant_type)
            )
            hrtf_type_all = any(str(value).strip().lower() == "all" for value in requested_hrtf_types)
            for hrtf_type in hrtf_types:
                hrtf_type_config = self.config.hrtf.types[hrtf_type]
                try:
                    hrtf_sample_rates = self.sanitize_download_values(
                        hrtf_variant_sample_rate,
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
                    hrtf_versions = self.sanitize_download_values(
                        hrtf_variant_version,
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
                            relative_path = hrtf_type_config.path_pattern.format(
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
                            checksum = self.get_checksum(
                                "hrtf",
                                relative_path,
                                hrtf_type=hrtf_type,
                                hrtf_version=None if hrtf_version is None else str(hrtf_version),
                                hrtf_sample_rate=sample_rate_value,
                            )
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
                                    "url": self.build_download_url(relative_path),
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
                DatasetSplitPlanner.sort_subject_ids(tuple(self.config.subject_ids))
            )
            mesh_types = self.sanitize_download_values(
                mesh_variant_type,
                tuple(self.config.mesh.types),
                None,
                "download_mesh_variant['type']",
            )
            requested_mesh_types = (
                (mesh_variant_type,)
                if isinstance(mesh_variant_type, (str, int)) or mesh_variant_type is None
                else tuple(mesh_variant_type)
            )
            mesh_type_all = any(str(value).strip().lower() == "all" for value in requested_mesh_types)
            if len(mesh_types) == 0 and "default" in self.config.mesh.types:
                mesh_types = ("default",)
            for mesh_type in mesh_types:
                mesh_type_config = self.config.mesh.types[mesh_type]
                try:
                    mesh_versions = self.sanitize_download_values(
                        mesh_variant_version,
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
                        relative_path = mesh_type_config.path_pattern.format(
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
                        checksum = self.get_checksum(
                            "mesh",
                            relative_path,
                            mesh_type=mesh_type,
                            mesh_version=None if mesh_version is None else str(mesh_version),
                        )
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
                                "url": self.build_download_url(relative_path),
                                "destination": destination,
                                "checksum": checksum,
                            }
                        )

        if "anthropometry" in resources:
            if self.config.anthropometry is None:
                raise ValueError(f"{self.config.name} does not provide official anthropometry")
            relative_path = self.config.anthropometry.path
            destination = self.compose_download_path(relative_path)
            download_jobs.append(
                {
                    "resource": "anthropometry",
                    "subject_id": None,
                    "relative_path": relative_path,
                    "url": self.build_download_url(relative_path),
                    "destination": destination,
                    "checksum": self.get_checksum("anthropometry", relative_path),
                }
            )

        if "metadata" in resources:
            if self.config.metadata is None:
                raise ValueError(f"{self.config.name} does not provide official metadata")
            relative_path = self.config.metadata.path
            destination = self.compose_download_path(relative_path)
            download_jobs.append(
                {
                    "resource": "metadata",
                    "subject_id": None,
                    "relative_path": relative_path,
                    "url": self.build_download_url(relative_path),
                    "destination": destination,
                    "checksum": self.get_checksum("metadata", relative_path),
                }
            )

        return download_jobs

    def download(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_variant: str | Mapping[str, object] | None = "all",
        download_mesh_variant: str | Mapping[str, object] | None = None,
    ) -> tuple[bool, str]:
        """Download or verify the selected official resources.

        This method validates the root, builds the plan, executes each job, tracks
        downloaded and verified files, and returns a human-readable summary. It raises
        one combined error if any planned job fails so callers get complete context
        instead of a silent partial dataset.

        Download execution follows the explicit download plan. It does not fall back
        to specs, dataset HRTF variants, or dataset mesh variants when a resource or
        variant is missing from the download arguments.

        Parameters
        ----------
        download_resources : str or sequence of str, default='all'
            Resource groups to download or verify. This value is not inferred from
            dataset specs.
        download_hrtf_variant : str, dict, or None, default='all'
            HRTF variant requested for download. This value is independent from any
            dataset construction HRTF variant.
        download_mesh_variant : str, dict, or None, default=None
            Mesh variant requested for download. This value is independent from any
            dataset construction mesh variant.

        Returns
        -------
        tuple[bool, str] ``True`` and a summary when at least one file was downloaded;
        ``False`` and a summary when all planned files already existed or no jobs were
        needed.

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
                        Path(job["destination"]),
                        checksum=job["checksum"],
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
        """Verify a file against its required SHA-256 checksum.

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
        None Raises when checksum validation fails.
        """
        expected = self.sanitize_checksum(checksum)
        current = self.compute_sha256(path)
        if current != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected}, got {current}"
            )

    def verify_archive_integrity(self, path: Path) -> None:
        """Verify archive containers can be opened and traversed.

        Archive downloads can be non-empty and checksum-valid while still being
        structurally corrupt. This method performs format-specific integrity checks for
        ZIP, TAR, and GZIP files before a download is accepted.

        Parameters
        ----------
        path : Path
            Downloaded file path.

        Returns
        -------
        None Raises when archive integrity checks fail.

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

    def verify_downloaded_file(self, path: Path, checksum: str) -> None:
        """Verify a downloaded file is present, non-empty, structurally valid, and
        checksum-valid.

        This method combines basic file checks, archive integrity checks, and required
        SHA-256 validation. It is used for both existing files and temporary downloads
        before dataset construction uses them.

        Parameters
        ----------
        path : Path
            Downloaded file path.
        checksum : str
            Expected SHA-256 checksum.

        Returns
        -------
        None Raises when the file is missing, empty, corrupt, or checksum-invalid.
        """
        if not path.exists():
            raise ValueError(f"Downloaded file is missing: {path}")
        if not path.is_file():
            raise ValueError(f"Downloaded path is not a file: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Downloaded file is empty: {path}")
        self.verify_archive_integrity(path)
        self.verify_checksum(path, checksum)
