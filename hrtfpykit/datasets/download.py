from pathlib import Path
import gzip
import hashlib
import tarfile
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urlparse

from .config import DatasetConfig
from .summary import download_summary
from .split import DatasetSubjectSplitPlanner

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class BaseDownload:
    def __init__(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        excluded_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
    ) -> None:
        self.config: type[DatasetConfig] | DatasetConfig = config
        self.root: Path = self.sanitize_root(Path(root))
        self.excluded_subject_ids = DatasetSubjectSplitPlanner.map_subject_ids(
            excluded_subject_ids,
            tuple(config.subject_ids),
        )

    @staticmethod
    def sanitize_root(root: Path) -> Path:
        normalized = Path(root).expanduser()
        if normalized.exists() and not normalized.is_dir():
            raise ValueError(f"Dataset root must be a directory, got file: {normalized}")
        return normalized.resolve()

    @staticmethod
    def validate_download_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"Only https downloads are allowed, got: {url}")
        if parsed.netloc.strip() == "":
            raise ValueError(f"Download URL is missing a host: {url}")
        return url

    @staticmethod
    def compute_sha256(path: Path) -> str:
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
        self.root = self.sanitize_root(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def compose_download_path(self, filename: str) -> Path:
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
    ) -> str | None:
        if self.config.download is None or self.config.download.checksums is None:
            return None
        checksums = self.config.download.checksums
        resource_checksums = checksums.get(resource)
        if resource_checksums is None:
            return None
        if resource == "hrtf":
            if hrtf_type is None:
                raise ValueError("HRTF checksum lookup requires a type")
            if not isinstance(resource_checksums, dict):
                raise ValueError("HRTF checksums must be grouped by type")
            type_checksums = resource_checksums.get(hrtf_type)
            if type_checksums is None:
                return None
            if isinstance(type_checksums, dict) and hrtf_version is not None and hrtf_version in type_checksums:
                version_checksums = type_checksums.get(hrtf_version)
                if not isinstance(version_checksums, dict):
                    raise ValueError("HRTF version checksums must be a filename dictionary")
                checksum = version_checksums.get(relative_path)
            else:
                if not isinstance(type_checksums, dict):
                    raise ValueError("HRTF type checksums must be a filename dictionary")
                checksum = type_checksums.get(relative_path)
        elif isinstance(resource_checksums, dict):
            checksum = resource_checksums.get(relative_path)
        elif isinstance(resource_checksums, str):
            checksum = resource_checksums
        else:
            raise ValueError(f"{resource} checksums must be a string or filename dictionary")
        if checksum is None:
            return None
        if not isinstance(checksum, str):
            raise ValueError(f"{resource} checksum for {relative_path} must be a string")
        return checksum

    def has_checksum_map(self, resource: str) -> bool:
        if self.config.download is None or self.config.download.checksums is None:
            return False
        return isinstance(self.config.download.checksums.get(resource), dict)

    def get_included_subject_ids(self, subject_ids: tuple[str, ...]) -> tuple[str, ...]:
        excluded_subject_ids_set = set(self.excluded_subject_ids)
        return tuple(
            subject_id for subject_id in subject_ids if subject_id not in excluded_subject_ids_set
        )

    def download_file(
        self,
        url: str,
        destination: Path,
        checksum: str | None = None,
    ) -> str:
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
        download_hrtf_type: str | tuple[str, ...] | list[str] | None = "all",
        download_hrtf_sample_rate: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        download_hrtf_version: str | tuple[str, ...] | list[str] | None = None,
        download_mesh_type: str | tuple[str, ...] | list[str] | None = None,
        download_mesh_version: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, object]]:
        resources = self.sanitize_download_resources(download_resources)
        download_jobs: list[dict[str, object]] = []

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
            subject_numbers = DatasetSubjectSplitPlanner.build_subject_number_map(
                DatasetSubjectSplitPlanner.sort_subject_ids(tuple(self.config.subject_ids))
            )
            hrtf_types = self.sanitize_download_values(
                download_hrtf_type,
                tuple(self.config.hrtf.types),
                self.config.hrtf.default_type,
                "download_hrtf_type",
            )
            requested_hrtf_types = (
                (download_hrtf_type,)
                if isinstance(download_hrtf_type, (str, int)) or download_hrtf_type is None
                else tuple(download_hrtf_type)
            )
            hrtf_type_all = any(str(value).strip().lower() == "all" for value in requested_hrtf_types)
            for hrtf_type in hrtf_types:
                hrtf_type_config = self.config.hrtf.types[hrtf_type]
                try:
                    hrtf_sample_rates = self.sanitize_download_values(
                        download_hrtf_sample_rate,
                        hrtf_type_config.sample_rates,
                        hrtf_type_config.default_sample_rate,
                        "download_hrtf_sample_rate",
                    )
                except ValueError:
                    if hrtf_type_all:
                        continue
                    raise
                if len(hrtf_sample_rates) == 0:
                    hrtf_sample_rates = (None,)
                try:
                    hrtf_versions = self.sanitize_download_values(
                        download_hrtf_version,
                        hrtf_type_config.versions,
                        hrtf_type_config.default_version,
                        "download_hrtf_version",
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
                            )
                            download_jobs.append(
                                {
                                    "resource": "hrtf",
                                    "subject_id": subject_id,
                                    "hrtf_type": hrtf_type,
                                    "hrtf_sample_rate": sample_rate_value,
                                    "hrtf_version": hrtf_version,
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
            subject_numbers = DatasetSubjectSplitPlanner.build_subject_number_map(
                DatasetSubjectSplitPlanner.sort_subject_ids(tuple(self.config.subject_ids))
            )
            mesh_types = self.sanitize_download_values(
                download_mesh_type,
                tuple(self.config.mesh.types),
                self.config.mesh.default_type,
                "download_mesh_type",
            )
            requested_mesh_types = (
                (download_mesh_type,)
                if isinstance(download_mesh_type, (str, int)) or download_mesh_type is None
                else tuple(download_mesh_type)
            )
            mesh_type_all = any(str(value).strip().lower() == "all" for value in requested_mesh_types)
            if len(mesh_types) == 0 and "default" in self.config.mesh.types:
                mesh_types = ("default",)
            for mesh_type in mesh_types:
                mesh_type_config = self.config.mesh.types[mesh_type]
                try:
                    mesh_versions = self.sanitize_download_values(
                        download_mesh_version,
                        mesh_type_config.versions,
                        mesh_type_config.default_version,
                        "download_mesh_version",
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
                        checksum = self.get_checksum("mesh", relative_path)
                        if checksum is None and self.has_checksum_map("mesh"):
                            continue
                        download_jobs.append(
                            {
                                "resource": "mesh",
                                "subject_id": subject_id,
                                "mesh_type": mesh_type,
                                "mesh_version": mesh_version,
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

        return download_jobs

    def download(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_type: str | tuple[str, ...] | list[str] | None = "all",
        download_hrtf_sample_rate: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        download_hrtf_version: str | tuple[str, ...] | list[str] | None = None,
        download_mesh_type: str | tuple[str, ...] | list[str] | None = None,
        download_mesh_version: str | tuple[str, ...] | list[str] | None = None,
    ) -> tuple[bool, str]:
        self.validate_download_root()
        download_jobs = self.build_download_plan(
            download_resources=download_resources,
            download_hrtf_type=download_hrtf_type,
            download_hrtf_sample_rate=download_hrtf_sample_rate,
            download_hrtf_version=download_hrtf_version,
            download_mesh_type=download_mesh_type,
            download_mesh_version=download_mesh_version,
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

    def verify_checksum(self, path: Path, checksum: str | None) -> None:
        if checksum is None:
            return
        expected = self.sanitize_checksum(checksum)
        current = self.compute_sha256(path)
        if current != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {expected}, got {current}"
            )

    def verify_archive_integrity(self, path: Path) -> None:
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
        if not path.exists():
            raise ValueError(f"Downloaded file is missing: {path}")
        if not path.is_file():
            raise ValueError(f"Downloaded path is not a file: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Downloaded file is empty: {path}")
        self.verify_archive_integrity(path)
        self.verify_checksum(path, checksum)
