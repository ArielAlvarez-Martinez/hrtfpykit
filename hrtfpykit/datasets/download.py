from pathlib import Path
import gzip
import hashlib
import tarfile
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urlparse

from .config import DatasetConfig

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

def normalize_download_resources(
    requested: str | tuple[str, ...] | list[str],
    available: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(requested, str):
        requested_values = (requested,)
    else:
        requested_values = tuple(requested)
    normalized = tuple(
        str(value).strip().lower() for value in requested_values
    )
    if "all" in normalized:
        return tuple(value for value in available if value != "all")
    invalid = [value for value in normalized if value not in available]
    if invalid:
        raise ValueError(f"Unsupported download_resources: {invalid}")
    return normalized


def normalize_root(root: Path) -> Path:
    normalized = Path(root).expanduser()
    if normalized.exists() and not normalized.is_dir():
        raise ValueError(f"Dataset root must be a directory, got file: {normalized}")
    return normalized.resolve()


def validate_download_root(root: Path) -> Path:
    validated_root = normalize_root(root)
    validated_root.mkdir(parents=True, exist_ok=True)
    return validated_root


def validate_download_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Only https downloads are allowed, got: {url}")
    if parsed.netloc.strip() == "":
        raise ValueError(f"Download URL is missing a host: {url}")
    return url


def resolve_download_path(root: Path, filename: str) -> Path:
    candidate = Path(filename)
    if candidate.is_absolute():
        raise ValueError(f"Download filename must be relative: {filename}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"Download filename must not escape root: {filename}")
    destination = (root / candidate).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Resolved download path escapes root: {destination}") from exc
    return destination


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if len(chunk) == 0:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_checksum(checksum: str) -> str:
    value = str(checksum).strip().lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Checksums must be SHA-256 hex digests")
    return value


def verify_checksum(path: Path, checksum: str | None) -> None:
    if checksum is None:
        return
    expected = normalize_checksum(checksum)
    current = compute_sha256(path)
    if current != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {current}"
        )


def verify_archive_integrity(path: Path) -> None:
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


def verify_downloaded_file(path: Path, checksum: str | None) -> None:
    if not path.exists():
        raise ValueError(f"Downloaded file is missing: {path}")
    if not path.is_file():
        raise ValueError(f"Downloaded path is not a file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Downloaded file is empty: {path}")
    verify_archive_integrity(path)
    verify_checksum(path, checksum)


def normalize_download_hrtf_versions(
    requested: str,
    dataset_variants: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if dataset_variants is None or len(dataset_variants) == 0:
        raise ValueError("Dataset variants are missing")
    available_versions = tuple(str(value).strip().lower() for value in dataset_variants)
    requested_value = str(requested).strip().lower()
    if requested_value == "all":
        return available_versions
    if requested_value not in available_versions:
        raise ValueError(
            f"Unsupported download_hrtf_version {requested!r}. "
            f"Expected one of {available_versions + ('all',)}"
        )
    return (requested_value,)


class BaseDownload:
    def __init__(
        self,
        config: DatasetConfig,
        root: str | Path,
        excluded_subject_ids: tuple[str, ...] = tuple(),
    ) -> None:
        self.config = config
        self.root = normalize_root(Path(root))
        self.excluded_subject_ids = tuple(dict.fromkeys(excluded_subject_ids))

    def normalize_download_resources(
        self,
        requested: str | tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        if self.config.download is None:
            raise ValueError(f"{self.config.name} does not define downloadable resources")
        return normalize_download_resources(
            requested,
            tuple(self.config.download.available_resources),
        )

    def normalize_download_hrtf_versions(
        self,
        requested: str,
    ) -> tuple[str, ...]:
        dataset_variants = (
            None if self.config.hrtf is None else tuple(self.config.hrtf.variants)
        )
        return normalize_download_hrtf_versions(requested, dataset_variants)

    def validate_download_root(self) -> Path:
        self.root = validate_download_root(self.root)
        return self.root

    def resolve_download_path(self, filename: str) -> Path:
        return resolve_download_path(self.root, filename)

    def build_download_url(self, filename: str) -> str:
        if self.config.download is None:
            raise ValueError(f"{self.config.name} does not define an official download base URL")
        validated_base_url = validate_download_url(self.config.download.base_url.rstrip("/"))
        return f"{validated_base_url}/{filename}"

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
    ) -> None:
        validated_url = validate_download_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            try:
                verify_downloaded_file(destination, checksum)
                return
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
            verify_downloaded_file(temporary_path, checksum)
            temporary_path.replace(destination)
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
        download_hrtf_version: str = "all",
    ) -> list[tuple[str, Path, str | None]]:
        resources = self.normalize_download_resources(download_resources)
        download_jobs: list[tuple[str, Path, str | None]] = []

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
            for version in self.normalize_download_hrtf_versions(download_hrtf_version):
                for subject_id in subject_ids:
                    relative_path = self.config.hrtf.path_pattern.format(
                        subject_id=subject_id,
                        variant=version,
                    )
                    destination = self.resolve_download_path(relative_path)
                    checksum = (
                        None
                        if self.config.hrtf.checksums is None
                        else self.config.hrtf.checksums.get(relative_path)
                    )
                    download_jobs.append(
                        (self.build_download_url(relative_path), destination, checksum)
                    )

        if "mesh" in resources:
            if self.config.mesh is None:
                raise ValueError(f"{self.config.name} does not provide official mesh data")
            mesh_extension = self.config.mesh.official_extension
            if mesh_extension is None:
                if len(self.config.mesh.extensions) == 0:
                    raise ValueError(f"{self.config.name} mesh extensions are missing")
                mesh_extension = self.config.mesh.extensions[0]
            mesh_subject_ids = (
                tuple(self.config.subject_ids)
                if self.config.mesh.subject_ids is None
                else tuple(self.config.mesh.subject_ids)
            )
            subject_ids = self.get_included_subject_ids(
                mesh_subject_ids
            )
            for subject_id in subject_ids:
                relative_path = self.config.mesh.path_pattern.format(
                    subject_id=subject_id,
                    extension=mesh_extension,
                )
                destination = self.resolve_download_path(relative_path)
                checksum = (
                    None
                    if self.config.mesh.checksums is None
                    else self.config.mesh.checksums.get(relative_path)
                )
                download_jobs.append(
                    (self.build_download_url(relative_path), destination, checksum)
                )

        if "anthropometry" in resources:
            if self.config.anthropometry is None:
                raise ValueError(f"{self.config.name} does not provide official anthropometry")
            relative_path = self.config.anthropometry.path
            destination = self.resolve_download_path(relative_path)
            download_jobs.append(
                (
                    self.build_download_url(relative_path),
                    destination,
                    self.config.anthropometry.checksum,
                )
            )

        return download_jobs

    def download(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_version: str = "all",
    ) -> None:
        self.validate_download_root()
        download_jobs = self.build_download_plan(
            download_resources=download_resources,
            download_hrtf_version=download_hrtf_version,
        )
        if len(download_jobs) == 0:
            return
        if tqdm is None:
            for url, destination, checksum in download_jobs:
                self.download_file(url, destination, checksum=checksum)
            return
        with tqdm(total=len(download_jobs), desc=f"{self.config.name} download", unit="file") as progress_bar:
            for url, destination, checksum in download_jobs:
                self.download_file(url, destination, checksum=checksum)
                progress_bar.update(1)
