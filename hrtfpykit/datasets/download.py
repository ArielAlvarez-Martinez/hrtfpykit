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


class BaseDownload:
    def __init__(
        self,
        config: type[DatasetConfig] | DatasetConfig,
        root: str | Path,
        excluded_subject_ids: tuple[str, ...] = tuple(),
    ) -> None:
        self.config: type[DatasetConfig] | DatasetConfig = config
        self.root: Path = self.normalize_root(Path(root))
        self.excluded_subject_ids: tuple[str, ...] = tuple(dict.fromkeys(excluded_subject_ids))

    @staticmethod
    def preview_values(values: list[str] | tuple[str, ...], limit: int = 5) -> str:
        if len(values) == 0:
            return "none"
        preview = ", ".join(str(value) for value in values[:limit])
        if len(values) > limit:
            preview = f"{preview}, ..."
        return preview

    @staticmethod
    def normalize_root(root: Path) -> Path:
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
    def normalize_checksum(checksum: str) -> str:
        value = str(checksum).strip().lower()
        if value.startswith("sha256:"):
            value = value.split(":", 1)[1]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Checksums must be SHA-256 hex digests")
        return value

    def normalize_download_resources(
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

    def normalize_download_hrtf_variants(
        self,
        requested: str,
    ) -> tuple[str, ...]:
        dataset_variants = (
            None if self.config.hrtf is None else tuple(self.config.hrtf.variants)
        )
        if dataset_variants is None or len(dataset_variants) == 0:
            raise ValueError("Dataset variants are missing")
        available_versions = tuple(str(value).strip().lower() for value in dataset_variants)
        requested_value = str(requested).strip().lower()
        if requested_value == "all":
            return available_versions
        if requested_value not in available_versions:
            raise ValueError(
                f"Unsupported download_hrtf_variant {requested!r}. "
                f"Expected one of {available_versions + ('all',)}"
            )
        return (requested_value,)

    def validate_download_root(self) -> Path:
        self.root = self.normalize_root(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def resolve_download_path(self, filename: str) -> Path:
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
        variant: str | None = None,
    ) -> str | None:
        if self.config.download is None or self.config.download.checksums is None:
            return None
        checksums = self.config.download.checksums
        resource_checksums = checksums.get(resource)
        if resource_checksums is None:
            return None
        if resource == "hrtf":
            if variant is None:
                raise ValueError("HRTF checksum lookup requires a variant")
            if not isinstance(resource_checksums, dict):
                raise ValueError("HRTF checksums must be grouped by variant")
            variant_checksums = resource_checksums.get(variant)
            if variant_checksums is None:
                return None
            if not isinstance(variant_checksums, dict):
                raise ValueError("HRTF variant checksums must be a filename dictionary")
            checksum = variant_checksums.get(relative_path)
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
        download_hrtf_variant: str = "all",
    ) -> list[dict[str, object]]:
        resources = self.normalize_download_resources(download_resources)
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
            for variant in self.normalize_download_hrtf_variants(download_hrtf_variant):
                for subject_id in subject_ids:
                    relative_path = self.config.hrtf.path_pattern.format(
                        subject_id=subject_id,
                        variant=variant,
                    )
                    destination = self.resolve_download_path(relative_path)
                    checksum = self.get_checksum(
                        "hrtf",
                        relative_path,
                        variant=variant,
                    )
                    download_jobs.append(
                        {
                            "resource": "hrtf",
                            "subject_id": subject_id,
                            "variant": variant,
                            "relative_path": relative_path,
                            "url": self.build_download_url(relative_path),
                            "destination": destination,
                            "checksum": checksum,
                        }
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
                checksum = self.get_checksum("mesh", relative_path)
                if checksum is None and self.has_checksum_map("mesh"):
                    continue
                download_jobs.append(
                    {
                        "resource": "mesh",
                        "subject_id": subject_id,
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
            destination = self.resolve_download_path(relative_path)
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

    def format_download_summary(
        self,
        download_jobs: list[dict[str, object]],
        downloaded_count: int,
        verified_count: int,
        failures: list[str],
    ) -> str:
        resources: dict[str, int] = {}
        subject_ids: set[str] = set()
        variants: set[str] = set()
        for job in download_jobs:
            resource = str(job["resource"])
            resources[resource] = resources.get(resource, 0) + 1
            subject_id = job.get("subject_id")
            if subject_id is not None:
                subject_ids.add(str(subject_id))
            variant = job.get("variant")
            if variant is not None:
                variants.add(str(variant))
        lines = [
            f"{self.config.name} download summary",
            f"  root: {self.root}",
            f"  planned_files: {len(download_jobs)}",
            f"  downloaded_files: {downloaded_count}",
            f"  verified_existing_files: {verified_count}",
            f"  failed_files: {len(failures)}",
            f"  subjects: {len(subject_ids)}",
        ]
        if len(variants) > 0:
            lines.append(f"  variants: {', '.join(sorted(variants))}")
        if len(resources) > 0:
            lines.append(
                "  resources: "
                + ", ".join(f"{resource}={count}" for resource, count in sorted(resources.items()))
            )
        if len(failures) == 0:
            lines.append(f"  status: {self.config.name} dataset downloaded successfully")
        else:
            lines.append(
                "  failure_examples: " + self.preview_values(failures)
            )
            lines.append(f"  status: {self.config.name} dataset download finished with errors")
        return "\n".join(lines)

    def download(
        self,
        download_resources: str | tuple[str, ...] | list[str] = "all",
        download_hrtf_variant: str = "all",
    ) -> None:
        self.validate_download_root()
        download_jobs = self.build_download_plan(
            download_resources=download_resources,
            download_hrtf_variant=download_hrtf_variant,
        )
        if len(download_jobs) == 0:
            print(
                f"{self.config.name} download summary\n"
                f"  root: {self.root}\n"
                "  planned_files: 0\n"
                "  status: nothing to download"
            )
            return
        downloaded_count = 0
        verified_count = 0
        failures: list[str] = []
        progress_bar = (
            None
            if tqdm is None
            else tqdm(total=len(download_jobs), desc=f"{self.config.name} download", unit="file")
        )
        try:
            for job in download_jobs:
                try:
                    status = self.download_file(
                        str(job["url"]),
                        Path(job["destination"]),
                        checksum=job["checksum"],
                    )
                except ValueError as exc:
                    failures.append(f"{job['relative_path']}: {exc}")
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue
                if status == "downloaded":
                    downloaded_count += 1
                else:
                    verified_count += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        summary = self.format_download_summary(
            download_jobs,
            downloaded_count,
            verified_count,
            failures,
        )
        print(summary)
        if len(failures) > 0:
            raise ValueError(summary)

    def verify_checksum(self, path: Path, checksum: str | None) -> None:
        if checksum is None:
            return
        expected = self.normalize_checksum(checksum)
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
