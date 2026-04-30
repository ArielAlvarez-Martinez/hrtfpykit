from dataclasses import dataclass

from .checksums import HUTUBS_CHECKSUMS


@dataclass(frozen=True)
class HRTFConfig:
    variants: tuple[str, ...]
    default_variant: str
    path_pattern: str
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MeshConfig:
    path_pattern: str
    extensions: tuple[str, ...]
    subject_ids: tuple[str, ...] | None = None
    official_extension: str | None = None


@dataclass(frozen=True)
class AnthropometryConfig:
    path: str
    left_prefix: str
    right_prefix: str
    subject_column_candidates: tuple[str, ...] = (
        "subject_id",
        "subject",
        "id",
        "participant",
        "pp",
    )


@dataclass(frozen=True)
class ImageConfig:
    extensions: tuple[str, ...]
    supported_grouped_by: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VideoConfig:
    extensions: tuple[str, ...]
    supported_grouped_by: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DownloadConfig:
    base_url: str
    available_resources: tuple[str, ...]
    checksums: dict[str, object] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    subject_ids: tuple[str, ...]
    hrtf: HRTFConfig | None = None
    mesh: MeshConfig | None = None
    anthropometry: AnthropometryConfig | None = None
    image: ImageConfig | None = None
    video: VideoConfig | None = None
    download: DownloadConfig | None = None


@dataclass(frozen=True)
class HUTUBSConfig(DatasetConfig):
    name: str = "HUTUBS"
    subject_ids: tuple[str, ...] = tuple(f"pp{index}" for index in range(1, 97))
    hrtf: HRTFConfig | None = HRTFConfig(
        variants=("measured", "simulated"),
        default_variant="measured",
        path_pattern="{subject_id}_HRIRs_{variant}.sofa",
    )
    mesh: MeshConfig | None = MeshConfig(
        path_pattern="{subject_id}_3DheadMesh{extension}",
        extensions=(".ply", ".stl"),
        official_extension=".ply",
    )
    anthropometry: AnthropometryConfig | None = AnthropometryConfig(
        path="AntrhopometricMeasures.csv",
        left_prefix="L_",
        right_prefix="R_",
    )
    image: ImageConfig | None = ImageConfig(
        extensions=(
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        ),
        supported_grouped_by=(
            ("subject",),
            ("subject", "ear"),
        ),
    )
    video: VideoConfig | None = VideoConfig(
        extensions=(
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        ),
        supported_grouped_by=(
            ("subject",),
            ("subject", "ear"),
        ),
    )
    download: DownloadConfig | None = DownloadConfig(
        base_url="https://sofacoustics.org/data/database/hutubs",
        available_resources=("all", "hrtf", "mesh", "anthropometry"),
        checksums=HUTUBS_CHECKSUMS,
    )
