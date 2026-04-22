from dataclasses import dataclass


@dataclass(frozen=True)
class HRTFConfig:
    variants: tuple[str, ...]
    default_variant: str
    path_pattern: str
    subject_ids: tuple[str, ...] | None = None
    checksums: dict[str, str] | None = None


@dataclass(frozen=True)
class MeshConfig:
    path_pattern: str
    extensions: tuple[str, ...]
    subject_ids: tuple[str, ...] | None = None
    official_extension: str | None = None
    checksums: dict[str, str] | None = None


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
    checksum: str | None = None


@dataclass(frozen=True)
class ImageConfig:
    extensions: tuple[str, ...]
    supported_align_by: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VideoConfig:
    extensions: tuple[str, ...]
    supported_align_by: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DownloadConfig:
    base_url: str
    available_resources: tuple[str, ...]


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

HUTUBS_CONFIG = DatasetConfig(
    name="HUTUBS",
    subject_ids=tuple(f"pp{index}" for index in range(1, 97)),
    hrtf=HRTFConfig(
        variants=("measured", "simulated"),
        default_variant="measured",
        path_pattern="{subject_id}_HRIRs_{variant}.sofa",
    ),
    mesh=MeshConfig(
        subject_ids=(
            "pp1",
            "pp2",
            "pp3",
            "pp4",
            "pp5",
            "pp6",
            "pp8",
            "pp9",
            "pp10",
            "pp11",
            "pp12",
            "pp16",
            "pp19",
            "pp20",
            "pp21",
            "pp22",
            "pp23",
            "pp29",
            "pp30",
            "pp31",
            "pp32",
            "pp33",
            "pp40",
            "pp41",
            "pp44",
            "pp45",
            "pp46",
            "pp47",
            "pp48",
            "pp49",
            "pp55",
            "pp57",
            "pp58",
            "pp59",
            "pp60",
            "pp61",
            "pp62",
            "pp63",
            "pp66",
            "pp67",
            "pp68",
            "pp69",
            "pp70",
            "pp71",
            "pp72",
            "pp73",
            "pp76",
            "pp77",
            "pp78",
            "pp80",
            "pp81",
            "pp82",
            "pp88",
            "pp89",
            "pp90",
            "pp91",
            "pp95",
            "pp96",
        ),
        path_pattern="{subject_id}_3DheadMesh{extension}",
        extensions=(".ply", ".stl"),
        official_extension=".ply",
    ),
    anthropometry=AnthropometryConfig(
        path="AntrhopometricMeasures.csv",
        left_prefix="L_",
        right_prefix="R_",
    ),
    image=ImageConfig(
        extensions=(
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        ),
        supported_align_by=(
            ("subject",),
            ("subject", "position"),
            ("subject", "ear"),
            ("subject", "position", "ear"),
        ),
    ),
    video=VideoConfig(
        extensions=(
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        ),
        supported_align_by=(
            ("subject",),
            ("subject", "position"),
            ("subject", "ear"),
            ("subject", "position", "ear"),
        ),
    ),
    download=DownloadConfig(
        base_url="https://sofacoustics.org/data/database/hutubs",
        available_resources=("all", "hrtf", "mesh", "anthropometry"),
    ),
)
