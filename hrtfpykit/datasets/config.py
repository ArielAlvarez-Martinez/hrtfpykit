from dataclasses import dataclass

from .checksums import HUTUBS_CHECKSUMS, SONICOM_CHECKSUMS


@dataclass(frozen=True)
class ResourceTypeConfig:
    path_pattern: str
    versions: tuple[str, ...] = ()
    default_version: str | None = None
    version_labels: dict[str, str] | None = None
    sample_rates: tuple[int | str, ...] = ()
    default_sample_rate: int | str | None = None
    sample_rate_labels: dict[int | str, str] | None = None


@dataclass(frozen=True)
class HRTFConfig:
    types: dict[str, ResourceTypeConfig]
    default_type: str
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MeshConfig:
    types: dict[str, ResourceTypeConfig]
    default_type: str | None = None
    extensions: tuple[str, ...] = (".ply",)
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AnthropometryConfig:
    path: str
    left_prefix: str
    right_prefix: str
    extensions: tuple[str, ...] = (".csv", ".mat")


@dataclass(frozen=True)
class MetadataConfig:
    path: str
    extensions: tuple[str, ...] = (".csv", ".mat")


@dataclass(frozen=True)
class ImageConfig:
    extensions: tuple[str, ...] = (".png",)


@dataclass(frozen=True)
class VideoConfig:
    extensions: tuple[str, ...] = (".mp4",)


@dataclass(frozen=True)
class DownloadConfig:
    base_url: str
    available_resources: tuple[str, ...]
    checksums: dict[str, object] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    subject_ids: tuple[str, ...]
    excluded_subject_ids: tuple[str, ...] = ()
    hrtf: HRTFConfig | None = None
    mesh: MeshConfig | None = None
    anthropometry: AnthropometryConfig | None = None
    metadata: MetadataConfig | None = None
    image: ImageConfig | None = None
    video: VideoConfig | None = None
    download: DownloadConfig | None = None


@dataclass(frozen=True)
class HUTUBSConfig(DatasetConfig):
    name: str = "HUTUBS"
    subject_ids: tuple[str, ...] = tuple(f"pp{index}" for index in range(1, 97))
    hrtf: HRTFConfig | None = HRTFConfig(
        default_type="measured",
        types={
            "measured": ResourceTypeConfig(
                path_pattern="{subject_id}_HRIRs_measured.sofa",
            ),
            "simulated": ResourceTypeConfig(
                path_pattern="{subject_id}_HRIRs_simulated.sofa",
            ),
        },
    )
    mesh: MeshConfig | None = MeshConfig(
        default_type=None,
        types={
            "default": ResourceTypeConfig(
                path_pattern="{subject_id}_3DheadMesh.ply",
            ),
        },
    )
    anthropometry: AnthropometryConfig | None = AnthropometryConfig(
        path="AntrhopometricMeasures.csv",
        left_prefix="L_",
        right_prefix="R_",
    )
    image: ImageConfig | None = ImageConfig()
    video: VideoConfig | None = VideoConfig()
    download: DownloadConfig | None = DownloadConfig(
        base_url="https://sofacoustics.org/data/database/hutubs",
        available_resources=("all", "hrtf", "mesh", "anthropometry"),
        checksums=HUTUBS_CHECKSUMS,
    )


@dataclass(frozen=True)
class SONICOMConfig(DatasetConfig):
    name: str = "SONICOM"
    subject_ids: tuple[str, ...] = tuple(f"P{index:04d}" for index in range(1, 401))
    excluded_subject_ids: tuple[str, ...] = (
        "P0253",
        "P0258",
        "P0270",
        "P0272",
        "P0275",
        "P0396",
    )
    hrtf: HRTFConfig | None = HRTFConfig(
        default_type="measured",
        types={
            "measured": ResourceTypeConfig(
                path_pattern="{subject_id}/HRTF/HRTF/{sample_rate_label}/{subject_id}_{version}_{sample_rate_label}.sofa",
                sample_rates=(44100, 48000, 96000),
                default_sample_rate=44100,
                sample_rate_labels={
                    44100: "44kHz",
                    48000: "48kHz",
                    96000: "96kHz",
                },
                versions=(
                    "Raw",
                    "Raw_NoITD",
                    "Windowed",
                    "Windowed_NoITD",
                    "FreeFieldComp",
                    "FreeFieldComp_NoITD",
                    "FreeFieldCompMinPhase",
                    "FreeFieldCompMinPhase_NoITD",
                ),
                default_version="Windowed",
            ),
            "synthetic": ResourceTypeConfig(
                path_pattern="{subject_id}/SYNTHETIC_HRTF/HRIR_SONICOM_{sample_rate}.sofa",
                sample_rates=(44100, 48000),
                default_sample_rate=44100,
                versions=("generic",),
                default_version="generic",
            ),
        },
    )
    mesh: MeshConfig | None = MeshConfig(
        default_type="scanned",
        types={
            "scanned": ResourceTypeConfig(
                path_pattern="{subject_id}/3DSCAN/{subject_id}{version_label}",
                versions=("raw", "point_cloud", "watertight"),
                default_version="watertight",
                version_labels={
                    "raw": ".stl",
                    "point_cloud": "_Project1.asc",
                    "watertight": "_watertight.stl",
                },
            ),
            "synthetic": ResourceTypeConfig(
                path_pattern="{subject_id}/SYNTHETIC_HRTF/{subject_id}_{version}.stl",
                versions=("preprocessed", "plugged", "graded_left", "graded_right"),
                default_version="plugged",
            ),
        },
        extensions=(".stl", ".asc"),
    )
    metadata: MetadataConfig | None = MetadataConfig(
        path="metadata_and_readme/metadata.csv",
        extensions=(".csv",),
    )
    download: DownloadConfig | None = DownloadConfig(
        base_url="https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET",
        available_resources=("metadata", "hrtf", "mesh"),
        checksums=SONICOM_CHECKSUMS,
    )
