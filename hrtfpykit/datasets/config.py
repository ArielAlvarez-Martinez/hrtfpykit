from dataclasses import dataclass

from .checksums import HUTUBS_CHECKSUMS, SONICOM_CHECKSUMS


@dataclass(frozen=True)
class ResourceTypeConfig:
    """Describe one concrete resource type inside a dataset configuration.

    Parameters
    ----------
    path_pattern : str
        Relative path template used to locate or download resource files.
    versions : tuple of str, default=()
        Supported processing or geometry versions for this resource type.
    version_labels : dict or None, default=None
        Optional mapping from public version names to path-template labels.
    sample_rates : tuple of int or str, default=()
        Supported sample-rate variants for this resource type.
    sample_rate_labels : dict or None, default=None
        Optional mapping from public sample-rate values to path-template labels.

    Returns
    -------
    ResourceTypeConfig Immutable resource-type description used by scanners and
    downloaders.

    Use Cases
    ---------
    - Describe measured and synthetic HRTF layouts.
    - Describe mesh variants such as raw, scanned, or watertight.
    - Keep resource path formatting separate from dataset constructor defaults.
    """

    path_pattern: str
    versions: tuple[str, ...] = ()
    version_labels: dict[str, str] | None = None
    sample_rates: tuple[int | str, ...] = ()
    sample_rate_labels: dict[int | str, str] | None = None


@dataclass(frozen=True)
class HRTFConfig:
    """Describe the HRTF resources available in a dataset.

    Parameters
    ----------
    types : dict[str, ResourceTypeConfig]
        Mapping from HRTF type names to resource path descriptions.
    subject_ids : tuple of str or None, default=None
        Optional subject list specific to HRTF resources. ``None`` uses the
        dataset-level subjects.

    Returns
    -------
    HRTFConfig Immutable HRTF resource description used by dataset scanning and
    downloading.

    Use Cases
    ---------
    - Declare measured, simulated, or synthetic HRTF resource families.
    - Restrict HRTF resources to a subset of dataset subjects.
    - Keep available variants separate from user-selected dataset defaults.
    """

    types: dict[str, ResourceTypeConfig]
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MeshConfig:
    """Describe mesh resources available in a dataset.

    Parameters
    ----------
    types : dict[str, ResourceTypeConfig]
        Mapping from mesh type names to resource path descriptions.
    extensions : tuple of str, default=('.ply',)
        File extensions accepted when scanning mesh resources.
    subject_ids : tuple of str or None, default=None
        Optional subject list specific to mesh resources. ``None`` uses the
        dataset-level subjects.

    Returns
    -------
    MeshConfig Immutable mesh resource description used by dataset scanning and
    downloading.

    Use Cases
    ---------
    - Declare scanned or synthetic mesh resources.
    - Support multiple mesh file extensions.
    - Restrict mesh resources to a subset of dataset subjects.
    """

    types: dict[str, ResourceTypeConfig]
    extensions: tuple[str, ...] = (".ply",)
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AnthropometryConfig:
    """Describe the official anthropometry table for a dataset.

    Parameters
    ----------
    path : str
        Relative path to the anthropometry table.
    left_prefix : str
        Prefix used for left-ear anthropometry fields.
    right_prefix : str
        Prefix used for right-ear anthropometry fields.
    extensions : tuple of str, default=('.csv', '.mat')
        Supported table file extensions.

    Returns
    -------
    AnthropometryConfig Immutable anthropometry resource description.

    Use Cases
    ---------
    - Register physical measurement tables for dataset construction.
    - Configure ear-specific field prefixes.
    - Support CSV or MAT anthropometry resources.
    """

    path: str
    left_prefix: str
    right_prefix: str
    extensions: tuple[str, ...] = (".csv", ".mat")


@dataclass(frozen=True)
class MetadataConfig:
    """Describe the official metadata table for a dataset.

    Parameters
    ----------
    path : str
        Relative path to the metadata table.
    extensions : tuple of str, default=('.csv', '.mat')
        Supported table file extensions.

    Returns
    -------
    MetadataConfig Immutable metadata resource description.

    Use Cases
    ---------
    - Register general subject annotations for dataset construction.
    - Keep metadata separate from anthropometry resources.
    - Support CSV or MAT metadata resources.
    """

    path: str
    extensions: tuple[str, ...] = (".csv", ".mat")


@dataclass(frozen=True)
class ImageConfig:
    """Describe image resources available in a dataset.

    Parameters
    ----------
    extensions : tuple of str, default=('.png',)
        Supported image file extensions.

    Returns
    -------
    ImageConfig Immutable image resource description.

    Use Cases
    ---------
    - Configure subject image scanning.
    - Restrict image datasets to selected file formats.
    - Pair image specs with acoustic specs.
    """

    extensions: tuple[str, ...] = (".png",)


@dataclass(frozen=True)
class VideoConfig:
    """Describe video resources available in a dataset.

    Parameters
    ----------
    extensions : tuple of str, default=('.mp4',)
        Supported video file extensions.

    Returns
    -------
    VideoConfig Immutable video resource description.

    Use Cases
    ---------
    - Configure subject video scanning.
    - Restrict video datasets to selected file formats.
    - Pair video specs with acoustic specs.
    """

    extensions: tuple[str, ...] = (".mp4",)


@dataclass(frozen=True)
class DownloadConfig:
    """Describe official downloadable resources for a dataset.

    Parameters
    ----------
    base_url : str
        HTTPS base URL used to compose download URLs.
    available_resources : tuple of str
        Resource names supported by the dataset downloader.
    checksums : dict or None, default=None
        Optional SHA-256 checksum map used for secure verification.

    Returns
    -------
    DownloadConfig Immutable download description consumed by ``BaseDownload``.

    Use Cases
    ---------
    - Register official dataset download endpoints.
    - Limit downloads to supported resource groups.
    - Attach checksums for integrity validation.
    """

    base_url: str
    available_resources: tuple[str, ...]
    checksums: dict[str, object] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Describe the resources and subjects that make up a dataset family.

    Parameters
    ----------
    name : str
        Public dataset name used in summaries and errors.
    subject_ids : tuple of str
        Canonical subject IDs accepted by the dataset.
    excluded_subject_ids : tuple of str, default=()
        Dataset-level subject exclusions applied before user exclusions.
    hrtf, mesh, anthropometry, metadata, image, video : config or None
        Optional resource descriptions available to specs.
    download : DownloadConfig or None, default=None
        Optional official download description.

    Returns
    -------
    DatasetConfig Immutable dataset-family description consumed by
    ``BaseDataset``.

    Use Cases
    ---------
    - Add a new dataset integration.
    - Declare official resource layouts without selecting runtime defaults.
    - Share one config between resource scanning and downloading.
    """

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
    """Static configuration for the HUTUBS dataset family.

    The dataclass fields declare HUTUBS subject IDs, HRTF path templates, mesh
    resources, anthropometry resources, and official download metadata.
    """

    name: str = "HUTUBS"
    subject_ids: tuple[str, ...] = tuple(f"pp{index}" for index in range(1, 97))
    hrtf: HRTFConfig | None = HRTFConfig(
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
        available_resources=("hrtf", "mesh", "anthropometry"),
        checksums=HUTUBS_CHECKSUMS,
    )


@dataclass(frozen=True)
class SONICOMConfig(DatasetConfig):
    """Static configuration for the SONICOM dataset family.

    The dataclass fields declare SONICOM subject IDs, HRTF variants, mesh variants,
    metadata resources, dataset-level exclusions, and official download metadata.
    """

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
        types={
            "measured": ResourceTypeConfig(
                path_pattern="{subject_id}/HRTF/HRTF/{sample_rate_label}/{subject_id}_{version}_{sample_rate_label}.sofa",
                sample_rates=(44100, 48000, 96000),
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
            ),
            "synthetic": ResourceTypeConfig(
                path_pattern="{subject_id}/SYNTHETIC_HRTF/HRIR_SONICOM_{sample_rate}.sofa",
                sample_rates=(44100, 48000),
                versions=("generic",),
            ),
        },
    )
    mesh: MeshConfig | None = MeshConfig(
        types={
            "scanned": ResourceTypeConfig(
                path_pattern="{subject_id}/3DSCAN/{subject_id}{version_label}",
                versions=("raw", "point_cloud", "watertight"),
                version_labels={
                    "raw": ".stl",
                    "point_cloud": "_Project1.asc",
                    "watertight": "_watertight.stl",
                },
            ),
            "synthetic": ResourceTypeConfig(
                path_pattern="{subject_id}/SYNTHETIC_HRTF/{subject_id}_{version}.stl",
                versions=("preprocessed", "plugged", "graded_left", "graded_right"),
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
