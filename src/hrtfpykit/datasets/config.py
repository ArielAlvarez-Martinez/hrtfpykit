from dataclasses import dataclass
from typing import cast

from .checksums import ARI_CHECKSUMS, HUTUBS_CHECKSUMS, SONICOM_CHECKSUMS


ARI_HRTF_PATHS = dict(
    sorted(
        (
            (filename.split("_", 1)[1][:-len(".sofa")], filename)
            for filename in ARI_CHECKSUMS["hrtf"]
            if filename.startswith("hrtf ") and filename.endswith(".sofa")
        ),
        key=lambda item: int(item[0][2:]),
    )
)
ARI_SUBJECT_IDS = tuple(ARI_HRTF_PATHS)


@dataclass(frozen=True)
class ResourceTypeConfig:
    """Describe one concrete variant family for a dataset resource.

    :class:`~hrtfpykit.datasets.config.ResourceTypeConfig` is the low-level
    schema used by HRTF and mesh resource configurations. It defines the
    relative path rule for one resource type, plus optional version and sample
    rate axes that expand the rule into concrete subject files during local
    scanning or download planning.

    Path templates are formatted by the dataset resource scanners and
    downloader. HRTF templates can use placeholders such as subject_id,
    subject_number, type, hrtf_type, sample_rate,
    hrtf_sample_rate, sample_rate_label, version,
    hrtf_version, version_label, hrtf_version_label, and
    variant. Mesh templates can use the subject placeholders plus
    type, mesh_type, version, mesh_version,
    version_label, mesh_version_label, and variant.

    Attributes
    ----------
    path_pattern : str or dict[str, str]
        Relative path template used to locate local files and build official
        download URLs, or a mapping from subject ID to relative resource path
        for datasets whose official files do not share one filename template.
        Mapping keys must use the canonical dataset subject IDs.
    versions : tuple of str
        Supported processing, compensation, or geometry versions for this
        resource type. An empty tuple means the type has no version selector.
    version_labels : dict[str, str] or None
        Optional mapping from public version names to labels used inside
        path_pattern. When absent, the public version string is used.
    sample_rates : tuple of int or str
        Supported sample-rate variants. An empty tuple means the type has no
        sample-rate selector.
    sample_rate_labels : dict[int | str, str] or None
        Optional mapping from public sample-rate values to labels used inside
        path_pattern. When absent, the public sample-rate value is converted
        to text.

    Notes
    -----
    This dataclass is frozen and performs no validation itself. Variant keys and
    values are validated by :class:`~hrtfpykit.datasets.build.DatasetBuilder`
    for dataset construction and by
    :class:`~hrtfpykit.datasets.download.BaseDownload` for download planning.

    """

    path_pattern: str | dict[str, str]
    versions: tuple[str, ...] = ()
    version_labels: dict[str, str] | None = None
    sample_rates: tuple[int | str, ...] = ()
    sample_rate_labels: dict[int | str, str] | None = None


@dataclass(frozen=True)
class HRTFConfig:
    """Describe HRTF or HRIR resources available in a dataset family.

    :class:`~hrtfpykit.datasets.config.HRTFConfig` declares the resource types
    that can satisfy acoustic specs such as
    :class:`~hrtfpykit.datasets.HRTFSpec`,
    :class:`~hrtfpykit.datasets.ITDSpec`,
    :class:`~hrtfpykit.datasets.ILDSpec`, and
    :class:`~hrtfpykit.datasets.SHSpec`. During dataset construction, the
    selected HRTF variant is validated against the configured types and then
    used to scan local subject files. During download planning, the same type
    definitions are expanded into official download jobs.

    Attributes
    ----------
    types : dict[str, ResourceTypeConfig]
        Mapping from public HRTF type names to resource path descriptions.
        Common type names include ``measured``, ``simulated``, and
        ``synthetic``.
    subject_ids : tuple of str or None
        Optional HRTF-specific subject list. None means the dataset-level
        :attr:`~hrtfpykit.datasets.config.DatasetConfig.subject_ids` are used.

    Notes
    -----
    The configuration name uses HRTF for the dataset resource family even
    when files store HRIR data in SOFA SimpleFreeFieldHRIR form. The higher
    level HRTF object handles time/frequency-domain access.

    """

    types: dict[str, ResourceTypeConfig]
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MeshConfig:
    """Describe subject mesh resources available in a dataset family.

    :class:`~hrtfpykit.datasets.config.MeshConfig` declares mesh resource types
    and the file extensions accepted during local scanning. Mesh specs use this
    configuration to find the geometry file for each selected subject, while the
    downloader uses the same path templates to build official mesh download jobs.

    Attributes
    ----------
    types : dict[str, ResourceTypeConfig]
        Mapping from public mesh type names to resource path descriptions.
    extensions : tuple of str
        File extensions accepted when scanning mesh resources. Extensions should
        include the leading dot.
    subject_ids : tuple of str or None
        Optional mesh-specific subject list. None means the dataset-level
        :attr:`~hrtfpykit.datasets.config.DatasetConfig.subject_ids` are used.

    Notes
    -----
    If a dataset defines a mesh type named ``default``, the resource scanner can
    use it when no explicit dataset mesh variant was provided.

    """

    types: dict[str, ResourceTypeConfig]
    extensions: tuple[str, ...] = (".ply",)
    subject_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AnthropometryConfig:
    """Describe an official anthropometry table resource.

    :class:`~hrtfpykit.datasets.config.AnthropometryConfig` tells the dataset
    resource scanner where the default anthropometry table lives and which file
    extensions are valid. The left and right prefixes support datasets whose
    physical measurements contain ear-specific columns, such as HUTUBS pinna or
    ear measurements.

    Attributes
    ----------
    path : str
        Relative path from the dataset root to the anthropometry table.
    left_prefix : str
        Prefix used to identify left-ear anthropometry fields.
    right_prefix : str
        Prefix used to identify right-ear anthropometry fields.
    extensions : tuple of str
        Supported table file extensions. Extensions should include the leading
        dot.

    Notes
    -----
    The config only describes the default official table. Individual
    :class:`~hrtfpykit.datasets.AnthropometrySpec` objects can still provide path or extension overrides.

    """

    path: str
    left_prefix: str
    right_prefix: str
    extensions: tuple[str, ...] = (".csv", ".mat")


@dataclass(frozen=True)
class MetadataConfig:
    """Describe an official subject metadata table resource.

    :class:`~hrtfpykit.datasets.config.MetadataConfig` declares the default
    table used by :class:`~hrtfpykit.datasets.MetadataSpec`. The resource
    scanner resolves this path relative to the dataset root, validates the
    requested extension, and loads rows or columns into the dataset state.

    Attributes
    ----------
    path : str
        Relative path from the dataset root to the metadata table.
    extensions : tuple of str
        Supported table file extensions. Extensions should include the leading
        dot.

    Notes
    -----
    The config only describes the default official table. Individual
    :class:`~hrtfpykit.datasets.MetadataSpec` objects can provide path or extension overrides.

    """

    path: str
    extensions: tuple[str, ...] = (".csv", ".mat")


@dataclass(frozen=True)
class ImageConfig:
    """Describe subject image resources available in a dataset family.

    :class:`~hrtfpykit.datasets.config.ImageConfig` declares the default image
    path and accepted extensions used by image resource scanning. Image specs use
    it when indexing subject-level or subject-ear image files under the dataset
    root.

    Attributes
    ----------
    path : str or None
        Relative path from the dataset root to the default image resource folder.
    extensions : tuple of str
        Supported image file extensions. Extensions should include the leading
        dot.

    """

    path: str | None = None
    extensions: tuple[str, ...] = (".png",)


@dataclass(frozen=True)
class VideoConfig:
    """Describe subject video resources available in a dataset family.

    :class:`~hrtfpykit.datasets.config.VideoConfig` declares the default video
    path and accepted extensions used by video resource scanning. Video specs use
    it when indexing subject-level or subject-ear video files under the dataset
    root.

    Attributes
    ----------
    path : str or None
        Relative path from the dataset root to the default video resource folder.
    extensions : tuple of str
        Supported video file extensions. Extensions should include the leading
        dot.

    """

    path: str | None = None
    extensions: tuple[str, ...] = (".mp4",)


@dataclass(frozen=True)
class DownloadConfig:
    """Describe official downloadable resources for a dataset family.

    :class:`~hrtfpykit.datasets.config.DownloadConfig` is consumed by
    :class:`~hrtfpykit.datasets.download.BaseDownload`. It defines the HTTPS base
    URL, the resource groups that can be downloaded, and the checksum mapping
    used to verify every planned file.

    Attributes
    ----------
    base_url : str
        Default HTTPS base URL used to compose resource download URLs.
        Resource-specific base URLs override this value when declared in
        ``resource_base_urls``.
    available_resources : tuple of str
        Resource group names accepted by the downloader, such as ``hrtf``,
        ``mesh``, ``metadata``, or ``anthropometry``.
    checksums : dict[str, object] or None
        Optional SHA-256 checksum map used for secure verification. The nested
        shape depends on the resource family and variant axes.
    resource_base_urls : dict[str, str] or None
        Optional mapping from resource group names to HTTPS base URLs. Use this
        when one dataset hosts different official resource families on different
        servers or URL roots while preserving the same resource-relative paths.

    Notes
    -----
    Download selection is independent from dataset construction selection. A
    dataset class can download one variant while later constructing samples from
    another local variant.

    """

    base_url: str
    available_resources: tuple[str, ...]
    checksums: dict[str, object] | None = None
    resource_base_urls: dict[str, str] | None = None


@dataclass(frozen=True)
class DatasetConfig:
    """Describe the subjects, resources, and downloads for a dataset family.

    :class:`~hrtfpykit.datasets.config.DatasetConfig` is the top-level
    declarative schema consumed by
    :class:`~hrtfpykit.datasets.base.BaseDataset`,
    :class:`~hrtfpykit.datasets.resources.DatasetResources`, and
    :class:`~hrtfpykit.datasets.download.BaseDownload`. Concrete dataset classes
    pass a config subclass or instance into the shared dataset builder so resource
    discovery, subject mapping, split planning, and download planning all use the
    same source of truth.

    Attributes
    ----------
    name : str
        Public dataset name used in summaries, errors, and download reports.
    subject_ids : tuple of str
        Canonical subject identifiers accepted by the dataset.
    excluded_subject_ids : tuple of str
        Dataset-level subject exclusions applied before user-provided
        exclusions.
    hrtf : HRTFConfig or None
        HRTF/HRIR resource configuration, when the dataset provides acoustic
        files.
    mesh : MeshConfig or None
        Mesh resource configuration, when the dataset provides subject geometry.
    anthropometry : AnthropometryConfig or None
        Anthropometry table configuration, when available.
    metadata : MetadataConfig or None
        Metadata table configuration, when available.
    image : ImageConfig or None
        Image resource configuration, when available.
    video : VideoConfig or None
        Video resource configuration, when available.
    download : DownloadConfig or None
        Official download configuration, when supported.

    Notes
    -----
    None resource fields mean the dataset family does not declare that
    resource type. Specs requesting an undeclared resource are rejected by the
    dataset construction workflow or resource scanner.

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
class ARIConfig(DatasetConfig):
    """Built-in configuration for the ARI HRTF database.

    :class:`~hrtfpykit.datasets.config.ARIConfig` declares the ARI subject IDs,
    official HRTF SOFA paths, download base URL, and SHA-256 checksums used by
    :class:`~hrtfpykit.datasets.ARI`. ARI filenames do not follow one shared
    subject template, so the HRTF configuration stores a subject path map from
    canonical IDs such as ``nh2`` or ``nh720`` to the corresponding official
    SOFA filename.

    Attributes
    ----------
    name : str
        Dataset family name.
    subject_ids : tuple of str
        ARI subject IDs available through the current official HRTF checksum
        map.
    hrtf : HRTFConfig or None
        HRTF resource configuration. ARI currently exposes one combined HRTF
        resource family backed by subject-specific SOFA paths.
    download : DownloadConfig or None
        Official ARI download configuration for HRTF resources.

    Notes
    -----
    The official ARI HRTF files are distributed in b, c, and d filename groups.
    The files included in this configuration are treated as one compatible ARI
    HRTF collection because they share the same source grid, IR shape, and
    sample rate. Workflows that need a narrower subset can exclude subject IDs
    when constructing :class:`~hrtfpykit.datasets.ARI`.

    """

    name: str = "ARI"
    subject_ids: tuple[str, ...] = ARI_SUBJECT_IDS
    hrtf: HRTFConfig | None = HRTFConfig(
        types={
            "hrtf": ResourceTypeConfig(
                path_pattern=ARI_HRTF_PATHS,
            ),
        },
    )
    download: DownloadConfig | None = DownloadConfig(
        base_url="https://sofacoustics.org/data/database/ari",
        available_resources=("hrtf",),
        checksums=cast(dict[str, object], ARI_CHECKSUMS),
    )


@dataclass(frozen=True)
class HUTUBSConfig(DatasetConfig):
    """Built-in configuration for the HUTUBS dataset family.

    :class:`~hrtfpykit.datasets.config.HUTUBSConfig` declares the subject IDs
    and official resource layout used by :class:`~hrtfpykit.datasets.HUTUBS`.
    It provides measured and
    simulated HRIR SOFA resources, a default head mesh resource, the official
    anthropometry table, optional image and video scanning defaults, and official
    download metadata for HRTF, mesh, and anthropometry resources.

    Attributes
    ----------
    name : str
        Public dataset name, ``HUTUBS``.
    subject_ids : tuple of str
        HUTUBS subject identifiers ``pp1`` through ``pp96``.
    hrtf : HRTFConfig
        Measured and simulated HUTUBS SOFA file templates.
    mesh : MeshConfig
        Default HUTUBS 3D head mesh template.
    anthropometry : AnthropometryConfig
        Official HUTUBS anthropometry table and left/right field prefixes.
    image, video : ImageConfig, VideoConfig
        Default media extension declarations for optional local media resources.
    download : DownloadConfig
        Official HUTUBS download base URL, downloadable resource groups, and
        checksum map.

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
    """Built-in configuration for the SONICOM dataset family.

    :class:`~hrtfpykit.datasets.config.SONICOMConfig` declares the subject IDs
    and official resource layout used by :class:`~hrtfpykit.datasets.SONICOM`.
    It provides measured and
    synthetic HRTF resource variants, scanned and synthetic mesh variants, the
    official metadata table, dataset-level subject exclusions, and download
    metadata for metadata, HRTF, and mesh resources.

    Attributes
    ----------
    name : str
        Public dataset name, ``SONICOM``.
    subject_ids : tuple of str
        SONICOM subject identifiers ``P0001`` through ``P0400``.
    excluded_subject_ids : tuple of str
        Dataset-level exclusions applied before resource scanning and split
        planning.
    hrtf : HRTFConfig
        Measured and synthetic SONICOM HRTF templates with sample-rate and
        version selectors.
    mesh : MeshConfig
        Scanned and synthetic SONICOM mesh templates with version selectors.
    metadata : MetadataConfig
        Official SONICOM metadata table configuration.
    download : DownloadConfig
        Official SONICOM download base URL, downloadable resource groups, and
        checksum map.

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
