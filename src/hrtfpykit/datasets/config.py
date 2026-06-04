from dataclasses import dataclass, field
from typing import ClassVar, cast

from .checksums import ARI_CHECKSUMS, HUTUBS_CHECKSUMS, SONICOM_CHECKSUMS, TUBERLIN_HUTUBS_CHECKSUMS


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
    local_path_patterns: tuple[str, ...] = ()
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
    local_path_patterns : tuple of str
        Additional local table paths accepted by the resource scanner. The
        configured path remains the fallback path reported when no candidate
        exists.

    Notes
    -----
    The config only describes the default official table. Individual
    :class:`~hrtfpykit.datasets.AnthropometrySpec` objects can still provide path or extension overrides.

    """

    path: str
    left_prefix: str
    right_prefix: str
    extensions: tuple[str, ...] = (".csv", ".mat")
    local_path_patterns: tuple[str, ...] = ()


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
    local_path_patterns : tuple of str
        Additional local table paths accepted by the resource scanner. The
        configured path remains the fallback path reported when no candidate
        exists.

    Notes
    -----
    The config only describes the default official table. Individual
    :class:`~hrtfpykit.datasets.MetadataSpec` objects can provide path or extension overrides.

    """

    path: str
    extensions: tuple[str, ...] = (".csv", ".mat")
    local_path_patterns: tuple[str, ...] = ()


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
class EcosystemCatalogRule:
    database_key: str
    filename_regex: str
    relative_path_pattern: str
    checksum_key: str = "filename"
    subject_id_field: str | None = "Dataset Name"
    hrtf_type: str | None = None
    mesh_type: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class DownloadServerConfig:
    """Describe one official download source for a dataset family.

    A dataset can expose multiple official sources for the same logical
    resources. For example, SONICOM can use the Imperial transfer server or the
    SONICOM ecosystem, while HUTUBS can use SOFAcoustics or TU Berlin
    DepositOnce archives. This config keeps the server-specific download
    metadata separate from local resource scanning.

    Attributes
    ----------
    base_url : str
        Server root used by the downloader. Direct path-pattern downloaders join
        this URL with a resource relative path. Server-specific downloaders can
        still use absolute URLs stored in database_urls or archives.
    available_resources : tuple of str
        Logical resources the server can provide, such as ``hrtf``, ``mesh``,
        ``anthropometry``, or ``metadata``.
    download_exclude_subject_ids : tuple of str
        Subject IDs excluded by this download server before user-provided
        download exclusions are applied. These exclusions affect download
        planning only.
    checksums : dict or None
        Expected file checksums for this server. Keys are server-relative
        download identities, not necessarily the only local paths accepted by
        the resource scanner.
    resource_base_urls : dict[str, str] or None
        Optional per-resource URL roots for one-off resources whose official
        host differs from base_url.
    database_urls : dict[str, str or tuple[str, ...]]
        Server-specific database JSON URLs used by downloaders that discover
        file URLs from a remote listing.
    archives : dict[str, tuple[dict[str, str], ...]]
        Archive resources used by archive-based downloaders. Each archive entry
        stores at least a name and URL.
    supports_filter : dict[str, bool]
        Declares whether this server can filter downloads by resource selector.
        A False value means the downloader may need to fetch a larger archive or
        complete resource group even when a narrower dataset selection is used.

    Notes
    -----
    Local scanning is controlled by resource configs such as
    :class:`ResourceTypeConfig`, :class:`AnthropometryConfig`, and
    :class:`MetadataConfig`. DownloadServerConfig only describes where official
    files come from and how the downloader should identify them.

    """

    base_url: str
    available_resources: tuple[str, ...]
    download_exclude_subject_ids: tuple[str, ...] = ()
    checksums: dict[str, object] | None = None
    resource_base_urls: dict[str, str] | None = None
    database_urls: dict[str, str | tuple[str, ...]] = field(default_factory=dict)
    catalog_rules: dict[str, tuple[EcosystemCatalogRule, ...]] = field(default_factory=dict)
    archives: dict[str, tuple[dict[str, str], ...]] = field(default_factory=dict)
    supports_filter: dict[str, bool] = field(default_factory=dict)


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
    download_servers : dict[str, DownloadServerConfig] or None
        Named official download sources available for this dataset.

    Notes
    -----
    None resource fields mean the dataset family does not declare that
    resource type. Specs requesting an undeclared resource are rejected by the
    dataset construction workflow or resource scanner.

    """

    name: str
    subject_ids: tuple[str, ...]
    hrtf: HRTFConfig | None = None
    mesh: MeshConfig | None = None
    anthropometry: AnthropometryConfig | None = None
    metadata: MetadataConfig | None = None
    image: ImageConfig | None = None
    video: VideoConfig | None = None
    download_servers: dict[str, DownloadServerConfig] | None = None


@dataclass(frozen=True)
class ARIConfig(DatasetConfig):
    """Built-in configuration for the ARI HRTF database.

    :class:`~hrtfpykit.datasets.config.ARIConfig` declares the ARI subject IDs,
    official HRTF SOFA paths, ARI CSV resource paths, download base URLs,
    and SHA-256 checksums used by
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
    anthropometry : AnthropometryConfig or None
        ARI anthropometry CSV resource configuration.
    metadata : MetadataConfig or None
        ARI metadata CSV resource configuration.
    download_servers : dict[str, DownloadServerConfig] or None
        Official ARI download sources for HRTF, anthropometry, and metadata
        resources.

    Notes
    -----
    The official ARI HRTF files are distributed in b, c, and d filename groups.
    The files included in this configuration are treated as one compatible ARI
    HRTF collection because they share the same source grid, IR shape, and
    sample rate. Workflows that need a narrower subset can exclude subject IDs
    when constructing :class:`~hrtfpykit.datasets.ARI`.

    """

    hrtf_paths: ClassVar[dict[str, str]] = dict(
        sorted(
            (
                (filename.split("_", 1)[1][:-len(".sofa")], filename)
                for filename in ARI_CHECKSUMS["hrtf"]
                if filename.startswith("hrtf ") and filename.endswith(".sofa")
            ),
            key=lambda item: int(item[0][2:]),
        )
    )
    anthropometry_metadata_base_url: ClassVar[str] = (
        "https://raw.githubusercontent.com/"
        "ArielAlvarez-Martinez/ari_anthropometry_and_metadata/v1.0"
    )

    name: str = "ARI"
    subject_ids: tuple[str, ...] = tuple(hrtf_paths)
    hrtf: HRTFConfig | None = HRTFConfig(
        types={
            "hrtf": ResourceTypeConfig(
                path_pattern=hrtf_paths,
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/hrtf/{filename}",
                ),
            ),
        },
    )
    anthropometry: AnthropometryConfig | None = AnthropometryConfig(
        path="anthro.csv",
        left_prefix="L_",
        right_prefix="R_",
        extensions=(".csv",),
        local_path_patterns=(
            "anthropometry/anthro.csv",
            "anthropometry/*.csv",
            "anthro/anthro.csv",
            "anthro/*.csv",
        ),
    )
    metadata: MetadataConfig | None = MetadataConfig(
        path="metadata.csv",
        extensions=(".csv",),
        local_path_patterns=(
            "metadata/metadata.csv",
            "metadata/*.csv",
        ),
    )
    download_servers: dict[str, DownloadServerConfig] | None = field(
        default_factory=lambda: {
            "sofacoustics": DownloadServerConfig(
                base_url="https://sofacoustics.org/data/database/ari",
                available_resources=("hrtf", "anthropometry", "metadata"),
                checksums=cast(dict[str, object], ARI_CHECKSUMS),
                resource_base_urls={
                    "anthropometry": "https://raw.githubusercontent.com/ArielAlvarez-Martinez/ari_anthropometry_and_metadata/v1.0",
                    "metadata": "https://raw.githubusercontent.com/ArielAlvarez-Martinez/ari_anthropometry_and_metadata/v1.0",
                },
                supports_filter={
                    "subject": True,
                    "resource": True,
                    "hrtf_variant": True,
                    "mesh_variant": False,
                },
            ),
            "sonicom-ecosystem": DownloadServerConfig(
                base_url="https://ecosystem.sonicom.eu",
                available_resources=("hrtf",),
                checksums=cast(dict[str, object], ARI_CHECKSUMS),
                database_urls={
                    "hrtf": (
                        "https://ecosystem.sonicom.eu/databases/14/download?type=json",
                        "https://ecosystem.sonicom.eu/databases/16/download?type=json",
                        "https://ecosystem.sonicom.eu/databases/18/download?type=json",
                        "https://ecosystem.sonicom.eu/databases/63/download?type=json",
                        "https://ecosystem.sonicom.eu/databases/64/download?type=json",
                    ),
                },
                catalog_rules={
                    "hrtf": (
                        EcosystemCatalogRule(
                            database_key="hrtf",
                            filename_regex=r"^hrtf [bcd]_(?P<subject_id>.+)\.sofa$",
                            relative_path_pattern="{filename}",
                            subject_id_field=None,
                            hrtf_type="hrtf",
                        ),
                    ),
                },
                supports_filter={
                    "subject": True,
                    "resource": True,
                    "hrtf_variant": True,
                    "mesh_variant": False,
                },
            ),
        }
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
    download_servers : dict[str, DownloadServerConfig]
        Official HUTUBS download sources, downloadable resource groups, and
        checksum maps.

    """

    name: str = "HUTUBS"
    subject_ids: tuple[str, ...] = tuple(f"pp{index}" for index in range(1, 97))
    hrtf: HRTFConfig | None = HRTFConfig(
        types={
            "measured": ResourceTypeConfig(
                path_pattern="{subject_id}_HRIRs_measured.sofa",
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/hrtf/measured/{filename}",
                ),
            ),
            "simulated": ResourceTypeConfig(
                path_pattern="{subject_id}_HRIRs_simulated.sofa",
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/hrtf/simulated/{filename}",
                ),
            ),
        },
    )
    mesh: MeshConfig | None = MeshConfig(
        types={
            "default": ResourceTypeConfig(
                path_pattern="{subject_id}_3DheadMesh.ply",
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/mesh/{filename}",
                    "{subject_id}/mesh/default/{filename}",
                ),
            ),
        },
        subject_ids=tuple(
            sorted(
                (filename.split("_", 1)[0] for filename in cast(dict[str, str], HUTUBS_CHECKSUMS["mesh"])),
                key=lambda subject_id: int(subject_id[2:]),
            )
        ),
    )
    anthropometry: AnthropometryConfig | None = AnthropometryConfig(
        path="AntrhopometricMeasures.csv",
        left_prefix="L_",
        right_prefix="R_",
        local_path_patterns=(
            "anthropometry/AntrhopometricMeasures.csv",
            "anthropometry/*.csv",
            "anthro/AntrhopometricMeasures.csv",
            "anthro/*.csv",
        ),
    )
    image: ImageConfig | None = ImageConfig()
    video: VideoConfig | None = VideoConfig()
    download_servers: dict[str, DownloadServerConfig] | None = field(
        default_factory=lambda: {
            "sofacoustics": DownloadServerConfig(
                base_url="https://sofacoustics.org/data/database/hutubs",
                available_resources=("hrtf", "mesh", "anthropometry"),
                checksums=HUTUBS_CHECKSUMS,
                supports_filter={
                    "subject": True,
                    "resource": True,
                    "hrtf_variant": True,
                    "mesh_variant": True,
                },
            ),
            "tu-berlin": DownloadServerConfig(
                base_url="https://depositonce.tu-berlin.de/bitstreams",
                available_resources=("hrtf", "mesh", "anthropometry"),
                checksums=cast(dict[str, object], TUBERLIN_HUTUBS_CHECKSUMS),
                archives={
                    "hrtf": (
                        {
                            "name": "HRIRs.zip",
                            "url": "https://depositonce.tu-berlin.de/bitstreams/9f8b8874-c567-43fa-9085-eac010599a66/download",
                        },
                    ),
                    "mesh": (
                        {
                            "name": "3D head meshes.zip",
                            "url": "https://depositonce.tu-berlin.de/bitstreams/7153f32e-f630-4b5e-9674-445f9797887d/download",
                        },
                    ),
                    "anthropometry": (
                        {
                            "name": "Antrhopometric measures.zip",
                            "url": "https://depositonce.tu-berlin.de/bitstreams/21612c81-9b16-477f-af01-0eb775acb253/download",
                        },
                    ),
                },
                supports_filter={
                    "subject": False,
                    "resource": True,
                    "hrtf_variant": False,
                    "mesh_variant": False,
                },
            ),
        }
    )

@dataclass(frozen=True)
class SONICOMConfig(DatasetConfig):
    """Built-in configuration for the SONICOM dataset family.

    :class:`~hrtfpykit.datasets.config.SONICOMConfig` declares the subject IDs
    and official resource layout used by :class:`~hrtfpykit.datasets.SONICOM`.
    It provides measured and
    synthetic HRTF resource variants, scanned and synthetic mesh variants, the
    official metadata table, and download metadata for metadata, HRTF, and mesh
    resources.

    Attributes
    ----------
    name : str
        Public dataset name, ``SONICOM``.
    subject_ids : tuple of str
        SONICOM subject identifiers ``P0001`` through ``P0405``.
    hrtf : HRTFConfig
        Measured and synthetic SONICOM HRTF templates with sample-rate and
        version selectors.
    mesh : MeshConfig
        Scanned and synthetic SONICOM mesh templates with version selectors.
    metadata : MetadataConfig
        Official SONICOM metadata table configuration.
    download_servers : dict[str, DownloadServerConfig]
        Official SONICOM download sources, downloadable resource groups, and
        checksum maps.

    """

    name: str = "SONICOM"
    subject_ids: tuple[str, ...] = tuple(f"P{index:04d}" for index in range(1, 406))
    hrtf: HRTFConfig | None = HRTFConfig(
        types={
            "measured": ResourceTypeConfig(
                path_pattern="{subject_id}/HRTF/HRTF/{sample_rate_label}/{subject_id}_{version}_{sample_rate_label}.sofa",
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/hrtf/measured/{filename}",
                    "{subject_id}/hrtf/measured/{sample_rate}/{filename}",
                    "{subject_id}/hrtf/measured/{sample_rate_label}/{filename}",
                    "{subject_id}/hrtf/measured/{version}/{sample_rate}/{filename}",
                    "{subject_id}/hrtf/measured/{version}/{sample_rate_label}/{filename}",
                ),
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
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/hrtf/synthetic/{filename}",
                    "{subject_id}/hrtf/synthetic/{sample_rate}/{filename}",
                ),
                sample_rates=(44100, 48000),
                versions=("generic",),
            ),
        },
    )
    mesh: MeshConfig | None = MeshConfig(
        types={
            "scanned": ResourceTypeConfig(
                path_pattern="{subject_id}/3DSCAN/{subject_id}{version_label}",
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/mesh/scanned/{filename}",
                    "{subject_id}/mesh/scanned/{version}/{filename}",
                ),
                versions=("raw", "point_cloud", "watertight"),
                version_labels={
                    "raw": ".stl",
                    "point_cloud": "_Project1.asc",
                    "watertight": "_watertight.stl",
                },
            ),
            "synthetic": ResourceTypeConfig(
                path_pattern="{subject_id}/SYNTHETIC_HRTF/{subject_id}_{version}.stl",
                local_path_patterns=(
                    "{subject_id}/{filename}",
                    "{subject_id}/mesh/synthetic/{filename}",
                    "{subject_id}/mesh/synthetic/{version}/{filename}",
                ),
                versions=("preprocessed", "plugged", "graded_left", "graded_right"),
            ),
        },
        extensions=(".stl", ".asc"),
    )
    metadata: MetadataConfig | None = MetadataConfig(
        path="metadata_and_readme/metadata.csv",
        extensions=(".csv",),
        local_path_patterns=(
            "metadata.csv",
            "metadata/metadata.csv",
            "metadata_and_readme/*.csv",
        ),
    )
    download_servers: dict[str, DownloadServerConfig] | None = field(
        default_factory=lambda: {
            "imperial": DownloadServerConfig(
                base_url="https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET",
                available_resources=("metadata", "hrtf", "mesh"),
                download_exclude_subject_ids=(
                    "P0253",
                    "P0258",
                    "P0270",
                    "P0272",
                    "P0275",
                    "P0396",
                ),
                checksums=SONICOM_CHECKSUMS,
                supports_filter={
                    "subject": True,
                    "resource": True,
                    "hrtf_variant": True,
                    "mesh_variant": True,
                },
            ),
            "sonicom-ecosystem": DownloadServerConfig(
                base_url="https://ecosystem.sonicom.eu",
                available_resources=("hrtf", "mesh"),
                checksums=SONICOM_CHECKSUMS,
                database_urls={
                    "measured": "https://ecosystem.sonicom.eu/databases/3/download?type=json",
                    "synthetic": "https://ecosystem.sonicom.eu/databases/20/download?type=json",
                },
                catalog_rules={
                    "hrtf": (
                        EcosystemCatalogRule(
                            database_key="measured",
                            filename_regex=r"^(?P<subject_id>P\d{4})_(?P<version>.+)_(?P<sample_rate_label>\d+kHz)\.sofa$",
                            relative_path_pattern="{subject_id}/HRTF/HRTF/{sample_rate_label}/{filename}",
                            hrtf_type="measured",
                        ),
                        EcosystemCatalogRule(
                            database_key="synthetic",
                            filename_regex=r"^HRIR_SONICOM_(?P<sample_rate>\d+)\.sofa$",
                            relative_path_pattern="{subject_id}/SYNTHETIC_HRTF/{filename}",
                            hrtf_type="synthetic",
                            version="generic",
                        ),
                    ),
                    "mesh": (
                        EcosystemCatalogRule(
                            database_key="synthetic",
                            filename_regex=r"^(?P<subject_id>P\d{4})\.stl(?:\.stl)?$",
                            relative_path_pattern="{subject_id}/3DSCAN/{subject_id}.stl",
                            mesh_type="scanned",
                            version="raw",
                        ),
                        EcosystemCatalogRule(
                            database_key="synthetic",
                            filename_regex=r"^(?P<subject_id>P\d{4})_watertight\.stl$",
                            relative_path_pattern="{subject_id}/3DSCAN/{filename}",
                            mesh_type="scanned",
                            version="watertight",
                        ),
                        EcosystemCatalogRule(
                            database_key="synthetic",
                            filename_regex=r"^(?P<subject_id>P\d{4})_(?P<version>.+)\.stl$",
                            relative_path_pattern="{subject_id}/SYNTHETIC_HRTF/{filename}",
                            mesh_type="synthetic",
                        ),
                    ),
                },
                supports_filter={
                    "subject": True,
                    "resource": True,
                    "hrtf_variant": True,
                    "mesh_variant": True,
                },
            ),
        }
    )
