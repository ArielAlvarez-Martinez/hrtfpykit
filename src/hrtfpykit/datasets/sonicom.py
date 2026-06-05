from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Callable

from .base import BaseDataset
from .config import SONICOMConfig
from .download import ImperialDownload, SONICOMEcosystemDownload
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ILDSpec,
    ITDSpec,
    MetadataSpec,
    MeshSpec,
    SHSpec,
    VideoSpec,
)

class SONICOM(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        dataset_hrtf_variant: str | Mapping[str, object] = {"type": "measured", "sample_rate": 44100, "version": "FreeFieldComp"},
        dataset_mesh_variant: str | Mapping[str, object] =  {"type": "scanned", "version": "watertight"},
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "hrtf",
        download_hrtf_variant: str | Mapping[str, object] | None = {"type": "measured", "sample_rate": 44100, "version": "FreeFieldComp"},
        download_mesh_variant: str | Mapping[str, object] | None =  {"type": "scanned", "version": "watertight"},
        download_server: str = "imperial",
        verify_checksum: bool = True,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        download_exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        :class:`~hrtfpykit.datasets.SONICOM` resolves the local SONICOM
        resources declared by :class:`~hrtfpykit.datasets.config.SONICOMConfig`.
        It exposes measured and synthetic HRTF SOFA variants, scanned and
        synthetic mesh variants, and subject metadata through the shared
        integer-indexed dataset interface. The dataset applies subject
        exclusions and split selection after resource scanning so samples only
        use subjects with every required local resource family.

        Samples are driven by input and target specs. Acoustic specs load a
        subject HRTF with :func:`~hrtfpykit.hrtf.load_hrtf`. If
        ``dataset_hrtf_transform`` is provided, it is applied to that loaded HRTF
        first. Acoustic specs then operate on the dataset-level HRTF version,
        optionally apply their own HRTF transform, and finally extract
        time-domain values, frequency-domain values, ITD, ILD, or
        spherical-harmonic coefficients. Resource specs can add mesh and metadata
        values to the same sample. Subjects missing any required resource family
        are removed before row construction.

        Download selection is independent from dataset construction selection.
        ``download_resources``, ``download_hrtf_variant``, and
        ``download_mesh_variant`` control which official files are downloaded.
        ``dataset_hrtf_variant`` and ``dataset_mesh_variant`` control which local
        files are scanned and loaded after the download step. The dataset does
        not infer download resources from inputs or target and does not copy
        dataset variants into download variants. SONICOM can download from
        ``imperial`` or ``sonicom-ecosystem``. ``imperial`` provides configured
        metadata, HRTF, and mesh files through direct paths.
        ``sonicom-ecosystem`` provides HRTF and mesh files listed by ecosystem
        database catalogs; it does not provide the SONICOM metadata table through
        this downloader.

        Users can also download or prepare files manually and copy them under
        ``root``. Measured SONICOM HRTFs are accepted in the official layout such
        as ``P0001/HRTF/HRTF/44kHz/P0001_FreeFieldComp_44kHz.sofa`` and in
        semantic alternatives such as ``P0001/hrtf/measured/{filename}``,
        ``P0001/hrtf/measured/44100/{filename}``,
        ``P0001/hrtf/measured/44kHz/{filename}``, or
        ``P0001/hrtf/measured/FreeFieldComp/44kHz/{filename}``. Synthetic HRTFs
        are accepted in ``P0001/SYNTHETIC_HRTF/HRIR_SONICOM_44100.sofa`` or
        ``P0001/hrtf/synthetic/44100/HRIR_SONICOM_44100.sofa`` style layouts.
        Scanned and synthetic meshes are accepted in their official folders or
        under ``P0001/mesh/scanned/...`` and ``P0001/mesh/synthetic/...``.
        Metadata is accepted as ``metadata_and_readme/metadata.csv``,
        ``metadata_and_readme/*.csv``, ``metadata/metadata.csv``, or
        ``metadata.csv``.

        Parameters
        ----------
        root : str or Path
            Local SONICOM dataset root.
        dataset_hrtf_variant : dict or str
            SONICOM HRTF variant used for dataset construction. Valid HRTF types
            are ``measured`` and ``synthetic``. ``measured`` supports sample
            rates ``44100``, ``48000``, and ``96000`` with versions ``Raw``,
            ``Raw_NoITD``, ``Windowed``, ``Windowed_NoITD``, ``FreeFieldComp``,
            ``FreeFieldComp_NoITD``, ``FreeFieldCompMinPhase``, and
            ``FreeFieldCompMinPhase_NoITD``. ``synthetic`` supports sample rates
            ``44100`` and ``48000`` with version ``generic``. A full dict uses
            ``{"type": ..., "sample_rate": ..., "version": ...}``; a string
            selects an HRTF type only when the remaining axes can be resolved by
            the dataset configuration.
        dataset_mesh_variant : dict or str
            SONICOM mesh variant used for dataset construction. Valid mesh types
            are ``scanned`` and ``synthetic``. ``scanned`` supports versions
            ``raw``, ``point_cloud``, and ``watertight``. ``synthetic`` supports
            versions ``preprocessed``, ``plugged``, ``graded_left``, and
            ``graded_right``. A full dict uses ``{"type": ..., "version":
            ...}``.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to every loaded HRTF before any acoustic
            spec is evaluated. Spec-level HRTF transforms are applied after this
            dataset-level transform and before value extraction or derived cue
            calculation.
        download : bool, default=False
            If True, downloads selected official SONICOM resources before dataset
            construction.
        download_resources : {``metadata``, ``hrtf``, ``mesh``, ``all``} or sequence of str, default=``hrtf``
            Official resources requested for download. This value is not inferred
            from inputs or target. ``imperial`` supports ``metadata``, ``hrtf``,
            and ``mesh``. ``sonicom-ecosystem`` supports ``hrtf`` and ``mesh``.
            Passing ``all`` requests every resource provided by the selected
            download server.
        download_hrtf_variant : dict, str, or None
            HRTF variant values requested for download. The valid HRTF type,
            sample-rate, and version combinations are the same as
            ``dataset_hrtf_variant``: measured HRTFs use sample rates ``44100``,
            ``48000``, or ``96000`` with measured versions, while synthetic HRTFs
            use sample rates ``44100`` or ``48000`` with version ``generic``.
            This value is independent from ``dataset_hrtf_variant`` and is only
            used when HRTF resources are requested.
        download_mesh_variant : dict, str, or None
            Mesh variant values requested for download. The valid mesh type and
            version combinations are the same as ``dataset_mesh_variant``:
            ``scanned`` with ``raw``, ``point_cloud``, or ``watertight``;
            ``synthetic`` with ``preprocessed``, ``plugged``, ``graded_left``, or
            ``graded_right``. This value is independent from
            ``dataset_mesh_variant`` and is only used when mesh is requested.
        download_server : {``imperial``, ``sonicom-ecosystem``}, default=``imperial``
            Official server used when ``download=True``. ``imperial`` downloads
            direct files from the original Imperial transfer server and supports
            metadata, HRTF, mesh, subject, and variant filtering. ``sonicom-
            ecosystem`` reads configured ecosystem database JSON endpoints and
            downloads only HRTF and mesh files available in those entries.
        verify_checksum : bool, default=True
            Whether official SHA-256 checksums are verified during resource
            download. Keeping this enabled is the recommended behavior. Set it to
            False only when you intentionally want to skip checksum verification;
            file existence, non-empty checks, and archive integrity checks still
            run.
        exclude_subject_ids : str, int, sequence, or None, default=None
            SONICOM subjects excluded before scanning and splitting.
        download_exclude_subject_ids : str, int, sequence, or None, default=None
            SONICOM subjects excluded only from the download request. This does
            not change dataset construction; use exclude_subject_ids to exclude
            subjects from scanning, splitting, and samples.
        inputs : spec, sequence of specs, or None, default=None
            Specs resolved under ``sample["inputs"]``. None builds samples
            without an inputs group unless target specs are provided.
        target : spec, sequence of specs, or None, default=None
            Specs resolved under ``sample["target"]``. None builds samples
            without a target group unless input specs are provided.
        split : {``all``, ``train``, ``validation``, ``test``}, default=``all``
            Subject split used by this dataset instance.
        split_ratio : tuple of float, default=(0.8, 0.1, 0.1)
            Train, validation, and test split ratios.
        split_seed : int, default=0
            Random seed used for deterministic split assignment.
        verbose : bool, default=False
            If True, prints resource and dataset summaries. Download summaries
            print whenever files are downloaded.

        Returns
        -------
        SONICOM
            Dataset object supporting indexed sample extraction and subject HRTF
            loading.

        Examples
        --------
        Build a training split from measured 44.1 kHz FreeFieldComp HRTFs,
        scanned watertight meshes, and the SONICOM metadata table:

        >>> from hrtfpykit.datasets import HRTFSpec, MeshSpec, MetadataSpec, SONICOM
        >>> dataset = SONICOM(
        ...     root="datasets/sonicom",
        ...     dataset_hrtf_variant={
        ...         "type": "measured",
        ...         "sample_rate": 44100,
        ...         "version": "FreeFieldComp",
        ...     },
        ...     dataset_mesh_variant={
        ...         "type": "scanned",
        ...         "version": "watertight",
        ...     },
        ...     inputs=[
        ...         HRTFSpec(
        ...             domain="frequency",
        ...             signal="tf_magnitude_db",
        ...             index_by=("subject", "position", "ear"),
        ...             ears="both",
        ...             position_index=True,
        ...             ear_index=True,
        ...             name="magnitude_db",
        ...         ),
        ...         MeshSpec(name="head_mesh"),
        ...         MetadataSpec(name="subject_metadata"),
        ...     ],
        ...     split="train",
        ...     split_ratio=(0.8, 0.1, 0.1),
        ...     split_seed=42,
        ... )
        >>> sample = dataset[0]

        """
        config = SONICOMConfig()
        if download:
            download_servers = config.download_servers
            if download_servers is None:
                raise ValueError("SONICOM does not define downloadable resources")
            selected_download_server = download_server
            if selected_download_server not in download_servers:
                raise ValueError(
                    f"SONICOM download_server accepts {tuple(download_servers)}; got {download_server!r}"
                )
            downloader_class = SONICOMEcosystemDownload if selected_download_server == "sonicom-ecosystem" else ImperialDownload
            downloaded, download_report = downloader_class(
                config=config,
                root=root,
                excluded_subject_ids=download_exclude_subject_ids,
                verify_checksum=verify_checksum,
                download_server=selected_download_server,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=download_hrtf_variant,
                download_mesh_variant=download_mesh_variant,
            )
            if downloaded or verbose:
                print(download_report)

        super().__init__(
            root=root,
            config=config,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            dataset_hrtf_variant=dict(dataset_hrtf_variant) if isinstance(dataset_hrtf_variant, Mapping) else dataset_hrtf_variant,
            dataset_mesh_variant=dict(dataset_mesh_variant) if isinstance(dataset_mesh_variant, Mapping) else dataset_mesh_variant,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            verbose=verbose,
        )
