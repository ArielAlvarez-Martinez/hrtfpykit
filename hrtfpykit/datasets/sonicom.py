from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Callable

from .base import BaseDataset
from .config import SONICOMConfig
from .download import BaseDownload
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
        verify_checksum: bool = True,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """Dataset interface for local or downloadable SONICOM resources.

        :class:`~hrtfpykit.datasets.SONICOM` turns SONICOM HRTF, mesh, and
        metadata layouts into the shared
        :class:`~hrtfpykit.datasets.base.BaseDataset` API. It resolves measured
        and synthetic HRTF variants, scanned or synthetic mesh variants, subject
        metadata, subject exclusions, and split selection before exposing samples
        through the shared integer-indexed dataset interface.

        Samples are driven by input and target specs. Acoustic specs
        load a subject HRTF with :func:`~hrtfpykit.hrtf.load_hrtf` and extract
        time-domain, frequency-domain, ITD, ILD, or spherical-harmonic values.
        Resource specs can add mesh and metadata values to the same sample.
        Subjects missing any required resource family are removed before row
        construction.

        Download selection is independent from dataset construction selection.
        download_resources, download_hrtf_variant, and download_mesh_variant
        control which official files are downloaded. dataset_hrtf_variant and
        dataset_mesh_variant control which local files are scanned and loaded
        after the download step. The dataset does not infer download resources
        from inputs or target and does not copy dataset variants into download
        variants.

        Parameters
        ----------
        root : str or Path
            Local SONICOM dataset root.
        dataset_hrtf_variant : dict or str
            SONICOM HRTF variant used for dataset construction. Full SONICOM HRTF
            variants use type, sample_rate, and version keys.
        dataset_mesh_variant : dict or str
            SONICOM mesh variant used for dataset construction. Full SONICOM mesh
            variants use type and version keys.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to loaded HRTFs before spec extraction.
        download : bool, default=False
            If True, downloads selected official SONICOM resources before dataset
            construction.
        download_resources : str or sequence of str, default=``hrtf``
            Official resources requested for download. This value is not inferred
            from inputs or target.
        download_hrtf_variant : dict, str, or None
            HRTF variant values requested for download. This value is independent
            from dataset_hrtf_variant.
        download_mesh_variant : dict, str, or None
            Mesh variant values requested for download. This value is independent
            from dataset_mesh_variant.
        verify_checksum : bool, default=True
            Whether official SHA-256 checksums are verified during resource
            download. Keeping this enabled is the recommended behavior. Set it to
            False only when you intentionally want to skip checksum verification;
            file existence, non-empty checks, and archive integrity checks still
            run.
        exclude_subject_ids : str, int, sequence, or None, default=None
            SONICOM subjects excluded before scanning and splitting.
        inputs : spec, sequence of specs, or None, default=None
            Specs exposed under sample inputs.
        target : spec, sequence of specs, or None, default=None
            Specs exposed under sample targets.
        split : {``all``, ``train``, ``validation``, ``test``}, default=``all``
            Subject split used by this dataset instance.
        split_ratio : tuple of float, default=(0.8, 0.1, 0.1)
            Train, validation, and test split ratios.
        split_seed : int, default=0
            Random seed used for deterministic split assignment.
        verbose : bool, default=False
            If True, prints resource and dataset summaries. Download summaries print
            whenever files are downloaded.

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
        if download:
            downloaded, download_report = BaseDownload(
                config=SONICOMConfig,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
                verify_checksum=verify_checksum,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=download_hrtf_variant,
                download_mesh_variant=download_mesh_variant,
            )
            if downloaded:
                print(download_report)

        super().__init__(
            root=root,
            config=SONICOMConfig,
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
