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
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """Dataset interface for local or downloadable SONICOM resources.

        ``SONICOM`` turns SONICOM HRTF, mesh, and metadata layouts into the shared
        ``BaseDataset`` API. The constructor optionally downloads selected resources,
        passes explicit HRTF and mesh variants into the build pipeline, and lets
        specs decide which resource families participate in subject intersection and
        sample extraction.

        ``SONICOM`` is a concrete ``BaseDataset`` implementation for the SONICOM
        dataset. It exposes measured and synthetic HRTF resource variants, selected
        sample-rate/version combinations, scanned or synthetic meshes, and metadata.
        Specs passed through ``inputs`` and ``target`` determine which resources are
        required and how each sample is extracted.

        Download selection is independent from dataset construction selection.
        ``download_resources``, ``download_hrtf_variant``, and ``download_mesh_variant``
        control which official files are downloaded. ``dataset_hrtf_variant`` and
        ``dataset_mesh_variant`` control which local files are scanned and loaded
        after the download step. The constructor does not infer download resources
        from ``inputs`` or ``target`` and does not copy dataset variants into download
        variants.

        Parameters
        ----------
        root : str or Path
            Local SONICOM dataset root.
        dataset_hrtf_variant : dict or str
            SONICOM HRTF variant used for dataset construction. Full SONICOM HRTF
            variants use ``type``, ``sample_rate``, and ``version`` keys.
        dataset_mesh_variant : dict or str
            SONICOM mesh variant used for dataset construction. Full SONICOM mesh
            variants use ``type`` and ``version`` keys.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to loaded HRTFs before spec extraction.
        download : bool, default=False
            If ``True``, downloads selected official SONICOM resources before dataset
            construction.
        download_resources : str or sequence of str, default='hrtf'
            Official resources requested for download. This value is not inferred
            from ``inputs`` or ``target``.
        download_hrtf_variant : dict, str, or None
            HRTF variant values requested for download. This value is independent
            from ``dataset_hrtf_variant``.
        download_mesh_variant : dict, str, or None
            Mesh variant values requested for download. This value is independent
            from ``dataset_mesh_variant``.
        exclude_subject_ids : str, int, sequence, or None, default=None
            SONICOM subjects excluded before scanning and splitting.
        inputs : spec, sequence of specs, or None, default=None
            Specs exposed under ``sample['inputs']``.
        target : spec, sequence of specs, or None, default=None
            Specs exposed under ``sample['target']``.
        split : {'all', 'train', 'validation', 'test'}, default='all'
            Subject split used by this dataset instance.
        split_ratio : tuple of float, default=(0.8, 0.1, 0.1)
            Train, validation, and test split ratios.
        split_seed : int, default=0
            Random seed used for deterministic split assignment.
        verbose : bool, default=False
            If ``True``, prints resource and dataset summaries. Download summaries print
            whenever files are downloaded.

        Returns
        -------
        SONICOM Dataset object supporting indexed sample extraction and subject HRTF
        loading.

        Examples
        --------
        >>> from hrtfpykit.datasets import SONICOM
        >>> from hrtfpykit.datasets.specs import HRTFSpec, MetadataSpec
        >>> dataset = SONICOM(
        ...     root="datasets/sonicom",
        ...     inputs=[HRTFSpec(), MetadataSpec()],
        ...     dataset_hrtf_variant={
        ...         "type": "measured",
        ...         "sample_rate": 44100,
        ...         "version": "FreeFieldComp",
        ...     },
        ... )
        >>> sample = dataset[0]
        """
        if download:
            downloaded, download_report = BaseDownload(
                config=SONICOMConfig,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
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
