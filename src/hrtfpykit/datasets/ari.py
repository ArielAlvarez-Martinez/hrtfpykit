from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from .base import BaseDataset
from .config import ARIConfig
from .download import BaseDownload
from .specs import (
    AnthropometrySpec,
    HRTFSpec,
    ILDSpec,
    ITDSpec,
    ImageSpec,
    MeshSpec,
    MetadataSpec,
    SHSpec,
    VideoSpec,
)


class ARI(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "hrtf",
        verify_checksum: bool = True,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """Build an ARI dataset instance.

        :class:`~hrtfpykit.datasets.ARI` resolves the local ARI HRTF SOFA
        resources declared by :class:`~hrtfpykit.datasets.config.ARIConfig`,
        applies subject exclusions and split selection, and returns samples
        defined by the requested input and target specs.

        Acoustic specs load one subject HRTF with
        :func:`~hrtfpykit.hrtf.load_hrtf`. When ``dataset_hrtf_transform`` is
        provided, the loaded HRTF is transformed before specs extract IR/TF
        values or calculate derived values such as ITD, ILD, or
        spherical harmonic coefficients.

        Parameters
        ----------
        root : str or Path
            Local ARI dataset root.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to every loaded HRTF before any acoustic
            spec is evaluated. Spec transforms are applied after this dataset
            transform and before value extraction or derived cue calculation.
        download : bool, default=False
            If True, downloads selected official ARI resources before dataset
            construction.
        download_resources : str or sequence of str, default=``hrtf``
            Official resources requested for download. The current ARI
            configuration provides official HRTF SOFA files.
        verify_checksum : bool, default=True
            Whether official SHA-256 checksums are verified during resource
            download. Keeping this enabled is the recommended behavior. Set it to
            False only when checksum verification should be skipped; file
            existence and non empty checks still run.
        exclude_subject_ids : str, int, sequence, or None, default=None
            ARI subjects excluded before scanning and splitting.
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
            If True, prints resource and dataset summaries. Download summaries
            print whenever files are downloaded.

        Returns
        -------
        ARI
            Dataset object supporting indexed sample extraction and subject HRTF
            loading.

        Notes
        -----
        The official ARI HRTF files are distributed in b, c, and d filename
        groups. This class treats the included files as one compatible ARI HRTF
        collection because they share the same source grid, IR shape, and sample
        rate, so they can be used inside the same ARI dataset instance.

        The ARI dataset does not expose a public group selector. To use only a
        specific subset, exclude the subjects outside that subset with
        ``exclude_subject_ids`` before construction.

        Examples
        --------
        Build an ARI dataset that returns full HRIR arrays:

        >>> from hrtfpykit.datasets import ARI, HRTFSpec
        >>> dataset = ARI(
        ...     root="datasets/ari",
        ...     inputs=HRTFSpec(
        ...         domain="time",
        ...         signal="ir",
        ...         index_by=("subject",),
        ...         name="hrir",
        ...     ),
        ... )
        >>> sample = dataset[0]
        >>> sample["inputs"]["hrir"].shape
        (1550, 2, 256)

        """
        if download:
            downloaded, download_report = BaseDownload(
                config=ARIConfig,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
                verify_checksum=verify_checksum,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=None,
                download_mesh_variant=None,
            )
            if downloaded:
                print(download_report)

        super().__init__(
            root=root,
            config=ARIConfig,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            dataset_hrtf_variant=None,
            dataset_mesh_variant=None,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            verbose=verbose,
        )
