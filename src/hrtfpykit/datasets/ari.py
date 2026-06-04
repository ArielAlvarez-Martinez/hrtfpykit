from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from .base import BaseDataset
from .config import ARIConfig
from .download import SOFAcousticsDownload, SONICOMEcosystemDownload
from .sanitize import sanitize_grouped_by
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
        download_server: str = "sofacoustics",
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
        """Build an ARI dataset instance.

        :class:`~hrtfpykit.datasets.ARI` resolves the local ARI resources
        declared by :class:`~hrtfpykit.datasets.config.ARIConfig`, including
        HRTF SOFA files, anthropometry CSV data, and metadata CSV data. It
        applies subject exclusions and split selection, and returns samples
        defined by the requested input and target specs. When ``download=True``,
        ARI can download resources from SOFAcoustics or from the SONICOM
        ecosystem, depending on ``download_server``. Download selection is
        independent from dataset construction: ``download_resources`` controls
        which official files are fetched, while ``inputs`` and ``target``
        control which local resources are required for samples.

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
        download_resources : {``hrtf``, ``anthropometry``, ``metadata``, ``all``} or sequence of str, default=``hrtf``
            Official resources requested for download. ``sofacoustics`` supports
            ``hrtf``, ``anthropometry``, and ``metadata``. ``sonicom-ecosystem``
            supports only ``hrtf`` for ARI. Passing ``all`` requests every
            resource provided by the selected download server.
        download_server : {``sofacoustics``, ``sonicom-ecosystem``}, default=``sofacoustics``
            Official server used when ``download=True``. ``sofacoustics``
            downloads configured ARI files directly from SOFAcoustics, with
            anthropometry and metadata fetched from their configured companion
            URLs. ``sonicom-ecosystem`` reads the SONICOM ecosystem database JSON
            entries and downloads ARI HRTF files listed there.
        verify_checksum : bool, default=True
            Whether official SHA-256 checksums are verified during resource
            download. Keeping this enabled is the recommended behavior. Set it to
            False only when checksum verification should be skipped; file
            existence and non empty checks still run.
        exclude_subject_ids : str, int, sequence, or None, default=None
            ARI subjects excluded before scanning and splitting.
        download_exclude_subject_ids : str, int, sequence, or None, default=None
            ARI subjects excluded only from the download request. This does not
            change dataset construction; use exclude_subject_ids to exclude
            subjects from scanning, splitting, and samples.
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
        config = ARIConfig()
        if download:
            download_servers = config.download_servers
            if download_servers is None:
                raise ValueError("ARI does not define downloadable resources")
            selected_download_server = download_server
            if selected_download_server not in download_servers:
                raise ValueError(
                    f"ARI download_server accepts {tuple(download_servers)}; got {download_server!r}"
                )
            downloader_class = SONICOMEcosystemDownload if selected_download_server == "sonicom-ecosystem" else SOFAcousticsDownload
            downloaded, download_report = downloader_class(
                config=config,
                root=root,
                excluded_subject_ids=download_exclude_subject_ids,
                verify_checksum=verify_checksum,
                download_server=selected_download_server,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=None,
                download_mesh_variant=None,
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
            dataset_hrtf_variant=None,
            dataset_mesh_variant=None,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            verbose=verbose,
        )
        self._state.anthropometry_value_selector = self._select_anthropometry_value

    def _select_anthropometry_value(
        self,
        spec: AnthropometrySpec,
        row: dict[str, str | int | None],
        value: object,
    ) -> object:
        """Select ARI anthropometry fields for an optional ear context.

        ARI anthropometry columns store shared measurements with names such as
        ``x1`` and ear-specific measurements with left and right prefixes such as
        ``L_a1`` and ``R_a1``. This selector filters the loaded subject row when
        an ear is requested by :class:`~hrtfpykit.datasets.AnthropometrySpec` or
        by an ear-indexed row. The selected ear keeps its matching prefixed
        fields and the shared non-ear fields.

        Parameters
        ----------
        spec : AnthropometrySpec
            Anthropometry spec requesting the value.
        row : dict
            Current dataset row context.
        value : object
            Loaded anthropometry value selected by the generic table resolver.

        Returns
        -------
        object
            Filtered value containing the requested ARI ear fields and shared
            measurements. Non-dictionary values are returned unchanged.

        """
        if not isinstance(value, dict):
            return value
        grouped_by = sanitize_grouped_by(spec.grouped_by)

        selected_ear = str(spec.ear).strip().lower() if spec.ear else None
        if selected_ear == "both" or selected_ear == "":
            selected_ear = None

        if "ear" in grouped_by:
            if selected_ear not in {"left", "right"}:
                row_ear = row.get("ear")
                if row_ear is not None and str(row_ear).strip().lower() in {"left", "right"}:
                    selected_ear = str(row_ear).strip().lower()

        if selected_ear is None:
            return value

        config = self._state.config.anthropometry if self._state.config is not None else None
        left_prefix = "L_"
        right_prefix = "R_"
        if config is not None:
            left_prefix = str(config.left_prefix)
            right_prefix = str(config.right_prefix)
        target_prefix = left_prefix if selected_ear == "left" else right_prefix

        return {
            key: feature
            for key, feature in value.items()
            if str(key).startswith(target_prefix)
            or (
                not str(key).startswith(left_prefix)
                and not str(key).startswith(right_prefix)
            )
        }
