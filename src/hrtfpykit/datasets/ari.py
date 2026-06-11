from collections.abc import Mapping, Sequence
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
        dataset_hrtf_variant: str | Mapping[str, object] = "NH",
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "hrtf",
        download_hrtf_variant: str | Mapping[str, object] | None = "NH",
        download_server: str = "sofacoustics",
        verify_checksum: bool = True,
        subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        download_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        download_exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        preload_hrtfs: bool = False,
        check_sofa_against_conventions: bool = False,
        sofa_open: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        :class:`~hrtfpykit.datasets.ARI` resolves the local ARI resources
        declared by :class:`~hrtfpykit.datasets.config.ARIConfig`. It exposes
        the official NH HRTF SOFA collection, ARI anthropometry CSV data, and
        subject metadata CSV data through the shared integer-indexed dataset
        interface. The NH HRTFs are distributed in ``b``, ``c``, and ``d``
        filename groups; ``dataset_hrtf_variant="NH"`` scans the compatible
        configured collection, while a versioned NH variant selects one group.

        Samples are driven by input and target specs. Acoustic specs load one
        subject HRTF with :func:`~hrtfpykit.hrtf.load_hrtf`. If
        ``dataset_hrtf_transform`` is provided, it is applied to that loaded HRTF
        first. Acoustic specs then operate on the dataset-level HRTF version,
        optionally apply their own HRTF transform, and finally extract
        time-domain values, frequency-domain values, ITD, ILD, or
        spherical-harmonic coefficients. Resource specs can add anthropometry and
        metadata values to the same sample. Subjects missing any required
        resource family are removed before row construction.

        Download selection is independent from dataset construction selection.
        ``download_resources`` and ``download_hrtf_variant`` control which
        official files are downloaded. ``dataset_hrtf_variant`` controls which
        local HRTF files are scanned and loaded after the download step. The
        dataset does not infer download resources from inputs or target and does
        not copy ``dataset_hrtf_variant`` into ``download_hrtf_variant``. ARI can
        download from ``sofacoustics`` or ``sonicom-ecosystem``. ``sofacoustics``
        provides configured HRTF, anthropometry, and metadata files.
        ``sonicom-ecosystem`` provides ARI HRTF files listed by ecosystem
        database catalogs.

        Users can also download or prepare files manually and copy them under
        ``root``. ARI HRTFs are accepted as official root-level filenames such as
        ``hrtf b_nh2.sofa`` and as local alternatives such as
        ``nh2/hrtf b_nh2.sofa``, ``nh2/hrtf/hrtf b_nh2.sofa``,
        ``nh2/hrtf/nh/hrtf b_nh2.sofa``, or
        ``nh2/hrtf/nh/b/hrtf b_nh2.sofa``. Anthropometry is accepted as
        ``anthro.csv``, ``anthropometry/anthro.csv``, ``anthropometry/*.csv``,
        ``anthro/anthro.csv``, or ``anthro/*.csv``. Metadata is accepted as
        ``metadata.csv``, ``metadata/metadata.csv``, or ``metadata/*.csv``.

        Parameters
        ----------
        root : str or Path
            Local ARI dataset root.
        dataset_hrtf_variant : {``NH``} or dict, default=``NH``
            ARI HRTF variant used for dataset construction. ``NH`` selects the
            full configured NH collection across the ``b``, ``c``, and ``d``
            filename groups. A dict may use ``{"type": "NH", "version": "b"}``,
            ``{"type": "NH", "version": "c"}``, or ``{"type": "NH",
            "version": "d"}`` to scan only one filename group.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to every loaded HRTF before any acoustic
            spec is evaluated. Spec-level HRTF transforms run after this
            dataset-level transform and before value extraction or derived cue
            calculation.
        download : bool, default=False
            If True, downloads selected official ARI resources before dataset
            construction.
        download_resources : {``hrtf``, ``anthropometry``, ``metadata``, ``all``} or sequence of str, default=``hrtf``
            Official resources requested for download. ``sofacoustics`` supports
            ``hrtf``, ``anthropometry``, and ``metadata``. ``sonicom-ecosystem``
            supports only ``hrtf`` for ARI. Passing ``all`` requests every
            resource provided by the selected download server.
        download_hrtf_variant : {``all``, ``NH``}, dict, or None, default=``NH``
            ARI HRTF variant requested for download. ``all`` downloads every
            configured ARI HRTF family. ``NH`` downloads the full NH collection.
            A dict may use ``{"type": "NH", "version": "b"}``, ``{"type":
            "NH", "version": "c"}``, or ``{"type": "NH", "version": "d"}``
            to download one ARI filename group. This value is independent from
            ``dataset_hrtf_variant``.
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
            existence and non-empty checks still run.
        subject_ids : str, int, sequence, or None, default=None
            Optional subjects used as the initial dataset construction scope.
            Exclusions are applied after this inclusion filter. None uses every
            configured subject.
        exclude_subject_ids : str, int, sequence, or None, default=None
            ARI subjects excluded before scanning and splitting.
        download_subject_ids : str, int, sequence, or None, default=None
            Optional subjects used as the initial download scope. Download
            exclusions are applied after this inclusion filter. None uses every
            configured subject supported by the selected download server.
        download_exclude_subject_ids : str, int, sequence, or None, default=None
            ARI subjects excluded only from the download request. This does not
            change dataset construction; use exclude_subject_ids to exclude
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
        preload_hrtfs : bool, default=False
            If True, load and cache selected subject HRTFs during construction.
            If False, HRTFs load on first access and stay cached until
            :meth:`~hrtfpykit.datasets.base.BaseDataset.clear_cache` is called.
        check_sofa_against_conventions : bool, default=False
            Whether dataset HRTF loading runs SOFA convention checks before
            reading files. False keeps normal dataset construction quiet for
            files with accepted custom SOFA fields.
        sofa_open : bool, default=False
            Whether HRTFs loaded by the dataset keep their backing SOFA netCDF
            datasets open. False closes the handle after arrays and source
            positions are loaded.
        verbose : bool, default=True
            If True, prints the download summary when ``download=True``, then
            prints resource and dataset summaries after construction. If False,
            these summaries are not printed.

        Returns
        -------
        ARI
            Dataset object supporting indexed sample extraction and subject HRTF
            loading.

        Notes
        -----
        The configured ARI NH groups share the same source grid, IR shape, and
        sample rate, so they can be used as one compatible dataset collection or
        selected by filename group when a workflow needs a narrower subset.

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
            _, download_report = downloader_class(
                config=config,
                root=root,
                subject_ids=download_subject_ids,
                excluded_subject_ids=download_exclude_subject_ids,
                verify_checksum=verify_checksum,
                download_server=selected_download_server,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=download_hrtf_variant,
                download_mesh_variant=None,
            )
            if verbose:
                print(download_report)

        super().__init__(
            root=root,
            config=config,
            dataset_hrtf_transform=dataset_hrtf_transform,
            subject_ids=subject_ids,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            dataset_hrtf_variant=dict(dataset_hrtf_variant) if isinstance(dataset_hrtf_variant, Mapping) else dataset_hrtf_variant,
            dataset_mesh_variant=None,
            split=split,
            split_ratio=split_ratio,
            split_seed=split_seed,
            preload_hrtfs=preload_hrtfs,
            check_sofa_against_conventions=check_sofa_against_conventions,
            sofa_open=sofa_open,
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
