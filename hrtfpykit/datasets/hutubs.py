from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Callable

from .base import BaseDataset
from .config import HUTUBSConfig
from .download import BaseDownload
from .sanitize import sanitize_grouped_by
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


class HUTUBS(BaseDataset):
    def __init__(
        self,
        root: str | Path,
        dataset_hrtf_variant: str | Mapping[str, object] = "measured",
        dataset_hrtf_transform: Callable[[object], object] | None = None,
        download: bool = False,
        download_resources: str | tuple[str, ...] | list[str] = "hrtf",
        download_hrtf_variant: str | Mapping[str, object] = "measured",
        exclude_subject_ids: str | int | tuple[str | int, ...] | list[str | int] | None = None,
        inputs: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        target: HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec | Sequence[HRTFSpec | ITDSpec | ILDSpec | SHSpec | MeshSpec | AnthropometrySpec | MetadataSpec | ImageSpec | VideoSpec] | None = None,
        split: str = "all",
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        """Dataset interface for local or downloadable HUTUBS resources.

        :class:`~hrtfpykit.datasets.HUTUBS` turns the HUTUBS resource layout
        into the shared :class:`~hrtfpykit.datasets.base.BaseDataset` API. It
        maps HUTUBS subject identifiers to local resource paths, supports
        measured and simulated HRTF variants, handles optional anthropometry,
        mesh, image, and video resources, and exposes samples through the shared
        integer-indexed dataset interface. HUTUBS-specific anthropometry field
        selection is applied for left/right measurements.

        Samples are defined entirely by input and target specs. The dataset
        scans the requested resource families, intersects available subjects, applies
        exclusions and split selection, and builds row contexts for subject-,
        position-, ear-, frequency-, or sample-indexed data. At access time, the
        selected subject HRTF is loaded through
        :func:`~hrtfpykit.hrtf.load_hrtf`, and the requested spec values are
        extracted into sample inputs and sample targets.

        Download selection is independent from dataset construction selection.
        download_resources and download_hrtf_variant control which official
        files are downloaded. dataset_hrtf_variant controls which local HRTF files
        are scanned and loaded after the download step. The dataset does not infer
        download resources from inputs or target and does not copy
        dataset_hrtf_variant into download_hrtf_variant.

        Parameters
        ----------
        root : str or Path
            Local HUTUBS dataset root.
        dataset_hrtf_variant : {``measured``, ``simulated``} or dict, default=``measured``
            HUTUBS HRTF resource variant used for dataset construction.
        dataset_hrtf_transform : callable or None, default=None
            Optional transform applied to loaded HRTFs before spec extraction.
        download : bool, default=False
            If True, downloads selected official HUTUBS resources before dataset
            construction.
        download_resources : str or sequence of str, default=``hrtf``
            Official resources requested for download. This value is not inferred
            from inputs or target.
        download_hrtf_variant : str or dict, default=``measured``
            HRTF variant requested for download. This value is independent from
            dataset_hrtf_variant.
        exclude_subject_ids : str, int, sequence, or None, default=None
            HUTUBS subjects excluded before scanning and splitting.
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
        HUTUBS
            Dataset object supporting indexed sample extraction and subject HRTF
            loading.

        """
        if isinstance(dataset_hrtf_variant, Mapping):
            unknown_keys = set(dataset_hrtf_variant) - {"type"}
            if len(unknown_keys) > 0:
                raise ValueError(
                    f"Unsupported dataset_hrtf_variant keys {tuple(sorted(unknown_keys))}. "
                    "HUTUBS only supports the 'type' key"
                )
            hrtf_type_value = dataset_hrtf_variant.get("type")
        else:
            hrtf_type_value = dataset_hrtf_variant
        hrtf_type = str(hrtf_type_value).strip().lower()
        if HUTUBSConfig.hrtf is None:
            raise ValueError("HUTUBS config does not define HRTF metadata")
        if hrtf_type not in HUTUBSConfig.hrtf.types:
            raise ValueError(
                f"Unsupported dataset_hrtf_variant {hrtf_type!r}. Expected one of {tuple(HUTUBSConfig.hrtf.types)}"
            )
        if download:
            downloaded, download_report = BaseDownload(
                config=HUTUBSConfig,
                root=root,
                excluded_subject_ids=exclude_subject_ids,
            ).download(
                download_resources=download_resources,
                download_hrtf_variant=download_hrtf_variant,
                download_mesh_variant=None,
            )
            if downloaded:
                print(download_report)
        super().__init__(
            root=root,
            config=HUTUBSConfig,
            dataset_hrtf_transform=dataset_hrtf_transform,
            exclude_subject_ids=exclude_subject_ids,
            inputs=inputs,
            target=target,
            dataset_hrtf_variant=hrtf_type,
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
        """Select HUTUBS anthropometry fields for an optional ear context.

        HUTUBS anthropometry columns use left and right prefixes for ear-specific
        measurements. This selector filters those prefixed fields according to the
        current spec or row ear while preserving shared non-ear fields, keeping
        generic table loading independent from HUTUBS naming conventions.

        Parameters
        ----------
        spec : AnthropometrySpec
            Anthropometry spec requesting the value.
        row : dict
            Current dataset row context.
        value : object
            Raw anthropometry value selected by the generic table resolver.

        Returns
        -------
        object
            Filtered value containing the requested ear-specific fields and shared
            fields.

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
