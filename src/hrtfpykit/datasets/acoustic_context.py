from dataclasses import dataclass

import numpy as np
from typing import TYPE_CHECKING, Any, cast

from ..utils.coordinates import get_spherical_positions
from ..utils.planes import get_frontal_plane, get_horizontal_plane, get_median_plane
from .sanitize import sanitize_index_by, sanitize_positions
from .specs import HRTFSpec, ITDSpec, ILDSpec, SHSpec
from .specs_workflow import DatasetSpecWorkflow

if TYPE_CHECKING:
    from .base import BaseDataset
    from ..hrtf.hrtf import HRTF


@dataclass(frozen=True)
class DatasetAcousticContextPlan:
    """Immutable acoustic-axis plan produced during dataset construction.

    :class:`~hrtfpykit.datasets.acoustic_context.DatasetAcousticContextPlan` is
    the data transfer object returned by
    :meth:`~hrtfpykit.datasets.acoustic_context.DatasetAcousticContext.build`
    and consumed by :class:`~hrtfpykit.datasets.build.DatasetBuilder`. It
    separates the full acoustic context discovered from a representative HRTF
    resource from the subsets that participate in row generation. This lets
    dataset properties expose both the complete source/frequency/sample axes and
    the smaller selected axes requested by specs.

    Full context fields such as ``positions``, ``frequency_bins``, and
    ``sample_indices`` describe the selected dataset resource as a whole. Selected
    fields such as ``selected_position_indices`` and
    ``selected_frequency_indices`` describe the axes that expand dataset rows when
    ``index_by`` includes ``position``, ``frequency``, or ``samples``.

    Attributes
    ----------
    sample_rate : float or None
        Dataset-level sample rate inferred from the representative subject HRTF.
    positions : numpy.ndarray or None
        Full source-position grid from the representative HRTF in degrees.
    azimuth_angles, elevation_angles : numpy.ndarray or None
        Unique azimuth and elevation angles available in the full source grid.
    frequency_bins : numpy.ndarray or None
        Full frequency-bin axis from the representative HRTF, when available.
    sample_indices : numpy.ndarray or None
        Full time-sample index axis from the representative HRIR data.
    selected_position_indices : tuple of int
        Source-position indices selected by position-indexed specs.
    selected_azimuth_angles, selected_elevation_angles : numpy.ndarray or None
        Unique azimuth and elevation angles for selected_position_indices.
    selected_frequency_indices : tuple of int
        Frequency-bin indices selected by frequency-indexed specs.
    selected_sample_indices : tuple of int
        Time-sample indices selected by sample-indexed specs.
    spec_position_indices : tuple
        Per-spec position selections stored as (id(spec), indices) pairs.
    spec_frequency_indices : tuple
        Per-HRTFSpec frequency selections stored as (id(spec), indices) pairs.

    """
    sample_rate: float | None
    positions: np.ndarray | None
    azimuth_angles: np.ndarray | None
    elevation_angles: np.ndarray | None
    frequency_bins: np.ndarray | None
    sample_indices: np.ndarray | None
    selected_position_indices: tuple[int, ...]
    selected_azimuth_angles: np.ndarray | None
    selected_elevation_angles: np.ndarray | None
    selected_frequency_indices: tuple[int, ...]
    selected_sample_indices: tuple[int, ...]
    spec_position_indices: tuple[tuple[int, tuple[int, ...]], ...]
    spec_frequency_indices: tuple[tuple[int, tuple[int, ...]], ...]


class DatasetAcousticContext:
    """Resolve acoustic axes and selected row context for dataset specs.

    :class:`~hrtfpykit.datasets.acoustic_context.DatasetAcousticContext` is the
    construction-time utility that inspects a representative selected-subject
    HRTF object and turns acoustic specs into dataset axes. It derives the
    sample rate, source positions, frequency bins, sample indices, selected row
    axes, selected angle summaries, and per-spec position mappings used by the
    value selectors.

    The utility is intentionally separate from
    :class:`~hrtfpykit.datasets.base.BaseDataset` and
    :class:`~hrtfpykit.datasets.build.DatasetBuilder`. The builder orchestrates
    dataset construction, while this class handles acoustic consistency checks
    for HRTF, ITD, ILD, and spherical-harmonic specs.
    """

    @staticmethod
    def resolve_position_indices(
        positions: str | tuple[int, ...] | list[int] | np.ndarray,
        plane: str | tuple[object, ...] | dict[str, object] | None,
        hrtf: "HRTF",
    ) -> list[int]:
        """Resolve source-position indices for one position-selectable spec.

        The resolver accepts either explicit position indices or a plane selector
        and converts the request into concrete source indices from the sample HRTF
        source grid. A plane selector can be expressed as a string, a tuple, or a
        dictionary. Plane selection is exclusive with custom positions because
        the dataset needs one unambiguous position subset for each acoustic spec.

        Supported plane names are ``horizontal``, ``median``, and
        ``frontal``. A string selector uses the default plane angle: 0
        degrees for horizontal and median planes, and 90 degrees for frontal
        planes. Tuple selectors use (plane, angle) or
        (plane, angle, angle_unit). Dictionary selectors read ``plane`` and
        optionally ``angle`` or ``plane_angle`` plus ``angle_unit``.

        Parameters
        ----------
        positions : str or sequence of int
            Explicit position selection or ``all``. Custom indices are valid
            only when plane is None.
        plane : str, tuple, dict, or None
            Optional plane selector used instead of explicit position indices.
        hrtf : :class:`~hrtfpykit.hrtf.HRTF`
            :class:`~hrtfpykit.hrtf.HRTF` object used to inspect available
            source positions and resolve plane membership.

        Returns
        -------
        list of int
            Source-position indices selected for the spec.

        Raises
        ------
        ValueError
            If custom positions are combined with a plane selector, if a tuple
            plane selector has an invalid shape, if the plane name is unsupported,
            or if delegated position/plane sanitization rejects the request.

        Notes
        -----
        Plane selection uses the same HRTF plane helpers exposed by
        :mod:`~hrtfpykit.hrtf.planes`, so indices refer to the real nearest plane
        in the HRTF source grid rather than to an idealized analytical grid.

        """
        position_count = int(hrtf.Sources.get_positions().shape[0])
        if plane is None:
            return sanitize_positions(positions, position_count)
        if not isinstance(positions, str) or str(positions).strip().lower() != "all":
            raise ValueError("plane selection cannot be combined with custom positions")
        if isinstance(plane, str):
            plane_key = str(plane).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = default_angle
            angle_unit = "degrees"
        elif isinstance(plane, tuple):
            if len(plane) not in {2, 3} or not isinstance(plane[0], str):
                raise ValueError(
                    "Plane selection must be ('horizontal'|'median'|'frontal', angle[, angle_unit])"
                )
            plane_key = str(plane[0]).strip().lower()
            angle = float(cast(Any, plane[1]))
            angle_unit = "degrees" if len(plane) == 2 else str(plane[2]).strip().lower()
        else:
            plane_key = str(plane.get("plane")).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = float(cast(Any, plane.get("angle", plane.get("plane_angle", default_angle))))
            angle_unit = str(plane.get("angle_unit", "degrees")).strip().lower()
        if plane_key not in {"horizontal", "median", "frontal"}:
            raise ValueError("plane must be horizontal, median, or frontal")
        if plane_key == "horizontal":
            indices, _ = get_horizontal_plane(
                hrtf=hrtf,
                elevation=float(angle),
                angle_unit=angle_unit,
            )
        elif plane_key == "median":
            indices, _ = get_median_plane(
                hrtf=hrtf,
                azimuth=float(angle),
                angle_unit=angle_unit,
            )
        else:
            indices, _ = get_frontal_plane(
                hrtf=hrtf,
                azimuth=float(angle),
                angle_unit=angle_unit,
            )
        return [int(index) for index in np.asarray(indices, dtype=int).reshape(-1)]

    @staticmethod
    def resolve_frequency_indices(
        frequencies: float | list[float] | tuple[float, ...] | np.ndarray | None,
        frequency_bands: tuple[float, float] | list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None,
        domain: str,
        hrtf: "HRTF",
    ) -> list[int]:
        """Resolve one HRTFSpec frequency selector against an HRTF TF grid."""
        if frequencies is not None and frequency_bands is not None:
            raise ValueError("HRTFSpec frequencies and frequency_bands are mutually exclusive")
        if frequencies is not None or frequency_bands is not None:
            if str(domain).strip().lower() != "frequency":
                raise ValueError("HRTFSpec frequencies and frequency_bands require domain=frequency")
        if hrtf.TF.frequency_bins is None:
            raise ValueError("HRTFSpec frequency selection requires available TF frequency_bins")
        frequency_bins = np.asarray(hrtf.TF.frequency_bins, dtype=float).reshape(-1)
        if frequency_bins.size == 0:
            raise ValueError("HRTFSpec frequency selection requires available TF frequency_bins")
        if frequencies is None and frequency_bands is None:
            return list(range(int(frequency_bins.size)))
        if frequencies is not None:
            raw_frequency_values = np.asarray(frequencies, dtype=object).reshape(-1)
            if any(isinstance(value, bool | np.bool_) for value in raw_frequency_values.tolist()):
                raise ValueError("HRTFSpec frequencies must contain finite, non-negative value(s)")
            try:
                frequency_values = np.asarray(frequencies, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                raise ValueError("HRTFSpec frequencies must contain finite, non-negative value(s)") from None
            if frequency_values.size == 0:
                raise ValueError("HRTFSpec frequencies must contain at least one value")
            if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values < 0.0):
                raise ValueError("HRTFSpec frequencies must contain finite, non-negative value(s)")
            nearest_indices = [
                int(np.argmin(np.abs(frequency_bins - float(frequency))))
                for frequency in frequency_values
            ]
            return list(dict.fromkeys(nearest_indices))
        raw_bands = np.asarray(frequency_bands, dtype=object)
        if any(isinstance(value, bool | np.bool_) for value in raw_bands.reshape(-1).tolist()):
            raise ValueError("HRTFSpec frequency_bands must contain finite, non-negative values")
        try:
            bands = np.asarray(frequency_bands, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("HRTFSpec frequency_bands must contain (minimum, maximum) pairs") from None
        if bands.ndim == 1 and bands.size == 2:
            bands = bands.reshape(1, 2)
        if bands.ndim != 2 or bands.shape[0] == 0 or bands.shape[1] != 2:
            raise ValueError("HRTFSpec frequency_bands must contain (minimum, maximum) pairs")
        if not np.all(np.isfinite(bands)) or np.any(bands < 0.0):
            raise ValueError("HRTFSpec frequency_bands must contain finite, non-negative values")
        if np.any(bands[:, 0] > bands[:, 1]):
            raise ValueError("HRTFSpec frequency_bands minimum must not exceed maximum")
        selected_mask = np.zeros(frequency_bins.shape, dtype=bool)
        for minimum, maximum in bands:
            selected_mask |= (frequency_bins >= float(minimum)) & (frequency_bins <= float(maximum))
        selected_indices = np.flatnonzero(selected_mask).astype(int).tolist()
        if len(selected_indices) == 0:
            raise ValueError("HRTFSpec frequency_bands selected no available TF bins")
        return selected_indices

    @classmethod
    def build(cls, dataset: "BaseDataset") -> DatasetAcousticContextPlan:
        """Build acoustic context for a constructed dataset state.

        The build step inspects one selected subject HRTF to derive dataset-level
        acoustic metadata and to validate the row axes requested by acoustic specs.
        It separates full resource context from selected row context so dataset
        properties can report the original source grid, frequency bins, and sample
        axis alongside the subsets used by indexed rows.

        HRTF, ITD, and ILD specs can select positions directly or through planes.
        When any of those specs are position-indexed, all position-indexed specs
        must select the same position axis because
        :class:`~hrtfpykit.datasets.base.BaseDataset` builds one shared row table.
        HRTFSpec frequency selectors are resolved against the representative
        subject after dataset-level and spec-level HRTF transforms.
        Frequency-indexed specs must agree on the selected frequency bins, and
        sample-indexed specs must agree on the number of HRIR samples for the
        same reason.

        If the dataset contains no acoustic specs, the method returns an empty
        plan without loading an HRTF. This allows metadata-, mesh-, image-, or
        video-only datasets to use the same construction pipeline.

        Parameters
        ----------
        dataset : :class:`~hrtfpykit.datasets.base.BaseDataset`
            Dataset with config, specs, resource paths, selected subjects, cache,
            and split state already initialized by
            :class:`~hrtfpykit.datasets.build.DatasetBuilder`.

        Returns
        -------
        DatasetAcousticContextPlan
            Acoustic context used by state assignment and row generation.

        Raises
        ------
        ValueError
            If position-indexed specs select different position axes, if
            frequency-indexed specs disagree on selected frequency bins, if a
            frequency-indexed HRTF or SH spec lacks frequency bins, if
            sample-indexed specs disagree on sample count, or if delegated HRTF
            loading and plane resolution fail.
        IndexError
            If acoustic specs are present but no subject was selected for the
            dataset split.

        Notes
        -----
        The representative subject is dataset._state.selected_subjects[0].
        The current implementation assumes validated dataset resources share the
        same acoustic axes across selected subjects.

        """
        state = dataset._state
        acoustic_specs = tuple(
            spec
            for spec in state.specs
            if isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec, SHSpec))
        )
        if len(acoustic_specs) == 0:
            return DatasetAcousticContextPlan(
                sample_rate=None,
                positions=None,
                azimuth_angles=None,
                elevation_angles=None,
                frequency_bins=None,
                sample_indices=None,
                selected_position_indices=(),
                selected_azimuth_angles=None,
                selected_elevation_angles=None,
                selected_frequency_indices=(),
                selected_sample_indices=(),
                spec_position_indices=(),
                spec_frequency_indices=(),
            )

        sample_subject_id = state.selected_subjects[0]
        sample_hrtf = dataset.get_subject_hrtf(sample_subject_id)
        sample_rate = (
            None if sample_hrtf.IR.sample_rate is None else float(sample_hrtf.IR.sample_rate)
        )
        positions = np.asarray(
            sample_hrtf.Sources.get_positions(angle_unit="degrees"),
            dtype=float,
        )
        frequency_bins = (
            None if sample_hrtf.TF.frequency_bins is None else np.asarray(sample_hrtf.TF.frequency_bins, dtype=float)
        )
        selected_frequency_indices = () if frequency_bins is None else tuple(range(int(frequency_bins.shape[0])))
        if sample_hrtf.IR.values is None:
            raise ValueError("Sample HRTF IR values are not available")
        sample_ir_values = sample_hrtf.IR.values
        sample_indices = np.arange(sample_ir_values.shape[-1], dtype=int)
        selected_sample_indices = tuple(range(int(sample_indices.shape[0])))

        position_axis: tuple[int, ...] | None = None
        position_axis_spec: str | None = None
        frequency_axis: tuple[int, ...] | None = None
        frequency_axis_spec: str | None = None
        sample_count: int | None = None
        sample_count_spec: str | None = None
        spec_position_indices: list[tuple[int, tuple[int, ...]]] = []
        spec_frequency_indices: list[tuple[int, tuple[int, ...]]] = []

        for spec in tuple(
            spec for spec in state.specs if isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec))
        ):
            indices = DatasetAcousticContext.resolve_position_indices(
                spec.positions,
                spec.plane,
                sample_hrtf,
            )
            spec_position_indices.append((id(spec), tuple(indices)))
            if "position" not in sanitize_index_by(spec.index_by):
                continue
            axis = tuple(indices)
            if position_axis is None:
                position_axis = axis
                position_axis_spec = DatasetSpecWorkflow.get_spec_name(spec)
            elif axis != position_axis:
                current_spec_name = DatasetSpecWorkflow.get_spec_name(spec)
                raise ValueError(
                    "All position-indexed specs in a dataset must use the same selected positions. "
                    f"{current_spec_name!r} selects {len(axis)} positions, but {position_axis_spec!r} selects {len(position_axis)}. "
                    "Pick one position selection for the full dataset."
                )

        for spec in tuple(spec for spec in state.specs if isinstance(spec, HRTFSpec)):
            selected_hrtf = sample_hrtf
            if spec.transform is not None:
                transform_cache_key = ("hrtf_transform", sample_subject_id, id(spec.transform))
                transformed_hrtf = state.cache.get(transform_cache_key)
                if transformed_hrtf is None:
                    transformed_hrtf = spec.transform(sample_hrtf)
                    state.cache[transform_cache_key] = transformed_hrtf
                selected_hrtf = cast(Any, transformed_hrtf)
            indices = DatasetAcousticContext.resolve_frequency_indices(
                spec.frequencies,
                spec.frequency_bands,
                str(spec.domain),
                cast(Any, selected_hrtf),
            )
            spec_frequency_indices.append((id(spec), tuple(indices)))

        for acoustic_spec in acoustic_specs:
            spec_name = DatasetSpecWorkflow.get_spec_name(acoustic_spec)
            spec_index_by = sanitize_index_by(acoustic_spec.index_by)
            if "frequency" in spec_index_by:
                if isinstance(acoustic_spec, HRTFSpec):
                    current_frequency_axis = dict(spec_frequency_indices)[id(acoustic_spec)]
                elif isinstance(acoustic_spec, SHSpec):
                    if sample_hrtf.TF.frequency_bins is None:
                        raise ValueError("Frequency-indexed specs require available HRTF frequency bins")
                    current_frequency_axis = tuple(range(int(np.asarray(sample_hrtf.TF.frequency_bins).reshape(-1).shape[0])))
                elif isinstance(acoustic_spec, ILDSpec):
                    fft_length = (
                        int(acoustic_spec.fft_length)
                        if acoustic_spec.fft_length is not None
                        else int(sample_ir_values.shape[-1])
                    )
                    current_frequency_axis = tuple(range(int(fft_length // 2 + 1)))
                else:
                    continue
                if frequency_axis is None:
                    frequency_axis = current_frequency_axis
                    frequency_axis_spec = spec_name
                elif current_frequency_axis != frequency_axis:
                    raise ValueError(
                        "All frequency-indexed specs in a dataset must use the same selected frequency bins. "
                        f"{spec_name!r} selects {len(current_frequency_axis)} bins, but "
                        f"{frequency_axis_spec!r} selects {len(frequency_axis)}. "
                        "Pick one frequency selection for the full dataset."
                    )
            if "samples" in spec_index_by:
                current_sample_count = int(sample_ir_values.shape[-1])
                if sample_count is None:
                    sample_count = current_sample_count
                    sample_count_spec = spec_name
                elif current_sample_count != sample_count:
                    raise ValueError(
                        "All sample-indexed specs in a dataset must use the same sample count. "
                        f"{spec_name!r} selects {current_sample_count} samples, "
                        f"but {sample_count_spec!r} selects {sample_count}. "
                        "Pick one sample selection for the full dataset."
                    )

        selected_position_indices = () if position_axis is None else position_axis
        if frequency_axis is not None:
            selected_frequency_indices = frequency_axis
        if sample_count is not None:
            selected_sample_indices = tuple(range(int(sample_count)))

        spherical_positions = np.asarray(
            get_spherical_positions(sample_hrtf.Sources, angle_unit="degrees"),
            dtype=float,
        )
        azimuth_angles = np.unique(np.round(spherical_positions[:, 0], 2))
        elevation_angles = np.unique(np.round(spherical_positions[:, 1], 2))
        if len(selected_position_indices) > 0:
            selected_spherical_positions = np.asarray(
                spherical_positions[list(selected_position_indices)],
                dtype=float,
            )
            selected_azimuth_angles = np.unique(np.round(selected_spherical_positions[:, 0], 2))
            selected_elevation_angles = np.unique(np.round(selected_spherical_positions[:, 1], 2))
        else:
            selected_azimuth_angles = None
            selected_elevation_angles = None

        return DatasetAcousticContextPlan(
            sample_rate=sample_rate,
            positions=positions,
            azimuth_angles=azimuth_angles,
            elevation_angles=elevation_angles,
            frequency_bins=frequency_bins,
            sample_indices=sample_indices,
            selected_position_indices=selected_position_indices,
            selected_azimuth_angles=selected_azimuth_angles,
            selected_elevation_angles=selected_elevation_angles,
            selected_frequency_indices=selected_frequency_indices,
            selected_sample_indices=selected_sample_indices,
            spec_position_indices=tuple(spec_position_indices),
            spec_frequency_indices=tuple(spec_frequency_indices),
        )
