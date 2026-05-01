from dataclasses import dataclass

import numpy as np
from typing import TYPE_CHECKING

from ..hrtf.coordinates import get_spherical_positions
from ..hrtf.planes import get_frontal_plane, get_horizontal_plane, get_median_plane
from .sanitize import sanitize_index_by
from .specs import HRTFSpec, ITDSpec, ILDSpec, SHSpec
from .specs_workflow import DatasetSpecWorkflow

if TYPE_CHECKING:
    from .base import BaseDataset
    from ..hrtf.hrtf import HRTF


@dataclass(frozen=True)
class DatasetAcousticContextPlan:
    dataset_sample_rate: float | None
    dataset_source_positions: np.ndarray | None
    available_azimuth_angles: np.ndarray | None
    available_elevation_angles: np.ndarray | None
    azimuth_angles: np.ndarray | None
    elevation_angles: np.ndarray | None
    frequency_bins: np.ndarray | None
    sample_indices: np.ndarray | None
    selected_position_indices: tuple[int, ...]
    selected_frequency_indices: tuple[int, ...]
    selected_sample_indices: tuple[int, ...]
    spec_position_indices: tuple[tuple[int, tuple[int, ...]], ...]


class DatasetAcousticContext:
    @staticmethod
    def resolve_position_indices(
        positions: str | tuple[int, ...] | list[int] | np.ndarray,
        plane: str | tuple[object, ...] | dict[str, object] | None,
        hrtf: "HRTF",
    ) -> list[int]:
        from .sanitize import sanitize_positions

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
            angle = plane[1]
            angle_unit = "degrees" if len(plane) == 2 else str(plane[2]).strip().lower()
        else:
            plane_key = str(plane.get("plane")).strip().lower()
            default_angle = 90.0 if plane_key == "frontal" else 0.0
            angle = plane.get("angle", plane.get("plane_angle", default_angle))
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
    def _filter_specs(
        specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...],
        spec_types: type[object] | tuple[type[object], ...],
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...]:
        return tuple(spec for spec in specs if isinstance(spec, spec_types))

    @staticmethod
    def _indexed_specs(
        specs: tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...],
    ) -> tuple[HRTFSpec | ITDSpec | ILDSpec | SHSpec, ...]:
        return DatasetAcousticContext._filter_specs(specs, (HRTFSpec, ITDSpec, ILDSpec, SHSpec))

    def build(self, dataset: "BaseDataset") -> DatasetAcousticContextPlan:
        acoustic_specs = dataset._get_specs((HRTFSpec, ITDSpec, ILDSpec, SHSpec))
        if len(acoustic_specs) == 0:
            return DatasetAcousticContextPlan(
                dataset_sample_rate=None,
                dataset_source_positions=None,
                available_azimuth_angles=None,
                available_elevation_angles=None,
                azimuth_angles=None,
                elevation_angles=None,
                frequency_bins=None,
                sample_indices=None,
                selected_position_indices=(),
                selected_frequency_indices=(),
                selected_sample_indices=(),
                spec_position_indices=(),
            )

        sample_subject_id = dataset._subject_ids[0]
        sample_hrtf = dataset.get_subject_hrtf(sample_subject_id)
        dataset_sample_rate = (
            None if sample_hrtf.IR.sample_rate is None else float(sample_hrtf.IR.sample_rate)
        )
        dataset_source_positions = np.asarray(
            sample_hrtf.Sources.get_positions(angle_unit="degrees"),
            dtype=float,
        )
        frequency_bins = (
            None if sample_hrtf.TF.frequency_bins is None else np.asarray(sample_hrtf.TF.frequency_bins, dtype=float)
        )
        selected_frequency_indices = () if frequency_bins is None else tuple(range(int(frequency_bins.shape[0])))
        sample_indices = np.arange(sample_hrtf.IR.values.shape[-1], dtype=int)
        selected_sample_indices = tuple(range(int(sample_indices.shape[0])))

        position_axis: tuple[int, ...] | None = None
        position_axis_spec: str | None = None
        frequency_count: int | None = None
        frequency_count_spec: str | None = None
        sample_count: int | None = None
        sample_count_spec: str | None = None
        spec_position_indices: list[tuple[int, tuple[int, ...]]] = []

        for spec in self._filter_specs(dataset._specs, (HRTFSpec, ITDSpec, ILDSpec)):
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

        for spec in self._indexed_specs(dataset._specs):
            spec_name = DatasetSpecWorkflow.get_spec_name(spec)
            spec_index_by = sanitize_index_by(spec.index_by)
            if "frequency" in spec_index_by:
                if isinstance(spec, (HRTFSpec, SHSpec)):
                    if sample_hrtf.TF.frequency_bins is None:
                        raise ValueError("Frequency-indexed specs require available HRTF frequency bins")
                    current_frequency_count = int(np.asarray(sample_hrtf.TF.frequency_bins).reshape(-1).shape[0])
                elif isinstance(spec, ILDSpec):
                    fft_length = (
                        int(spec.fft_length)
                        if spec.fft_length is not None
                        else int(sample_hrtf.IR.values.shape[-1])
                    )
                    current_frequency_count = int(fft_length // 2 + 1)
                else:
                    continue
                if frequency_count is None:
                    frequency_count = current_frequency_count
                    frequency_count_spec = spec_name
                elif current_frequency_count != frequency_count:
                    raise ValueError(
                        "All frequency-indexed specs in a dataset must use the same frequency-bin count. "
                        f"{spec_name!r} selects {current_frequency_count} bins, but "
                        f"{frequency_count_spec!r} selects {frequency_count}. "
                        "Pick one frequency selection for the full dataset."
                    )
            if "samples" in spec_index_by:
                current_sample_count = int(sample_hrtf.IR.values.shape[-1])
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
        if frequency_count is not None:
            selected_frequency_indices = tuple(range(int(frequency_count)))
        if sample_count is not None:
            selected_sample_indices = tuple(range(int(sample_count)))

        spherical_positions = np.asarray(
            get_spherical_positions(sample_hrtf.Sources, angle_unit="degrees"),
            dtype=float,
        )
        available_azimuth_angles = np.unique(np.round(spherical_positions[:, 0], 2))
        available_elevation_angles = np.unique(np.round(spherical_positions[:, 1], 2))
        if len(selected_position_indices) > 0:
            selected_spherical_positions = np.asarray(
                spherical_positions[list(selected_position_indices)],
                dtype=float,
            )
            azimuth_angles = np.unique(np.round(selected_spherical_positions[:, 0], 2))
            elevation_angles = np.unique(np.round(selected_spherical_positions[:, 1], 2))
        else:
            azimuth_angles = None
            elevation_angles = None

        return DatasetAcousticContextPlan(
            dataset_sample_rate=dataset_sample_rate,
            dataset_source_positions=dataset_source_positions,
            available_azimuth_angles=available_azimuth_angles,
            available_elevation_angles=available_elevation_angles,
            azimuth_angles=azimuth_angles,
            elevation_angles=elevation_angles,
            frequency_bins=frequency_bins,
            sample_indices=sample_indices,
            selected_position_indices=selected_position_indices,
            selected_frequency_indices=selected_frequency_indices,
            selected_sample_indices=selected_sample_indices,
            spec_position_indices=tuple(spec_position_indices),
        )
