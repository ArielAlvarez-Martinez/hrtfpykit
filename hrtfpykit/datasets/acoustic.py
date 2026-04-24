import numpy as np
import warnings

from .index import normalize_positions
from ..hrtf.coordinates import get_spherical_positions
from ..hrtf.planes import get_frontal_plane, get_horizontal_plane, get_median_plane
from ..main import load_hrtf


class DatasetAcousticContext:
    @staticmethod
    def resolve_positions_selection(
        positions: str | tuple[int, ...] | list[int] | np.ndarray,
        plane: str | tuple[object, ...] | dict[str, object] | None,
        hrtf,
    ) -> list[int]:
        position_count = int(hrtf.Sources.get_positions().shape[0])
        if plane is None:
            return normalize_positions(positions, position_count)
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
    def is_hrtf_object(value: object) -> bool:
        return (
            hasattr(value, "IR")
            and hasattr(value, "TF")
            and hasattr(value, "Sources")
            and hasattr(value, "transform")
        )

    @staticmethod
    def is_explicit_hrtf_transform(transform) -> bool:
        return bool(getattr(transform, "__hrtf_transform__", False))

    @staticmethod
    def is_raw_hrtf_transform_method(transform) -> bool:
        transform_module = str(getattr(transform, "__module__", ""))
        transform_qualname = str(getattr(transform, "__qualname__", ""))
        return transform_module.endswith(".transforms") and transform_qualname.startswith("Transform.")

    @classmethod
    def apply_hrtf_spec_transform(
        cls,
        hrtf,
        transform,
    ):
        transformed_hrtf = transform(hrtf)
        if not cls.is_hrtf_object(transformed_hrtf):
            raise ValueError(
                "HRTFTransform callables used in HRTFSpec.transform must return an HRTF object"
            )
        return transformed_hrtf

    def resolve_dataset_hrtf(self, subject_id: str, hrtf):
        if self.hrtf_transform is None:
            return hrtf
        transformed_hrtf = self._dataset_transformed_hrtf_cache.get(subject_id)
        if transformed_hrtf is not None:
            return transformed_hrtf
        transformed_hrtf = self.hrtf_transform(hrtf)
        if not self.is_hrtf_object(transformed_hrtf):
            raise ValueError("hrtf_transform must return an HRTF object")
        if self._cache_hrtf:
            self._dataset_transformed_hrtf_cache[subject_id] = transformed_hrtf
        return transformed_hrtf

    def get_subject_hrtf(self, subject_id: str | int):
        resolved_subject_id = self.resolve_dataset_subject_id(subject_id, self.subject_ids)
        if resolved_subject_id not in self._hrtf_paths:
            raise KeyError(
                f"Subject {subject_id!r} resolved to {resolved_subject_id!r} but does not have an available HRTF file"
            )
        path = self._hrtf_paths[resolved_subject_id]
        if not path.exists():
            warnings.warn(
                f"{self.name}: subject {resolved_subject_id} HRTF path is missing: {path}",
                stacklevel=2,
            )
            raise FileNotFoundError(
                f"HRTF path is missing for subject {resolved_subject_id}: {path}"
            )
        hrtf = self._hrtf_cache.get(resolved_subject_id)
        if hrtf is None:
            try:
                hrtf = load_hrtf(path)
            except Exception as exc:
                warnings.warn(
                    f"{self.name}: subject {resolved_subject_id} HRTF file could not be loaded: {path} ({exc})",
                    stacklevel=2,
                )
                raise
            if self._cache_hrtf:
                self._hrtf_cache[resolved_subject_id] = hrtf
        return self.resolve_dataset_hrtf(resolved_subject_id, hrtf)

    def reset_acoustic_context(self) -> None:
        self.sample_rate = None
        self.available_positions = None
        self.selected_positions = None
        self.available_azimuth_angles = None
        self.available_elevation_angles = None
        self.azimuth_angles = None
        self.elevation_angles = None
        self.frequency_bins = None
        self.sample_indices = None
        self._selected_position_indices = []
        self._selected_frequency_indices = []
        self._selected_sample_indices = []

    def configure_reference_hrtf(self, reference_hrtf) -> None:
        self.sample_rate = (
            None if reference_hrtf.IR.sample_rate is None else float(reference_hrtf.IR.sample_rate)
        )
        self.available_positions = np.asarray(
            reference_hrtf.Sources.get_positions(angle_unit="degrees"),
            dtype=float,
        )

    def configure_frequency_and_sample_axes(self, reference_hrtf) -> None:
        if reference_hrtf.TF.frequency_bins is not None:
            self.frequency_bins = np.asarray(reference_hrtf.TF.frequency_bins, dtype=float)
            self._selected_frequency_indices = list(range(int(self.frequency_bins.shape[0])))
        self.sample_indices = np.arange(reference_hrtf.IR.values.shape[-1], dtype=int)
        self._selected_sample_indices = list(range(int(self.sample_indices.shape[0])))

    def configure_spatial_context(self, reference_hrtf) -> None:
        if self.primary_spatial_spec is not None:
            self._selected_position_indices = self.resolve_positions_selection(
                self.primary_spatial_spec.positions,
                self.primary_spatial_spec.plane,
                reference_hrtf,
            )
            self.selected_positions = np.asarray(
                self.available_positions[self._selected_position_indices],
                dtype=float,
            )

    def configure_angle_context(self, reference_hrtf) -> None:
        spherical_positions = np.asarray(
            get_spherical_positions(reference_hrtf.Sources, angle_unit="degrees"),
            dtype=float,
        )
        self.available_azimuth_angles = np.unique(np.round(spherical_positions[:, 0], 2))
        self.available_elevation_angles = np.unique(np.round(spherical_positions[:, 1], 2))
        if self.primary_spatial_spec is not None:
            selected_spherical_positions = np.asarray(
                spherical_positions[self._selected_position_indices],
                dtype=float,
            )
            self.azimuth_angles = np.unique(np.round(selected_spherical_positions[:, 0], 2))
            self.elevation_angles = np.unique(np.round(selected_spherical_positions[:, 1], 2))

    def prepare_acoustic_context(self) -> None:
        self.reset_acoustic_context()
        if self.primary_hrtf_backed_spec is None:
            return
        reference_subject_id = self.subject_ids[0]
        reference_hrtf = self.get_subject_hrtf(reference_subject_id)
        self.configure_reference_hrtf(reference_hrtf)
        self.configure_frequency_and_sample_axes(reference_hrtf)
        self.configure_spatial_context(reference_hrtf)
        self.configure_angle_context(reference_hrtf)
