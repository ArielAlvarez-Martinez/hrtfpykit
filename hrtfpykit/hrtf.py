from functools import cached_property
from pathlib import Path

import numpy as np
from .analytics import Analytics
from .dsp import (
    apply_ir_crop,
    calculate_ir_from_tf,
    calculate_tf_from_ir,
)
from .plots import Plots
from .sofa.core import SOFA
from .spatial import Planes, Sources
from .domain import IR, TF
from .transforms import Transform


class HRTF(Plots):
    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        self.Sofa: SOFA | None = Sofa
        self.SOFAConventions: str | None = None
        self.fft_length: int | None = None

    @cached_property
    def IR(self) -> "IR":
        return IR(self)

    @cached_property
    def TF(self) -> "TF":
        return TF(self)

    @cached_property
    def Sources(self) -> "Sources":
        return Sources(self)

    @cached_property
    def Planes(self) -> "Planes":
        return Planes(self)

    @cached_property
    def transform(self) -> "Transform":
        return Transform(self)

    @property
    def Analytics(self) -> "Analytics":
        return Analytics(self)

    def __getitem__(self, ear: str) -> "HRTF":
        """Return an ear-selected HRTF view (`left`, `right`, or `both`)."""
        ear_key = str(ear).strip().lower()
        if ear_key not in {"both", "left", "right"}:
            raise KeyError("ear must be one of: left, right, both")
        if ear_key == "both":
            return self
        return self.select(ear=ear_key)

    def clone(self) -> "HRTF":
        sofa_clone = self.Sofa
        if self.Sofa is not None:
            try:
                sofa_clone = self.Sofa.clone()
            except ValueError:
                sofa_clone = self.Sofa
        hrtf = HRTF(Sofa=sofa_clone)
        hrtf.SOFAConventions = self.SOFAConventions
        hrtf.fft_length = self.fft_length
        if self.IR.values is not None:
            hrtf.IR.values = np.array(self.IR.values, copy=True)
        if self.IR.sample_rate is not None:
            hrtf.IR.sample_rate = float(self.IR.sample_rate)
        if self.TF.values is not None:
            hrtf.TF.values = np.array(self.TF.values, copy=True)
        if self.TF.frequency_bins is not None:
            hrtf.TF.frequency_bins = np.array(self.TF.frequency_bins, copy=True)
        return hrtf

    def select(
        self,
        positions: np.ndarray | list[list[float]] | list[float] | None = None,
        position_coordinate_system: str = "spherical",
        plane: str | None = None,
        plane_angle: float = 0.0,
        ear: str = "both",
        angle_unit: str = "degrees",
        start: int | None = None,
        end: int | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> "HRTF":
        transformed_hrtf = self.clone()
        selected_indices: np.ndarray | None = None
        ear_key = str(ear).strip().lower()
        if ear_key not in {"both", "left", "right"}:
            raise ValueError("ear must be one of: both, left, right")

        selecting_spatial = positions is not None or plane is not None
        if selecting_spatial:
            if transformed_hrtf.Sofa is None:
                raise ValueError("Spatial selection requires a loaded SOFA dataset")
            source_positions = transformed_hrtf.Sources.get_positions(angle_unit=angle_unit)
            if source_positions.ndim != 2 or source_positions.shape[-1] != 3:
                raise ValueError("Source positions grid must have shape (N, 3)")
            source_count = int(source_positions.shape[0])

            if positions is not None:
                positions_array = np.asarray(positions, dtype=float)
                if positions_array.ndim == 1:
                    positions_array = positions_array.reshape(1, -1)
                if positions_array.ndim != 2 or positions_array.shape[-1] not in {2, 3}:
                    raise ValueError("positions must have shape (K, 2) or (K, 3)")
                position_indices: list[int] = []
                for position in positions_array:
                    idx, _ = transformed_hrtf.Sources.get_position_index(
                        position=position,
                        coordinate_system=position_coordinate_system,
                        angle_unit=angle_unit,
                    )
                    if idx not in position_indices:
                        position_indices.append(int(idx))
                selected_indices = np.asarray(position_indices, dtype=int)

            if plane is not None:
                plane_key = str(plane).strip().lower()
                if plane_key == "horizontal":
                    plane_indices, _ = transformed_hrtf.Planes.get_horizontal_plane_indices(
                        elevation=plane_angle,
                        angle_unit=angle_unit,
                    )
                elif plane_key == "median":
                    plane_indices, _ = transformed_hrtf.Planes.get_median_plane_indices(
                        azimuth=plane_angle,
                        angle_unit=angle_unit,
                    )
                elif plane_key == "frontal":
                    plane_indices, _ = transformed_hrtf.Planes.get_frontal_plane_indices(
                        azimuth=plane_angle,
                        angle_unit=angle_unit,
                    )
                else:
                    raise ValueError("plane must be one of: horizontal, median, frontal")
                plane_indices = np.asarray(plane_indices, dtype=int)
                if selected_indices is None:
                    selected_indices = plane_indices
                else:
                    selected_indices = np.intersect1d(selected_indices, plane_indices)

            if selected_indices is None:
                selected_indices = np.arange(source_count, dtype=int)
            if selected_indices.size == 0:
                raise ValueError("Selection produced no source positions")

            if transformed_hrtf.IR.values is not None:
                transformed_hrtf.IR.values = np.take(
                    transformed_hrtf.IR.values,
                    selected_indices,
                    axis=0,
                )
            if transformed_hrtf.TF.values is not None:
                transformed_hrtf.TF.values = np.take(
                    transformed_hrtf.TF.values,
                    selected_indices,
                    axis=0,
                )

        cropping_ir = (
            start is not None
            or end is not None
            or start_seconds is not None
            or end_seconds is not None
        )
        if cropping_ir:
            if transformed_hrtf.IR.values is None:
                raise ValueError("IR data is not available")
            transformed_hrtf.IR.values = apply_ir_crop(
                transformed_hrtf.IR,
                start=start,
                end=end,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            calculate_tf_from_ir(
                transformed_hrtf.IR,
                fft_length=transformed_hrtf.fft_length,
            )

        if ear_key != "both":
            ear_index = 0 if ear_key == "left" else 1
            if transformed_hrtf.IR.values is not None:
                if transformed_hrtf.IR.values.shape[-2] <= ear_index:
                    raise ValueError(f"Requested ear '{ear_key}' is not available in IR data")
                transformed_hrtf.IR.values = np.take(
                    transformed_hrtf.IR.values,
                    indices=ear_index,
                    axis=-2,
                )
            if transformed_hrtf.TF.values is not None:
                if transformed_hrtf.TF.values.shape[-2] <= ear_index:
                    raise ValueError(f"Requested ear '{ear_key}' is not available in TF data")
                transformed_hrtf.TF.values = np.take(
                    transformed_hrtf.TF.values,
                    indices=ear_index,
                    axis=-2,
                )

        transformed_hrtf.Sources._positions = transformed_hrtf.Sources.get_positions(
            angle_unit=angle_unit
        )
        return transformed_hrtf

    @classmethod
    def load_hrtf(
        cls,
        path: str | Path,
        mode: str = "r",
        parallel: bool = False,
        check_sofa_against_conventions: bool = True,
        fft_length: int | None = None,
    ) -> "HRTF":
  
        Sofa = SOFA.load(
            path,
            mode=mode,
            parallel=parallel,
            check_sofa_against_conventions=check_sofa_against_conventions,
        )
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        global_attrs = Sofa.GlobalAttributes
        variables = Sofa.Variables
        if global_attrs is None or variables is None:
            raise ValueError("SOFA dataset is not loaded")

        try:
            convention = global_attrs.get("SOFAConventions").value
        except ValueError:
            convention = None
        if convention not in allowed:
            raise ValueError(
                "SOFAConventions is not an HRTF convention. "
                f"Expected one of {sorted(allowed)}, got {convention!r} "
                f"for {path!s}."
            )
        variable_names = set(variables.get_names())
       
        if convention == "SimpleFreeFieldHRIR":
            if "Data.IR" not in variable_names:
                raise ValueError(
                    "SimpleFreeFieldHRIR requires variable 'Data.IR', but it is missing."
                )
            ir = np.asarray(variables.get("Data.IR").value)
            if ir.size == 0 or np.all(ir == 0):
                raise ValueError(
                    "SimpleFreeFieldHRIR requires non empty 'Data.IR'."
                )
            if "Data.SamplingRate" not in variable_names:
                raise ValueError(
                    "SimpleFreeFieldHRIR requires variable 'Data.SamplingRate', but it is missing."
                )
            sample_rate_data = np.asarray(
                variables.get("Data.SamplingRate").value,
                dtype=float,
            )
            if sample_rate_data.size == 0 or np.all(sample_rate_data == 0):
                raise ValueError(
                    "SimpleFreeFieldHRIR requires non empty 'Data.SamplingRate'."
                )
            resolved_sample_rate = float(sample_rate_data.flat[0])
            if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
                raise ValueError(
                    "SimpleFreeFieldHRIR requires a finite, positive 'Data.SamplingRate' value."
                )

            tf, frequency_bins, fft_length_used = calculate_tf_from_ir(
                ir,
                resolved_sample_rate,
                fft_length=fft_length,
            )
            hrtf = cls(Sofa)
            hrtf.IR.values = ir
            hrtf.IR.sample_rate = resolved_sample_rate
            hrtf.TF.values = tf
            hrtf.TF.frequency_bins = frequency_bins
            hrtf.fft_length = fft_length
            if fft_length_used is not None:
                hrtf.fft_length = fft_length_used
            hrtf.SOFAConventions = convention
            return hrtf

        if convention == "SimpleFreeFieldHRTF":
            required_variables = ("Data.Real", "Data.Imag", "N")
            missing_variables = [name for name in required_variables if name not in variable_names]
            if missing_variables:
                raise ValueError(
                    "SimpleFreeFieldHRTF requires variables "
                    f"{required_variables}, but missing: {missing_variables}."
                )

            real = np.asarray(variables.get("Data.Real").value, dtype=float)
            if real.size == 0 or np.all(real == 0):
                raise ValueError(
                    "SimpleFreeFieldHRTF requires non empty 'Data.Real'."
                )

            imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
            if imag.size == 0 or np.all(imag == 0):
                raise ValueError(
                    "SimpleFreeFieldHRTF requires non empty 'Data.Imag'."
                )

            frequency_bins = np.asarray(variables.get("N").value, dtype=float)
            if frequency_bins.size == 0 or np.all(frequency_bins == 0):
                raise ValueError(
                    "SimpleFreeFieldHRTF requires non empty 'N'."
                )

            tf = real + 1j * imag
            tf_normalization = None
            if "Normalization" in variable_names:
                norm_data = np.asarray(variables.get("Normalization").value, dtype=float)
                if norm_data.size > 0:
                    tf_normalization = float(norm_data.flat[0])
            ir, sample_rate, fft_length_used = calculate_ir_from_tf(
                tf,
                frequency_bins=frequency_bins,
                tf_normalization=tf_normalization,
                normalization_action="undo",
            )
            if fft_length is not None and fft_length != fft_length_used:
                raise ValueError("FFT length does not match the provided frequency bins.")
            resolved_sample_rate = sample_rate
            hrtf = cls(Sofa)
            hrtf.IR.values = ir
            hrtf.IR.sample_rate = resolved_sample_rate
            hrtf.TF.values = tf
            hrtf.TF.frequency_bins = frequency_bins
            hrtf.fft_length = fft_length_used
            hrtf.SOFAConventions = convention
            return hrtf
