from functools import cached_property
from pathlib import Path

import numpy as np
from .coordinates import get_position_alias, get_position_queries
from .dsp import (
    ir_from_tf,
    tf_from_ir,
)
from .planes import (
    get_frontal_plane,
    get_horizontal_plane,
    get_median_plane,
)
from ..plots.hrtf_plots import HRTFPlots
from ..sofa import SOFA
from .sources import Sources
from .domain import IR, TF
from .transforms import Transform


class HRTF(HRTFPlots):
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
    def transform(self) -> "Transform":
        return Transform(self)

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
        if "Sources" in self.__dict__:
            hrtf.Sources.source_coordinate_system = self.Sources.source_coordinate_system
            if self.Sources._selected_indices is not None:
                hrtf.Sources._selected_indices = np.array(
                    self.Sources._selected_indices,
                    dtype=int,
                    copy=True,
                )
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
            current_source_indices = transformed_hrtf.Sources._selected_indices
            source_positions = transformed_hrtf.Sources.get_positions(angle_unit=angle_unit)
            if source_positions.ndim != 2 or source_positions.shape[-1] != 3:
                raise ValueError("Source positions grid must have shape (N, 3)")
            source_count = int(source_positions.shape[0])

            if positions is not None:
                position_indices: list[int] = []
                for position in get_position_queries(positions):
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
                    plane_indices, _ = get_horizontal_plane(
                        hrtf=transformed_hrtf,
                        elevation=plane_angle,
                        angle_unit=angle_unit,
                    )
                elif plane_key == "median":
                    plane_indices, _ = get_median_plane(
                        hrtf=transformed_hrtf,
                        azimuth=plane_angle,
                        angle_unit=angle_unit,
                    )
                elif plane_key == "frontal":
                    plane_indices, _ = get_frontal_plane(
                        hrtf=transformed_hrtf,
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

            if current_source_indices is None:
                source_selected_indices = np.asarray(selected_indices, dtype=int)
            else:
                source_selected_indices = np.take(
                    np.asarray(current_source_indices, dtype=int),
                    np.asarray(selected_indices, dtype=int),
                    axis=0,
                )
            transformed_hrtf.Sources._selected_indices = source_selected_indices

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
            ir_values = transformed_hrtf.IR.values
            if not isinstance(ir_values, np.ndarray):
                raise ValueError("IR data must be a NumPy array")
            if ir_values.ndim == 0:
                raise ValueError("IR data must have at least one dimension")

            using_sample_indices = start is not None or end is not None
            using_seconds = start_seconds is not None or end_seconds is not None
            if using_sample_indices and using_seconds:
                raise ValueError(
                    "Use either sample indices (start/end) or seconds (start_seconds/end_seconds)"
                )

            start_index = start
            end_index = end
            if using_seconds:
                if transformed_hrtf.IR.sample_rate is None:
                    raise ValueError("sample_rate is required when using seconds crop")
                resolved_sample_rate = transformed_hrtf.IR.sample_rate
                if isinstance(resolved_sample_rate, bool):
                    raise ValueError("sample_rate must be a finite, positive value.")
                try:
                    resolved_sample_rate = float(resolved_sample_rate)
                except (TypeError, ValueError):
                    raise ValueError("sample_rate must be a finite, positive value.") from None
                if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
                    raise ValueError("sample_rate must be a finite, positive value.")
                if start_seconds is not None:
                    if isinstance(start_seconds, bool):
                        raise ValueError("start_seconds must be a finite, non-negative value.")
                    try:
                        start_seconds = float(start_seconds)
                    except (TypeError, ValueError):
                        raise ValueError("start_seconds must be a finite, non-negative value.") from None
                    if not np.isfinite(start_seconds) or start_seconds < 0.0:
                        raise ValueError("start_seconds must be a finite, non-negative value.")
                    start_index = int(round(start_seconds * resolved_sample_rate))
                else:
                    start_index = None
                if end_seconds is not None:
                    if isinstance(end_seconds, bool):
                        raise ValueError("end_seconds must be a finite, non-negative value.")
                    try:
                        end_seconds = float(end_seconds)
                    except (TypeError, ValueError):
                        raise ValueError("end_seconds must be a finite, non-negative value.") from None
                    if not np.isfinite(end_seconds) or end_seconds < 0.0:
                        raise ValueError("end_seconds must be a finite, non-negative value.")
                    end_index = int(round(end_seconds * resolved_sample_rate))
                else:
                    end_index = None
            else:
                if start is not None:
                    if isinstance(start, bool) or not isinstance(start, int):
                        raise ValueError("start must be an integer")
                    if start < 0:
                        raise ValueError("start must be non-negative")
                if end is not None:
                    if isinstance(end, bool) or not isinstance(end, int):
                        raise ValueError("end must be an integer")
                    if end < 0:
                        raise ValueError("end must be non-negative")

            if start_index is not None and end_index is not None and start_index >= end_index:
                raise ValueError("Crop end must be greater than crop start")

            transformed_hrtf.IR.values = ir_values[..., slice(start_index, end_index)]
            tf_from_ir(
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

            tf, frequency_bins, fft_length_used = tf_from_ir(
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
            ir, sample_rate, fft_length_used = ir_from_tf(
                tf,
                frequency_bins=frequency_bins,
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
