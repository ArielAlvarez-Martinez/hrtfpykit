from functools import cached_property
from pathlib import Path

import numpy as np
from .analytics import Analytics
from .dsp import (
    calculate_ir_from_tf,
    calculate_tf_from_ir,
)
from .sofa.core import SOFA
from .spatial import Planes, Sources
from .domain import IR, TF
from .transforms import Transform


class HRTF:
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
            ir, sample_rate = calculate_ir_from_tf(
                tf,
                frequency_bins=frequency_bins,
                tf_normalization=tf_normalization,
                normalization_action="undo",
            )
            min_frequency_bin = float(np.min(frequency_bins))
            if min_frequency_bin < 0.0:
                fft_length_used = frequency_bins.size
            else:
                fft_length_used = 2 * (frequency_bins.size - 1)
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
