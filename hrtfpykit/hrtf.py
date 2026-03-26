import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
from .analytics import Analytics
from .dsp import calculate_ir_from_tf, calculate_tf_from_ir
from .sofa.core import SOFA
from .spatial import Sources
from .domain import IR, TF


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

    @property
    def Sources(self) -> "Sources":
        return Sources(self)

    @property
    def Analytics(self) -> "Analytics":
        return Analytics(self)

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
        if global_attrs is None:
            message = "Loaded SOFA dataset is unavailable; cannot verify HRTF convention."
            warnings.warn(message, UserWarning)
        else:
            try:
                convention = global_attrs.get("SOFAConventions").value
            except ValueError:
                convention = None
            if convention not in allowed:
                message = (
                    "SOFAConventions is not an HRTF convention. "
                    f"Expected one of {sorted(allowed)}, got {convention!r} "
                    f"for {path!s}."
                )
                warnings.warn(message, UserWarning)
        global_attrs = Sofa.GlobalAttributes
        variables = Sofa.Variables
        if global_attrs is None or variables is None:
            raise ValueError("SOFA dataset is not loaded")

        try:
            convention = global_attrs.get("SOFAConventions").value
        except ValueError:
            convention = None
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
            if frequency_bins is None:
                message = "Missing N frequency_bins; cannot compute IR from TF."
                warnings.warn(message, UserWarning)
            else:
                if frequency_bins.ndim == 1 and frequency_bins.size > 0:
                    if float(np.min(frequency_bins)) >= 0.0 and not np.isclose(frequency_bins[0], 0.0):
                        warnings.warn(
                            "Frequency axis should start at 0 Hz to compute IR from TF.",
                            UserWarning,
                        )
            tf_normalization = None
            if "Normalization" in variable_names:
                norm_data = np.asarray(variables.get("Normalization").value, dtype=float)
                if norm_data.size > 0:
                    tf_normalization = float(norm_data.flat[0])
            resolved_sample_rate = None
            fft_length_used = None
            if frequency_bins is not None and frequency_bins.ndim == 1 and frequency_bins.size >= 2:
                diffs = np.diff(frequency_bins)
                first = float(diffs[0])
                if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
                    if float(np.min(frequency_bins)) < 0.0:
                        expected_n_fft = frequency_bins.size
                    else:
                        expected_n_fft = 2 * (frequency_bins.size - 1)
                    if fft_length is not None and fft_length != expected_n_fft:
                        warnings.warn(
                            "FFT length does not match the provided frequency bins; using inferred length.",
                            UserWarning,
                        )
                    fft_length_used = expected_n_fft
                    resolved_sample_rate = first * expected_n_fft
            if fft_length_used is None:
                fft_length_used = fft_length or (2 * (tf.shape[-1] - 1))
            if frequency_bins is None:
                warnings.warn(
                    "Missing frequency_bins; cannot compute IR from TF.",
                    UserWarning,
                )
                ir, sample_rate = None, None
            else:
                ir, sample_rate = calculate_ir_from_tf(
                    tf,
                    frequency_bins=frequency_bins,
                    fft_length=fft_length_used or fft_length,
                    tf_normalization=tf_normalization,
                    normalization_action="undo",
                )
            if resolved_sample_rate is None and sample_rate is not None:
                resolved_sample_rate = sample_rate
            hrtf = cls(Sofa)
            hrtf.IR.values = ir
            hrtf.IR.sample_rate = resolved_sample_rate
            hrtf.TF.values = tf
            hrtf.TF.frequency_bins = frequency_bins
            hrtf.fft_length = fft_length_used
            if ir is None:
                message = "Unable to compute IR from TF with the provided frequency_bins."
                warnings.warn(message, UserWarning)
            if resolved_sample_rate is None:
                warnings.warn("Unable to infer samplerate from frequency_bins.", UserWarning)
            hrtf.SOFAConventions = convention
            return hrtf

        message = "Unable to determine HRTF domain from SOFA content."
        warnings.warn(message, UserWarning)
        hrtf = cls(Sofa)
        hrtf.fft_length = fft_length
        hrtf.SOFAConventions = convention
        return hrtf

