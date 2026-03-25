import warnings
from pathlib import Path

import numpy as np
from .analytics import Analytics
from .dsp import calculate_ir_from_tf, calculate_tf_from_ir
from .sofa.core import SOFA
from .spatial import Sources
from .domain import TimeDomain, FrequencyDomain


class HRTF:
    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        self.Sofa: SOFA | None = Sofa
        self.ir: np.ndarray | None = None
        self.tf: np.ndarray | None = None
        self.sample_rate: float | None = None
        self.frequency_bins: np.ndarray | None = None
        self.sofa_convention: str | None = None
        self.fft_length: int | None = None

    @property
    def TimeDomain(self) -> "TimeDomain":
        return TimeDomain(self)

    @property
    def FrequencyDomain(self) -> "FrequencyDomain":
        return FrequencyDomain(self)

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
        if convention == "SimpleFreeFieldHRIR" or "Data.IR" in variable_names:
            ir = np.asarray(variables.get("Data.IR").value)
            resolved_sample_rate = None
            if "Data.SamplingRate" in variable_names:
                data = np.asarray(
                    variables.get("Data.SamplingRate").value,
                    dtype=float,
                )
                if data.size > 0:
                    resolved_sample_rate = int(data.flat[0])
            tf = None
            frequency_bins = None
            fft_length_used = None
            if resolved_sample_rate is None:
                message = "Missing Data.SamplingRate; cannot compute TF from IR."
                warnings.warn(message, UserWarning)
            else:
                tf, frequency_bins, fft_length_used = calculate_tf_from_ir(
                    ir,
                    resolved_sample_rate,
                    fft_length=fft_length,
                )
            hrtf = cls(Sofa)
            hrtf.ir = ir
            hrtf.tf = tf
            hrtf.sample_rate = resolved_sample_rate
            hrtf.frequency_bins = frequency_bins
            hrtf.fft_length = fft_length
            if fft_length_used is not None:
                hrtf.fft_length = fft_length_used
            hrtf.sofa_convention = convention
            return hrtf

        if (
            convention == "SimpleFreeFieldHRTF"
            or ("Data.Real" in variable_names and "Data.Imag" in variable_names)
        ):
            real = np.asarray(variables.get("Data.Real").value, dtype=float)
            imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
            tf = real + 1j * imag
            frequency_bins = None
            if "N" in variable_names:
                data = np.asarray(variables.get("N").value, dtype=float)
                if data.size > 0:
                    frequency_bins = data
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
            hrtf.ir = ir
            hrtf.tf = tf
            hrtf.sample_rate = resolved_sample_rate
            hrtf.frequency_bins = frequency_bins
            hrtf.fft_length = fft_length_used
            if ir is None:
                message = "Unable to compute IR from TF with the provided frequency_bins."
                warnings.warn(message, UserWarning)
            if resolved_sample_rate is None:
                warnings.warn("Unable to infer samplerate from frequency_bins.", UserWarning)
            hrtf.sofa_convention = convention
            return hrtf

        message = "Unable to determine HRTF domain from SOFA content."
        warnings.warn(message, UserWarning)
        hrtf = cls(Sofa)
        hrtf.fft_length = fft_length
        hrtf.sofa_convention = convention
        return hrtf
