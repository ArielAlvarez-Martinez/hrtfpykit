from __future__ import annotations

from typing import TYPE_CHECKING

import warnings
import numpy as np

from .dsp import compute_ir_from_tf, compute_tf_from_ir

if TYPE_CHECKING:
    from .hrtf import HRTF
    from .sofa.core import SOFA


def from_sofa(
    hrtf_cls: type["HRTF"],
    Sofa: "SOFA",
    SampleRate_override: float | None = None,
    FFT_length: int | None = None,
) -> "HRTF":
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
        SampleRate = SampleRate_override
        if SampleRate is None:
            if "Data.SamplingRate" in variable_names:
                data = np.asarray(
                    variables.get("Data.SamplingRate").value,
                    dtype=float,
                )
                if data.size > 0:
                    SampleRate = int(data.flat[0])
        tf = None
        freqs = None
        n_fft_used = None
        if SampleRate is None:
            message = "Missing Data.SamplingRate; cannot compute TF from IR."
            warnings.warn(message, UserWarning)
        else:
            tf, freqs, n_fft_used = compute_tf_from_ir(
                ir,
                SampleRate,
                fft_length=FFT_length,
            )
        hrtf = hrtf_cls(Sofa)
        hrtf.ir = ir
        hrtf.tf = tf
        hrtf.sample_rate = SampleRate
        hrtf.frequency_bins = freqs
        hrtf.fft_length = FFT_length
        if n_fft_used is not None:
            hrtf.fft_length = n_fft_used
        hrtf.sofa_convention = convention
        return hrtf

    if (
        convention == "SimpleFreeFieldHRTF"
        or ("Data.Real" in variable_names and "Data.Imag" in variable_names)
    ):
        real = np.asarray(variables.get("Data.Real").value, dtype=float)
        imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
        tf = real + 1j * imag
        freqs = None
        if "N" in variable_names:
            data = np.asarray(variables.get("N").value, dtype=float)
            if data.size > 0:
                freqs = data
        if freqs is None:
            message = "Missing N frequency axis; cannot compute IR from TF."
            warnings.warn(message, UserWarning)
        else:
            if freqs.ndim == 1 and freqs.size > 0:
                if float(np.min(freqs)) >= 0.0 and not np.isclose(freqs[0], 0.0):
                    warnings.warn(
                        "Frequency axis should start at 0 Hz to compute IR from TF.",
                        UserWarning,
                    )
        normalization_value = None
        if "Normalization" in variable_names:
            norm_data = np.asarray(variables.get("Normalization").value, dtype=float)
            if norm_data.size > 0:
                normalization_value = float(norm_data.flat[0])
        ir, SampleRate, n_fft_used = compute_ir_from_tf(
            tf,
            freqs,
            fft_length=FFT_length,
            normalization=normalization_value,
        )
        if SampleRate_override is not None:
            SampleRate = SampleRate_override
        if ir is None:
            message = "Unable to compute IR from TF with the provided frequency axis."
            warnings.warn(message, UserWarning)
        if SampleRate is None:
            warnings.warn("Unable to infer samplerate from frequency axis.", UserWarning)
        hrtf = hrtf_cls(Sofa)
        hrtf.ir = ir
        hrtf.tf = tf
        hrtf.sample_rate = SampleRate
        hrtf.frequency_bins = freqs
        hrtf.fft_length = FFT_length
        if n_fft_used is not None:
            hrtf.fft_length = n_fft_used
        hrtf.sofa_convention = convention
        return hrtf

    message = "Unable to determine HRTF domain from SOFA content."
    warnings.warn(message, UserWarning)
    hrtf = hrtf_cls(Sofa)
    hrtf.fft_length = FFT_length
    hrtf.sofa_convention = convention
    return hrtf
