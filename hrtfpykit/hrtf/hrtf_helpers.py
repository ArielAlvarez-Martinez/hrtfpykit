from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import numpy as np

from ..sofa.sofa import SOFA
from ..dsp import ir_from_tf, prepend_missing_dc, tf_from_ir


@dataclass(frozen=True)
class ParsedSOFAData:
    convention: str
    ir: np.ndarray
    sample_rate: float
    tf: np.ndarray
    frequency_bins: np.ndarray
    fft_length: int | None


def _extract_sofa_convention(sofa: SOFA, path: str | Path | None) -> str:
    global_attrs = sofa.GlobalAttributes
    if global_attrs is None:
        raise ValueError("SOFA dataset is not loaded")
    try:
        convention = cast(Any, global_attrs.get("SOFAConventions")).value
    except ValueError:
        convention = None
    allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
    if convention not in allowed:
        if path is None:
            context = "dataset."
        else:
            context = f"{path!s}."
        raise ValueError(
            "SOFAConventions is not an HRTF convention. "
            f"Expected one of {sorted(allowed)}, got {convention!r} for {context}"
        )
    return str(convention)


def _parse_simple_free_field_hrir(
    sofa: SOFA,
    *,
    fft_length: int | None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int | None]:
    variables = sofa.Variables
    if variables is None:
        raise ValueError("SOFA dataset is not loaded")
    variable_names = set(variables.get_names())
    if "Data.IR" not in variable_names:
        raise ValueError(
            "SimpleFreeFieldHRIR requires variable 'Data.IR', but it is missing."
        )
    ir = np.asarray(cast(Any, variables.get("Data.IR")).value)
    if ir.size == 0 or np.all(ir == 0):
        raise ValueError("SimpleFreeFieldHRIR requires non empty 'Data.IR'.")
    if "Data.SamplingRate" not in variable_names:
        raise ValueError(
            "SimpleFreeFieldHRIR requires variable 'Data.SamplingRate', but it is missing."
        )
    sample_rate_data = np.asarray(
        cast(Any, variables.get("Data.SamplingRate")).value,
        dtype=float,
    )
    if sample_rate_data.size == 0 or np.all(sample_rate_data == 0):
        raise ValueError(
            "SimpleFreeFieldHRIR requires non empty 'Data.SamplingRate'."
        )
    sample_rate = float(sample_rate_data.flat[0])
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(
            "SimpleFreeFieldHRIR requires a finite, positive 'Data.SamplingRate' value."
        )
    tf, frequency_bins, fft_length_used = tf_from_ir(ir, sample_rate, fft_length=fft_length)
    return ir, sample_rate, np.asarray(tf), np.asarray(frequency_bins), fft_length_used


def _parse_simple_free_field_hrtf(
    sofa: SOFA,
    *,
    fft_length: int | None,
    mesh2hrtf_compatible: bool,
    mesh2hrtf_n_shift: int | None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int | None]:
    variables = sofa.Variables
    if variables is None:
        raise ValueError("SOFA dataset is not loaded")
    required_variables = ("Data.Real", "Data.Imag", "N")
    variable_names = set(variables.get_names())
    missing_variables = [name for name in required_variables if name not in variable_names]
    if missing_variables:
        raise ValueError(
            "SimpleFreeFieldHRTF requires variables "
            f"{required_variables}, but missing: {missing_variables}."
        )

    real = np.asarray(cast(Any, variables.get("Data.Real")).value, dtype=float)
    if real.size == 0:
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'Data.Real'.")
    imag = np.asarray(cast(Any, variables.get("Data.Imag")).value, dtype=float)
    if imag.size == 0 or imag.shape != real.shape:
        raise ValueError(
            "SimpleFreeFieldHRTF requires 'Data.Imag' with the same shape as 'Data.Real'."
        )

    tf = real + 1j * imag
    if not np.isfinite(tf).all():
        raise ValueError("SimpleFreeFieldHRTF has invalid complex TF values.")
    if tf.size == 0 or not np.any(tf):
        raise ValueError("SimpleFreeFieldHRTF requires non empty complex TF.")

    frequency_bins = np.asarray(cast(Any, variables.get("N")).value, dtype=float)
    if frequency_bins.size == 0 or np.all(frequency_bins == 0):
        raise ValueError("SimpleFreeFieldHRTF requires non empty 'N'.")

    tf, frequency_bins = prepend_missing_dc(tf, frequency_bins)
    ir, sample_rate, fft_length_used = ir_from_tf(
        tf,
        frequency_bins=frequency_bins,
        mesh2hrtf_compatible=mesh2hrtf_compatible,
        n_shift=mesh2hrtf_n_shift,
    )
    if fft_length is not None and fft_length != fft_length_used:
        raise ValueError("FFT length does not match the provided frequency bins.")
    return ir, float(sample_rate), tf, np.asarray(frequency_bins), fft_length_used


def parse_sofa_data(
    sofa: SOFA,
    *,
    path: str | Path | None = None,
    fft_length: int | None = None,
    mesh2hrtf_compatible: bool = False,
    mesh2hrtf_n_shift: int | None = 30,
) -> ParsedSOFAData:
    convention = _extract_sofa_convention(sofa, path=path)
    if convention == "SimpleFreeFieldHRIR":
        ir, sample_rate, tf, frequency_bins, fft_length_used = (
            _parse_simple_free_field_hrir(
                sofa,
                fft_length=fft_length,
            )
        )
    else:
        ir, sample_rate, tf, frequency_bins, fft_length_used = (
            _parse_simple_free_field_hrtf(
                sofa,
                fft_length=fft_length,
                mesh2hrtf_compatible=mesh2hrtf_compatible,
                mesh2hrtf_n_shift=mesh2hrtf_n_shift,
            )
        )

    return ParsedSOFAData(
        convention=convention,
        ir=np.asarray(ir),
        sample_rate=float(sample_rate),
        tf=np.asarray(tf),
        frequency_bins=np.asarray(frequency_bins),
        fft_length=int(fft_length_used) if fft_length_used is not None else None,
    )
