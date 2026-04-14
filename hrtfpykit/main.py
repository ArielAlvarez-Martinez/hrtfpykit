from pathlib import Path

import numpy as np

from .hrtf.dsp import ir_from_tf, tf_from_ir
from .hrtf.hrtf import HRTF
from .sofa import SOFA


def load_hrtf(
    path: str | Path,
    mode: str = "r",
    parallel: bool = False,
    check_sofa_against_conventions: bool = True,
    fft_length: int | None = None,
) -> HRTF:
    """Load a SOFA file as an :class:`HRTF` object.

    The loader supports both ``SimpleFreeFieldHRIR`` and
    ``SimpleFreeFieldHRTF`` conventions, validates required variables, and
    populates synchronized time and frequency representations in the returned
    object.

    Parameters
    ----------
    path : str | Path
        Path to the SOFA file.
    mode : str, default='r'
        File mode used by the SOFA API.
    parallel : bool, default=False
        Whether to enable parallel loading in the SOFA API.
    check_sofa_against_conventions : bool, default=True
        Whether to run convention checks when reading the SOFA file.
    fft_length : int | None, default=None
        Optional FFT length used when deriving TF from HRIR content.

    Returns
    -------
    HRTF
        Loaded HRTF object with ``IR``, ``TF``, ``SOFAConventions``, and
        ``fft_length`` populated.

    Use Cases
    ---------
    - Load HRIR-based SOFA files and work in both domains.
    - Load HRTF-based SOFA files while preserving original frequency bins.
    - Standardize project entrypoint as ``from hrtfpykit import load_hrtf``.

    Examples
    --------
    >>> from hrtfpykit import load_hrtf
    >>> hrtf = load_hrtf("hrtfs/P0001_FreeFieldComp_44kHz.sofa")
    >>> hrtf.SOFAConventions
    'SimpleFreeFieldHRIR'

    >>> hrtf_tf = load_hrtf("hrtfs/HRTF_TF.sofa")
    >>> hrtf_tf.SOFAConventions
    'SimpleFreeFieldHRTF'

    Best Practices
    --------------
    - Keep ``check_sofa_against_conventions=True`` in production pipelines.
    - Use ``fft_length`` only when a fixed transform size is explicitly needed.
    - Fail fast on malformed SOFA variables instead of bypassing validation.
    """
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
        hrtf = HRTF(Sofa)
        hrtf.IR.values = ir
        hrtf.IR.sample_rate = resolved_sample_rate
        hrtf.TF.values = tf
        hrtf.TF.frequency_bins = frequency_bins
        hrtf.fft_length = fft_length
        if fft_length_used is not None:
            hrtf.fft_length = fft_length_used
        hrtf.SOFAConventions = convention
        return hrtf

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
    hrtf = HRTF(Sofa)
    hrtf.IR.values = ir
    hrtf.IR.sample_rate = resolved_sample_rate
    hrtf.TF.values = tf
    hrtf.TF.frequency_bins = frequency_bins
    hrtf.fft_length = fft_length_used
    hrtf.SOFAConventions = convention
    return hrtf
