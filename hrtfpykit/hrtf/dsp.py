from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal
if TYPE_CHECKING:
    from .domain import IR, TF


def signal_duration(
    signal: np.ndarray | "IR",
    sample_rate: float | None = None,
) -> float:
    """Compute the duration of a time-domain signal.

    Duration is derived from the length of the final axis. The function
    accepts a raw NumPy array or an :class:`~hrtfpykit.hrtf.domain.IR` domain object, which makes it usable
    for standalone signals and for HRTF/HRIR data stored inside an
    :class:`~hrtfpykit.hrtf.domain.IR` container. Leading source or ear axes do
    not affect the result.

    Parameters
    ----------
    signal : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with
        :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`. The final axis is
        interpreted as time samples.
    sample_rate : float | None, default=None
        Sample rate in hertz. Required for NumPy input. When signal is an
        :class:`~hrtfpykit.hrtf.domain.IR` object and this value is omitted,
        :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is used.

    Returns
    -------
    float
        Duration in seconds, computed from the final sample axis and
        ``sample_rate``.

    Raises
    ------
    ValueError
        If signal data are missing, are not a NumPy array, have no sample
        axis, or if no finite positive sample rate can be resolved.
    """
    if isinstance(signal, np.ndarray):
        signal_values = signal
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(signal, "values") or not hasattr(signal, "sample_rate"):
            raise ValueError("signal must be a NumPy array or an IR instance")
        signal_values = signal.values
        resolved_sample_rate = sample_rate if sample_rate is not None else signal.sample_rate

    if signal_values is None:
        raise ValueError("Signal data is not available")
    if not isinstance(signal_values, np.ndarray):
        raise ValueError("Signal data must be a NumPy array")
    if signal_values.ndim == 0:
        raise ValueError("Signal data must have at least one dimension")

    if resolved_sample_rate is None:
        raise ValueError("sample_rate is required")
    if isinstance(resolved_sample_rate, bool):
        raise ValueError("sample_rate must be a finite, positive value.")
    try:
        resolved_sample_rate = float(resolved_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a finite, positive value.") from None
    if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
        raise ValueError("sample_rate must be a finite, positive value.")

    return float(signal_values.shape[-1]) / resolved_sample_rate


def magnitude(tf: np.ndarray | "TF") -> np.ndarray:
    """Return transfer-function magnitudes.

    This is the basic magnitude accessor for frequency-domain HRTF data. It
    accepts raw complex arrays and :class:`~hrtfpykit.hrtf.domain.TF` domain objects, and returns
    numpy.abs of the resolved values without modifying the input.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain transfer-function values or
        :class:`~hrtfpykit.hrtf.domain.TF` object with
        :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.

    Returns
    -------
    np.ndarray
        Magnitude values computed as abs(tf) with the same shape as the
        input.

    Raises
    ------
    ValueError
        If TF data are missing or are not stored as a NumPy array.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    return np.abs(tf_values)


def magnitude_to_db(
    magnitude: np.ndarray,
    reference: float | str = 1.0,
) -> np.ndarray:
    """Convert linear magnitude values to decibels.

    The conversion uses 20 * log10(magnitude / reference). Zero magnitude
    values are allowed and produce -inf in the output. Negative magnitudes
    are rejected because they are not valid linear magnitudes.

    Parameters
    ----------
    magnitude : np.ndarray
        Non-negative magnitude values.
    reference : float | {``max``}, default=1.0
        Positive reference magnitude used in the conversion
        20 * log10(magnitude / reference). The special value ``max``
        uses the maximum magnitude present in the input array.

    Returns
    -------
    np.ndarray
        Magnitude values in dB with the same shape as the input array.

    Raises
    ------
    ValueError
        If any magnitude is negative, if reference is not finite and
        positive, if reference is an unsupported string, or if
        reference=``max`` is requested for data with no positive maximum.
    """
    magnitude_values = np.asarray(magnitude, dtype=float)
    if np.any(magnitude_values < 0.0):
        raise ValueError("magnitude values must be non-negative")
    if isinstance(reference, str):
        reference_key = str(reference).strip().lower()
        if reference_key != "max":
            raise ValueError("reference must be a finite, positive float or 'max'")
        reference_value = float(np.max(magnitude_values))
        if not np.isfinite(reference_value) or reference_value <= 0.0:
            raise ValueError("reference='max' requires at least one positive magnitude value")
    else:
        reference_value = float(reference)
        if not np.isfinite(reference_value) or reference_value <= 0.0:
            raise ValueError("reference must be a finite, positive float or 'max'")
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(magnitude_values / reference_value)


def db_to_magnitude(
    magnitude_db: np.ndarray,
    reference: float | str = 1.0,
) -> np.ndarray:
    """Convert decibel magnitudes back to linear values.

    This is the inverse of magnitude-to-decibel conversion for numeric references. It
    evaluates reference * 10 ** (magnitude_db / 20) and preserves the input
    shape.

    Parameters
    ----------
    magnitude_db : np.ndarray
        Magnitude values in decibels.
    reference : float, default=1.0
        Positive reference magnitude used in the inverse conversion.
        ``max`` is not supported here.

    Returns
    -------
    np.ndarray
        Linear magnitude values with the same shape as the input array.

    Raises
    ------
    ValueError
        If reference is not finite and positive, or if the unsupported
        special value reference=``max`` is provided.
    """
    magnitude_db_values = np.asarray(magnitude_db, dtype=float)
    if isinstance(reference, str):
        reference_key = str(reference).strip().lower()
        if reference_key == "max":
            raise ValueError("db_to_magnitude does not accept reference='max'")
        raise ValueError("reference must be a finite, positive float")
    reference_value = float(reference)
    if not np.isfinite(reference_value) or reference_value <= 0.0:
        raise ValueError("reference must be a finite, positive float")
    return reference_value * (10.0 ** (magnitude_db_values / 20.0))


def magnitude_db(
    tf: np.ndarray | "TF",
    reference: float | str = 1.0,
) -> np.ndarray:
    """Return transfer-function magnitudes directly in decibels.

    This convenience wrapper computes linear magnitude calculation and
    then converts the result with
    magnitude-to-decibel conversion. It is intended for HRTF plotting,
    spectral comparison, and any workflow that needs TF magnitude on a dB
    scale.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object
        with :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.
    reference : float | {``max``}, default=1.0
        Positive reference magnitude used in the dB conversion. The special
        value ``max`` uses the maximum magnitude present in the input TF.

    Returns
    -------
    np.ndarray
        Magnitude values in dB with the same shape as the TF input.

    Raises
    ------
    ValueError
        If TF data cannot be resolved, if any magnitude is invalid, or if
        reference is not accepted by
        magnitude-to-decibel conversion.
    """
    magnitude_values = magnitude(tf)
    return magnitude_to_db(magnitude_values, reference=reference)


def phase(tf: np.ndarray | "TF", unit: str = "degrees") -> np.ndarray:
    """Return transfer-function phase values.

    Phase is computed with numpy.angle from the complex TF representation.
    The function accepts raw NumPy arrays and :class:`~hrtfpykit.hrtf.domain.TF` domain objects and returns
    one phase value for each complex TF value.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object
        with :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.
    unit : str, default=``degrees``
        Output unit. Degree and radian aliases are supported.

    Returns
    -------
    np.ndarray
        Phase values in the requested unit with the same shape as the input TF.

    Raises
    ------
    ValueError
        If TF data are missing, are not stored as a NumPy array, or unit is
        not a degree or radian alias.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    unit_key = str(unit).strip().lower()
    if unit_key in {"degrees", "degree", "deg"}:
        return np.angle(tf_values, deg=True)
    if unit_key in {"radians", "radian", "rad"}:
        return np.angle(tf_values, deg=False)
    raise ValueError("unit must be one of: degrees, radians")


def modify_phase(
    tf: np.ndarray | "TF",
    new_phase: np.ndarray,
    unit: str = "degrees",
) -> np.ndarray:
    """Replace TF phase values while preserving the original magnitude.

    The returned TF has the magnitude of tf and the phase supplied in
    new_phase. This is useful when building phase-normalized HRTFs,
    applying externally computed phase estimates, or synchronizing a modified
    phase array with an unchanged magnitude response.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object
        with :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.
    new_phase : np.ndarray
        Phase array with exactly the same shape as the TF values.
    unit : str, default=``degrees``
        Phase unit used by new_phase. Degree and radian aliases are
        supported.

    Returns
    -------
    np.ndarray
        Complex TF values with the original magnitude and the new phase.

    Raises
    ------
    ValueError
        If TF data are missing, are not stored as a NumPy array, if
        new_phase does not match the TF shape, or if unit is not a
        degree or radian alias.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")

    new_phase_values = np.asarray(new_phase, dtype=float)
    if new_phase_values.shape != tf_values.shape:
        raise ValueError("new_phase must match TF shape")

    unit_key = str(unit).strip().lower()
    if unit_key in {"degrees", "degree", "deg"}:
        phase_radians = np.deg2rad(new_phase_values)
    elif unit_key in {"radians", "radian", "rad"}:
        phase_radians = new_phase_values
    else:
        raise ValueError("unit must be one of: degrees, radians")

    magnitude_values = np.abs(tf_values)
    return magnitude_values * np.exp(1j * phase_radians)


def modify_magnitude(
    tf: np.ndarray | "TF",
    new_magnitude: np.ndarray,
    scale: str = "linear",
) -> np.ndarray:
    """Replace TF magnitude values while preserving the original phase.

    The returned TF keeps the phase of tf and uses new_magnitude as the
    magnitude response. Magnitudes can be supplied directly on a linear scale
    or as dB values referenced to 1.0. This helper is used by higher-level
    HRTF transforms that edit spectral level while preserving phase.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object
        with :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`.
    new_magnitude : np.ndarray
        Magnitude array with exactly the same shape as the TF values.
    scale : str, default=``linear``
        Scale of new_magnitude. Supported values are linear and db.

    Returns
    -------
    np.ndarray
        Complex TF values with the new magnitude and the original phase.

    Raises
    ------
    ValueError
        If TF data are missing, are not stored as a NumPy array, if
        new_magnitude does not match the TF shape, if scale is
        unsupported, or if resolved linear magnitudes are negative.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")

    new_magnitude_values = np.asarray(new_magnitude, dtype=float)
    if new_magnitude_values.shape != tf_values.shape:
        raise ValueError("new_magnitude must match TF shape")

    scale_key = str(scale).strip().lower()
    if scale_key == "linear":
        magnitude_values = new_magnitude_values
    elif scale_key in {"db", "decibel", "decibels"}:
        magnitude_values = db_to_magnitude(new_magnitude_values)
    else:
        raise ValueError("scale must be one of: linear, db")

    if np.any(magnitude_values < 0.0):
        raise ValueError("new_magnitude must be non-negative")

    phase_values = np.angle(tf_values)
    return magnitude_values * np.exp(1j * phase_values)


def tf_gain(
    tf: np.ndarray | "TF",
    gain: float | np.ndarray,
    scale: str = "db",
) -> np.ndarray:
    """Apply a scalar or broadcastable gain to TF values.

    Gain is applied as a real multiplier to the complex TF array, so phase is
    preserved exactly. The gain can be a scalar, a per-ear/per-position/per-bin
    array, or any array that NumPy can broadcast to the TF shape.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object with .values.
    gain : float | np.ndarray
        Gain applied to the TF magnitude while preserving phase. Scalar gains
        affect every source, ear, and bin equally. Array gains must be
        broadcast-compatible with the TF shape. In scale=``db``, negative
        values attenuate and positive values amplify.
    scale : {``linear``, ``db``}, default=``db``
        Scale used by gain.

    Returns
    -------
    np.ndarray
        Complex TF values after gain application, with the same shape as the
        input TF.

    Raises
    ------
    ValueError
        If TF data are missing, are not stored as a NumPy array, if gain is
        empty, non-finite, negative on a linear scale, not broadcast-compatible
        with the TF shape, or if scale is unsupported.

    Notes
    -----
    This function is a generic TF-domain gain utility. It multiplies the
    existing complex TF by a real gain factor and therefore preserves the
    original phase.

    In scale=``db``, the gain is converted with 10 ** (gain / 20). In
    scale=``linear``, gain values must be non-negative. Use negative dB to
    attenuate and positive dB to amplify.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")

    gain_values = np.asarray(gain, dtype=float)
    if gain_values.size == 0:
        raise ValueError("gain must be non-empty")
    if not np.all(np.isfinite(gain_values)):
        raise ValueError("gain must contain only finite values")

    scale_key = str(scale).strip().lower()
    if scale_key == "linear":
        if np.any(gain_values < 0.0):
            raise ValueError("linear gain values must be non-negative")
        gain_factor = gain_values
    elif scale_key in {"db", "decibel", "decibels"}:
        gain_factor = db_to_magnitude(gain_values, reference=1.0)
    else:
        raise ValueError("scale must be one of: linear, db")

    try:
        gain_factor = np.broadcast_to(gain_factor, tf_values.shape)
    except ValueError:
        raise ValueError("gain must be broadcast-compatible with TF shape") from None

    return tf_values * gain_factor


def real(tf: np.ndarray | "TF") -> np.ndarray:
    """Return the real part of transfer-function values.

    This helper extracts the real component of raw TF arrays or :class:`~hrtfpykit.hrtf.domain.TF` domain
    objects without modifying the input. It is primarily a convenience for
    domain properties and diagnostic workflows.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object with .values.

    Returns
    -------
    np.ndarray
        Real component of the TF values with the same shape as the input.

    Raises
    ------
    ValueError
        If TF data are missing or are not stored as a NumPy array.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    return np.real(tf_values)


def imag(tf: np.ndarray | "TF") -> np.ndarray:
    """Return the imaginary part of transfer-function values.

    This helper extracts the imaginary component of raw TF arrays or :class:`~hrtfpykit.hrtf.domain.TF`
    domain objects without modifying the input. It is primarily a convenience
    for domain properties and diagnostic workflows.

    Parameters
    ----------
    tf : np.ndarray | TF
        Frequency-domain array or :class:`~hrtfpykit.hrtf.domain.TF` object with .values.

    Returns
    -------
    np.ndarray
        Imaginary component of the TF values with the same shape as the input.

    Raises
    ------
    ValueError
        If TF data are missing or are not stored as a NumPy array.
    """
    if isinstance(tf, np.ndarray):
        tf_values = tf
    else:
        if not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_values = tf.values
    if tf_values is None:
        raise ValueError("TF data is not available")
    if not isinstance(tf_values, np.ndarray):
        raise ValueError("TF data must be a NumPy array")
    return np.imag(tf_values)


def upsampling(
    ir: np.ndarray | "IR",
    new_sample_rate: float,
    sample_rate: float | None = None,
) -> tuple[np.ndarray, float]:
    """Upsample an IR signal using polyphase resampling.

    The function resamples along the final axis and preserves all leading
    axes, including source and ear axes for HRIR data. It accepts raw NumPy
    arrays and :class:`~hrtfpykit.hrtf.domain.IR` domain objects. For :class:`~hrtfpykit.hrtf.domain.IR` input, :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is
    used when sample_rate is omitted.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with .values. The final axis
        is interpreted as time samples.
    new_sample_rate : float
        Target sample rate in hertz. It must be strictly greater than the
        current sample rate.
    sample_rate : float | None, default=None
        Source sample rate in hertz. Required for NumPy input. Optional for
        :class:`~hrtfpykit.hrtf.domain.IR` input when :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is available.

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple (resampled_ir, resolved_new_sample_rate).

    Raises
    ------
    ValueError
        If IR data are missing, empty, scalar, or not stored as a NumPy array,
        if either sample rate is not finite and positive, or if
        new_sample_rate is not greater than the current sample rate.

    Notes
    -----
    Polyphase resampling uses a rational approximation of
    new_sample_rate / sample_rate with denominator limited to 10000.
    """
    if isinstance(ir, np.ndarray):
        ir_values = ir
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(ir, "values") or not hasattr(ir, "sample_rate"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_values = ir.values
        resolved_sample_rate = sample_rate if sample_rate is not None else ir.sample_rate

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

    if resolved_sample_rate is None:
        raise ValueError("sample_rate is required")
    if isinstance(resolved_sample_rate, bool):
        raise ValueError("sample_rate must be a finite, positive value.")
    try:
        resolved_sample_rate = float(resolved_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a finite, positive value.") from None
    if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
        raise ValueError("sample_rate must be a finite, positive value.")

    if isinstance(new_sample_rate, bool):
        raise ValueError("new_sample_rate must be a finite, positive value.")
    try:
        new_sample_rate = float(new_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("new_sample_rate must be a finite, positive value.") from None
    if not np.isfinite(new_sample_rate) or new_sample_rate <= 0.0:
        raise ValueError("new_sample_rate must be a finite, positive value.")
    if new_sample_rate <= resolved_sample_rate:
        raise ValueError("new_sample_rate must be greater than current sample_rate for upsampling")

    ratio = Fraction(new_sample_rate / resolved_sample_rate).limit_denominator(10000)
    resampled_ir = signal.resample_poly(
        ir_values,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )
    return resampled_ir, new_sample_rate


def downsampling(
    ir: np.ndarray | "IR",
    new_sample_rate: float,
    sample_rate: float | None = None,
) -> tuple[np.ndarray, float]:
    """Downsample an IR signal using polyphase resampling.

    The function resamples along the final axis and preserves all leading
    axes, including source and ear axes for HRIR data. It accepts raw NumPy
    arrays and :class:`~hrtfpykit.hrtf.domain.IR` domain objects. For :class:`~hrtfpykit.hrtf.domain.IR` input, :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is
    used when sample_rate is omitted.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with .values. The final axis
        is interpreted as time samples.
    new_sample_rate : float
        Target sample rate in hertz. It must be strictly lower than the
        current sample rate.
    sample_rate : float | None, default=None
        Source sample rate in hertz. Required for NumPy input. Optional for
        :class:`~hrtfpykit.hrtf.domain.IR` input when :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is available.

    Returns
    -------
    tuple[np.ndarray, float]
        Tuple (resampled_ir, resolved_new_sample_rate).

    Raises
    ------
    ValueError
        If IR data are missing, empty, scalar, or not stored as a NumPy array,
        if either sample rate is not finite and positive, or if
        new_sample_rate is not lower than the current sample rate.

    Notes
    -----
    Polyphase resampling uses a rational approximation of
    new_sample_rate / sample_rate with denominator limited to 10000.
    """
    if isinstance(ir, np.ndarray):
        ir_values = ir
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(ir, "values") or not hasattr(ir, "sample_rate"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_values = ir.values
        resolved_sample_rate = sample_rate if sample_rate is not None else ir.sample_rate

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

    if resolved_sample_rate is None:
        raise ValueError("sample_rate is required")
    if isinstance(resolved_sample_rate, bool):
        raise ValueError("sample_rate must be a finite, positive value.")
    try:
        resolved_sample_rate = float(resolved_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a finite, positive value.") from None
    if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
        raise ValueError("sample_rate must be a finite, positive value.")

    if isinstance(new_sample_rate, bool):
        raise ValueError("new_sample_rate must be a finite, positive value.")
    try:
        new_sample_rate = float(new_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("new_sample_rate must be a finite, positive value.") from None
    if not np.isfinite(new_sample_rate) or new_sample_rate <= 0.0:
        raise ValueError("new_sample_rate must be a finite, positive value.")
    if new_sample_rate >= resolved_sample_rate:
        raise ValueError("new_sample_rate must be lower than current sample_rate for downsampling")

    ratio = Fraction(new_sample_rate / resolved_sample_rate).limit_denominator(10000)
    resampled_ir = signal.resample_poly(
        ir_values,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=-1,
    )
    return resampled_ir, new_sample_rate


def window(ir: np.ndarray | "IR", window_name: str) -> np.ndarray:
    """Apply a named time-domain window to IR samples.

    Windowing is applied along the final axis and is broadcast over all leading
    axes. This makes the function suitable for mono signals, binaural HRIRs,
    and full (positions, ears, samples) HRTF arrays.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with .values. The final axis
        is interpreted as time samples.
    window_name : str
        Window identifier. Supported values are hann, hamming,
        blackman, and rectangular.

    Returns
    -------
    np.ndarray
        Windowed IR values with the same shape as the input.

    Raises
    ------
    ValueError
        If IR data are missing, scalar, or not stored as a NumPy array, if the
        final axis contains no samples, or if window_name is unsupported.
    """
    if isinstance(ir, np.ndarray):
        ir_values = ir
    else:
        if not hasattr(ir, "values"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_values = ir.values
    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")
    length = ir_values.shape[-1]
    if length <= 0:
        raise ValueError("IR data must contain at least one sample")
    key = window_name.strip().lower()
    if key in {"hann", "hanning"}:
        window_values = np.hanning(length)
    elif key in {"rectangular"}:
        window_values = np.ones(length)
    elif key == "hamming":
        window_values = np.hamming(length)
    elif key == "blackman":
        window_values = np.blackman(length)
    else:
        raise ValueError(
            "window_name must be one of: hann, hamming, blackman, rectangular"
        )
    return ir_values * window_values

def padding(
    ir: np.ndarray | "IR",
    padding_length: int,
    location: str = "end",
    value: float | complex = 0,
) -> np.ndarray:
    """Pad impulse-response values along the sample axis.

    The function accepts either a raw NumPy array or an :class:`~hrtfpykit.hrtf.domain.IR` domain object
    and appends or prepends constant samples on the last axis. The spatial and
    ear axes are preserved exactly; only the time-sample length changes.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with .values. The final axis
        is interpreted as time samples.
    padding_length : int
        Number of samples added to the IR. 0 returns the original array.
    location : {``start``, ``end``}, default=``end``
        Side where the padding is applied.
    value : float | complex, default=0
        Constant value used in the padded region.

    Returns
    -------
    np.ndarray
        Padded IR array with the same leading shape as the input.

    Raises
    ------
    ValueError
        If IR data are missing, empty, or not stored as a NumPy array, if
        padding_length is not a non-negative integer, or if location is
        unsupported.
    """

    if isinstance(ir, np.ndarray):
        ir_values = ir
    elif hasattr(ir, "values") and hasattr(ir, "sample_rate"):
        ir_values = ir.values
    else:
        ir_values = None
    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if isinstance(padding_length, bool) or not isinstance(padding_length, int):
        raise ValueError("Padding must be an integer")
    if padding_length < 0:
        raise ValueError("Padding must be non-negative")
    if padding_length == 0:
        return ir_values
    location_key = location.strip().lower()
    if location_key == "start":
        before, after = padding_length, 0
    elif location_key == "end":
        before, after = 0, padding_length
    else:
        raise ValueError("Padding location must be 'start' or 'end'")
    pad_width = [(0, 0)] * (ir_values.ndim - 1) + [(before, after)]
    return np.pad(
        ir_values,
        pad_width,
        mode="constant",
        constant_values=value,
    )


def fir_filter(
    ir: np.ndarray | "IR",
    filter: str,
    sample_rate: float | None = None,
    cutoff: float | tuple[float, float] | None = None,
    num_taps: int = 101,
    window: str | None = None,
) -> np.ndarray:
    """Apply a finite impulse response filter to time-domain IR data.

    The filter is designed from the requested family, cutoff, and sample rate,
    then applied along the last axis of the IR array. The function supports raw
    NumPy arrays and :class:`~hrtfpykit.hrtf.domain.IR` domain objects so it can be used both inside the
    HRTF abstraction and in lower-level DSP workflows.

    Low-pass, high-pass, and band-pass filters are designed with
    scipy.signal.firwin and applied with centered same convolution.
    The output therefore keeps the same shape as the input.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with .values. The final axis
        is filtered independently.
    filter : str
        Filter family. Supported aliases are ``lowpass``, ``low-pass``,
        ``lp``, ``highpass``, ``high-pass``, ``hp``,
        ``bandpass``, ``band-pass``, and ``bp``.
    sample_rate : float | None, default=None
        Sample rate in hertz used to interpret cutoff and the Nyquist
        frequency.
    cutoff : float | tuple[float, float] | None, default=None
        Cutoff frequency in hertz. Use a scalar for low-pass or high-pass
        filtering and (low, high) for band-pass filtering.
    num_taps : int, default=101
        Positive odd FIR length.
    window : str | None, default=None
        FIR design window. None and ``rectangular`` use a boxcar
        window. Other supported values are ``hann``, ``hamming``, and
        ``blackman``.

    Returns
    -------
    np.ndarray
        Filtered IR values with the same shape as the input.

    Raises
    ------
    ValueError
        If IR data are missing, sample_rate or cutoff is omitted,
        num_taps is not a positive odd integer, window or filter is
        unsupported, or cutoff frequencies do not satisfy the required
        relation to Nyquist.
    """
    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        raise ValueError("IR data is not available")

    filter_type = str(filter).strip().lower()
    if sample_rate is None:
        raise ValueError("sample_rate is required for filters")
    if cutoff is None:
        raise ValueError("cutoff is required for filters")
    if isinstance(num_taps, bool) or not isinstance(num_taps, int):
        raise ValueError("num_taps must be an integer")
    if num_taps <= 0:
        raise ValueError("num_taps must be positive")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd")

    window_value = None
    if window is None:
        window_value = "boxcar"
    else:
        window_type = str(window).strip().lower()
        if window_type in {"hann", "hanning"}:
            window_value = "hann"
        elif window_type in {"rectangular"}:
            window_value = "boxcar"
        elif window_type == "hamming":
            window_value = "hamming"
        elif window_type == "blackman":
            window_value = "blackman"
        else:
            raise ValueError("window must be one of: hann, hamming, blackman, rectangular")
    nyquist = 0.5 * sample_rate
    if filter_type in {"lowpass", "low-pass", "lp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for lowpass")
        kernel_values = signal.firwin(
            num_taps,
            cutoff_value,
            window=window_value,
            pass_zero=True,
            fs=sample_rate,
        )
    elif filter_type in {"highpass", "high-pass", "hp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for highpass")
        kernel_values = signal.firwin(
            num_taps,
            cutoff_value,
            window=window_value,
            pass_zero=False,
            fs=sample_rate,
        )
    elif filter_type in {"bandpass", "band-pass", "bp"}:
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("cutoff must be (low, high) for bandpass")
        cutoff_low = float(cutoff[0])
        cutoff_high = float(cutoff[1])
        if cutoff_low <= 0.0 or cutoff_high >= nyquist or cutoff_low >= cutoff_high:
            raise ValueError("cutoff must satisfy 0 < low < high < Nyquist for bandpass")
        kernel_values = signal.firwin(
            num_taps,
            [cutoff_low, cutoff_high],
            window=window_value,
            pass_zero=False,
            fs=sample_rate,
        )
    else:
        raise ValueError("filter must be one of: lowpass, highpass, bandpass")

    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel_values, mode="same"),
        axis=-1,
        arr=ir,
    )


def iir_filter(
    ir: np.ndarray | "IR",
    filter: str,
    sample_rate: float | None = None,
    cutoff: float | tuple[float, float] | None = None,
    order: int = 10,
) -> np.ndarray:
    """Apply an IIR Butterworth filter to IR data.

    The filter is designed with scipy.signal.butter and applied with
    scipy.signal.lfilter along the final axis. The function accepts raw
    NumPy arrays and :class:`~hrtfpykit.hrtf.domain.IR` domain objects and keeps the input shape.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain samples or :class:`~hrtfpykit.hrtf.domain.IR` object with .values. The final axis
        is filtered independently.
    filter : str
        Filter family. Supported aliases are ``lowpass``, ``low-pass``,
        ``lp``, ``highpass``, ``high-pass``, ``hp``,
        ``bandpass``, ``band-pass``, and ``bp``.
    sample_rate : float | None, default=None
        Sample rate in hertz used to interpret cutoff and the Nyquist
        frequency.
    cutoff : float | tuple[float, float] | None, default=None
        Cutoff frequency in hertz. Use a scalar for low-pass or high-pass
        filtering and (low, high) for band-pass filtering.
    order : int, default=10
        Positive Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered IR values with the same shape as the input.

    Raises
    ------
    ValueError
        If IR data are missing, sample_rate or cutoff is omitted,
        order is not a positive integer, filter is unsupported, or
        cutoff frequencies do not satisfy the required relation to Nyquist.
    """
    if not isinstance(ir, np.ndarray):
        if hasattr(ir, "values"):
            ir = ir.values
        else:
            ir = None
    if ir is None:
        raise ValueError("IR data is not available")

    filter_type = str(filter).strip().lower()
    if sample_rate is None:
        raise ValueError("sample_rate is required for filters")
    if cutoff is None:
        raise ValueError("cutoff is required for filters")
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError("order must be an integer")
    if order <= 0:
        raise ValueError("order must be positive")

    nyquist = 0.5 * sample_rate
    if filter_type in {"lowpass", "low-pass", "lp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for lowpass")
        b, a = signal.butter(order, cutoff_value, btype="lowpass", fs=sample_rate)
    elif filter_type in {"highpass", "high-pass", "hp"}:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0.0 or cutoff_value >= nyquist:
            raise ValueError("cutoff must be between 0 and Nyquist for highpass")
        b, a = signal.butter(order, cutoff_value, btype="highpass", fs=sample_rate)
    elif filter_type in {"bandpass", "band-pass", "bp"}:
        if not isinstance(cutoff, tuple) or len(cutoff) != 2:
            raise ValueError("cutoff must be (low, high) for bandpass")
        cutoff_low = float(cutoff[0])
        cutoff_high = float(cutoff[1])
        if cutoff_low <= 0.0 or cutoff_high >= nyquist or cutoff_low >= cutoff_high:
            raise ValueError("cutoff must satisfy 0 < low < high < Nyquist for bandpass")
        b, a = signal.butter(
            order,
            [cutoff_low, cutoff_high],
            btype="bandpass",
            fs=sample_rate,
        )
    else:
        raise ValueError("filter must be one of: lowpass, highpass, bandpass")

    return signal.lfilter(b, a, ir, axis=-1)


def convolve(
    ir_1: np.ndarray | "IR",
    ir_2: np.ndarray | "IR",
    mode: str = "full",
    method: str = "auto",
) -> np.ndarray:
    """Convolve two IR inputs along the last axis.

    The function performs linear convolution independently over the final axis
    after broadcasting the leading dimensions of both inputs. It accepts raw
    arrays and :class:`~hrtfpykit.hrtf.domain.IR` domain objects. When both inputs are :class:`~hrtfpykit.hrtf.domain.IR` objects,
    their sample rates must be finite, positive, and equal.

    Parameters
    ----------
    ir_1 : np.ndarray | IR
        Time-domain array or :class:`~hrtfpykit.hrtf.domain.IR` object with .values and
        .sample_rate. This is the reference input for the API, so when
        mode=``same`` the output length follows ir_1.
    ir_2 : np.ndarray | IR
        Second time-domain array or :class:`~hrtfpykit.hrtf.domain.IR` object with .values and
        .sample_rate. It is convolved with ir_1 independently along
        the last axis.
    mode : {``full``, ``same``, ``valid``}, default=``full``
        Convolution output mode passed to scipy.signal.convolve.
    method : {``auto``, ``direct``, ``fft``}, default=``auto``
        Convolution method passed to scipy.signal.convolve.

    Returns
    -------
    np.ndarray
        Convolved values with the broadcast leading shape of ir_1 and
        ir_2 and the output length implied by mode.

    Raises
    ------
    ValueError
        If either IR input is missing, empty, scalar, or not stored as a NumPy
        array, if mode or method is unsupported, if both domain inputs
        provide incompatible sample rates, or if leading shapes are not
        broadcast-compatible.

    Notes
    -----
    This is a generic time-domain linear convolution utility. It does not
    apply any HRTF-specific interpretation beyond broadcasting and operating
    along the last axis.

    In particular, this function is not equivalent to frequency-domain
    recomposition helpers such as multiplying a DTF by a CTF on an existing
    FFT grid. Those workflows correspond to circular convolution on the chosen
    FFT length, while this function performs linear convolution and then
    applies the requested mode.

    When mode=``same``, SciPy returns the centered portion of the linear
    convolution with the length of ir_1. That crop is often convenient for
    signal processing, but it discards boundary samples and therefore should
    not be treated as an exact inverse-friendly decomposition step.
    """
    ir_sample_rate = None
    if isinstance(ir_1, np.ndarray):
        ir_values = ir_1
    else:
        if not hasattr(ir_1, "values") or not hasattr(ir_1, "sample_rate"):
            raise ValueError("ir_1 must be a NumPy array or an IR instance")
        ir_values = ir_1.values
        ir_sample_rate = ir_1.sample_rate

    kernel_sample_rate = None
    if isinstance(ir_2, np.ndarray):
        kernel_values = ir_2
    else:
        if not hasattr(ir_2, "values") or not hasattr(ir_2, "sample_rate"):
            raise ValueError("ir_2 must be a NumPy array or an IR instance")
        kernel_values = ir_2.values
        kernel_sample_rate = ir_2.sample_rate

    if ir_values is None:
        raise ValueError("IR data for ir_1 is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data for ir_1 must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data for ir_1 must be non-empty")
    if ir_values.ndim == 0:
        raise ValueError("IR data for ir_1 must have at least one dimension")

    if kernel_values is None:
        raise ValueError("IR data for ir_2 is not available")
    if not isinstance(kernel_values, np.ndarray):
        raise ValueError("IR data for ir_2 must be a NumPy array")
    if kernel_values.size == 0:
        raise ValueError("IR data for ir_2 must be non-empty")
    if kernel_values.ndim == 0:
        raise ValueError("IR data for ir_2 must have at least one dimension")

    mode_key = str(mode).strip().lower()
    if mode_key not in {"full", "same", "valid"}:
        raise ValueError("mode must be one of: full, same, valid")

    method_key = str(method).strip().lower()
    if method_key not in {"auto", "direct", "fft"}:
        raise ValueError("method must be one of: auto, direct, fft")

    if ir_sample_rate is not None and kernel_sample_rate is not None:
        if isinstance(ir_sample_rate, bool) or isinstance(kernel_sample_rate, bool):
            raise ValueError("IR inputs must have matching finite, positive sample_rate values")
        try:
            ir_sample_rate = float(ir_sample_rate)
            kernel_sample_rate = float(kernel_sample_rate)
        except (TypeError, ValueError):
            raise ValueError("IR inputs must have matching finite, positive sample_rate values") from None
        if (
            not np.isfinite(ir_sample_rate)
            or ir_sample_rate <= 0.0
            or not np.isfinite(kernel_sample_rate)
            or kernel_sample_rate <= 0.0
        ):
            raise ValueError("IR inputs must have matching finite, positive sample_rate values")
        if not np.isclose(ir_sample_rate, kernel_sample_rate, rtol=1e-8, atol=1e-8):
            raise ValueError("IR inputs must share the same sample_rate for convolution")

    try:
        leading_shape = np.broadcast_shapes(ir_values.shape[:-1], kernel_values.shape[:-1])
    except ValueError:
        raise ValueError("ir_1 and ir_2 leading shapes must be broadcast-compatible") from None

    ir_broadcast = np.broadcast_to(
        ir_values,
        leading_shape + (ir_values.shape[-1],),
    )
    kernel_broadcast = np.broadcast_to(
        kernel_values,
        leading_shape + (kernel_values.shape[-1],),
    )

    ir_flat = ir_broadcast.reshape(-1, ir_values.shape[-1])
    kernel_flat = kernel_broadcast.reshape(-1, kernel_values.shape[-1])

    first_result = signal.convolve(
        ir_flat[0],
        kernel_flat[0],
        mode=mode_key,
        method=method_key,
    )
    convolved_values = np.empty(
        (ir_flat.shape[0], first_result.shape[-1]),
        dtype=first_result.dtype,
    )
    convolved_values[0] = first_result

    for index in range(1, ir_flat.shape[0]):
        convolved_values[index] = signal.convolve(
            ir_flat[index],
            kernel_flat[index],
            mode=mode_key,
            method=method_key,
        )

    return convolved_values.reshape(leading_shape + (first_result.shape[-1],))


def deconvolve(
    ir_1: np.ndarray | "IR",
    ir_2: np.ndarray | "IR",
    fft_length: int | None = None,
    output_length: int | None = None,
    regularization: float = 1e-8,
) -> np.ndarray:
    """Estimate an IR by removing another IR through regularized deconvolution.

    Deconvolution is performed in the frequency domain along the final axis.
    Leading dimensions are broadcast between ir_1 and ir_2 so the
    function can operate on single signals, binaural responses, or full
    position-by-ear HRIR arrays. When both inputs are :class:`~hrtfpykit.hrtf.domain.IR` objects, their
    sample rates must be finite, positive, and equal.

    Parameters
    ----------
    ir_1 : np.ndarray | IR
        Time-domain array or :class:`~hrtfpykit.hrtf.domain.IR` object with .values and
        .sample_rate. This is the measured or mixed IR from which
        ir_2 is removed.
    ir_2 : np.ndarray | IR
        Time-domain array or :class:`~hrtfpykit.hrtf.domain.IR` object with .values and
        .sample_rate. This is the IR to remove from ir_1.
    fft_length : int | None, default=None
        FFT length used for the frequency-domain inversion. When omitted, the
        maximum of ir_1 length, ir_2 length, and output_length is
        used.
    output_length : int | None, default=None
        Number of samples returned along the last axis. When omitted, the
        length of ir_1 is used.
    regularization : float, default=1e-8
        Positive stabilization value added to the spectral denominator to
        avoid division by zero and reduce numerical blow-up.

    Returns
    -------
    np.ndarray
        Deconvolved values with the broadcast leading shape of ir_1 and
        ir_2 and the requested output_length.

    Raises
    ------
    ValueError
        If either IR input is missing, empty, scalar, or not stored as a NumPy
        array, if both domain inputs provide incompatible sample rates, if
        output_length or fft_length is invalid, if fft_length is
        shorter than the input/output lengths, if regularization is not
        finite and positive, or if leading shapes are not broadcast-compatible.

    Notes
    -----
    This is a generic regularized deconvolution utility under a matched
    linear time-invariant model, that is, a situation where ir_1 can be
    approximated as the convolution of a target signal with ir_2.

    In controlled DSP workflows this is useful for de-embedding a known
    system response or approximately undoing a synthetic convolution. In
    contrast, it should not be interpreted as a general room-removal method
    for arbitrary non-anechoic HRTF measurements. Real room effects are often
    direction-dependent, truncated, noisy, or only approximately described by
    one shared IR, and in those cases the recovered signal is only an
    approximation.

    The regularization term intentionally trades exact inversion for
    stability. It limits blow-up near spectral nulls, but it also means the
    output is not expected to perfectly reproduce the original target even
    when the model is close.
    """
    ir_1_sample_rate = None
    if isinstance(ir_1, np.ndarray):
        ir_1_values = ir_1
    else:
        if not hasattr(ir_1, "values") or not hasattr(ir_1, "sample_rate"):
            raise ValueError("ir_1 must be a NumPy array or an IR instance")
        ir_1_values = ir_1.values
        ir_1_sample_rate = ir_1.sample_rate

    ir_2_sample_rate = None
    if isinstance(ir_2, np.ndarray):
        ir_2_values = ir_2
    else:
        if not hasattr(ir_2, "values") or not hasattr(ir_2, "sample_rate"):
            raise ValueError("ir_2 must be a NumPy array or an IR instance")
        ir_2_values = ir_2.values
        ir_2_sample_rate = ir_2.sample_rate

    if ir_1_values is None:
        raise ValueError("IR data for ir_1 is not available")
    if not isinstance(ir_1_values, np.ndarray):
        raise ValueError("IR data for ir_1 must be a NumPy array")
    if ir_1_values.size == 0:
        raise ValueError("IR data for ir_1 must be non-empty")
    if ir_1_values.ndim == 0:
        raise ValueError("IR data for ir_1 must have at least one dimension")

    if ir_2_values is None:
        raise ValueError("IR data for ir_2 is not available")
    if not isinstance(ir_2_values, np.ndarray):
        raise ValueError("IR data for ir_2 must be a NumPy array")
    if ir_2_values.size == 0:
        raise ValueError("IR data for ir_2 must be non-empty")
    if ir_2_values.ndim == 0:
        raise ValueError("IR data for ir_2 must have at least one dimension")

    if ir_1_sample_rate is not None and ir_2_sample_rate is not None:
        if isinstance(ir_1_sample_rate, bool) or isinstance(ir_2_sample_rate, bool):
            raise ValueError("IR inputs must have matching finite, positive sample_rate values")
        try:
            ir_1_sample_rate = float(ir_1_sample_rate)
            ir_2_sample_rate = float(ir_2_sample_rate)
        except (TypeError, ValueError):
            raise ValueError("IR inputs must have matching finite, positive sample_rate values") from None
        if (
            not np.isfinite(ir_1_sample_rate)
            or ir_1_sample_rate <= 0.0
            or not np.isfinite(ir_2_sample_rate)
            or ir_2_sample_rate <= 0.0
        ):
            raise ValueError("IR inputs must have matching finite, positive sample_rate values")
        if not np.isclose(ir_1_sample_rate, ir_2_sample_rate, rtol=1e-8, atol=1e-8):
            raise ValueError("IR inputs must share the same sample_rate for deconvolution")

    if output_length is None:
        output_length_used = int(ir_1_values.shape[-1])
    else:
        if isinstance(output_length, bool) or not isinstance(output_length, int):
            raise ValueError("output_length must be an integer")
        if output_length <= 0:
            raise ValueError("output_length must be positive")
        output_length_used = int(output_length)

    minimum_fft_length = max(
        int(ir_1_values.shape[-1]),
        int(ir_2_values.shape[-1]),
        output_length_used,
    )
    if fft_length is None:
        fft_length_used = minimum_fft_length
    else:
        if isinstance(fft_length, bool) or not isinstance(fft_length, int):
            raise ValueError("fft_length must be an integer")
        if fft_length < minimum_fft_length:
            raise ValueError("fft_length must be greater than or equal to the input and output lengths")
        fft_length_used = int(fft_length)

    if isinstance(regularization, bool):
        raise ValueError("regularization must be a finite, positive value.")
    try:
        regularization = float(regularization)
    except (TypeError, ValueError):
        raise ValueError("regularization must be a finite, positive value.") from None
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be a finite, positive value.")

    try:
        leading_shape = np.broadcast_shapes(
            ir_1_values.shape[:-1],
            ir_2_values.shape[:-1],
        )
    except ValueError:
        raise ValueError("ir_1 and ir_2 leading shapes must be broadcast-compatible") from None

    ir_1_broadcast = np.broadcast_to(
        ir_1_values,
        leading_shape + (ir_1_values.shape[-1],),
    )
    ir_2_broadcast = np.broadcast_to(
        ir_2_values,
        leading_shape + (ir_2_values.shape[-1],),
    )

    ir_1_spectrum = np.fft.rfft(ir_1_broadcast, n=fft_length_used, axis=-1)
    ir_2_spectrum = np.fft.rfft(ir_2_broadcast, n=fft_length_used, axis=-1)
    denominator = np.square(np.abs(ir_2_spectrum)) + regularization
    deconvolved_spectrum = ir_1_spectrum * np.conj(ir_2_spectrum) / denominator
    deconvolved_values = np.fft.irfft(
        deconvolved_spectrum,
        n=fft_length_used,
        axis=-1,
    )
    deconvolved_values = np.real_if_close(deconvolved_values, tol=1000)
    return np.asarray(deconvolved_values[..., :output_length_used])


def minimum_phase(
    data: np.ndarray | "IR",
    method: str = "homomorphic",
    fft_length: int | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Convert IR data into a minimum-phase IR.

    The conversion is applied independently along the final axis and preserves
    the input shape. The function accepts raw NumPy arrays and :class:`~hrtfpykit.hrtf.domain.IR` domain
    objects. Inputs must be real-valued HRIR/time-domain data.

    ``homomorphic`` and ``real_cepstrum`` currently use the same
    real-cepstrum path based on the log magnitude spectrum. ``cepstrum``
    uses a complex cepstrum with unwrapped phase before constructing the
    minimum-phase cepstrum.

    Parameters
    ----------
    data : np.ndarray | IR
        Real-valued IR samples stored as a NumPy array or :class:`~hrtfpykit.hrtf.domain.IR` object.
    method : {``homomorphic``, ``cepstrum``, ``real_cepstrum``}, default=``homomorphic``
        Minimum-phase strategy. homomorphic and real_cepstrum use a
        log-magnitude real cepstrum, while cepstrum uses a complex
        cepstrum with unwrapped phase.
    fft_length : int | None, default=None
        Optional FFT length used for cepstral operations.
    epsilon : float, default=1e-12
        Positive floor applied to magnitude values before logarithms.

    Returns
    -------
    np.ndarray
        Minimum-phase IR array with the same trailing length as the resolved
        IR input.

    Raises
    ------
    ValueError
        If IR data are missing, empty, scalar, not stored as a NumPy array, or
        complex-valued, if method is unsupported, if epsilon is not
        finite and positive, or if fft_length is invalid or shorter than
        the IR length.
    """
    if isinstance(data, np.ndarray):
        ir_values = data
    else:
        if hasattr(data, "sample_rate"):
            ir_values = data.values
        else:
            raise ValueError("data must be a NumPy array or an IR instance")

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")

    method_key = str(method).strip().lower()
    if method_key not in {"homomorphic", "cepstrum", "real_cepstrum"}:
        raise ValueError("method must be one of: homomorphic, cepstrum, real_cepstrum")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    ir_real = np.real_if_close(ir_values, tol=1000)
    if np.iscomplexobj(ir_real):
        raise ValueError("IR data must be real-valued for minimum-phase conversion")
    ir_array = np.asarray(ir_real, dtype=float)
    ir_length = int(ir_array.shape[-1])

    if fft_length is None:
        fft_length_used = max(2, 2 * max(ir_length - 1, 1))
    else:
        if isinstance(fft_length, bool) or not isinstance(fft_length, int):
            raise ValueError("fft_length must be an integer")
        if fft_length < 2:
            raise ValueError("fft_length must be at least 2")
        fft_length_used = int(fft_length)
    if fft_length_used < ir_length:
        raise ValueError("fft_length must be greater than or equal to IR length")

    minimum_phase_values = np.empty_like(ir_array, dtype=float)
    ir_reshaped = ir_array.reshape(-1, ir_length)
    minimum_phase_reshaped = minimum_phase_values.reshape(-1, ir_length)

    for index in range(ir_reshaped.shape[0]):
        if method_key in {"homomorphic", "real_cepstrum"}:
            spectrum_values = np.fft.rfft(ir_reshaped[index], n=fft_length_used)
            magnitude_values = np.maximum(np.abs(spectrum_values), epsilon)
            log_magnitude = np.log(magnitude_values)
            cepstrum = np.fft.irfft(log_magnitude, n=fft_length_used)

            minimum_cepstrum = np.zeros(fft_length_used, dtype=float)
            minimum_cepstrum[0] = cepstrum[0]
            if fft_length_used % 2 == 0:
                half_index = fft_length_used // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]
                minimum_cepstrum[half_index] = cepstrum[half_index]
            else:
                half_index = (fft_length_used + 1) // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]

            minimum_spectrum = np.exp(np.fft.rfft(minimum_cepstrum, n=fft_length_used))
            minimum_ir = np.fft.irfft(minimum_spectrum, n=fft_length_used)
        else:
            spectrum_values = np.fft.fft(ir_reshaped[index], n=fft_length_used)
            magnitude_values = np.maximum(np.abs(spectrum_values), epsilon)
            unwrapped_phase = np.unwrap(np.angle(spectrum_values))
            complex_log_spectrum = np.log(magnitude_values) + 1j * unwrapped_phase
            cepstrum = np.fft.ifft(complex_log_spectrum, n=fft_length_used)

            minimum_cepstrum = np.zeros(fft_length_used, dtype=complex)
            minimum_cepstrum[0] = cepstrum[0]
            if fft_length_used % 2 == 0:
                half_index = fft_length_used // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]
                minimum_cepstrum[half_index] = cepstrum[half_index]
            else:
                half_index = (fft_length_used + 1) // 2
                minimum_cepstrum[1:half_index] = 2.0 * cepstrum[1:half_index]

            minimum_spectrum = np.exp(np.fft.fft(minimum_cepstrum, n=fft_length_used))
            minimum_ir = np.fft.ifft(minimum_spectrum, n=fft_length_used)
            minimum_ir = np.real_if_close(minimum_ir, tol=1000)
            if np.iscomplexobj(minimum_ir):
                minimum_ir = np.real(minimum_ir)

        minimum_phase_reshaped[index] = np.asarray(minimum_ir[:ir_length], dtype=float)

    return minimum_phase_values


def tf_from_ir(
    ir: np.ndarray | "IR",
    sample_rate: float | None = None,
    fft_length: int | None = None,
    window_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | "TF":
    """Compute TF values from IR values using an FFT.

    This function is the forward calculation step from the time domain to the
    frequency domain. For raw NumPy input it behaves like a pure DSP utility:
    it applies the optional window, computes a one-sided FFT along the last
    axis, and returns the resulting TF values, frequency bins, and resolved
    FFT length.

    When ir is an :class:`~hrtfpykit.hrtf.domain.IR` object, the function also acts as the main
    synchronization bridge between :class:`~hrtfpykit.hrtf.domain.IR` and :class:`~hrtfpykit.hrtf.domain.TF` inside an :class:`~hrtfpykit.hrtf.HRTF`
    instance. In that mode it updates the linked :attr:`~hrtfpykit.hrtf.HRTF.TF` object in place,
    rebuilds :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>` from the resolved sample rate and FFT
    length, and stores the resolved fft_length on the parent :class:`~hrtfpykit.hrtf.HRTF`.
    That is the expected recalculation step after editing :attr:`IR.values <hrtfpykit.hrtf.domain.IR.values>`,
    changing :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>`, or applying time-domain transforms that must
    stay consistent with the frequency-domain representation.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain IR samples or :class:`~hrtfpykit.hrtf.domain.IR` object. The FFT is computed along the
        final axis.
    sample_rate : float | None, default=None
        Sample rate in hertz for NumPy input. Optional for :class:`~hrtfpykit.hrtf.domain.IR` input when
        :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is available.
    fft_length : int | None, default=None
        FFT size. When omitted, the IR length along the final axis is used.
    window_name : str | None, default=None
        Optional time-domain window applied along the final axis before the
        FFT. Supported names are delegated to the windowing helper.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int] | TF
        For NumPy input, returns (tf_values, frequency_bins, fft_length_used).
        For :class:`~hrtfpykit.hrtf.domain.IR` input, returns the updated :class:`~hrtfpykit.hrtf.domain.TF` object linked to the same
        :class:`~hrtfpykit.hrtf.HRTF` instance.

    Raises
    ------
    ValueError
        If IR data are missing, empty, all zero, or not stored as a NumPy
        array, if no finite positive sample rate can be resolved, if
        fft_length is not a positive integer or has fewer than two points,
        or if the optional window name is invalid.
    """
    ir_object = None
    if isinstance(ir, np.ndarray):
        ir_values = ir
        if ir_values.size == 0 or np.all(ir_values == 0):
            raise ValueError("NumPy ir array requires non empty values.")
        resolved_sample_rate = sample_rate
    else:
        if not hasattr(ir, "_hrtf") or not hasattr(ir, "values") or not hasattr(ir, "sample_rate"):
            raise ValueError("ir must be a NumPy array or an IR instance")
        ir_object = ir
        ir_values = ir.values
        if ir_values is None:
            raise ValueError("IR data is not available; cannot compute TF.")
        if not isinstance(ir_values, np.ndarray):
            raise ValueError("IR.values must be a NumPy array.")
        if ir_values.size == 0 or np.all(ir_values == 0):
            raise ValueError("IR requires non empty 'values'.")
        resolved_sample_rate = sample_rate if sample_rate is not None else ir.sample_rate

    if resolved_sample_rate is None:
        if ir_object is None:
            raise ValueError("sample_rate is required when ir is a NumPy array")
        raise ValueError("sample_rate is required when IR.sample_rate is unavailable")
    try:
        resolved_sample_rate = float(resolved_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("sample_rate must be a finite, positive value.") from None
    if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
        raise ValueError("sample_rate must be a finite, positive value.")

    if fft_length is None:
        fft_length_used = int(ir_values.shape[-1])
    else:
        if isinstance(fft_length, bool) or not isinstance(fft_length, int):
            raise ValueError("fft_length must be an integer")
        if fft_length <= 0:
            raise ValueError("fft_length must be positive")
        fft_length_used = int(fft_length)

    if fft_length_used < 2:
        raise ValueError("FFT length must contain at least two points.")

    ir_used = ir_values
    if window_name:
        ir_used = window(ir_values, window_name)

    tf_values = np.fft.rfft(ir_used, n=fft_length_used, axis=-1)
    frequency_bins = np.fft.rfftfreq(fft_length_used, d=1.0 / resolved_sample_rate)
    if ir_object is not None:
        tf_object = ir_object._hrtf.TF
        tf_object.values = tf_values
        tf_object.frequency_bins = frequency_bins
        ir_object._hrtf.fft_length = fft_length_used
        return tf_object
    return tf_values, frequency_bins, fft_length_used


def ir_from_tf(
    tf: np.ndarray | "TF",
    frequency_bins: np.ndarray | None = None,
    sample_rate: float | None = None,
    mesh2hrtf_compatible: bool = False,
    n_shift: int | None = None,
) -> tuple[np.ndarray, float, int] | "IR":
    """Compute IR values from TF values using inverse FFT routines.

    This function is the inverse calculation step from the frequency domain to
    the time domain. For raw NumPy input it behaves like a pure DSP utility:
    it uses the provided or inferred one-sided frequency bins to reconstruct
    the IR with inverse real FFT and returns the IR values together with the
    resolved sample rate and FFT length.

    When tf is a :class:`~hrtfpykit.hrtf.domain.TF` object, the function also acts as the main
    synchronization bridge from :class:`~hrtfpykit.hrtf.domain.TF` back to :class:`~hrtfpykit.hrtf.domain.IR` inside an :class:`~hrtfpykit.hrtf.HRTF`
    instance. In that mode it updates the linked :attr:`~hrtfpykit.hrtf.HRTF.IR` object in place,
    restores :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` from the frequency-bin spacing, and stores the
    resolved fft_length on the parent :class:`~hrtfpykit.hrtf.HRTF`. That is the expected
    recalculation step after editing :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>`, changing
    :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`, or applying magnitude or phase operations that must
    remain consistent with the time-domain representation.

    Parameters
    ----------
    tf : np.ndarray | TF
        One-sided frequency-domain TF values or :class:`~hrtfpykit.hrtf.domain.TF` object. The inverse FFT
        is computed along the final axis.
    frequency_bins : np.ndarray | None, default=None
        Optional frequency-bin vector matching the TF length. When tf is a
        :class:`~hrtfpykit.hrtf.domain.TF` object and frequency_bins is None, tf.frequency_bins
        is used.
    sample_rate : float | None, default=None
        Sample rate used when one-sided frequency bins must be inferred for
        NumPy TF input.
    mesh2hrtf_compatible : bool, default=False
        If True, apply Mesh2HRTF-style reconstruction conventions:
        force Nyquist to real magnitude, conjugate the one-sided spectrum
        before irfft, and optionally circularly shift the resulting HRIR.
    n_shift : int | None, default=None
        Optional circular shift applied after reconstruction when
        mesh2hrtf_compatible=True.

    Returns
    -------
    tuple[np.ndarray, float, int] | IR
        For NumPy input, returns (ir_values, sample_rate, fft_length_used).
        For :class:`~hrtfpykit.hrtf.domain.TF` input, returns the updated :class:`~hrtfpykit.hrtf.domain.IR` object linked to the same
        :class:`~hrtfpykit.hrtf.HRTF` instance.

    Raises
    ------
    ValueError
        If TF data are missing, empty, all zero, too short, or not stored as a
        NumPy array, if frequency bins are missing, empty, not one-dimensional,
        not uniformly spaced, not increasing, negative, or incompatible with
        the TF length, if NumPy input cannot infer frequency bins from a finite
        positive sample rate, if the one-sided grid starts at an unsupported
        nonzero frequency, or if n_shift is invalid when Mesh2HRTF
        compatibility is enabled.

    Design Rules
    ------------
    - Only one-sided non-negative frequency bins are supported.
    - Frequency bins must be 1D, uniformly spaced, and increasing.
    - If DC is missing and bins start at one-bin step Δf, DC is inserted
      as 1+0j (0 dB attenuation at 0 Hz).
    """
    tf_object = None
    if isinstance(tf, np.ndarray):
        tf_values = tf
        if tf_values.size == 0 or np.all(tf_values == 0):
            raise ValueError("NumPy tf array requires non empty values.")
    else:
        if not hasattr(tf, "_hrtf") or not hasattr(tf, "values"):
            raise ValueError("tf must be a NumPy array or a TF instance")
        tf_object = tf
        tf_values = tf.values
        if tf_values is None:
            raise ValueError("TF data is not available; cannot compute IR.")
        if not isinstance(tf_values, np.ndarray):
            raise ValueError("TF.values must be a NumPy array.")
        if tf_values.size == 0 or np.all(tf_values == 0):
            raise ValueError("TF requires non empty 'values'.")

    tf_used = tf_values

    if tf_used.shape[-1] < 2:
        raise ValueError("TF length must contain at least two points.")

    if frequency_bins is None:
        if tf_object is not None:
            frequency_bins_array = tf_object.frequency_bins
            if frequency_bins_array is None:
                raise ValueError(
                    "TF.frequency_bins is required when frequency_bins is not provided."
                )
        if sample_rate is None:
            if tf_object is not None:
                frequency_bins_array = np.asarray(frequency_bins_array, dtype=float)
            else:
                raise ValueError(
                    "sample_rate is required when frequency_bins is not provided for NumPy TF."
                )
        if tf_object is None:
            try:
                resolved_sample_rate = float(sample_rate)
            except (TypeError, ValueError):
                raise ValueError("sample_rate must be a finite, positive value.") from None
            if not np.isfinite(resolved_sample_rate) or resolved_sample_rate <= 0.0:
                raise ValueError("sample_rate must be a finite, positive value.")
            inferred_fft_length = 2 * (tf_used.shape[-1] - 1)
            frequency_bins_array = np.fft.rfftfreq(
                inferred_fft_length,
                d=1.0 / resolved_sample_rate,
            )
    else:
        frequency_bins_array = np.asarray(frequency_bins, dtype=float)

    if frequency_bins_array.size == 0 or np.all(frequency_bins_array == 0):
        raise ValueError("TF requires non empty 'frequency_bins'.")

    if frequency_bins_array.ndim != 1 or frequency_bins_array.size != tf_used.shape[-1]:
        raise ValueError("frequency_bins must be 1D and match TF length")
    if frequency_bins_array.size < 2:
        raise ValueError("frequency_bins must contain at least two points")

    diffs = np.diff(frequency_bins_array)
    step = float(diffs[0])
    if step <= 0.0 or not np.allclose(diffs, step, rtol=1e-5, atol=1e-8):
        raise ValueError("frequency_bins must be uniformly spaced and increasing")
    if float(np.min(frequency_bins_array)) < 0.0:
        raise ValueError("Only one-sided non-negative frequency_bins are supported")

    if not np.isclose(frequency_bins_array[0], 0.0):
        # Compatibility path for one-sided TFs that omit DC (e.g., bins start at Δf).
        if np.isclose(frequency_bins_array[0], step, rtol=1e-5, atol=1e-8):
            frequency_bins_array = np.concatenate(
                [np.array([0.0], dtype=float), frequency_bins_array]
            )
            tf_used = np.concatenate(
                [
                    np.ones((*tf_used.shape[:-1], 1), dtype=tf_used.dtype),
                    tf_used,
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                "One-sided frequency_bins must start at 0 Hz or at one-bin step (missing DC case)"
            )

    expected_n_fft = 2 * (frequency_bins_array.size - 1)
    fft_length_used = expected_n_fft
    tf_for_irfft = np.asarray(tf_used, dtype=complex)
    if mesh2hrtf_compatible:
        tf_for_irfft = np.array(tf_for_irfft, copy=True)
        tf_for_irfft[..., -1] = np.abs(tf_for_irfft[..., -1])
        tf_for_irfft = np.conj(tf_for_irfft)

    ir_values = np.fft.irfft(tf_for_irfft, n=fft_length_used, axis=-1)
    if mesh2hrtf_compatible and n_shift is not None:
        if isinstance(n_shift, bool) or not isinstance(n_shift, int):
            raise ValueError("n_shift must be an integer")
        ir_values = np.roll(ir_values, int(n_shift), axis=-1)

    sample_rate = step * expected_n_fft
    if tf_object is not None:
        ir_object = tf_object._hrtf.IR
        ir_object.values = ir_values
        ir_object.sample_rate = sample_rate
        tf_object._hrtf.fft_length = fft_length_used
        return ir_object
    return ir_values, sample_rate, fft_length_used
