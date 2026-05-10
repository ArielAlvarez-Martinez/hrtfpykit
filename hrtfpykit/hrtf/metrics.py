from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

from .dsp import iir_filter, magnitude_to_db, tf_from_ir
from .coordinates import get_position_queries
from .planes import get_horizontal_plane, get_median_plane

if TYPE_CHECKING:
    from .hrtf import HRTF
    from .domain import IR


def itd(
    ir: np.ndarray | "IR",
    method: str = "threshold",
    sample_rate: float | None = None,
    output: str = "samples",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
) -> np.ndarray:
    """Estimate interaural time difference from binaural HRIR data.

    itd operates on the time-domain representation of an HRTF. It accepts
    either a raw NumPy array or an :class:`~hrtfpykit.hrtf.domain.IR` domain object and expects the final
    two axes to be (ear, samples). The first two ear channels are treated
    as left and right, respectively; any leading axes are preserved in the
    returned ITD array.

    Before estimation, both ear signals are low-pass filtered with a
    Butterworth IIR filter. The ``threshold`` method returns the first
    left/right onset difference whose level exceeds the per-ear peak level
    plus thresh_level dB. The ``maxiacce`` method estimates delay from
    the peak of the cross-correlation between Hilbert envelopes.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain impulse-response data or :class:`~hrtfpykit.hrtf.domain.IR` object with layout
        (..., ear, samples). The ear convention is 0=left and
        1=right.
    method : {``threshold``, ``maxiacce``}, default=``threshold``
        Estimator used to resolve the left/right delay. ``threshold`` is an
        onset detector, while ``maxiacce`` uses envelope cross-correlation.
    sample_rate : float | None, default=None
        Sample rate in hertz. Required for NumPy input. When ir is an
        :class:`~hrtfpykit.hrtf.domain.IR` object and this value is omitted, :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is used.
    output : {``seconds``, ``samples``}, default=``samples``
        Unit used for the returned ITD values.
    thresh_level : float, default=-10.0
        Threshold offset in decibels used by method=``threshold``. The
        effective threshold is computed independently for each ear as
        peak_level + thresh_level.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff frequency in hertz applied before ITD estimation. The
        value must be between zero and the Nyquist frequency.
    filter_order : int, default=10
        Positive Butterworth filter order used in the preprocessing stage.

    Returns
    -------
    np.ndarray
        ITD values with shape ir.shape[:-2] for NumPy input, or
        IR.values.shape[:-2] for :class:`~hrtfpykit.hrtf.domain.IR` input. Positive values mean the
        left-ear response is delayed relative to the right-ear response.

    Raises
    ------
    ValueError
        If IR data are missing, empty, not array-like, do not contain ear and
        sample axes, have fewer than two ear channels or fewer than two
        samples, if sample_rate is missing or invalid, if method or
        output is unsupported, if the filter configuration is invalid, or
        if threshold mode cannot find a valid onset in either ear.

    Notes
    -----
    Use this function when inspecting absolute ITD values for one HRTF or when
    implementing custom metrics. Use
    :func:`~hrtfpykit.hrtf.itd_difference` when comparing two
    HRTFs over a shared source grid.
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
    if ir_values.ndim < 2:
        raise ValueError("IR data must include at least channel and time axes")

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

    method_key = str(method).strip().lower()
    if method_key not in {"threshold", "maxiacce"}:
        raise ValueError("method must be one of: threshold, maxiacce")
    output_key = str(output).strip().lower()
    if output_key not in {"seconds", "samples"}:
        raise ValueError("output must be one of: seconds, samples")

    ir_channel_last = ir_values
    channel_count = ir_channel_last.shape[-2]
    if channel_count < 2:
        raise ValueError("IR ear axis must contain at least two channels (0=left, 1=right)")

    time_length = int(ir_channel_last.shape[-1])
    if time_length < 2:
        raise ValueError("IR time axis must contain at least two samples")

    flattened = ir_channel_last.reshape(-1, channel_count, time_length)
    itd_values = np.empty(flattened.shape[0], dtype=int)

    if isinstance(filter_order, bool) or not isinstance(filter_order, int):
        raise ValueError("filter_order must be an integer")
    if filter_order <= 0:
        raise ValueError("filter_order must be positive")

    left_signals = flattened[:, 0, :]
    right_signals = flattened[:, 1, :]
    left_processed = iir_filter(
        left_signals,
        filter="lowpass",
        sample_rate=resolved_sample_rate,
        cutoff=upper_cut_freq,
        order=filter_order,
    )
    right_processed = iir_filter(
        right_signals,
        filter="lowpass",
        sample_rate=resolved_sample_rate,
        cutoff=upper_cut_freq,
        order=filter_order,
    )

    if method_key == "threshold":
        if isinstance(thresh_level, bool):
            raise ValueError("thresh_level must be a finite value.")
        try:
            thresh_level = float(thresh_level)
        except (TypeError, ValueError):
            raise ValueError("thresh_level must be a finite value.") from None
        if not np.isfinite(thresh_level):
            raise ValueError("thresh_level must be a finite value.")

        for index in range(flattened.shape[0]):
            left_db = 0.5 * magnitude_to_db(np.square(left_processed[index]))
            right_db = 0.5 * magnitude_to_db(np.square(right_processed[index]))
            left_threshold = float(np.max(left_db)) + thresh_level
            right_threshold = float(np.max(right_db)) + thresh_level
            left_hits = np.where(left_db > left_threshold)[0]
            right_hits = np.where(right_db > right_threshold)[0]
            if left_hits.size == 0 or right_hits.size == 0:
                raise ValueError("threshold mode could not find a valid onset index")
            left_idx = int(left_hits[0])
            right_idx = int(right_hits[0])
            itd_values[index] = int(left_idx - right_idx)
    else:
        lags = signal.correlation_lags(time_length, time_length, mode="full")
        for index in range(flattened.shape[0]):
            left_env = np.abs(signal.hilbert(left_processed[index]))
            right_env = np.abs(signal.hilbert(right_processed[index]))
            cross_corr = signal.correlate(right_env, left_env, mode="full", method="fft")
            peak_lag = lags[int(np.argmax(np.abs(cross_corr)))]
            itd_values[index] = int(-peak_lag)

    if output_key == "seconds":
        itd_values = itd_values.astype(float) / resolved_sample_rate
    output_shape = ir_channel_last.shape[:-2]
    return itd_values.reshape(output_shape)


def ild(
    ir: np.ndarray | "IR",
    sample_rate: float | None = None,
    fft_length: int | None = None,
    mode: str = "broad-band",
    output: str = "db",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute interaural level difference from binaural HRIR data.

    ild measures the level ratio between the left and right ears from a
    time-domain HRIR array. It accepts raw NumPy data or an :class:`~hrtfpykit.hrtf.domain.IR` domain
    object and expects the final two axes to be (ear, samples). The first
    two ear channels are treated as left and right, respectively; any leading
    axes are preserved in the returned ILD array.

    In ``broad-band`` mode, the function computes the ratio between left and
    right RMS levels over the time axis. In ``frequency-dependent`` mode, it
    first converts the IR to a one-sided transfer function with
    real FFT conversion, then computes the left/right
    magnitude ratio per frequency bin.

    Parameters
    ----------
    ir : np.ndarray | IR
        Time-domain impulse-response data or :class:`~hrtfpykit.hrtf.domain.IR` object with layout
        (..., ear, samples). The ear convention is 0=left and
        1=right.
    sample_rate : float | None, default=None
        Sample rate in hertz used when mode=``frequency-dependent``. For
        :class:`~hrtfpykit.hrtf.domain.IR` input, :attr:`IR.sample_rate <hrtfpykit.hrtf.domain.IR.sample_rate>` is used when this value is omitted.
    fft_length : int | None, default=None
        FFT length used for the internal IR-to-TF conversion in
        mode=``frequency-dependent``. When omitted, the IR sample length is
        used by the conversion helper.
    mode : {``broad-band``, ``frequency-dependent``}, default=``broad-band``
        Level-difference mode. ``broad-band`` returns one value per leading
        IR entry; ``frequency-dependent`` returns one value per leading IR
        entry and frequency bin.
    output : {``db``, ``linear``}, default=``db``
        Output representation. ``linear`` returns the raw left/right level
        ratio; ``db`` returns 20 * log10(left / right).
    epsilon : float, default=1e-12
        Positive floor added to left and right magnitudes before division to
        avoid division by zero.

    Returns
    -------
    np.ndarray
        ILD values in the requested representation. mode=``broad-band``
        returns shape ir.shape[:-2] for NumPy input, or
        IR.values.shape[:-2] for :class:`~hrtfpykit.hrtf.domain.IR` input.
        mode=``frequency-dependent`` appends the one-sided frequency-bin
        axis. In dB output, positive values mean the left-ear level is greater
        than the right-ear level; in linear output, values greater than
        1.0 have the same meaning.

    Raises
    ------
    ValueError
        If IR data are missing, empty, not array-like, do not contain ear and
        sample axes, have fewer than two ear channels, if mode or
        output is unsupported, if epsilon is not finite and positive,
        or if frequency-dependent mode cannot resolve a valid sample rate or
        FFT conversion.

    Notes
    -----
    Use this function when inspecting absolute ILD values for one HRTF or when
    implementing custom metrics. Use
    :func:`~hrtfpykit.hrtf.ild_difference` when comparing two
    HRTFs over a shared source grid.
    """
    ir_object = None
    if isinstance(ir, np.ndarray):
        ir_values = ir
    elif hasattr(ir, "values"):
        ir_object = ir
        ir_values = ir.values
    else:
        raise ValueError("ir must be a NumPy array or an IR instance")

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim < 2:
        raise ValueError("IR data must include at least ear and time axes")

    output_key = str(output).strip().lower()
    if output_key not in {"db", "linear"}:
        raise ValueError("output must be one of: db, linear")

    mode_key = str(mode).strip().lower()
    if mode_key not in {"broad-band", "frequency-dependent"}:
        raise ValueError("mode must be one of: broad-band, frequency-dependent")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    if ir_object is not None and hasattr(ir_object, "sample_rate"):
        resolved_sample_rate = (
            sample_rate if sample_rate is not None else ir_object.sample_rate
        )
    else:
        resolved_sample_rate = sample_rate

    if ir_values.shape[-2] < 2:
        raise ValueError("Ear axis must contain at least two channels (0=left, 1=right)")

    if mode_key == "frequency-dependent":
        if resolved_sample_rate is None:
            raise ValueError("sample_rate is required for IR NumPy inputs")
        tf_values, _, _ = tf_from_ir(
            ir_values,
            sample_rate=resolved_sample_rate,
            fft_length=fft_length,
        )
    else:
        tf_values = None

    if mode_key == "broad-band":
        left_values = np.asarray(ir_values[..., 0, :], dtype=float)
        right_values = np.asarray(ir_values[..., 1, :], dtype=float)
        left_rms = np.sqrt(np.mean(np.square(left_values), axis=-1))
        right_rms = np.sqrt(np.mean(np.square(right_values), axis=-1))
        ild_linear = (left_rms + epsilon) / (right_rms + epsilon)
        if output_key == "linear":
            return ild_linear
        return magnitude_to_db(ild_linear)

    left_magnitude = np.abs(tf_values[..., 0, :])
    right_magnitude = np.abs(tf_values[..., 1, :])
    ild_linear = (left_magnitude + epsilon) / (right_magnitude + epsilon)

    if output_key == "linear":
        return ild_linear
    return magnitude_to_db(ild_linear)


def itd_difference(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    method: str = "threshold",
    output: str = "seconds",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
) -> np.ndarray:
    """Compute absolute per-position ITD differences between two HRTFs.

    The metric estimates interaural time difference for each input HRTF,
    validates that both HRTFs expose the same source positions, and returns
    ``abs(itd_a - itd_b)``. Here, ``itd_a`` and ``itd_b`` are the per-position
    ITD arrays estimated from ``hrtf_a`` and ``hrtf_b``. The result is an
    unsigned timing-change magnitude for each source position in the current
    HRTF view.

    Source positions are compared through :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` in degrees.
    If either input already represents a selected spatial subset, both HRTFs
    must expose the same selected source grid in the same order.

    Parameters
    ----------
    hrtf_a : HRTF
        First :class:`~hrtfpykit.hrtf.hrtf.HRTF` object used in the comparison. It must provide IR data,
        an IR sample rate, and a :class:`~hrtfpykit.hrtf.sources.Sources` grid matching ``hrtf_b``.
    hrtf_b : HRTF
        Second :class:`~hrtfpykit.hrtf.hrtf.HRTF` object used in the comparison. It must provide IR data,
        an IR sample rate, and a :class:`~hrtfpykit.hrtf.sources.Sources` grid matching ``hrtf_a``.
    method : {``threshold``, ``maxiacce``}, default=``threshold``
        ITD estimator used for both HRTFs.
    output : {``seconds``, ``samples``}, default=``seconds``
        Unit of returned absolute ITD differences. ``samples`` requires
        equal sample rates in both HRTFs. ``seconds`` converts each ITD with
        that HRTF's own sample rate before subtraction.
    thresh_level : float, default=-10.0
        Threshold offset in decibels used when ``method`` is ``threshold``.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff frequency in hertz used by filtered ITD estimation.
    filter_order : int, default=10
        Positive Butterworth low-pass order used by filtered ITD estimation.

    Returns
    -------
    np.ndarray
        Absolute ITD differences per source position. For standard HRTF data
        with shape (positions, ears, samples), the result has shape
        (positions,).

    Raises
    ------
    ValueError
        If either input is not an HRTF-like object with :class:`~hrtfpykit.hrtf.domain.IR` and :class:`~hrtfpykit.hrtf.sources.Sources`,
        if IR values or sample rates are missing, if output or ITD estimator
        options are invalid, if sample-domain output is requested for unequal
        sample rates, if source grids differ in shape or values, or if the
        calculated ITD arrays do not have matching shapes.

    Notes
    -----
    This function intentionally returns absolute differences, so the sign of
    the timing change is not retained.

    Examples
    --------
    >>> from hrtfpykit.hrtf import itd_difference
    >>> itd_diff = itd_difference(hrtf_a, hrtf_b)
    >>> itd_diff.shape
    (hrtf_a.IR.values.shape[0],)
    """
    for label, hrtf in (("hrtf_a", hrtf_a), ("hrtf_b", hrtf_b)):
        if not hasattr(hrtf, "IR") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"{label} must be an HRTF instance")
        if hrtf.IR.values is None:
            raise ValueError(f"{label} IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"{label} IR sample_rate is required")

    output_key = str(output).strip().lower()
    if output_key not in {"seconds", "samples"}:
        raise ValueError("output must be one of: seconds, samples")
    if output_key == "samples" and not np.isclose(
        float(hrtf_a.IR.sample_rate),
        float(hrtf_b.IR.sample_rate),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "output='samples' requires equal sample_rate in both HRTFs"
        )

    source_positions_a = np.asarray(hrtf_a.Sources.get_positions(angle_unit="degrees"), dtype=float)
    source_positions_b = np.asarray(hrtf_b.Sources.get_positions(angle_unit="degrees"), dtype=float)
    if source_positions_a.shape != source_positions_b.shape:
        raise ValueError("HRTFs must have the same number of source positions")
    if not np.allclose(source_positions_a, source_positions_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same source positions for ITD difference")

    itd_a = np.asarray(
        itd(
            hrtf_a.IR,
            method=method,
            output=output_key,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        ),
        dtype=float,
    )
    itd_b = np.asarray(
        itd(
            hrtf_b.IR,
            method=method,
            output=output_key,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        ),
        dtype=float,
    )
    if itd_a.shape != itd_b.shape:
        raise ValueError("Calculated ITD arrays must have matching shapes")
    return np.abs(itd_a - itd_b)


def ild_difference(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    mode: str = "broad-band",
    output: str = "db",
    fft_length: int | None = None,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute absolute per-position ILD differences between two HRTFs.

    The metric estimates interaural level difference for each input HRTF,
    validates that both HRTFs expose the same source positions, and returns
    ``abs(ild_a - ild_b)``. Here, ``ild_a`` and ``ild_b`` are the ILD arrays
    estimated from ``hrtf_a`` and ``hrtf_b``. The result is an unsigned
    level-ratio change magnitude for each source position, and optionally for
    each frequency bin.

    Source positions are compared through :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` in degrees.
    If either input already represents a selected spatial subset, both HRTFs
    must expose the same selected source grid in the same order.

    Parameters
    ----------
    hrtf_a : HRTF
        First :class:`~hrtfpykit.hrtf.hrtf.HRTF` object used in the comparison. It must provide IR data,
        an IR sample rate, and a :class:`~hrtfpykit.hrtf.sources.Sources` grid matching ``hrtf_b``.
    hrtf_b : HRTF
        Second :class:`~hrtfpykit.hrtf.hrtf.HRTF` object used in the comparison. It must provide IR data,
        an IR sample rate, and a :class:`~hrtfpykit.hrtf.sources.Sources` grid matching ``hrtf_a``.
    mode : {``broad-band``, ``frequency-dependent``}, default=``broad-band``
        ILD mode used for both HRTFs. Broad-band mode compares RMS level ratios,
        while frequency-dependent mode compares per-bin magnitude ratios after
        IR-to-TF conversion.
    output : {``db``, ``linear``}, default=``db``
        Output representation used for both HRTFs. ``db`` compares dB ILD
        values; ``linear`` compares raw left/right ratios.
    fft_length : int | None, default=None
        FFT length used by the internal IR-to-TF conversion when
        ``mode`` is ``frequency-dependent``.
    epsilon : float, default=1e-12
        Positive floor used before left/right division.

    Returns
    -------
    np.ndarray
        Absolute ILD differences. For standard HRTF data, broad-band mode
        returns shape (positions,) and frequency-dependent mode returns
        shape (positions, frequency_bins).

    Raises
    ------
    ValueError
        If either input is not an HRTF-like object with :class:`~hrtfpykit.hrtf.domain.IR` and :class:`~hrtfpykit.hrtf.sources.Sources`,
        if IR values or sample rates are missing, if mode or output is
        unsupported, if source grids differ in shape or values, if
        frequency-dependent mode is requested with unequal sample rates, if
        epsilon or FFT conversion settings are invalid, or if calculated
        ILD arrays do not have matching shapes.

    Notes
    -----
    This function intentionally returns absolute differences, so the sign of
    the level-ratio change is not retained.

    Examples
    --------
    >>> from hrtfpykit.hrtf import ild_difference
    >>> ild_diff = ild_difference(hrtf_a, hrtf_b)
    >>> ild_diff.shape
    (hrtf_a.IR.values.shape[0],)
    """
    for label, hrtf in (("hrtf_a", hrtf_a), ("hrtf_b", hrtf_b)):
        if not hasattr(hrtf, "IR") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"{label} must be an HRTF instance")
        if hrtf.IR.values is None:
            raise ValueError(f"{label} IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"{label} IR sample_rate is required")

    mode_key = str(mode).strip().lower()
    if mode_key not in {"broad-band", "frequency-dependent"}:
        raise ValueError("mode must be one of: broad-band, frequency-dependent")
    output_key = str(output).strip().lower()
    if output_key not in {"db", "linear"}:
        raise ValueError("output must be one of: db, linear")

    source_positions_a = np.asarray(hrtf_a.Sources.get_positions(angle_unit="degrees"), dtype=float)
    source_positions_b = np.asarray(hrtf_b.Sources.get_positions(angle_unit="degrees"), dtype=float)
    if source_positions_a.shape != source_positions_b.shape:
        raise ValueError("HRTFs must have the same number of source positions")
    if not np.allclose(source_positions_a, source_positions_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same source positions for ILD difference")

    if mode_key == "frequency-dependent" and not np.isclose(
        float(hrtf_a.IR.sample_rate),
        float(hrtf_b.IR.sample_rate),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "mode='frequency-dependent' requires equal sample_rate in both HRTFs"
        )

    ild_a = np.asarray(
        ild(
            hrtf_a.IR,
            sample_rate=float(hrtf_a.IR.sample_rate),
            fft_length=fft_length,
            mode=mode_key,
            output=output_key,
            epsilon=epsilon,
        ),
        dtype=float,
    )
    ild_b = np.asarray(
        ild(
            hrtf_b.IR,
            sample_rate=float(hrtf_b.IR.sample_rate),
            fft_length=fft_length,
            mode=mode_key,
            output=output_key,
            epsilon=epsilon,
        ),
        dtype=float,
    )
    if ild_a.shape != ild_b.shape:
        raise ValueError("Calculated ILD arrays must have matching shapes")
    return np.abs(ild_a - ild_b)


def lsd(
    hrtf_a: "HRTF",
    hrtf_b: "HRTF",
    ear: str = "both",
    plane: str = "all",
    elevation: float = 0.0,
    positions: np.ndarray | list | tuple | str | None = None,
    frequencies: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    reduction: str = "none",
    epsilon: float = 1e-12,
) -> np.ndarray | float:
    """Compute log-spectral distortion (LSD) between two HRTFs in dB.

    LSD compares two frequency-domain HRTFs in the logarithmic magnitude
    domain. Both inputs must provide :attr:`TF.values <hrtfpykit.hrtf.domain.TF.values>` with shape
    (positions, ears, frequency_bins), matching :attr:`TF.frequency_bins <hrtfpykit.hrtf.domain.TF.frequency_bins>`, and
    matching source positions from :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` in degrees.

    The comparison can be restricted to one ear, both ears, all source
    positions, a measured horizontal or median plane, explicit source-position
    queries, explicit frequency queries, or a reduced scalar. Position queries
    are resolved against :attr:`~hrtfpykit.hrtf.hrtf.HRTF.Sources` in spherical degrees and then
    intersected with the selected plane. Frequency queries are mapped to the
    nearest available TF bins and duplicate bin selections are removed.

    The selected ear axis is averaged after the per-ear LSD calculation.
    reduction=``none`` returns absolute dB differences. The other reductions
    use root-mean-square dB differences over the requested axes before the ear
    average.

    Parameters
    ----------
    hrtf_a : HRTF
        First :class:`~hrtfpykit.hrtf.hrtf.HRTF` object used in the comparison. It must provide TF values,
        frequency bins, and a source grid matching hrtf_b.
    hrtf_b : HRTF
        Second :class:`~hrtfpykit.hrtf.hrtf.HRTF` object used in the comparison. It must provide TF values,
        frequency bins, and a source grid matching hrtf_a.
    ear : {``left``, ``right``, ``both``}, default=``both``
        Ear channel selection. ``left`` uses ear channel 0, ``right``
        uses ear channel 1, and ``both`` evaluates both channels before
        averaging over ears.
    plane : {``all``, ``horizontal``, ``median``}, default=``all``
        Spatial subset used before comparison. ``all`` uses the current
        full source grid, ``horizontal`` uses the nearest measured
        horizontal plane at elevation, and ``median`` uses the canonical
        median plane at azimuth 0 degrees.
    elevation : float, default=0.0
        Requested elevation in degrees used only when plane=``horizontal``.
        The nearest measured elevation in hrtf_a is used.
    positions : np.ndarray | list | tuple | str | None, default=None
        Optional source-position selector. Accepts one query or a collection
        of queries in the format accepted by
        :func:`~hrtfpykit.hrtf.coordinates.get_position_queries`, including
        named positions such as ``front`` and numeric spherical queries with
        shape (2,) or (3,). Resolved position indices are intersected
        with the selected plane.
    frequencies : float | list[float] | tuple[float, ...] | np.ndarray | None, default=None
        Optional frequency selector in hertz. Each requested frequency is
        mapped to the nearest available TF bin. None selects all available
        bins from 20 Hz through 20 kHz, inclusive, which excludes DC for
        typical one-sided FFT grids.
    reduction : {``none``, ``locations``, ``frequencies``, ``global``}, default=``none``
        Aggregation mode. ``none`` keeps selected positions and frequencies,
        ``locations`` reduces over positions and returns one value per
        selected frequency, ``frequencies`` reduces over frequencies and
        returns one value per selected position, and ``global`` reduces over
        positions and frequencies to one scalar.
    epsilon : float, default=1e-12
        Positive lower bound applied to magnitudes before conversion to dB.
        This avoids invalid values from log10(0).

    Returns
    -------
    np.ndarray | float
        LSD values in dB. With reduction=``none``, the usual output shape is
        (selected_positions, selected_frequencies); if one frequency is
        selected, the frequency axis is squeezed and the output has shape
        (selected_positions,). With reduction=``locations``, the output
        is indexed by selected frequency, or returned as a scalar when only
        one frequency is selected. With reduction=``frequencies``, the
        output is indexed by selected position. With reduction=``global``,
        the output is a scalar.

    Raises
    ------
    ValueError
        If either input is not an HRTF-like object with :class:`~hrtfpykit.hrtf.domain.TF` and :class:`~hrtfpykit.hrtf.sources.Sources`,
        if TF values or frequency bins are missing, if source positions, TF
        shapes, or frequency bins do not match, if TF values are not arranged
        as (positions, ears, frequency_bins), if the ear axis has fewer
        than two channels, if ear, plane, or reduction is
        unsupported, if epsilon is not finite and positive, if selected
        planes or position filters produce no source positions, if frequency
        selectors are invalid or select no bins, or if
        reduction=``frequencies`` is requested for a single selected
        frequency.

    Notes
    -----
    Use reduction=``none`` for heatmaps and per-bin diagnostics,
    reduction=``frequencies`` for one spatial error value per source
    position, reduction=``locations`` for one spectral error curve, and
    reduction=``global`` for a single comparison score.

    Examples
    --------
    >>> lsd(hrtf_a, hrtf_b, reduction="global")
    >>> lsd(hrtf_a, hrtf_b, ear="left", reduction="none")
    >>> lsd(hrtf_a, hrtf_b, ear="left", reduction="locations")
    >>> lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="right",
    ...     plane="horizontal",
    ...     elevation=0.0,
    ...     reduction="frequencies",
    ... )
    >>> lsd(
    ...     hrtf_a,
    ...     hrtf_b,
    ...     ear="left",
    ...     plane="median",
    ...     frequencies=4000.0,
    ...     reduction="global",
    ... )
    """
    for label, hrtf in (("hrtf_a", hrtf_a), ("hrtf_b", hrtf_b)):
        if not hasattr(hrtf, "TF") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"{label} must be an HRTF instance")
        if hrtf.TF.values is None:
            raise ValueError(f"{label} TF data is not available")
        if hrtf.TF.frequency_bins is None:
            raise ValueError(f"{label} TF frequency_bins are required")

    plane_key = str(plane).strip().lower()
    if plane_key not in {"all", "horizontal", "median"}:
        raise ValueError("plane must be one of: all, horizontal, median")

    reduction_key = str(reduction).strip().lower()
    if reduction_key not in {"none", "locations", "frequencies", "global"}:
        raise ValueError("reduction must be one of: none, locations, frequencies, global")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    source_positions_a = np.asarray(hrtf_a.Sources.get_positions(angle_unit="degrees"), dtype=float)
    source_positions_b = np.asarray(hrtf_b.Sources.get_positions(angle_unit="degrees"), dtype=float)
    if source_positions_a.shape != source_positions_b.shape:
        raise ValueError("HRTFs must have the same number of source positions")
    if not np.allclose(source_positions_a, source_positions_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same source positions for LSD")

    tf_a = np.asarray(hrtf_a.TF.values)
    tf_b = np.asarray(hrtf_b.TF.values)
    if tf_a.ndim != 3 or tf_b.ndim != 3:
        raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
    if tf_a.shape != tf_b.shape:
        raise ValueError("HRTFs must have matching TF shapes for LSD")
    if tf_a.shape[0] != source_positions_a.shape[0]:
        raise ValueError("TF positions axis must match source positions count")
    if tf_a.shape[1] < 2:
        raise ValueError("TF ear axis must contain at least two channels (0=left, 1=right)")

    frequency_bins_a = np.asarray(hrtf_a.TF.frequency_bins, dtype=float).reshape(-1)
    frequency_bins_b = np.asarray(hrtf_b.TF.frequency_bins, dtype=float).reshape(-1)
    if frequency_bins_a.shape != frequency_bins_b.shape:
        raise ValueError("HRTFs must have matching TF frequency_bins")
    if not np.allclose(frequency_bins_a, frequency_bins_b, atol=1e-8, rtol=0.0):
        raise ValueError("HRTFs must share the same TF frequency_bins for LSD")
    if frequency_bins_a.size != tf_a.shape[-1]:
        raise ValueError("TF frequency axis length must match frequency_bins length")

    ear_key = str(ear).strip().lower()
    if ear_key not in {"left", "right", "both"}:
        raise ValueError("ear must be one of: left, right, both")
    if ear_key == "left":
        selected_ear_indices = np.array([0], dtype=int)
    elif ear_key == "right":
        selected_ear_indices = np.array([1], dtype=int)
    else:
        selected_ear_indices = np.array([0, 1], dtype=int)

    if plane_key == "all":
        selected_positions = np.arange(source_positions_a.shape[0], dtype=int)
    elif plane_key == "horizontal":
        selected_positions, _ = get_horizontal_plane(hrtf=hrtf_a, elevation=float(elevation))
    else:
        selected_positions, _ = get_median_plane(hrtf=hrtf_a, azimuth=0.0)
    selected_positions = np.asarray(selected_positions, dtype=int).reshape(-1)
    if selected_positions.size == 0:
        raise ValueError("Selected plane has no source positions")

    if positions is not None:
        position_queries = get_position_queries(positions)
        selected_from_queries: list[int] = []
        for query in position_queries:
            position_index, _ = hrtf_a.Sources.get_position_index(
                query,
                coordinate_system="spherical",
            )
            selected_from_queries.append(int(position_index))
        selected_from_queries_array = np.asarray(selected_from_queries, dtype=int)
        selected_positions = np.intersect1d(
            selected_positions,
            selected_from_queries_array,
            assume_unique=False,
        )
        if selected_positions.size == 0:
            raise ValueError("No source positions matched the provided positions and plane filters")

    if frequencies is None:
        selected_frequency_indices = np.where(
            (frequency_bins_a >= 20.0) & (frequency_bins_a <= 20000.0)
        )[0]
        
        if selected_frequency_indices.size == 0:
            raise ValueError(
                "No frequency bins available in the default LSD range [20.0, 20000.0] Hz"
            )
    else:
        if isinstance(frequencies, bool):
            raise ValueError("frequencies must be finite value(s)")
        frequency_values = np.asarray(frequencies, dtype=float).reshape(-1)
        if frequency_values.size == 0:
            raise ValueError("frequencies must contain at least one value when provided")
        if not np.all(np.isfinite(frequency_values)):
            raise ValueError("frequencies must be finite value(s)")
        nearest_frequency_indices = [
            int(np.argmin(np.abs(frequency_bins_a - float(target_frequency))))
            for target_frequency in frequency_values
        ]
        selected_frequency_indices = np.asarray(
            tuple(dict.fromkeys(nearest_frequency_indices)),
            dtype=int,
        )

    tf_values_a = np.asarray(
        tf_a[np.ix_(selected_positions, selected_ear_indices, selected_frequency_indices)],
        dtype=complex,
    )
    tf_values_b = np.asarray(
        tf_b[np.ix_(selected_positions, selected_ear_indices, selected_frequency_indices)],
        dtype=complex,
    )
    if tf_values_a.shape != tf_values_b.shape:
        raise ValueError("Selected TF slices must have matching shapes")

    magnitude_a = np.maximum(np.abs(tf_values_a), epsilon)
    magnitude_b = np.maximum(np.abs(tf_values_b), epsilon)
    db_values_a = magnitude_to_db(magnitude_a)
    db_values_b = magnitude_to_db(magnitude_b)
    difference_db = db_values_a - db_values_b

    if reduction_key == "none":
        absolute_difference_db = np.abs(difference_db)
        absolute_difference_db = np.mean(absolute_difference_db, axis=1)
        if absolute_difference_db.shape[-1] == 1:
            lsd_output: np.ndarray | float = absolute_difference_db[:, 0]
        else:
            lsd_output = absolute_difference_db
    elif reduction_key == "locations":
        rms_over_locations = np.sqrt(np.mean(np.square(difference_db), axis=0))
        rms_over_locations = np.mean(rms_over_locations, axis=0)
        if rms_over_locations.size == 1:
            lsd_output = float(rms_over_locations[0])
        else:
            lsd_output = rms_over_locations
    elif reduction_key == "frequencies":
        if difference_db.shape[-1] == 1:
            raise ValueError("reduction='frequencies' requires multiple selected frequencies")
        rms_over_frequencies = np.sqrt(np.mean(np.square(difference_db), axis=-1))
        rms_over_frequencies = np.mean(rms_over_frequencies, axis=1)
        lsd_output = rms_over_frequencies
    else:
        rms_over_positions_frequencies = np.sqrt(np.mean(np.square(difference_db), axis=(0, 2)))
        lsd_output = float(np.mean(rms_over_positions_frequencies))

    return lsd_output
