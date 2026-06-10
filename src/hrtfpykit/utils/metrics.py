from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import signal

from .dsp import iir_filter, magnitude_to_db
from .coordinates import get_position_queries
from .planes import get_horizontal_plane, get_median_plane

if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF


def itd(
    hrtf: "HRTF",
    method: str = "threshold",
    output: str = "samples",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
    absolute: bool = False,
) -> np.ndarray:
    """Estimate interaural time difference for an HRTF.

    ``itd`` is an HRTF-level metric. It reads the time-domain HRIR data and
    sample rate from ``hrtf.IR`` and expects the final two IR axes to be
    ``(ear, samples)``. The first two ear channels are treated as left and
    right, respectively; any leading source axes are preserved in the returned
    ITD array.

    The default output is signed: positive values mean that the left-ear
    response is delayed relative to the right-ear response. Set
    ``absolute=True`` when only cue magnitude is needed.

    Before estimation, both ear signals are low-pass filtered with a
    Butterworth IIR filter. The ``threshold`` method returns the first
    left/right onset difference whose level exceeds the per-ear peak level
    plus ``thresh_level`` dB. The ``maxiacce`` method estimates delay from the
    peak of the cross-correlation between Hilbert envelopes.

    Parameters
    ----------
    hrtf : HRTF
        HRTF object that provides an ``IR`` domain with ``values`` laid out as
        ``(..., ear, samples)`` and a finite positive ``sample_rate``.
    method : {``threshold``, ``maxiacce``}, default=``threshold``
        Estimator used to resolve the left/right delay. ``threshold`` is an
        onset detector, while ``maxiacce`` uses envelope cross-correlation.
    output : {``time``, ``samples``}, default=``samples``
        Unit used for the returned ITD values. ``time`` returns microseconds.
    thresh_level : float, default=-10.0
        Threshold offset in decibels used by method=``threshold``. The
        effective threshold is computed independently for each ear as
        peak_level + thresh_level.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff frequency in hertz applied before ITD estimation. The
        value must be between zero and the Nyquist frequency.
    filter_order : int, default=10
        Positive Butterworth filter order used in the preprocessing stage.
    absolute : bool, default=False
        If False, return signed ITD. If True, return ``abs(ITD)`` in the
        selected output unit.

    Returns
    -------
    np.ndarray
        ITD values with shape ``hrtf.IR.values.shape[:-2]``. Positive signed
        values mean the left-ear response is delayed relative to the right-ear
        response.

    Raises
    ------
    ValueError
        If the input is not an HRTF-like object, IR data are missing, empty,
        not array-like, do not contain ear and sample axes, have fewer than
        two ear channels or fewer than two samples, if sample rate is missing
        or invalid, if method or output is unsupported, if absolute is not a
        boolean, if the filter configuration is invalid, or if threshold mode
        cannot find a valid onset in either ear.

    Examples
    --------
    Estimate one signed ITD value per source position:

    >>> from hrtfpykit.hrtf import load_hrtf, itd
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> itd_samples = itd(hrtf, output="samples")
    >>> itd_samples.shape
    (793,)

    Return ITD magnitude in microseconds:

    >>> abs_itd_time = itd(hrtf, output="time", absolute=True)
    >>> abs_itd_time.shape
    (793,)

    """
    if not hasattr(hrtf, "IR"):
        raise ValueError("hrtf must be an HRTF instance")

    ir = cast(Any, hrtf).IR
    if not hasattr(ir, "values") or not hasattr(ir, "sample_rate"):
        raise ValueError("hrtf must provide an IR domain with values and sample_rate")
    ir_values = ir.values
    resolved_sample_rate = ir.sample_rate

    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim < 2:
        raise ValueError("IR data must include at least channel and time axes")

    if resolved_sample_rate is None:
        raise ValueError("IR sample_rate is required")
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
    if output_key not in {"time", "samples"}:
        raise ValueError("output must be one of: time, samples")
    if not isinstance(absolute, bool):
        raise ValueError("absolute must be a boolean")

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

    if output_key == "time":
        itd_values = 1_000_000.0 * itd_values.astype(float) / resolved_sample_rate
    if absolute:
        itd_values = np.abs(itd_values)
    output_shape = ir_channel_last.shape[:-2]
    return itd_values.reshape(output_shape)


def ild(
    hrtf: "HRTF",
    mode: str = "broad-band",
    epsilon: float = 1e-12,
    absolute: bool = False,
) -> np.ndarray:
    """Compute interaural level difference for an HRTF.

    ``ild`` is an HRTF-level metric. It reads the active domain data from
    ``hrtf`` and returns interaural level differences in decibels between the
    first two ear channels, where ear index 0 is the left ear and ear index 1
    is the right ear. Any leading source or batch axes are preserved in the
    output.

    In ``mode="broad-band"``, ILD is calculated from the time-domain
    impulse responses in ``hrtf.IR.values`` by comparing the RMS level of the
    left and right ears over the final sample axis. In
    ``mode="frequency-dependent"``, ILD is calculated directly from
    ``hrtf.TF.values`` by comparing left and right magnitudes at each
    frequency bin.

    By default, the result is signed. Positive values mean the left ear has a
    greater level than the right ear. Negative values mean the right ear has a
    greater level than the left ear. Use ``absolute=True`` to return
    ``abs(ILD_db)``.

    Parameters
    ----------
    hrtf : HRTF
        HRTF object that provides ``IR`` and ``TF`` domain views. Broad-band
        mode requires ``hrtf.IR.values`` with layout ``(..., ear, samples)``.
        Frequency-dependent mode requires ``hrtf.TF.values`` with layout
        ``(..., ear, frequency)``.
    mode : {``"broad-band"``, ``"frequency-dependent"``}, default=``"broad-band"``
        ILD calculation mode. ``"broad-band"`` returns one ILD value per
        leading entry. ``"frequency-dependent"`` returns one ILD value per
        leading entry and frequency bin.
    epsilon : float, default=1e-12
        Positive floor added to left and right levels before division to avoid
        division by zero.
    absolute : bool, default=False
        If False, return signed ILD values in dB. If True, return
        ``abs(ILD_db)``.

    Returns
    -------
    numpy.ndarray
        ILD values in dB. Broad-band mode returns
        shape ``hrtf.IR.values.shape[:-2]``. Frequency-dependent mode returns
        shape ``hrtf.TF.values.shape[:-2] + (hrtf.TF.values.shape[-1],)``.

    Raises
    ------
    ValueError
        If ``hrtf`` is not an HRTF-like object with ``IR`` and ``TF`` domains,
        if the selected domain values are missing, empty, not NumPy arrays, do
        not contain ear and sample/frequency axes, or contain fewer than two ear
        channels; if ``mode`` is unsupported; if ``epsilon`` is not finite and
        positive; or if ``absolute`` is not a boolean.

    Examples
    --------
    Compute signed broad-band ILD for every source position in a loaded HRTF:

    >>> from hrtfpykit.hrtf import ild, load_hrtf
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> broad_band = ild(hrtf, mode="broad-band")
    >>> broad_band.shape
    (793,)

    Compute a frequency-dependent ILD matrix:

    >>> frequency_dependent = ild(hrtf, mode="frequency-dependent")
    >>> frequency_dependent.shape
    (793, 129)

    Return unsigned broad-band ILD when side is not needed:

    >>> unsigned_ild = ild(hrtf, mode="broad-band", absolute=True)
    >>> unsigned_ild.shape
    (793,)

    """
    if not hasattr(hrtf, "IR") or not hasattr(hrtf, "TF"):
        raise ValueError("hrtf must be an HRTF instance")

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
    if not isinstance(absolute, bool):
        raise ValueError("absolute must be a boolean")

    if mode_key == "broad-band":
        ir = cast(Any, hrtf).IR
        if not hasattr(ir, "values"):
            raise ValueError("hrtf must provide an IR domain with values")
        ir_values = ir.values
        if ir_values is None:
            raise ValueError("IR data is not available")
        if not isinstance(ir_values, np.ndarray):
            raise ValueError("IR data must be a NumPy array")
        if ir_values.size == 0:
            raise ValueError("IR data must be non-empty")
        if ir_values.ndim < 2:
            raise ValueError("IR data must include at least ear and time axes")
        if ir_values.shape[-2] < 2:
            raise ValueError("IR ear axis must contain at least two channels (0=left, 1=right)")
        left_values = np.asarray(ir_values[..., 0, :], dtype=float)
        right_values = np.asarray(ir_values[..., 1, :], dtype=float)
        left_rms = np.sqrt(np.mean(np.square(left_values), axis=-1))
        right_rms = np.sqrt(np.mean(np.square(right_values), axis=-1))
        ild_linear = (left_rms + epsilon) / (right_rms + epsilon)
    else:
        tf = cast(Any, hrtf).TF
        if not hasattr(tf, "values"):
            raise ValueError("hrtf must provide a TF domain with values")
        tf_values = tf.values
        if tf_values is None:
            raise ValueError("TF data is not available")
        if not isinstance(tf_values, np.ndarray):
            raise ValueError("TF data must be a NumPy array")
        if tf_values.size == 0:
            raise ValueError("TF data must be non-empty")
        if tf_values.ndim < 2:
            raise ValueError("TF data must include at least ear and frequency axes")
        if tf_values.shape[-2] < 2:
            raise ValueError("TF ear axis must contain at least two channels (0=left, 1=right)")
        left_magnitude = np.abs(tf_values[..., 0, :])
        right_magnitude = np.abs(tf_values[..., 1, :])
        ild_linear = (left_magnitude + epsilon) / (right_magnitude + epsilon)

    ild_values = magnitude_to_db(ild_linear)
    if absolute:
        ild_values = np.abs(ild_values)
    return ild_values


def rms(
    hrtf: "HRTF",
    output: str = "db",
    reference: float | str = 1.0,
    reduction_axis: str | tuple[str, ...] | None = None,
    reduction_method: str = "mean",
) -> np.ndarray:
    """Compute RMS values for an HRTF.

    ``rms`` is an HRTF metric. It reads HRIR data from ``hrtf.IR.values``
    and always computes the first RMS over the final sample axis. For standard
    HRTF data with shape ``(sources, ears, samples)``, this returns one RMS
    value for each source and ear.

    ``reduction_axis`` applies only after the sample axis RMS is computed. It
    selects remaining HRTF axes such as source positions and ears.
    ``reduction_method`` then chooses how those RMS values are reduced:
    ``"mean"`` averages them, while ``"rms"`` applies a second RMS over the
    selected axes.

    Reduction is applied in the selected output representation. With
    ``output="db"``, RMS values are converted to dB before reduction, so
    ``reduction_method="mean"`` averages dB values. With
    ``output="linear"``, reduction is applied to linear RMS amplitudes.

    Parameters
    ----------
    hrtf : HRTF
        HRTF object that provides ``IR.values`` with the final axis interpreted
        as time samples. Standard data use layout ``(..., ear, samples)``.
    output : {"db", "linear"}
        Output representation. The default is ``"db"``. ``"linear"`` returns
        RMS amplitudes. ``"db"`` converts RMS amplitudes to dB values.
    reference : float or "max"
        Reference used when ``output="db"``. The default is 1.0. ``"max"``
        uses the maximum RMS value before output conversion and domain
        reduction.
    reduction_axis : {"source", "ear", "global"}, tuple of str, or None
        HRTF axis or axes reduced after the first RMS calculation. The default
        is None, which returns the natural source by ear RMS array.
        ``"source"`` reduces all leading source axes and preserves ears.
        ``"ear"`` reduces the ear axis and preserves source axes.
        ``"global"`` reduces source and ear axes. The plural aliases
        ``"sources"`` and ``"ears"`` are also accepted.
    reduction_method : {"mean", "rms"}
        Method used for ``reduction_axis``. The default is ``"mean"``.
        ``"mean"`` computes the arithmetic average of the RMS values in the
        selected output representation. ``"rms"`` computes a second RMS over
        the selected axes. This parameter never changes the first RMS over the
        sample axis.

    Returns
    -------
    numpy.ndarray
        RMS values in the selected output representation after the requested
        reduction.

    Raises
    ------
    ValueError
        If the HRTF object or IR data are invalid; if ``output``,
        ``reduction_axis``, or ``reduction_method`` is unsupported; or if a
        requested domain axis is unavailable for the RMS array shape.

    Examples
    --------
    Compute one RMS value per source and ear:

    >>> from hrtfpykit.hrtf import load_hrtf, rms
    >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> rms_values = rms(hrtf, output="db")
    >>> rms_values.shape
    (793, 2)

    Average dB RMS values across source positions and ears:

    >>> average_level = rms(hrtf, output="db", reduction_axis="global")
    >>> average_level.shape
    ()

    Apply a second RMS reduction across ears after the sample axis RMS:

    >>> ear_reduced = rms(hrtf, output="linear", reduction_axis="ear", reduction_method="rms")
    >>> ear_reduced.shape
    (793,)

    """
    if not hasattr(hrtf, "IR"):
        raise ValueError("hrtf must be an HRTF instance")

    ir = cast(Any, hrtf).IR
    if not hasattr(ir, "values"):
        raise ValueError("hrtf must provide an IR domain with values")
    ir_values = ir.values
    if ir_values is None:
        raise ValueError("IR data is not available")
    if not isinstance(ir_values, np.ndarray):
        raise ValueError("IR data must be a NumPy array")
    if ir_values.size == 0:
        raise ValueError("IR data must be non-empty")
    if ir_values.ndim == 0:
        raise ValueError("IR data must have at least one dimension")
    if np.iscomplexobj(ir_values):
        raise ValueError("IR data must be real-valued for RMS calculation")

    output_key = str(output).strip().lower()
    if output_key not in {"linear", "db"}:
        raise ValueError("output must be one of: db, linear")

    reduction_method_key = str(reduction_method).strip().lower()
    if reduction_method_key not in {"mean", "rms"}:
        raise ValueError("reduction_method must be one of: mean, rms")

    reduction_axes: tuple[str, ...]
    if reduction_axis is None:
        reduction_axes = ()
    elif isinstance(reduction_axis, str):
        reduction_axis_key = reduction_axis.strip().lower()
        if reduction_axis_key == "global":
            reduction_axes = ("source", "ear")
        elif reduction_axis_key in {"source", "sources"}:
            reduction_axes = ("source",)
        elif reduction_axis_key in {"ear", "ears"}:
            reduction_axes = ("ear",)
        else:
            raise ValueError("reduction_axis must be source, ear, global, a tuple of source/ear, or None")
    elif isinstance(reduction_axis, tuple):
        if len(reduction_axis) == 0:
            raise ValueError("reduction_axis tuple cannot be empty")
        reduction_axes_list: list[str] = []
        for reduction_axis_value in reduction_axis:
            reduction_axis_key = str(reduction_axis_value).strip().lower()
            if reduction_axis_key == "global":
                raise ValueError("reduction_axis='global' cannot be combined with other axes")
            if reduction_axis_key in {"source", "sources"}:
                reduction_axes_list.append("source")
            elif reduction_axis_key in {"ear", "ears"}:
                reduction_axes_list.append("ear")
            else:
                raise ValueError("reduction_axis tuple entries must be source or ear")
        if len(set(reduction_axes_list)) != len(reduction_axes_list):
            raise ValueError("reduction_axis tuple cannot contain repeated entries")
        reduction_axes = tuple(reduction_axes_list)
    else:
        raise ValueError("reduction_axis must be a string, tuple of strings, or None")

    ir_float = np.asarray(ir_values, dtype=float)
    rms_values = np.sqrt(np.mean(np.square(ir_float), axis=-1))
    resolved_reference = reference
    if isinstance(reference, str) and reference.strip().lower() == "max":
        resolved_reference = float(np.max(rms_values))
        if not np.isfinite(resolved_reference) or resolved_reference <= 0.0:
            raise ValueError("reference='max' requires at least one positive RMS value")

    if output_key == "db":
        rms_values = magnitude_to_db(rms_values, reference=resolved_reference)

    selected_axes: list[int] = []
    for reduction_axis_key in reduction_axes:
        if reduction_axis_key == "ear":
            if rms_values.ndim < 1:
                raise ValueError("reduction_axis='ear' requires an ear axis")
            selected_axes.append(rms_values.ndim - 1)
        elif reduction_axis_key == "source":
            if rms_values.ndim < 2:
                raise ValueError("reduction_axis='source' requires at least one source axis")
            selected_axes.extend(range(rms_values.ndim - 1))

    if selected_axes:
        selected_axes_tuple = tuple(selected_axes)
        if reduction_method_key == "mean":
            rms_values = np.mean(rms_values, axis=selected_axes_tuple)
        else:
            rms_values = np.sqrt(np.mean(np.square(rms_values), axis=selected_axes_tuple))

    return np.asarray(rms_values)


def itd_difference(
    hrtf_reference: "HRTF",
    hrtfs: "HRTF | list[HRTF] | tuple[HRTF, ...]",
    method: str = "threshold",
    output: str = "time",
    thresh_level: float = -10.0,
    upper_cut_freq: float = 3000.0,
    filter_order: int = 10,
    absolute: bool = False,
    reduction_axis: str | None = None,
    reduction_method: str = "mean",
) -> np.ndarray:
    """Compute ITD differences from a reference HRTF.

    ``itd_difference`` compares one reference HRTF against one or more HRTFs.
    It first estimates signed ITD values with :func:`itd`, then subtracts the
    reference values from each compared HRTF. If ``hrtfs`` is one HRTF, no
    leading comparison axis is added. If ``hrtfs`` contains several HRTFs, the
    first axis indexes the compared
    HRTF ITD arrays, so standard HRTF data returns shape
    ``(len(itds), sources)``.

    The default result is signed. Positive values mean the compared HRTF has a
    greater ITD value than the reference at the same source position. Set
    ``absolute=True`` to return difference magnitudes.

    With a selected ``reduction_axis``, use ``absolute=True`` and
    ``reduction_method="mean"`` to compute ITD MAE over the selected axes.
    Use ``reduction_method="rms"`` to compute RMS ITD error. With
    ``absolute=False`` and ``reduction_method="mean"``, signs are kept and
    the result is mean signed ITD error.

    Parameters
    ----------
    hrtf_reference : HRTF
        Reference HRTF. It must provide IR data, an IR sample rate, and source
        positions.
    hrtfs : HRTF or sequence of HRTF
        HRTF object or objects compared against ``hrtf_reference``. Every HRTF
        must use the same source grid as the reference.
    method : {``"threshold"``, ``"maxiacce"``}, default=``"threshold"``
        ITD estimator passed to :func:`itd`.
    output : {``"time"``, ``"samples"``}, default=``"time"``
        Unit used before subtraction. ``"time"`` returns microseconds.
        ``"samples"`` requires matching sample rates across the reference and all compared HRTFs.
    thresh_level : float, default=-10.0
        Threshold offset passed to :func:`itd` when ``method="threshold"``.
    upper_cut_freq : float, default=3000.0
        Low-pass cutoff frequency passed to :func:`itd`.
    filter_order : int, default=10
        Filter order passed to :func:`itd`.
    absolute : bool, default=False
        If False, return signed differences ``compared - reference``. If True,
        return ``abs(compared - reference)``.
    reduction_axis : {``"itds"``, ``"sources"``, ``"global"``} or None, default=None
        Axis reduced after differences are computed. None returns every compared
        ITD difference array. ``"itds"`` reduces the compared HRTF ITD axis and
        preserves source positions. ``"sources"`` reduces source positions and
        preserves the compared HRTF ITD axis when several HRTF ITD arrays are provided.
        ``"global"`` reduces all axes.
    reduction_method : {``"mean"``, ``"rms"``}, default=``"mean"``
        Reduction method. ``"mean"`` computes the arithmetic mean over the
        selected axes. Use it with ``absolute=True`` to compute MAE. Use
        ``"rms"`` to compute RMS error over the selected axes.

    Returns
    -------
    numpy.ndarray
        ITD differences after the requested reduction. Without reduction, a
        single compared HRTF returns ``(sources,)`` for standard data. Several
        compared HRTF ITD arrays return ``(len(itds), sources)``.

    Raises
    ------
    ValueError
        If any input is not an HRTF object, if ``hrtfs`` is empty, if IR data
        or sample rates are missing, if source grids differ, if sample output is
        requested for different sample rates, if calculated ITD arrays have
        different shapes, or if an option value is unsupported.

    Examples
    --------
    Compare one processed HRTF against a reference and keep one value per source:

    >>> from hrtfpykit.hrtf import itd_difference, load_hrtf
    >>> reference = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> processed = reference.transform.add_itd(20, unit="samples")
    >>> values = itd_difference(
    ...     reference,
    ...     processed,
    ...     output="samples",
    ...     absolute=True,
    ...     reduction_axis="itds",
    ... )
    >>> values.shape
    (793,)

    Return one global RMS score in microseconds:

    >>> score = itd_difference(
    ...     reference,
    ...     processed,
    ...     output="time",
    ...     absolute=True,
    ...     reduction_axis="global",
    ...     reduction_method="rms",
    ... )
    >>> score.shape
    ()
    """
    output_key = str(output).strip().lower()
    if output_key not in {"time", "samples"}:
        raise ValueError("output must be one of: time, samples")
    if not isinstance(absolute, bool):
        raise ValueError("absolute must be a boolean")

    reduction_method_key = str(reduction_method).strip().lower()
    if reduction_method_key not in {"mean", "rms"}:
        raise ValueError("reduction_method must be one of: mean, rms")

    if reduction_axis is None:
        reduction_axis_key = None
    else:
        reduction_axis_key = str(reduction_axis).strip().lower()
        if reduction_axis_key in {"", "none"}:
            reduction_axis_key = None
        elif reduction_axis_key not in {"itd", "itds", "source", "sources", "global"}:
            raise ValueError("reduction_axis must be itds, sources, global, or None")

    if isinstance(hrtfs, (list, tuple)):
        compared_hrtfs = tuple(hrtfs)
    else:
        compared_hrtfs = (hrtfs,)
    if len(compared_hrtfs) == 0:
        raise ValueError("hrtfs must contain at least one HRTF")

    if not hasattr(hrtf_reference, "IR") or not hasattr(hrtf_reference, "Sources"):
        raise ValueError("hrtf_reference must be an HRTF instance")
    if hrtf_reference.IR.values is None:
        raise ValueError("hrtf_reference IR data is not available")
    if hrtf_reference.IR.sample_rate is None:
        raise ValueError("hrtf_reference IR sample_rate is required")

    reference_sample_rate = hrtf_reference.IR.sample_rate
    if isinstance(reference_sample_rate, bool):
        raise ValueError("hrtf_reference IR sample_rate must be a finite, positive value")
    try:
        reference_sample_rate = float(reference_sample_rate)
    except (TypeError, ValueError):
        raise ValueError("hrtf_reference IR sample_rate must be a finite, positive value") from None
    if not np.isfinite(reference_sample_rate) or reference_sample_rate <= 0.0:
        raise ValueError("hrtf_reference IR sample_rate must be a finite, positive value")

    for compared_index, hrtf in enumerate(compared_hrtfs):
        if not hasattr(hrtf, "IR") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"hrtfs[{compared_index}] must be an HRTF instance")
        if hrtf.IR.values is None:
            raise ValueError(f"hrtfs[{compared_index}] IR data is not available")
        if hrtf.IR.sample_rate is None:
            raise ValueError(f"hrtfs[{compared_index}] IR sample_rate is required")
        compared_sample_rate = hrtf.IR.sample_rate
        if isinstance(compared_sample_rate, bool):
            raise ValueError(f"hrtfs[{compared_index}] IR sample_rate must be a finite, positive value")
        try:
            compared_sample_rate = float(compared_sample_rate)
        except (TypeError, ValueError):
            raise ValueError(f"hrtfs[{compared_index}] IR sample_rate must be a finite, positive value") from None
        if not np.isfinite(compared_sample_rate) or compared_sample_rate <= 0.0:
            raise ValueError(f"hrtfs[{compared_index}] IR sample_rate must be a finite, positive value")
        if output_key == "samples" and not np.isclose(
            reference_sample_rate,
            compared_sample_rate,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("output='samples' requires equal sample_rate in all HRTFs")

    reference_positions = np.asarray(hrtf_reference.Sources.get_positions(angle_unit="degrees"), dtype=float)
    for compared_index, hrtf in enumerate(compared_hrtfs):
        compared_positions = np.asarray(hrtf.Sources.get_positions(angle_unit="degrees"), dtype=float)
        if reference_positions.shape != compared_positions.shape:
            raise ValueError("HRTFs must have the same number of source positions")
        if not np.allclose(reference_positions, compared_positions, atol=1e-8, rtol=0.0):
            raise ValueError("HRTFs must share the same source positions for ITD difference")

    reference_itd = np.asarray(
        itd(
            hrtf_reference,
            method=method,
            output=output_key,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        ),
        dtype=float,
    )
    difference_arrays: list[np.ndarray] = []
    for compared_index, hrtf in enumerate(compared_hrtfs):
        compared_itd = np.asarray(
            itd(
                hrtf,
                method=method,
                output=output_key,
                thresh_level=thresh_level,
                upper_cut_freq=upper_cut_freq,
                filter_order=filter_order,
            ),
            dtype=float,
        )
        if reference_itd.shape != compared_itd.shape:
            raise ValueError("Calculated ITD arrays must have matching shapes")
        difference_arrays.append(compared_itd - reference_itd)

    difference_values = np.stack(difference_arrays, axis=0)
    if absolute:
        difference_values = np.abs(difference_values)

    reduction_axes: tuple[int, ...]
    if reduction_axis_key is None:
        if len(compared_hrtfs) == 1:
            return np.asarray(difference_values[0])
        return np.asarray(difference_values)
    if reduction_axis_key in {"itd", "itds"}:
        reduction_axes = (0,)
    elif reduction_axis_key in {"source", "sources"}:
        reduction_axes = tuple(range(1, difference_values.ndim))
        if len(reduction_axes) == 0:
            raise ValueError("reduction_axis='sources' requires a source axis")
    else:
        reduction_axes = tuple(range(difference_values.ndim))

    if reduction_method_key == "mean":
        reduced_values = np.asarray(np.mean(difference_values, axis=reduction_axes))
    else:
        reduced_values = np.asarray(np.sqrt(np.mean(np.square(difference_values), axis=reduction_axes)))
    if len(compared_hrtfs) == 1 and reduced_values.ndim > 0 and reduced_values.shape[0] == 1:
        reduced_values = np.squeeze(reduced_values, axis=0)
    return np.asarray(reduced_values)


def ild_difference(
    hrtf_reference: "HRTF",
    hrtfs: "HRTF | list[HRTF] | tuple[HRTF, ...]",
    mode: str = "broad-band",
    epsilon: float = 1e-12,
    absolute: bool = False,
    reduction_axis: str | None = None,
    reduction_method: str = "mean",
) -> np.ndarray:
    """Compute ILD differences from a reference HRTF.

    ``ild_difference`` compares one reference HRTF against one or more HRTFs.
    It first computes signed ILD values with :func:`ild`, then subtracts the
    reference values from each compared HRTF. If ``hrtfs`` is one HRTF, no
    leading comparison axis is added. If ``hrtfs`` contains several HRTFs, the
    first axis indexes the compared
    HRTF ILD arrays. Broad-band mode then returns shape
    ``(len(ilds), sources)`` for standard data, and frequency-dependent mode
    returns shape ``(len(ilds), sources, frequency_bins)``.

    The default result is signed. Positive values mean the compared HRTF has a
    greater ILD value than the reference at the same source position. Set
    ``absolute=True`` to return difference magnitudes.

    With a selected ``reduction_axis``, use ``absolute=True`` and
    ``reduction_method="mean"`` to compute ILD MAE over the selected axes.
    Use ``reduction_method="rms"`` to compute RMS ILD error. With
    ``absolute=False`` and ``reduction_method="mean"``, signs are kept and
    the result is mean signed ILD error.

    Parameters
    ----------
    hrtf_reference : HRTF
        Reference HRTF. It must provide source positions and the domains needed
        by the selected ILD mode.
    hrtfs : HRTF or sequence of HRTF
        HRTF object or objects compared against ``hrtf_reference``. Every HRTF
        must use the same source grid as the reference. Frequency-dependent mode
        also requires matching TF frequency bins.
    mode : {``"broad-band"``, ``"frequency-dependent"``}, default=``"broad-band"``
        ILD mode passed to :func:`ild`.
    epsilon : float, default=1e-12
        Positive floor passed to :func:`ild`.
    absolute : bool, default=False
        If False, return signed differences ``compared - reference``. If True,
        return ``abs(compared - reference)``.
    reduction_axis : {``"ilds"``, ``"sources"``, ``"global"``} or None, default=None
        Axis reduced after differences are computed. None returns every compared
        ILD difference array. ``"ilds"`` reduces the compared HRTF ILD axis and
        preserves source positions. ``"sources"`` reduces source positions and
        preserves the compared HRTF ILD axis when several HRTF ILD arrays are provided.
        ``"global"`` reduces all axes.
    reduction_method : {``"mean"``, ``"rms"``}, default=``"mean"``
        Reduction method. ``"mean"`` computes the arithmetic mean over the
        selected axes. Use it with ``absolute=True`` to compute MAE. Use
        ``"rms"`` to compute RMS error over the selected axes.

    Returns
    -------
    numpy.ndarray
        ILD differences after the requested reduction. Without reduction, a
        single compared HRTF returns ``(sources,)`` in broad-band mode and
        ``(sources, frequency_bins)`` in frequency-dependent mode. Several
        compared HRTF ILD arrays keep a leading ILD comparison axis.

    Raises
    ------
    ValueError
        If any input is not an HRTF object, if ``hrtfs`` is empty, if the
        source grids differ, if frequency-dependent mode is requested with
        missing or different frequency bins, if ILD arrays have different
        shapes, or if an option value is unsupported.

    Examples
    --------
    Compare one processed HRTF against a reference and keep one value per source:

    >>> from hrtfpykit.hrtf import ild_difference, load_hrtf
    >>> reference = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> processed = reference.transform.apply_gain(-1.0, scale="db")
    >>> values = ild_difference(reference, processed, absolute=True, reduction_axis="ilds")
    >>> values.shape
    (793,)

    Average absolute ILD differences from several HRTFs into one source curve:

    >>> values = ild_difference(
    ...     reference,
    ...     [processed, processed],
    ...     absolute=True,
    ...     reduction_axis="ilds",
    ... )
    >>> values.shape
    (793,)

    Return one global RMS score:

    >>> score = ild_difference(
    ...     reference,
    ...     processed,
    ...     absolute=True,
    ...     reduction_axis="global",
    ...     reduction_method="rms",
    ... )
    >>> score.shape
    ()
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in {"broad-band", "frequency-dependent"}:
        raise ValueError("mode must be one of: broad-band, frequency-dependent")
    if not isinstance(absolute, bool):
        raise ValueError("absolute must be a boolean")

    reduction_method_key = str(reduction_method).strip().lower()
    if reduction_method_key not in {"mean", "rms"}:
        raise ValueError("reduction_method must be one of: mean, rms")

    if reduction_axis is None:
        reduction_axis_key = None
    else:
        reduction_axis_key = str(reduction_axis).strip().lower()
        if reduction_axis_key in {"", "none"}:
            reduction_axis_key = None
        elif reduction_axis_key not in {"ild", "ilds", "source", "sources", "global"}:
            raise ValueError("reduction_axis must be ilds, sources, global, or None")

    if isinstance(hrtfs, (list, tuple)):
        compared_hrtfs = tuple(hrtfs)
    else:
        compared_hrtfs = (hrtfs,)
    if len(compared_hrtfs) == 0:
        raise ValueError("hrtfs must contain at least one HRTF")

    if not hasattr(hrtf_reference, "Sources") or not hasattr(hrtf_reference, "IR") or not hasattr(hrtf_reference, "TF"):
        raise ValueError("hrtf_reference must be an HRTF instance")
    for compared_index, hrtf in enumerate(compared_hrtfs):
        if not hasattr(hrtf, "Sources") or not hasattr(hrtf, "IR") or not hasattr(hrtf, "TF"):
            raise ValueError(f"hrtfs[{compared_index}] must be an HRTF instance")

    reference_positions = np.asarray(hrtf_reference.Sources.get_positions(angle_unit="degrees"), dtype=float)
    for compared_index, hrtf in enumerate(compared_hrtfs):
        compared_positions = np.asarray(hrtf.Sources.get_positions(angle_unit="degrees"), dtype=float)
        if reference_positions.shape != compared_positions.shape:
            raise ValueError("HRTFs must have the same number of source positions")
        if not np.allclose(reference_positions, compared_positions, atol=1e-8, rtol=0.0):
            raise ValueError("HRTFs must share the same source positions for ILD difference")

    if mode_key == "frequency-dependent":
        if hrtf_reference.TF.frequency_bins is None:
            raise ValueError("hrtf_reference TF frequency_bins are required")
        reference_frequency_bins = np.asarray(hrtf_reference.TF.frequency_bins, dtype=float)
        for compared_index, hrtf in enumerate(compared_hrtfs):
            if hrtf.TF.frequency_bins is None:
                raise ValueError(f"hrtfs[{compared_index}] TF frequency_bins are required")
            compared_frequency_bins = np.asarray(hrtf.TF.frequency_bins, dtype=float)
            if reference_frequency_bins.shape != compared_frequency_bins.shape:
                raise ValueError("HRTFs must have matching TF frequency bins for frequency-dependent ILD difference")
            if not np.allclose(reference_frequency_bins, compared_frequency_bins, atol=1e-8, rtol=0.0):
                raise ValueError("HRTFs must share the same TF frequency bins for frequency-dependent ILD difference")

    reference_ild = np.asarray(
        ild(
            hrtf_reference,
            mode=mode_key,
            epsilon=epsilon,
        ),
        dtype=float,
    )
    difference_arrays: list[np.ndarray] = []
    for compared_index, hrtf in enumerate(compared_hrtfs):
        compared_ild = np.asarray(
            ild(
                hrtf,
                mode=mode_key,
                epsilon=epsilon,
            ),
            dtype=float,
        )
        if reference_ild.shape != compared_ild.shape:
            raise ValueError("Calculated ILD arrays must have matching shapes")
        difference_arrays.append(compared_ild - reference_ild)

    difference_values = np.stack(difference_arrays, axis=0)
    if absolute:
        difference_values = np.abs(difference_values)

    reduction_axes: tuple[int, ...]
    if reduction_axis_key is None:
        if len(compared_hrtfs) == 1:
            return np.asarray(difference_values[0])
        return np.asarray(difference_values)
    if reduction_axis_key in {"ild", "ilds"}:
        reduction_axes = (0,)
    elif reduction_axis_key in {"source", "sources"}:
        if mode_key == "frequency-dependent":
            reduction_axes = tuple(range(1, difference_values.ndim - 1))
        else:
            reduction_axes = tuple(range(1, difference_values.ndim))
        if len(reduction_axes) == 0:
            raise ValueError("reduction_axis='sources' requires a source axis")
    else:
        reduction_axes = tuple(range(difference_values.ndim))

    if reduction_method_key == "mean":
        reduced_values = np.asarray(np.mean(difference_values, axis=reduction_axes))
    else:
        reduced_values = np.asarray(np.sqrt(np.mean(np.square(difference_values), axis=reduction_axes)))
    if len(compared_hrtfs) == 1 and reduced_values.ndim > 0 and reduced_values.shape[0] == 1:
        reduced_values = np.squeeze(reduced_values, axis=0)
    return np.asarray(reduced_values)


def lsd(
    hrtf_reference: "HRTF",
    hrtfs: "HRTF | list[HRTF] | tuple[HRTF, ...]",
    ear: str = "both",
    plane: str = "all",
    plane_angle: float = 0.0,
    positions: np.ndarray | list | tuple | str | None = None,
    frequencies: float | list[float] | tuple[float, ...] | np.ndarray | None = None,
    frequency_bands: tuple[float, float] | list[tuple[float, float]] | tuple[tuple[float, float], ...] | np.ndarray | None = None,
    reduction_axis: str | tuple[str, ...] | list[str] | None = None,
    reduction_method: str = "mean",
    epsilon: float = 1e-12,
) -> np.ndarray | float:
    """Compute log-spectral distortion from a reference HRTF.

    ``lsd`` compares one reference HRTF against one or more HRTFs in the
    logarithmic magnitude domain. The metric first computes dB magnitude
    differences for each selected source, ear, and frequency bin. It then
    applies RMS over frequency, producing one LSD value per selected source and
    selected ear.

    If ``hrtfs`` is one HRTF, no leading comparison axis is added. If ``hrtfs``
    contains several HRTFs, the first axis indexes the compared HRTF LSD arrays.
    With ``ear="both"`` and no reduction, one compared HRTF returns
    ``(selected_sources, 2)`` and several compared HRTF LSD arrays return
    ``(len(lsds), selected_sources, 2)``.

    Parameters
    ----------
    hrtf_reference : HRTF
        Reference HRTF. It must provide TF values, frequency bins, and source
        positions.
    hrtfs : HRTF or sequence of HRTF
        HRTF object or objects compared against ``hrtf_reference``. Every HRTF
        must use the same source grid, TF shape, and frequency bins as the
        reference.
    ear : {``"left"``, ``"right"``, ``"both"``}, default=``"both"``
        Ear channel selection. ``"left"`` uses ear channel 0, ``"right"`` uses
        ear channel 1, and ``"both"`` keeps both ears unless the ear axis is
        reduced.
    plane : {``"all"``, ``"horizontal"``, ``"median"``}, default=``"all"``
        Spatial subset used before comparison.
    plane_angle : float, default=0.0
        Plane coordinate in degrees. For ``plane="horizontal"`` this is
        spherical elevation. For ``plane="median"`` this is lateral-polar
        lateral angle.
    positions : np.ndarray | list | tuple | str | None, default=None
        Optional source-position selector. Queries are resolved on the
        reference source grid and intersected with the selected plane.
    frequencies : float, sequence of float, numpy.ndarray, or None, default=None
        Optional frequency selector in hertz. Each requested frequency is
        mapped to the nearest available TF bin. Mutually exclusive with
        ``frequency_bands``.
    frequency_bands : pair, sequence of pairs, numpy.ndarray, or None, default=None
        Optional inclusive frequency band or bands in hertz. Mutually exclusive
        with ``frequencies``.
    reduction_axis : {``"lsds"``, ``"sources"``, ``"ears"``, ``"global"``}, sequence, or None, default=None
        Axis or axes reduced after LSD values are computed. None returns the
        natural LSD array. ``"lsds"`` reduces the compared HRTF LSD axis and
        preserves sources and ears. ``"sources"`` reduces source positions.
        ``"ears"`` reduces ears and requires ``ear="both"``. ``"global"``
        reduces all axes.
    reduction_method : {``"mean"``, ``"rms"``}, default=``"mean"``
        Reduction method. ``"mean"`` computes the arithmetic mean of LSD values.
        ``"rms"`` computes the root mean square over the selected axis.
    epsilon : float, default=1e-12
        Positive lower bound applied to magnitudes before conversion to dB.

    Returns
    -------
    numpy.ndarray or float
        LSD values in dB after the requested reduction. A single compared HRTF
        has no leading comparison axis. Several compared HRTF LSD arrays keep
        that axis unless ``reduction_axis`` includes ``"lsds"`` or ``"global"``. A
        full global reduction returns a scalar float.

    Raises
    ------
    ValueError
        If any input is not an HRTF object, if ``hrtfs`` is empty, if TF values
        or frequency bins are missing, if source grids, TF shapes, or frequency
        bins differ, if selected data are not arranged as
        ``(positions, ears, frequency_bins)``, if option values are unsupported,
        if ``ears`` reduction is requested without ``ear="both"``, if epsilon
        is not finite and positive, or if source or frequency selectors produce
        no data.

    Examples
    --------
    Compare one processed HRTF against a reference and keep one value per source:

    >>> from hrtfpykit.hrtf import load_hrtf, lsd
    >>> reference = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
    >>> processed = reference.transform.apply_gain(gain=10, scale="db")
    >>> values = lsd(reference, processed, ear="left")
    >>> values.shape
    (793,)

    Return one global RMS score from two compared HRTF LSD arrays:

    >>> score = lsd(
    ...     reference,
    ...     [processed, processed],
    ...     ear="both",
    ...     reduction_axis="global",
    ...     reduction_method="rms",
    ... )
    >>> isinstance(score, float)
    True
    """
    if isinstance(hrtfs, (list, tuple)):
        compared_hrtfs = tuple(hrtfs)
    else:
        compared_hrtfs = (hrtfs,)
    if len(compared_hrtfs) == 0:
        raise ValueError("hrtfs must contain at least one HRTF")

    if not hasattr(hrtf_reference, "TF") or not hasattr(hrtf_reference, "Sources"):
        raise ValueError("hrtf_reference must be an HRTF instance")
    if hrtf_reference.TF.values is None:
        raise ValueError("hrtf_reference TF data is not available")
    if hrtf_reference.TF.frequency_bins is None:
        raise ValueError("hrtf_reference TF frequency_bins are required")
    for compared_index, hrtf in enumerate(compared_hrtfs):
        if not hasattr(hrtf, "TF") or not hasattr(hrtf, "Sources"):
            raise ValueError(f"hrtfs[{compared_index}] must be an HRTF instance")
        if hrtf.TF.values is None:
            raise ValueError(f"hrtfs[{compared_index}] TF data is not available")
        if hrtf.TF.frequency_bins is None:
            raise ValueError(f"hrtfs[{compared_index}] TF frequency_bins are required")

    plane_key = str(plane).strip().lower()
    if plane_key not in {"all", "horizontal", "median"}:
        raise ValueError("plane must be one of: all, horizontal, median")

    reduction_method_key = str(reduction_method).strip().lower()
    if reduction_method_key not in {"mean", "rms"}:
        raise ValueError("reduction_method must be one of: mean, rms")

    global_reduction = False
    if reduction_axis is None:
        reduction_axes: tuple[str, ...] = ()
    elif isinstance(reduction_axis, str):
        reduction_axis_key = reduction_axis.strip().lower()
        if reduction_axis_key in {"", "none"}:
            reduction_axes = ()
        elif reduction_axis_key == "global":
            global_reduction = True
            reduction_axes = ("lsds", "sources", "ears")
        elif reduction_axis_key in {"lsd", "lsds"}:
            reduction_axes = ("lsds",)
        elif reduction_axis_key in {"source", "sources"}:
            reduction_axes = ("sources",)
        elif reduction_axis_key in {"ear", "ears"}:
            reduction_axes = ("ears",)
        else:
            raise ValueError("reduction_axis must be lsds, sources, ears, global, a sequence of axes, or None")
    elif isinstance(reduction_axis, tuple | list):
        normalized_axes: list[str] = []
        for axis in reduction_axis:
            axis_key = str(axis).strip().lower()
            if axis_key in {"", "none"}:
                raise ValueError("reduction_axis='none' cannot be combined with other axes")
            if axis_key == "global":
                raise ValueError("reduction_axis='global' cannot be combined with other axes")
            if axis_key in {"lsd", "lsds"}:
                normalized_axis = "lsds"
            elif axis_key in {"source", "sources"}:
                normalized_axis = "sources"
            elif axis_key in {"ear", "ears"}:
                normalized_axis = "ears"
            else:
                raise ValueError("reduction_axis entries must be lsds, sources, or ears")
            if normalized_axis not in normalized_axes:
                normalized_axes.append(normalized_axis)
        reduction_axes = tuple(normalized_axes)
    else:
        raise ValueError("reduction_axis must be a string, sequence of strings, or None")

    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a finite, positive value.")
    try:
        epsilon = float(epsilon)
    except (TypeError, ValueError):
        raise ValueError("epsilon must be a finite, positive value.") from None
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite, positive value.")

    reference_positions = np.asarray(hrtf_reference.Sources.get_positions(angle_unit="degrees"), dtype=float)
    for compared_index, hrtf in enumerate(compared_hrtfs):
        compared_positions = np.asarray(hrtf.Sources.get_positions(angle_unit="degrees"), dtype=float)
        if reference_positions.shape != compared_positions.shape:
            raise ValueError("HRTFs must have the same number of source positions")
        if not np.allclose(reference_positions, compared_positions, atol=1e-8, rtol=0.0):
            raise ValueError("HRTFs must share the same source positions for LSD")

    reference_tf = np.asarray(hrtf_reference.TF.values)
    if reference_tf.ndim != 3:
        raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
    if reference_tf.shape[0] != reference_positions.shape[0]:
        raise ValueError("TF positions axis must match source positions count")
    if reference_tf.shape[1] < 2:
        raise ValueError("TF ear axis must contain at least two channels (0=left, 1=right)")

    reference_frequency_bins = np.asarray(hrtf_reference.TF.frequency_bins, dtype=float).reshape(-1)
    if reference_frequency_bins.size != reference_tf.shape[-1]:
        raise ValueError("TF frequency axis length must match frequency_bins length")
    for compared_index, hrtf in enumerate(compared_hrtfs):
        compared_tf = np.asarray(hrtf.TF.values)
        if compared_tf.ndim != 3:
            raise ValueError("TF values must have shape (positions, ears, frequency_bins)")
        if reference_tf.shape != compared_tf.shape:
            raise ValueError("HRTFs must have matching TF shapes for LSD")
        compared_frequency_bins = np.asarray(hrtf.TF.frequency_bins, dtype=float).reshape(-1)
        if reference_frequency_bins.shape != compared_frequency_bins.shape:
            raise ValueError("HRTFs must have matching TF frequency_bins")
        if not np.allclose(reference_frequency_bins, compared_frequency_bins, atol=1e-8, rtol=0.0):
            raise ValueError("HRTFs must share the same TF frequency_bins for LSD")

    ear_key = str(ear).strip().lower()
    if ear_key not in {"left", "right", "both"}:
        raise ValueError("ear must be one of: left, right, both")
    if ear_key == "left":
        selected_ear_indices = np.array([0], dtype=int)
    elif ear_key == "right":
        selected_ear_indices = np.array([1], dtype=int)
    else:
        selected_ear_indices = np.array([0, 1], dtype=int)
    if "ears" in reduction_axes and ear_key != "both":
        if global_reduction:
            reduction_axes = tuple(axis for axis in reduction_axes if axis != "ears")
        else:
            raise ValueError("reduction axis 'ears' can only be used when ear='both'")

    if plane_key == "all":
        selected_positions = np.arange(reference_positions.shape[0], dtype=int)
    elif plane_key == "horizontal":
        selected_positions, _ = get_horizontal_plane(hrtf=hrtf_reference, plane_angle=float(plane_angle))
    else:
        selected_positions, _ = get_median_plane(hrtf=hrtf_reference, plane_angle=float(plane_angle))
    selected_positions = np.asarray(selected_positions, dtype=int).reshape(-1)
    if selected_positions.size == 0:
        raise ValueError("Selected plane has no source positions")

    if positions is not None:
        position_queries = get_position_queries(positions)
        selected_from_queries: list[int] = []
        for query in position_queries:
            position_index, _ = hrtf_reference.Sources.get_position_index(
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

    if frequencies is not None and frequency_bands is not None:
        raise ValueError("frequencies and frequency_bands are mutually exclusive")
    if frequencies is None and frequency_bands is None:
        selected_frequency_indices = np.where(
            (reference_frequency_bins >= 20.0) & (reference_frequency_bins <= 20000.0)
        )[0]
        if selected_frequency_indices.size == 0:
            raise ValueError("No frequency bins available in the default LSD range [20.0, 20000.0] Hz")
    elif frequencies is not None:
        raw_frequency_values = np.asarray(frequencies, dtype=object).reshape(-1)
        if any(isinstance(value, bool | np.bool_) for value in raw_frequency_values.tolist()):
            raise ValueError("frequencies must contain finite, non-negative value(s)")
        try:
            frequency_values = np.asarray(frequencies, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError("frequencies must contain finite, non-negative value(s)") from None
        if frequency_values.size == 0:
            raise ValueError("frequencies must contain at least one value when provided")
        if not np.all(np.isfinite(frequency_values)) or np.any(frequency_values < 0.0):
            raise ValueError("frequencies must contain finite, non-negative value(s)")
        nearest_frequency_indices = [
            int(np.argmin(np.abs(reference_frequency_bins - float(target_frequency))))
            for target_frequency in frequency_values
        ]
        selected_frequency_indices = np.asarray(tuple(dict.fromkeys(nearest_frequency_indices)), dtype=int)
    else:
        raw_bands = np.asarray(frequency_bands, dtype=object)
        if any(isinstance(value, bool | np.bool_) for value in raw_bands.reshape(-1).tolist()):
            raise ValueError("frequency_bands must contain finite, non-negative values")
        try:
            bands = np.asarray(frequency_bands, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("frequency_bands must contain (minimum, maximum) pairs") from None
        if bands.ndim == 1 and bands.size == 2:
            bands = bands.reshape(1, 2)
        if bands.ndim != 2 or bands.shape[0] == 0 or bands.shape[1] != 2:
            raise ValueError("frequency_bands must contain (minimum, maximum) pairs")
        if not np.all(np.isfinite(bands)) or np.any(bands < 0.0):
            raise ValueError("frequency_bands must contain finite, non-negative values")
        if np.any(bands[:, 0] > bands[:, 1]):
            raise ValueError("frequency_bands minimum must not exceed maximum")
        selected_mask = np.zeros(reference_frequency_bins.shape, dtype=bool)
        for minimum, maximum in bands:
            selected_mask |= (reference_frequency_bins >= float(minimum)) & (reference_frequency_bins <= float(maximum))
        selected_frequency_indices = np.flatnonzero(selected_mask).astype(int)
        if selected_frequency_indices.size == 0:
            raise ValueError("frequency_bands selected no available TF bins")

    reference_slice = np.asarray(
        reference_tf[np.ix_(selected_positions, selected_ear_indices, selected_frequency_indices)],
        dtype=complex,
    )
    reference_db = magnitude_to_db(np.maximum(np.abs(reference_slice), epsilon))
    lsd_arrays: list[np.ndarray] = []
    for hrtf in compared_hrtfs:
        compared_tf = np.asarray(hrtf.TF.values)
        compared_slice = np.asarray(
            compared_tf[np.ix_(selected_positions, selected_ear_indices, selected_frequency_indices)],
            dtype=complex,
        )
        if reference_slice.shape != compared_slice.shape:
            raise ValueError("Selected TF slices must have matching shapes")
        compared_db = magnitude_to_db(np.maximum(np.abs(compared_slice), epsilon))
        difference_db = reference_db - compared_db
        lsd_values = np.sqrt(np.mean(np.square(difference_db), axis=-1))
        if ear_key != "both":
            lsd_values = np.squeeze(lsd_values, axis=1)
        lsd_arrays.append(np.asarray(lsd_values, dtype=float))

    lsd_output = np.stack(lsd_arrays, axis=0)
    if len(reduction_axes) == 0:
        if len(compared_hrtfs) == 1:
            return np.asarray(lsd_output[0])
        return np.asarray(lsd_output)

    selected_axes: list[int] = []
    for reduction_axis_key in reduction_axes:
        if reduction_axis_key == "lsds":
            selected_axes.append(0)
        elif reduction_axis_key == "sources":
            selected_axes.append(1)
        elif reduction_axis_key == "ears":
            if ear_key != "both":
                raise ValueError("reduction axis 'ears' can only be used when ear='both'")
            selected_axes.append(lsd_output.ndim - 1)
    selected_axes_tuple = tuple(dict.fromkeys(selected_axes))
    if reduction_method_key == "mean":
        reduced_values = np.asarray(np.mean(lsd_output, axis=selected_axes_tuple))
    else:
        reduced_values = np.asarray(np.sqrt(np.mean(np.square(lsd_output), axis=selected_axes_tuple)))
    if len(compared_hrtfs) == 1 and reduced_values.ndim > 0 and reduced_values.shape[0] == 1:
        reduced_values = np.squeeze(reduced_values, axis=0)
    if reduced_values.ndim == 0:
        return float(reduced_values)
    return np.asarray(reduced_values)
