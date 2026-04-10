from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

from .options import FrequencyAxisOptions


def build_frequency_axis(
    scale: str,
    default_ticks: tuple[float, ...],
    default_labels: tuple[str, ...],
    frequency_bins: np.ndarray | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    options: FrequencyAxisOptions | None = None,
) -> FrequencyAxisOptions:
    frequency_axis_options = FrequencyAxisOptions() if options is None else options
    resolved_frequency_bins = None
    if frequency_bins is not None:
        resolved_frequency_bins = np.asarray(frequency_bins, dtype=float)
        if resolved_frequency_bins.ndim != 1 or resolved_frequency_bins.size == 0:
            raise ValueError("frequency_bins must be a non-empty 1D array")
    requested_freq_min = (
        frequency_axis_options.freq_min if freq_min is None else freq_min
    )
    if requested_freq_min is None:
        if resolved_frequency_bins is None:
            raise ValueError("freq_min is required when frequency_bins are not provided")
        if scale == "log":
            positive_frequency_bins = resolved_frequency_bins[
                resolved_frequency_bins > 0.0
            ]
            if positive_frequency_bins.size == 0:
                raise ValueError(
                    "frequency_bins must include a positive value for logarithmic frequency axis"
                )
            resolved_freq_min = float(np.min(positive_frequency_bins))
        else:
            resolved_freq_min = float(np.min(resolved_frequency_bins))
    else:
        resolved_freq_min = float(requested_freq_min)
    requested_freq_max = (
        frequency_axis_options.freq_max if freq_max is None else freq_max
    )
    if requested_freq_max is None:
        if resolved_frequency_bins is None:
            raise ValueError("freq_max is required when frequency_bins are not provided")
        resolved_freq_max = float(np.max(resolved_frequency_bins))
    else:
        resolved_freq_max = float(requested_freq_max)
    resolved_margin_ratio = (
        0.03
        if frequency_axis_options.margin_ratio is None
        else float(frequency_axis_options.margin_ratio)
    )
    if not np.isfinite(resolved_freq_min) or not np.isfinite(resolved_freq_max):
        raise ValueError("freq_min and freq_max must be finite values")
    if resolved_freq_min >= resolved_freq_max:
        raise ValueError("freq_min must be smaller than freq_max")
    if scale == "log" and resolved_freq_min <= 0.0:
        raise ValueError("freq_min must be positive for logarithmic frequency axis")
    if resolved_margin_ratio < 0.0:
        raise ValueError("margin_ratio must be non-negative")

    resolved_ticks = (
        tuple(float(tick) for tick in default_ticks)
        if frequency_axis_options.ticks is None
        else tuple(float(tick) for tick in frequency_axis_options.ticks)
    )
    resolved_labels = (
        tuple(str(label) for label in default_labels)
        if frequency_axis_options.labels is None
        and frequency_axis_options.ticks is None
        else (
            tuple(f"{tick / 1000.0:g}" for tick in resolved_ticks)
            if frequency_axis_options.labels is None
            else tuple(str(label) for label in frequency_axis_options.labels)
        )
    )
    if len(resolved_ticks) != len(resolved_labels):
        raise ValueError("frequency axis ticks and labels must have the same length")
    if scale == "log" and any(tick <= 0.0 for tick in resolved_ticks):
        raise ValueError("frequency axis ticks must be positive for logarithmic axis")

    visible_pairs = tuple(
        (tick, tick_label)
        for tick, tick_label in zip(resolved_ticks, resolved_labels)
        if resolved_freq_min <= tick <= resolved_freq_max
    )
    return FrequencyAxisOptions(
        ticks=tuple(tick for tick, _ in visible_pairs),
        labels=tuple(tick_label for _, tick_label in visible_pairs),
        freq_min=resolved_freq_min,
        freq_max=resolved_freq_max,
        margin_ratio=resolved_margin_ratio,
    )


def apply_frequency_axis(
    scale: str,
    ax: plt.Axes,
    axis: str,
    label: str | None = None,
    options: FrequencyAxisOptions | None = None,
) -> None:
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis accepts 'x', 'y', or 'z'")
    resolved_frequency_axis = FrequencyAxisOptions() if options is None else options
    ticks_khz = [tick / 1000.0 for tick in resolved_frequency_axis.ticks or ()]
    resolved_freq_min_khz = float(resolved_frequency_axis.freq_min) / 1000.0
    resolved_freq_max_khz = float(resolved_frequency_axis.freq_max) / 1000.0
    margin_ratio = float(resolved_frequency_axis.margin_ratio)

    if axis == "x":
        axis_object = ax.xaxis
        set_scale = ax.set_xscale
        set_limits = ax.set_xlim
        set_label = ax.set_xlabel
    elif axis == "y":
        axis_object = ax.yaxis
        set_scale = ax.set_yscale
        set_limits = ax.set_ylim
        set_label = ax.set_ylabel
    else:
        axis_object = getattr(ax, "zaxis", None)
        set_scale = getattr(ax, "set_zscale", None)
        set_limits = getattr(ax, "set_zlim", None)
        set_label = getattr(ax, "set_zlabel", None)
        if axis_object is None or set_scale is None or set_limits is None:
            raise ValueError("z-axis formatting requires a matplotlib 3D axis")
        if set_label is None:
            raise ValueError("z-axis labeling requires a matplotlib 3D axis")
    if label is not None:
        set_label(label)

    if scale == "log":
        set_scale("log")
        log_min = np.log10(resolved_freq_min_khz)
        log_max = np.log10(resolved_freq_max_khz)
        margin_log = (log_max - log_min) * margin_ratio
        axis_min = 10 ** (log_min - margin_log)
        axis_max = 10 ** (log_max + margin_log)
    else:
        set_scale("linear")
        margin_linear = (resolved_freq_max_khz - resolved_freq_min_khz) * margin_ratio
        axis_min = resolved_freq_min_khz - margin_linear
        axis_max = resolved_freq_max_khz + margin_linear
    set_limits(axis_min, axis_max)
    axis_object.set_major_locator(FixedLocator(ticks_khz))
    axis_object.set_major_formatter(FixedFormatter(resolved_frequency_axis.labels or ()))
    axis_object.set_minor_locator(NullLocator())
    axis_object.set_minor_formatter(NullFormatter())
    if hasattr(axis_object, "offsetText"):
        axis_object.offsetText.set_visible(False)
