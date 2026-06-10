from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    """Default plot labels shared by HRTF plotting components.

    :class:`~hrtfpykit.plots.labels.Labels` centralizes the text used by axis
    formatters, heatmap colorbars, polar radial axes, comparison plots, and
    spatial annotations. The plotting layer imports these class attributes
    instead of repeating label strings in individual methods, which keeps HRTF,
    HRIR, ITD, ILD, LSD, and source-grid figures visually consistent.

    Frequency values plotted by the high-level HRTF helpers are displayed in
    kilohertz, so the frequency axis label and frequency tick labels use kHz
    text even when lower-level axis configuration receives tick positions in
    hertz. Time-domain impulse-response plots use milliseconds or sample indices
    depending on the selected x-axis mode. Absolute ITD and ILD labels are used
    as radial labels in polar summary plots.

    Notes
    -----
    The class is used as a constants namespace. Plot functions access values as
    class attributes, for example Labels.frequency or Labels.label_box;
    instantiation is not required.

    Attributes
    ----------
    frequency : str
        Default frequency-axis label used by magnitude spectra, heatmaps, and
        comparison plots.
    magnitude_db, magnitude_linear : str
        Colorbar or y-axis labels for decibel and linear magnitude displays.
    ild, itd : str
        Axis or colorbar labels for interaural level difference and interaural
        time difference values.
    time, samples : str
        X-axis labels for impulse-response plots in milliseconds or discrete sample
        indices.
    impulse_response : str
        Default y-axis label for time-domain HRIR amplitude plots.
    itd_time, ild_db : str
        Radial labels for absolute ITD and unsigned ILD polar plots.
    azimuth, elevation, lateral, polar : str
        Direction-coordinate labels used by spatial axes and plane plots.
    frequency_tick_labels_log, frequency_tick_labels_linear : tuple[str, ...]
        Default tick-label text for logarithmic and linear frequency axes. The
        corresponding tick positions are defined by the frequency-axis classes.
    three_d_x_label, three_d_y_label, three_d_z_label : str
        Cartesian axis labels used by 3D source-grid visualizations.
    compare_itd_difference_time, compare_itd_difference_samples : str
        Colorbar labels for ITD-difference comparison plots in microseconds
        or samples.
    compare_ild_difference_db : str
        Colorbar label for ILD-difference comparison plots in decibels.
    compare_lsd_db : str
        Label for log-spectral-distance comparison values.
    label_box : dict[str, object]
        Matplotlib text bounding-box style used by orientation annotations in
        source-grid plots.

    """

    frequency = "Frequency(kHz)"
    magnitude_db = "Magnitude (dB)"
    magnitude_linear = "Magnitude"
    ild = "ILD (dB)"
    itd = "ITD (µs)"
    time = "Time (ms)"
    samples = "Samples"
    impulse_response = "Amplitude"
    energy_db = "Energy (dB)"
    itd_time = "Absolute ITD (µs)"
    ild_db = "Absolute ILD (dB)"
    azimuth = "Azimuth (°)"
    elevation = "Elevation (°)"
    lateral = "Lateral (°)"
    polar = "Polar (°)"
    frequency_tick_labels_log = (
        "0.25",
        "0.5",
        "1",
        "2",
        "4",
        "8",
        "16",
        "20",
    )
    frequency_tick_labels_linear = (
        "2",
        "4",
        "6",
        "8",
        "10",
        "12",
        "14",
        "16",
        "18",
        "20",
    )
    three_d_x_label = "X (m)"
    three_d_y_label = "Y (m)"
    three_d_z_label = "Z (m)"
    compare_itd_difference_time = "ITD Difference (µs)"
    compare_itd_difference_samples = "ITD Difference (samples)"
    compare_ild_difference_db = "ILD Difference (dB)"
    compare_lsd_db = "LSD (dB)"
    label_box = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.88,
    }
