from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    frequency = "Frequency(kHz)"
    magnitude_db = "Magnitude (dB)"
    magnitude_linear = "Magnitude"
    ild = "ILD (dB)"
    itd = "ITD (s)"
    time = "Time (s)"
    samples = "Samples"
    impulse_response = "Amplitude"
    itd_seconds = "Absolute ITD (s)"
    ild_db = "Absolute ILD (dB)"
    azimuth = "Azimuth (degrees)"
    elevation = "Elevation (degrees)"
    lateral = "Lateral (degrees)"
    polar = "Polar (degrees)"
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
    label_box = {
        "boxstyle": "round,pad=0.18",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.88,
    }
