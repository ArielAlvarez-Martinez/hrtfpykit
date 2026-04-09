from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    frequency: str = "Frequency(kHz)"
    magnitude_db: str = "Magnitude (dB)"
    magnitude_linear: str = "Magnitude"
    ild: str = "ILD (dB)"
    itd: str = "ITD (s)"
    time: str = "Time (s)"
    samples: str = "Samples"
    impulse_response: str = "Amplitude"
    itd_seconds: str = "Absolute ITD (s)"
    ild_db: str = "Absolute ILD (dB)"
    azimuth: str = "Azimuth (degrees)"
    elevation: str = "Elevation (degrees)"
    lateral: str = "Lateral (degrees)"
    polar: str = "Polar (degrees)"
