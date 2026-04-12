from __future__ import annotations

"""Default plotting configuration values used across plot layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Margins:
    """Default subplot spacing and figure-margin values."""

    top: float = 0.9
    bottom: float = 0.1
    left: float = 0.1
    right: float = 0.9
    wspace: float = 0.35
    hspace: float = 0.35


@dataclass(frozen=True)
class FigureSize:
    """Default figure width and height values."""

    width: float = 8
    height: float = 6


@dataclass(frozen=True)
class RC:
    """Default font-size values for plot text elements."""

    legend_title: float = 10
    legend: float = 9
    ticks: float = 7
    axis_labels: float = 9
    default: float = 10
    axis_title: float = 10
    fig_title: float = 12
