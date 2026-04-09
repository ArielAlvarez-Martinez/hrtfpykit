from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Margins:
    top: float = 0.9
    bottom: float = 0.1
    left: float = 0.1
    right: float = 0.9
    wspace: float = 0.35
    hspace: float = 0.35


@dataclass(frozen=True)
class FigureSize:
    width: float = 8
    height: float = 6


@dataclass(frozen=True)
class RC:
    legend_title: float = 10
    legend: float = 9
    ticks: float = 7
    axis_labels: float = 9
    default: float = 10
    axis_title: float = 10
    fig_title: float = 12


__all__ = ["Margins", "FigureSize", "RC"]
