from __future__ import annotations

"""Layout definitions used to build plot figure grids."""

from dataclasses import dataclass, field

from .default import FigureSize, Margins


@dataclass(frozen=True)
class Layout:
    """Base layout configuration for subplot geometry and spacing."""

    code: int
    rows: int
    cols: int
    positions: tuple[str, ...]
    figsize: FigureSize | tuple[float, float] = field(default_factory=FigureSize)
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.07
    subplot_title_y: float = 0.92


@dataclass(frozen=True)
class Layout_1(Layout):
    """Single-subplot layout with one position named ``main``."""

    code: int = 1
    rows: int = 1
    cols: int = 1
    positions: tuple[str, ...] = ("main",)
    figsize: FigureSize | tuple[float, float] = field(default_factory=FigureSize)
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.90


@dataclass(frozen=True)
class Layout_2Vertical(Layout):
    """Two-row vertical layout with ``top`` and ``bottom`` positions."""

    code: int = 21
    rows: int = 2
    cols: int = 1
    positions: tuple[str, ...] = ("top", "bottom")
    figsize: FigureSize | tuple[float, float] = field(default_factory=FigureSize)
    margins: Margins = field(default_factory=Margins)
    sharex: bool = True
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.90


@dataclass(frozen=True)
class Layout_2Horizontal(Layout):
    """Two-column horizontal layout with ``left`` and ``right`` positions."""

    code: int = 22
    rows: int = 1
    cols: int = 2
    positions: tuple[str, ...] = ("left", "right")
    figsize: FigureSize | tuple[float, float] = field(
        default_factory=lambda: FigureSize(width=12, height=6)
    )
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.98


@dataclass(frozen=True)
class Layout_3(Layout):
    """Four-subplot 2x2 layout used for up to four position views."""

    code: int = 3
    rows: int = 2
    cols: int = 2
    positions: tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")
    figsize: FigureSize | tuple[float, float] = field(
        default_factory=lambda: FigureSize(width=10, height=7)
    )
    margins: Margins = field(default_factory=Margins)
    sharex: bool = False
    sharey: bool = False
    figure_title_offset: float = 0.08
    subplot_title_y: float = 0.98
