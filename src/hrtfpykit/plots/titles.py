from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure as MatplotlibFigure
import numpy as np

from ..utils.coordinates import get_position_alias


@dataclass(frozen=True)
class Titles:
    """Title templates and helper methods used by hrtfpykit plots.

    :class:`~hrtfpykit.plots.titles.Titles` centralizes the user-facing text used
    by the plotting layer. The class stores format templates for source
    positions, planes, ear labels, and comparison figures, and exposes small
    static helpers that apply those templates consistently across HRTF, SH, and
    comparison plots.

    The helpers do not own Matplotlib state. They either return formatted strings or
    apply titles to caller-provided Matplotlib figures and axes. This keeps title
    behavior consistent while allowing each plot method to control layout, axes,
    legends, and rendering.

    Attributes
    ----------
    spherical_alias, spherical_position : str
        Templates for spherical source-position titles with and without a known
        cardinal alias.
    cartesian_alias, cartesian_position : str
        Templates for cartesian source-position titles.
    lateral_polar_alias, lateral_polar_position : str
        Templates for lateral-polar source-position titles.
    horizontal_plane, horizontal_plane_elevation, median_plane : str
        Templates for plane-level figure titles.
    elevation_spectrum : str
        Template for elevation-spectrum figure titles.
    left_ear, right_ear : str
        Standard ear labels used by comparison plots.
    compare_itd_difference, compare_ild_difference, compare_lsd, compare_lsd_plane : str
        Figure-title labels used by comparison metric plots.
    """

    spherical_alias = "{name} : [Azimuth= {az}°, Elevation= {el}°]"
    spherical_position = "Position : [Azimuth= {az}°, Elevation= {el}°]"
    cartesian_alias = "{name} : [x= {x}, y= {y}, z= {z}]"
    cartesian_position = "Position : [x= {x}, y= {y}, z= {z}]"
    lateral_polar_alias = "{name} : [Lateral= {lateral}°, Polar= {polar}°]"
    lateral_polar_position = "Position : [Lateral= {lateral}°, Polar= {polar}°]"
    horizontal_plane = "Horizontal Plane"
    horizontal_plane_elevation = "Horizontal Plane : [Elevation= {angle}°]"
    median_plane = "Median Plane"
    elevation_spectrum = "Elevation Spectrum : [Azimuth= {angle}°]"
    left_ear = "Left Ear"
    right_ear = "Right Ear"
    compare_itd_difference = "ITD Difference"
    compare_ild_difference = "ILD Difference"
    compare_lsd = "LSD"
    compare_lsd_plane = "LSD Plane"

    @staticmethod
    def create_position_title(
        selected_positions: np.ndarray,
    ) -> str:
        """Create a subplot title for a spherical source position.

        The helper formats the first two values in selected_positions as
        azimuth and elevation in degrees. If the position matches one of the
        built-in cardinal aliases from
        :func:`~hrtfpykit.hrtf.coordinates.get_position_alias`, the alias is
        capitalized and used as the title prefix. Otherwise the generic
        ``Position`` prefix is used.

        Parameters
        ----------
        selected_positions : np.ndarray
            Spherical position values in degrees. The first two values are interpreted
            as [azimuth, elevation]. A third radius value may be present and is
            ignored by the title formatting.

        Returns
        -------
        str
            Formatted subplot title, for example
            ``Front : [Azimuth= 0.0°, Elevation= 0.0°]`` or
            ``Position : [Azimuth= 30.0°, Elevation= 10.0°]``.

        Raises
        ------
        ValueError
            If the position cannot be interpreted by the alias resolver.

        """
        position_alias = get_position_alias(
            selected_positions,
            coordinate_system="spherical",
        )
        title_name = None if position_alias is None else position_alias.capitalize()
        title_template = (
            Titles.spherical_position
            if title_name is None
            else Titles.spherical_alias
        )
        title_values: dict[str, object] = {
            "az": float(selected_positions[0]),
            "el": float(selected_positions[1]),
        }
        if title_name is not None:
            title_values["name"] = title_name
        return title_template.format(**title_values)

    @staticmethod
    def create_plane_title(
        plane: str,
        elevation_angle: float = 0.0,
    ) -> str:
        """Create a figure title for horizontal or median plane plots.

        Horizontal-plane titles include the elevation angle only when the angle is
        not numerically close to zero. Median-plane titles do not use
        elevation_angle because the median plane is selected by azimuth in the
        HRTF plane utilities.

        Parameters
        ----------
        plane : str
            Plane name. Supported values are ``horizontal`` and ``median``.
            Matching is case-insensitive after surrounding whitespace is removed.
        elevation_angle : float, default=0.0
            Horizontal-plane elevation in degrees. Used only when plane is
            ``horizontal``.

        Returns
        -------
        str
            Formatted figure title for the requested plane.

        Raises
        ------
        ValueError
            If plane is not ``horizontal`` or ``median``.

        """
        plane_key = str(plane).strip().lower()
        if plane_key == "horizontal":
            if np.isclose(float(elevation_angle), 0.0, atol=1e-8, rtol=0.0):
                return Titles.horizontal_plane
            return Titles.horizontal_plane_elevation.format(
                angle=float(elevation_angle)
            )
        if plane_key == "median":
            return Titles.median_plane
        raise ValueError("plane accepts horizontal or median")

    @staticmethod
    def create_elevation_spectrum_title(
        real_azimuth: float,
    ) -> str:
        """Create a figure title for elevation-spectrum plots.

        Parameters
        ----------
        real_azimuth : float
            Azimuth angle in degrees represented by the selected elevation spectrum.

        Returns
        -------
        str
            Formatted elevation-spectrum title.
        """
        return Titles.elevation_spectrum.format(angle=float(real_azimuth))

    @staticmethod
    def create_subplots_titles(
        ax: Axes,
        title: str,
    ) -> None:
        """Apply a title to one Matplotlib subplot axis.

        This is the shared subplot-title entry point used by plot methods before
        rendering figure-level titles. It delegates directly to Axes.set_title so
        the title participates in Matplotlib's normal axis layout behavior.

        Parameters
        ----------
        ax : Axes
            Target subplot axis.
        title : str
            Title text applied to ax.

        Returns
        -------
        None

        """
        ax.set_title(title)

    @staticmethod
    def create_figure_title(
        fig: MatplotlibFigure,
        axes: np.ndarray,
        figure_title_y: float,
        title: str,
    ) -> None:
        """Apply a centered figure title based on visible subplot bounds.

        The title is horizontally centered over the currently visible axes rather
        than always centered over the entire figure. This keeps titles visually
        aligned when a layout hides unused axes. When an axis has the
        hrtfpykit_subplot_title_y_with_figure_title attribute, its subplot-title
        y-position is adjusted before the figure title is added so subplot titles and
        figure titles do not overlap.

        Parameters
        ----------
        fig : MatplotlibFigure
            Target Matplotlib figure.
        axes : np.ndarray
            Array of subplot axes used to compute the visible horizontal bounds.
        figure_title_y : float
            Vertical figure-title position in figure coordinates.
        title : str
            Figure title text.

        Returns
        -------
        None

        """
        visible_axes = tuple(ax for ax in axes if ax.get_visible())
        figure_title_x = 0.5
        if visible_axes:
            axis_positions = tuple(ax.get_position() for ax in visible_axes)
            figure_title_x = (
                min(position.x0 for position in axis_positions)
                + max(position.x1 for position in axis_positions)
            ) / 2.0
            for ax in visible_axes:
                subplot_title_y = getattr(
                    ax,
                    "hrtfpykit_subplot_title_y_with_figure_title",
                    None,
                )
                if subplot_title_y is not None:
                    ax.title.set_y(float(subplot_title_y))
        fig.suptitle(title, x=figure_title_x, y=figure_title_y)
