from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure as MatplotlibFigure
import numpy as np

from ..hrtf.coordinates import get_position_alias


@dataclass(frozen=True)
class Titles:
    """Central title templates and title helpers for plot methods."""

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

    @staticmethod
    def create_position_title(
        selected_positions: np.ndarray,
    ) -> str:
        """Create a subplot title for a spherical source position.

        Parameters
        ----------
        selected_positions : np.ndarray
            Position values as ``[azimuth, elevation]`` in degrees.

        Returns
        -------
        str
            Formatted position title, optionally using a known alias.
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
        title_values = {
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

        Parameters
        ----------
        plane : str
            Plane name. Supported values are ``"horizontal"`` and ``"median"``.
        elevation_angle : float, default=0.0
            Horizontal-plane elevation in degrees.

        Returns
        -------
        str
            Formatted plane title.
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
        """Create a figure title for elevation-spectrum plots."""
        return Titles.elevation_spectrum.format(angle=float(real_azimuth))

    @staticmethod
    def create_subplots_titles(
        ax: Axes,
        title: str,
    ) -> None:
        """Apply a subplot title on a single axis."""
        ax.set_title(title)

    @staticmethod
    def create_figure_title(
        fig: MatplotlibFigure,
        axes: np.ndarray,
        figure_title_y: float,
        title: str,
    ) -> None:
        """Apply a centered figure title based on visible subplot bounds.

        Parameters
        ----------
        fig : MatplotlibFigure
            Target Matplotlib figure.
        axes : np.ndarray
            Array of subplot axes used to compute visible bounds.
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
