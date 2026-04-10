from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from .labels import Labels
from ..hrtf.coordinates import spherical_to_cartesian
from ..hrtf.sources import Sources


class ThreeDimensional(ABC):
    @staticmethod
    @abstractmethod
    def configure_axis(
        ax: plt.Axes,
        cartesian_positions: np.ndarray,
    ) -> float:
        raise NotImplementedError


class ThreeDimensional1(ThreeDimensional):
    view_elev: float = 22.0
    view_azim: float = -37.0
    arrow_color: str = "#303030"
    arrow_linewidth: float = 2.8
    arrow_length_ratio: float = 0.32
    arrow_delta_ratio: float = 0.50
    arrow_label_offset_ratio: float = 0.10
    right_label_vertical_offset_ratio: float = 0.18

    @staticmethod
    def configure_axis(
        ax: plt.Axes,
        cartesian_positions: np.ndarray,
    ) -> float:
        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)

        ax.set_xlabel(Labels.three_d_x_label)
        ax.set_ylabel(Labels.three_d_y_label)
        ax.set_zlabel(Labels.three_d_z_label)
        ax.view_init(
            elev=ThreeDimensional1.view_elev,
            azim=ThreeDimensional1.view_azim,
        )

        x_center = (float(np.min(x_values)) + float(np.max(x_values))) / 2.0
        y_center = (float(np.min(y_values)) + float(np.max(y_values))) / 2.0
        z_center = (float(np.min(z_values)) + float(np.max(z_values))) / 2.0
        axis_span = max(
            float(np.max(x_values) - np.min(x_values)),
            float(np.max(y_values) - np.min(y_values)),
            float(np.max(z_values) - np.min(z_values)),
            1.0,
        )
        axis_half_span = axis_span / 2.0
        ax.set_xlim(x_center - axis_half_span, x_center + axis_half_span)
        ax.set_ylim(y_center - axis_half_span, y_center + axis_half_span)
        ax.set_zlim(z_center - axis_half_span, z_center + axis_half_span)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        return axis_half_span

    @staticmethod
    def create_direction_markers(
        ax: plt.Axes,
        sources: Sources,
        axis_half_span: float,
    ) -> None:
        _, front_position = sources.get_position_index(
            np.array([0.0, 0.0], dtype=float),
            coordinate_system="spherical",
            angle_unit="degrees",
        )
        _, right_position = sources.get_position_index(
            np.array([270.0, 0.0], dtype=float),
            coordinate_system="spherical",
            angle_unit="degrees",
        )
        _, up_position = sources.get_position_index(
            np.array([0.0, 90.0], dtype=float),
            coordinate_system="spherical",
            angle_unit="degrees",
        )

        front_tail = spherical_to_cartesian(front_position, angle_unit="degrees")
        right_tail = spherical_to_cartesian(right_position, angle_unit="degrees")
        up_tail = spherical_to_cartesian(up_position, angle_unit="degrees")
        front_direction = front_tail / max(float(np.linalg.norm(front_tail)), 1e-12)
        right_direction = right_tail / max(float(np.linalg.norm(right_tail)), 1e-12)
        up_direction = up_tail / max(float(np.linalg.norm(up_tail)), 1e-12)
        arrow_delta_radius = ThreeDimensional1.arrow_delta_ratio * axis_half_span

        ax.quiver(
            *front_tail,
            *(front_direction * arrow_delta_radius),
            color=ThreeDimensional1.arrow_color,
            linewidth=ThreeDimensional1.arrow_linewidth,
            arrow_length_ratio=ThreeDimensional1.arrow_length_ratio,
        )
        ax.text(
            *(
                front_tail
                + front_direction
                * (
                    arrow_delta_radius
                    + ThreeDimensional1.arrow_label_offset_ratio * axis_half_span
                )
            ),
            "Front",
            color=ThreeDimensional1.arrow_color,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="center",
            bbox=Labels.label_box,
        )

        ax.quiver(
            *right_tail,
            *(right_direction * arrow_delta_radius),
            color=ThreeDimensional1.arrow_color,
            linewidth=ThreeDimensional1.arrow_linewidth,
            arrow_length_ratio=ThreeDimensional1.arrow_length_ratio,
        )
        ax.text(
            *(
                right_tail
                + right_direction
                * (
                    arrow_delta_radius
                    + ThreeDimensional1.arrow_label_offset_ratio * axis_half_span
                )
                + np.array(
                    [0.0, 0.0, ThreeDimensional1.right_label_vertical_offset_ratio * axis_half_span]
                )
            ),
            "Right",
            color=ThreeDimensional1.arrow_color,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="bottom",
            bbox=Labels.label_box,
        )

        ax.quiver(
            *up_tail,
            *(up_direction * arrow_delta_radius),
            color=ThreeDimensional1.arrow_color,
            linewidth=ThreeDimensional1.arrow_linewidth,
            arrow_length_ratio=ThreeDimensional1.arrow_length_ratio,
        )
        ax.text(
            *(
                up_tail
                + up_direction
                * (
                    arrow_delta_radius
                    + ThreeDimensional1.arrow_label_offset_ratio * axis_half_span
                )
            ),
            "Up",
            color=ThreeDimensional1.arrow_color,
            fontweight="bold",
            fontsize=11,
            ha="left",
            va="bottom",
            bbox=Labels.label_box,
        )
