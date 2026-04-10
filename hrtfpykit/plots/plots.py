from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from .axis import (
    AmplitudeAxis,
    Axis,
    AzimuthAnglesAxis,
    ElevationAnglesAxis,
    FrequencyLinearAxis,
    FrequencyLogAxis,
    MagnitudeAxis,
    PolarAnglesAxis,
    SampleAxis,
    TimeAxis,
)
from .colorbar import ColorBar
from .default import Margins
from .figure import Figure
from .labels import Labels
from .layouts import Layout_1, Layout_2Horizontal, Layout_2Vertical, Layout_3
from .options import (
    AxisOptions,
    AzimuthAxisOptions,
    FigureOptions,
    FrequencyAxisOptions,
    HeatmapOptions,
    LegendOptions,
    PlotOptions,
)
from .three_dimensional import ThreeDimensional1
from .titles import Titles
from ..hrtf.coordinates import (
    get_named_positions,
    get_position_queries,
    get_source_positions,
    spherical_to_lateral_polar,
)
from ..hrtf.dsp import magnitude_to_db, tf_from_ir
from ..hrtf.metrics import calculate_ild, calculate_itd
from ..hrtf.planes import (
    get_frontal_plane,
    get_horizontal_plane,
    get_median_plane,
)


if TYPE_CHECKING:
    from ..hrtf.hrtf import HRTF
class Polar:
    theta_tick_step: float = 30.0

    @staticmethod
    def create_horizontal_plane_curve(
        hrtf: "HRTF",
        values: np.ndarray,
        elevation: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        indices, real_elevation = get_horizontal_plane(
            hrtf=hrtf,
            elevation=elevation,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Horizontal plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=hrtf.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        azimuth_values = np.mod(np.asarray(spherical_positions[:, 0], dtype=float), 360.0)
        plane_values = np.asarray(values, dtype=float)[indices]
        if plane_values.ndim != 1:
            plane_values = np.asarray(plane_values, dtype=float).reshape(-1)

        sort_indices = np.argsort(azimuth_values)
        sorted_azimuth_values = azimuth_values[sort_indices]
        sorted_plane_values = plane_values[sort_indices]
        if sorted_azimuth_values.size > 1:
            theta_values = np.deg2rad(
                np.concatenate(
                    (
                        sorted_azimuth_values,
                        np.array([sorted_azimuth_values[0] + 360.0], dtype=float),
                    )
                )
            )
            radial_values = np.concatenate(
                (
                    sorted_plane_values,
                    np.array([sorted_plane_values[0]], dtype=float),
                )
            )
        else:
            theta_values = np.deg2rad(sorted_azimuth_values)
            radial_values = sorted_plane_values
        return theta_values, radial_values, sorted_plane_values, float(real_elevation)


class HRTFPlots:
    #  Inheritance. All methods will accept a instance of HRTF , then HRTF will inherit from HRTFPlots
    def plot_magnitude(
        self: "HRTF",
        positions: str | list | tuple | np.ndarray = ("front", "back", "left", "right"),
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = 1.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot HRTF magnitude responses for up to four source positions.

        This method draws one to four magnitude-response panels selected from
        the current source grid. Positions are always resolved in spherical
        coordinates and may be provided numerically or through the supported
        aliases such as ``"front"`` and ``"left"``.

        Parameters
        ----------
        positions : str | list | tuple | np.ndarray, default=("front", "back", "left", "right")
            One position or a collection of positions. Named aliases such as
            ``"front"``, ``"back"``, ``"left"``, and ``"right"`` are accepted.
            Numeric queries must use spherical coordinates in degrees as
            ``[azimuth, elevation]``, for example ``[0.0, 0.0]`` for the front
            direction. Up to four positions can be shown in one figure.
        x_axis : {"linear", "log"}, default="linear"
            Frequency scale used on the x axis.
        unit : {"db", "linear"}, default="db"
            Magnitude representation used on the y axis.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, left and right
            responses are drawn together in each subplot.
        reference : float | {"max"}, default=1.0
            Reference used when ``unit="db"``. ``"max"`` normalizes the plotted
            magnitude to the maximum selected value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, frequency-axis, and per-panel overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress generated default subplot titles. Explicit
            titles provided through axis or panel options are still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Compare magnitude responses across several source positions.
        - Inspect left, right, or binaural magnitude structure at one location.
        - Generate figures for later display with ``show=False``.

        Examples
        --------
        Load a measured HRTF and compare two practical listening directions:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_magnitude(
        ...     positions=["front", "left"],
        ...     ear="both",
        ...     show=False,
        ... )

        Plot two explicit spherical directions using ``[azimuth, elevation]`` queries:

        >>> hrtf.plot_magnitude(
        ...     positions=[[0.0, 0.0], [90.0, 0.0]],
        ...     ear="left",
        ...     show=False,
        ... )

        Window one direction before plotting its magnitude on a log-frequency axis:

        >>> front = hrtf.select(positions="front")
        >>> windowed = front.transform.apply_window("hann")
        >>> windowed.plot_magnitude(
        ...     positions="front",
        ...     unit="linear",
        ...     x_axis="log",
        ...     show=False,
        ... )
        """
        if unit not in {"db", "linear"}:
            raise AttributeError(
                "unit accepts : db or linear"
            )
        if x_axis not in {"log", "linear"}:
            raise AttributeError(
                "x_axis accepts log or linear"
            )
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        position_queries = get_position_queries(positions)
        position_count = len(position_queries)
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_magnitude accepts up to 4 positions")

        if position_count == 1:
            resolved_layout = Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        elif position_count == 2:
            resolved_layout = Layout_2Vertical(
                figsize=figure_options.figsize or Layout_2Vertical().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_3(
                figsize=figure_options.figsize or Layout_3().figsize,
                margins=resolved_margins,
            )
        layout = Figure(resolved_layout)
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        selected_position_info = [
            self.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            for selected_position_query in position_queries
        ]
        tf_magnitude = self.TF.magnitude
        if unit == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                selected_indices = [selected_index for selected_index, _ in selected_position_info]
                reference_values = np.asarray(tf_magnitude[selected_indices], dtype=float)
                if ear != "both" and reference_values.ndim >= 3:
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[1] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in TF data")
                    reference_values = reference_values[:, ear_index, :]
                plot_reference = float(np.max(reference_values))
                tf_values = magnitude_to_db(tf_magnitude, reference=plot_reference)
            else:
                tf_values = magnitude_to_db(tf_magnitude, reference=reference)
        else:
            tf_values = tf_magnitude
        magnitude_legend_location = "upper right" if x_axis == "linear" else "upper left"

        for index, (_, selected_positions) in enumerate(selected_position_info):
            ax = layout.get_axis(index)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(index))
            if not titles and resolved_axis_options.title is None:
                resolved_axis_options = resolved_axis_options.merge(AxisOptions(title=""))
            frequency_axis = (
                FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
            )
            resolved_frequency_axis = frequency_axis.build(
                frequency_bins=frequency_bins_hz,
                freq_min=freq_min,
                freq_max=freq_max,
                options=resolved_axis_options.frequency_axis,
            )
            frequency_mask = (
                (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
                & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
            )
            if not np.any(frequency_mask):
                raise ValueError("Selected frequency range produced no TF bins")
            frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
            frequency_label = (
                Labels.frequency
                if resolved_axis_options.xlabel is None
                else resolved_axis_options.xlabel
            )
            idxs = int(selected_position_info[index][0])
            selected_positions = np.asarray(selected_positions, dtype=float)
            y_values = np.asarray(tf_values[idxs][..., frequency_mask], dtype=float)

            if ear == "both":
                if y_values.ndim < 2 or y_values.shape[0] < 2:
                    raise ValueError("Both ears requested but TF data does not contain two ear channels")
                ax.plot(frequency_khz, y_values[0, :], color='blue')
                ax.plot(frequency_khz, y_values[1, :], color='red')
            else:
                if y_values.ndim == 1:
                    selected_y_values = y_values.reshape(-1)
                else:
                    ear_index = 0 if ear == "left" else 1
                    if y_values.shape[0] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in TF data")
                    selected_y_values = np.asarray(y_values[ear_index], dtype=float).reshape(-1)
                ax.plot(frequency_khz, selected_y_values, color='blue')

            frequency_axis.apply(
                ax=ax,
                axis="x",
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            MagnitudeAxis.apply(
                ax=ax,
                axis="y",
                unit=unit,
                options=resolved_axis_options,
            )
            layout.apply_panel(
                ax=ax,
                selected_positions=selected_positions,
                ear=ear,
                options=resolved_axis_options,
                legend_location=magnitude_legend_location,
            )

        if position_count < layout.axes.size:
            layout.hide_unused_axes(position_count)

        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_amplitude(
        self: "HRTF",
        positions: str | list | tuple | np.ndarray = ("front", "back", "left", "right"),
        ear: str = "both",
        x_axis: str = "time",
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot HRIR amplitude responses for up to four source positions.

        This method displays time-domain HRIR waveforms for one to four source
        positions from the current grid. Positions are resolved in spherical
        coordinates and may be provided numerically or through the supported
        aliases such as ``"front"`` and ``"right"``.

        Parameters
        ----------
        positions : str | list | tuple | np.ndarray, default=("front", "back", "left", "right")
            One position or a collection of positions. Named aliases such as
            ``"front"``, ``"back"``, ``"left"``, and ``"right"`` are accepted.
            Numeric queries must use spherical coordinates in degrees as
            ``[azimuth, elevation]``, for example ``[0.0, 0.0]`` for the front
            direction. Up to four positions can be shown in one figure.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, left and right
            ear waveforms are drawn together in each subplot.
        x_axis : {"time", "samples"}, default="time"
            Horizontal axis used for the waveform plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, and per-panel overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress generated default subplot titles. Explicit
            titles provided through axis or panel options are still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect HRIR waveform shape for one or several directions.
        - Compare left and right ear impulse responses at the same position.
        - Generate waveform figures for later display with ``show=False``.

        Examples
        --------
        Load an HRIR set and compare the front and right directions in samples:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_amplitude(
        ...     positions=["front", "right"],
        ...     x_axis="samples",
        ...     show=False,
        ... )

        Plot two explicit spherical directions in the time domain:

        >>> hrtf.plot_amplitude(
        ...     positions=[[0.0, 0.0], [270.0, 0.0]],
        ...     ear="both",
        ...     show=False,
        ... )

        Remove ITD from the front direction and inspect the left-ear waveform:

        >>> aligned_front = hrtf.select(positions="front").transform.delete_itd()
        >>> aligned_front.plot_amplitude(
        ...     positions="front",
        ...     ear="left",
        ...     show=False,
        ... )
        """
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        if x_axis not in {"time", "samples"}:
            raise AttributeError(
                "x_axis accepts : time or samples"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if x_axis == "time" and self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required when x_axis='time'")

        position_queries = get_position_queries(positions)
        position_count = len(position_queries)
        if position_count == 0:
            raise ValueError("At least one position is required")
        if position_count > 4:
            raise ValueError("plot_amplitude accepts up to 4 positions")

        if position_count == 1:
            resolved_layout = Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        elif position_count == 2:
            resolved_layout = Layout_2Vertical(
                figsize=figure_options.figsize or Layout_2Vertical().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_3(
                figsize=figure_options.figsize or Layout_3().figsize,
                margins=resolved_margins,
            )
        layout = Figure(resolved_layout)
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        ir_values = np.asarray(self.IR.values, dtype=float)
        if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
            raise ValueError("IR values must contain at least one sample")
        sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
        if x_axis == "time":
            x_values = sample_indexes / float(self.IR.sample_rate)
        else:
            x_values = sample_indexes

        for index, selected_position_query in enumerate(position_queries):
            ax = layout.get_axis(index)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(index))
            if not titles and resolved_axis_options.title is None:
                resolved_axis_options = resolved_axis_options.merge(AxisOptions(title=""))
            idxs, selected_positions = self.Sources.get_position_index(
                selected_position_query,
                coordinate_system="spherical",
            )
            selected_positions = np.asarray(selected_positions, dtype=float)
            y_values = np.asarray(ir_values[idxs], dtype=float)

            if ear == "both":
                if y_values.ndim < 2 or y_values.shape[0] < 2:
                    raise ValueError("Both ears requested but IR data does not contain two ear channels")
                ax.plot(x_values, y_values[0, :], color="blue")
                ax.plot(x_values, y_values[1, :], color="red")
            else:
                if y_values.ndim == 1:
                    selected_y_values = y_values.reshape(-1)
                else:
                    ear_index = 0 if ear == "left" else 1
                    if y_values.shape[0] <= ear_index:
                        raise ValueError(f"Requested ear '{ear}' is not available in IR data")
                    selected_y_values = np.asarray(y_values[ear_index], dtype=float).reshape(-1)
                ax.plot(x_values, selected_y_values, color="blue")

            if x_axis == "time":
                TimeAxis.apply(
                    ax=ax,
                    axis="x",
                    options=resolved_axis_options,
                )
            else:
                SampleAxis.apply(
                    ax=ax,
                    axis="x",
                    options=resolved_axis_options,
                )
            AmplitudeAxis.apply(
                ax=ax,
                axis="y",
                options=resolved_axis_options,
            )
            layout.apply_panel(
                ax=ax,
                selected_positions=selected_positions,
                position_coordinate_system="spherical",
                ear=ear,
                options=resolved_axis_options,
            )

        if position_count < layout.axes.size:
            layout.hide_unused_axes(position_count)

        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_amplitude_and_magnitude(
        self: "HRTF",
        position: str | list | np.ndarray = "front",
        ear: str = "both",
        amplitude_x_axis: str = "time",
        magnitude_x_axis: str = "linear",
        magnitude: str = "db",
        reference: float | str = 1.0,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot amplitude and magnitude views for a single source position.

        This method creates a two-panel summary for one source direction. The
        top panel shows the HRIR amplitude response and the bottom panel shows
        the corresponding HRTF magnitude response for the same position. The
        amplitude panel uses ``amplitude_x_axis`` and the magnitude panel uses
        ``magnitude_x_axis``.

        Parameters
        ----------
        position : str | list | np.ndarray, default="front"
            Position query to plot. Exactly one position is accepted. Named
            aliases such as ``"front"``, ``"back"``, ``"left"``, and ``"right"``
            are accepted. Numeric queries must use spherical coordinates in
            degrees as ``[azimuth, elevation]``, for example ``[0.0, 0.0]``.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display in both subplots.
        amplitude_x_axis : {"time", "samples"}, default="time"
            Horizontal axis used for the amplitude subplot.
        magnitude_x_axis : {"linear", "log"}, default="linear"
            Frequency-axis scale used on the magnitude subplot.
        magnitude : {"db", "linear"}, default="db"
            Magnitude representation used on the bottom subplot.
        reference : float | {"max"}, default=1.0
            Reference used when ``magnitude="db"`` for the magnitude subplot.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, frequency-axis, and panel overrides.
            Frequency-range control for the magnitude subplot should be passed
            through ``options.axis.frequency_axis`` or the bottom-panel axis
            override.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title. Explicit
            figure titles provided through ``options.figure.title`` are still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect time-domain and frequency-domain behavior for the same direction.
        - Compare left and right ear waveform and magnitude structure together.
        - Create a compact two-panel summary for one position.

        Examples
        --------
        Load one direction and inspect its HRIR and HRTF together:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa").select(positions="front")
        >>> hrtf.plot_amplitude_and_magnitude(show=False)

        Plot a windowed version of the same direction with both ears and a log-frequency axis:

        >>> windowed = hrtf.transform.apply_window("hann")
        >>> windowed.plot_amplitude_and_magnitude(
        ...     ear="both",
        ...     amplitude_x_axis="samples",
        ...     magnitude_x_axis="log",
        ...     magnitude="linear",
        ...     show=False,
        ... )

        Plot one explicit spherical direction using an ``[azimuth, elevation]`` query:

        >>> hrtf.plot_amplitude_and_magnitude(
        ...     position=[90.0, 0.0],
        ...     ear="left",
        ...     show=False,
        ... )
        """
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        if amplitude_x_axis not in {"time", "samples"}:
            raise AttributeError(
                "amplitude_x_axis accepts : time or samples"
            )
        if magnitude_x_axis not in {"log", "linear"}:
            raise AttributeError(
                "magnitude_x_axis accepts log or linear"
            )
        if magnitude not in {"db", "linear"}:
            raise AttributeError(
                "magnitude accepts : db or linear"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")
        if amplitude_x_axis == "time" and self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required when amplitude_x_axis='time'")

        position_queries = get_position_queries(position)
        if len(position_queries) != 1:
            raise ValueError(
                "plot_amplitude_and_magnitude accepts exactly one position"
            )
        selected_position_query = position_queries[0]

        layout = Figure(
            Layout_2Vertical(
                figsize=(8, 8) if figure_options.figsize is None else figure_options.figsize,
                margins=resolved_margins,
                sharex=False,
            )
        )
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        top_axis_options = axis_options.merge(panel_axis_options.get(0))
        bottom_axis_options = axis_options.merge(panel_axis_options.get(1))
        top_axis_panel_options = top_axis_options.merge(AxisOptions(title=""))
        bottom_axis_panel_options = bottom_axis_options.merge(AxisOptions(title=""))

        idxs, selected_positions = self.Sources.get_position_index(
            selected_position_query,
            coordinate_system="spherical",
        )
        selected_positions = np.asarray(selected_positions, dtype=float)

        ir_values = np.asarray(self.IR.values, dtype=float)
        if ir_values.ndim < 2 or ir_values.shape[-1] == 0:
            raise ValueError("IR values must contain at least one sample")
        sample_indexes = np.arange(ir_values.shape[-1], dtype=float)
        x_values = (
            sample_indexes / float(self.IR.sample_rate)
            if amplitude_x_axis == "time"
            else sample_indexes
        )
        ir_y_values = np.asarray(ir_values[idxs], dtype=float)

        ir_ax = layout.get_axis("top")
        if ear == "both":
            if ir_y_values.ndim < 2 or ir_y_values.shape[0] < 2:
                raise ValueError(
                    "Both ears requested but IR data does not contain two ear channels"
                )
            ir_ax.plot(x_values, ir_y_values[0, :], color="blue")
            ir_ax.plot(x_values, ir_y_values[1, :], color="red")
        else:
            if ir_y_values.ndim == 1:
                selected_ir_y_values = ir_y_values.reshape(-1)
            else:
                ear_index = 0 if ear == "left" else 1
                if ir_y_values.shape[0] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in IR data"
                    )
                selected_ir_y_values = np.asarray(
                    ir_y_values[ear_index],
                    dtype=float,
                ).reshape(-1)
            ir_ax.plot(x_values, selected_ir_y_values, color="blue")

        if amplitude_x_axis == "time":
            TimeAxis.apply(
                ax=ir_ax,
                axis="x",
                options=top_axis_options,
            )
        else:
            SampleAxis.apply(
                ax=ir_ax,
                axis="x",
                options=top_axis_options,
            )
        AmplitudeAxis.apply(
            ax=ir_ax,
            axis="y",
            options=top_axis_panel_options,
        )
        layout.apply_panel(
            ax=ir_ax,
            selected_positions=selected_positions,
            ear=ear,
            options=top_axis_panel_options,
        )

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        magnitude_frequency_axis = (
            FrequencyLogAxis
            if magnitude_x_axis == "log"
            else FrequencyLinearAxis
        )
        resolved_frequency_axis = magnitude_frequency_axis.build(
            frequency_bins=frequency_bins_hz,
            options=bottom_axis_options.frequency_axis,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0
        tf_magnitude = self.TF.magnitude
        if magnitude == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                reference_values = np.asarray(tf_magnitude[idxs], dtype=float)
                if ear != "both" and reference_values.ndim >= 2:
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[0] <= ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in TF data"
                        )
                    reference_values = reference_values[ear_index]
                plot_reference = float(np.max(reference_values))
                tf_values = magnitude_to_db(tf_magnitude, reference=plot_reference)
            else:
                tf_values = magnitude_to_db(tf_magnitude, reference=reference)
        else:
            tf_values = tf_magnitude
        magnitude_y_values = np.asarray(tf_values[idxs][..., frequency_mask], dtype=float)

        magnitude_ax = layout.get_axis("bottom")
        if ear == "both":
            if magnitude_y_values.ndim < 2 or magnitude_y_values.shape[0] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            magnitude_ax.plot(frequency_khz, magnitude_y_values[0, :], color="blue")
            magnitude_ax.plot(frequency_khz, magnitude_y_values[1, :], color="red")
        else:
            if magnitude_y_values.ndim == 1:
                selected_magnitude_y_values = magnitude_y_values.reshape(-1)
            else:
                ear_index = 0 if ear == "left" else 1
                if magnitude_y_values.shape[0] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in TF data"
                    )
                selected_magnitude_y_values = np.asarray(
                    magnitude_y_values[ear_index],
                    dtype=float,
                ).reshape(-1)
            magnitude_ax.plot(frequency_khz, selected_magnitude_y_values, color="blue")

        magnitude_legend_location = (
            "upper right" if magnitude_x_axis == "linear" else "upper left"
        )
        frequency_label = (
            Labels.frequency
            if bottom_axis_options.xlabel is None
            else bottom_axis_options.xlabel
        )
        magnitude_frequency_axis.apply(
            ax=magnitude_ax,
            axis="x",
            label=frequency_label,
            options=resolved_frequency_axis,
        )
        MagnitudeAxis.apply(
            ax=magnitude_ax,
            axis="y",
            unit=magnitude,
            options=bottom_axis_panel_options,
        )
        layout.apply_panel(
            ax=magnitude_ax,
            selected_positions=selected_positions,
            ear=ear,
            options=bottom_axis_panel_options,
            legend_location=magnitude_legend_location,
        )

        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_position_title(selected_positions=selected_positions),
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_spectrum_plane(
        self: "HRTF",
        plane: str = "horizontal",
        elevation_angle: float = 0.0,
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = "max",
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot a frequency-angle spectrum heatmap for an HRTF plane.

        This method plots a heatmap where frequency is shown on the horizontal
        axis and the plane angle is shown on the vertical axis. The horizontal
        plane may be selected at another elevation through ``elevation_angle``.
        The median plane remains the canonical sagittal path.

        Parameters
        ----------
        plane : {"horizontal", "median"}, default="horizontal"
            Plane to visualize. ``"horizontal"`` uses a horizontal plane
            selected by elevation. ``"median"`` uses the canonical median
            plane defined by the front-back sagittal path.
        elevation_angle : float, default=0.0
            Target elevation used when ``plane="horizontal"``. The nearest
            available horizontal plane in the grid is selected. This parameter
            is not used for the median plane.
        x_axis : {"linear", "log"}, default="linear"
            Frequency scale used on the x axis.
        unit : {"db", "linear"}, default="db"
            Magnitude representation used for the heatmap values.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, a separate panel
            is created for each ear.
        reference : float | {"max"}, default="max"
            Reference used when ``unit="db"``. ``"max"`` normalizes the plotted
            plane to its maximum value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, heatmap, and panel overrides. For the
            horizontal plane, ``options.axis.azimuth_axis`` can be used to choose
            the azimuth plotting convention, for example ``"-180-180"`` or
            ``"0-360"``.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress generated default panel and figure titles.
            Explicit titles provided through figure, axis, or panel options are
            still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect a horizontal-plane spectrum over azimuth.
        - Inspect the median-plane spectrum over polar angle.
        - Compare left and right ear spectral structure in the same plane.
        - Create plane-based HRTF heatmaps without showing them immediately.

        Examples
        --------
        Load a measured HRTF and inspect the horizontal-plane spectrum for the left ear:

        >>> from hrtfpykit import HRTF
        >>> from hrtfpykit.plots.plots import AxisOptions, AzimuthAxisOptions, PlotOptions
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_spectrum_plane(
        ...     plane="horizontal",
        ...     ear="left",
        ...     show=False,
        ... )

        Replot the same plane with a signed azimuth convention for interpretation:

        >>> hrtf.plot_spectrum_plane(
        ...     plane="horizontal",
        ...     options=PlotOptions(
        ...         axis=AxisOptions(
        ...             azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ...         )
        ...     ),
        ...     show=False,
        ... )
        """
        if plane not in ("horizontal", "median"):
            raise AttributeError(
                "plot_spectrum_plane plane accepts horizontal or median"
            )
        if isinstance(elevation_angle, bool):
            raise AttributeError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise AttributeError("elevation_angle must be a finite value")
        if unit not in {"db", "linear"}:
            raise AttributeError(
                "unit accepts : db or linear"
            )
        if x_axis not in {"log", "linear"}:
            raise AttributeError(
                "x_axis accepts log or linear"
            )
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = AxisOptions(
            azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ).merge(plot_options.axis)
        heatmap_options = HeatmapOptions(cmap="jet").merge(plot_options.heatmap)
        heatmap_frequency_axis_options = (
            FrequencyAxisOptions()
            if axis_options.frequency_axis is None
            else axis_options.frequency_axis
        ).merge(FrequencyAxisOptions(margin_ratio=0.0))

        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        plane_key = str(plane).strip().lower()
        if plane_key != "horizontal" and not np.isclose(
            elevation_angle,
            0.0,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "elevation_angle only applies when plane='horizontal'"
            )
        if ear == "both":
            resolved_layout = Layout_2Horizontal(
                figsize=figure_options.figsize or Layout_2Horizontal().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        layout = Figure(resolved_layout)
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        if plane_key == "horizontal":
            indices, real_plane_elevation = get_horizontal_plane(
                hrtf=self,
                elevation=elevation_angle,
                angle_unit="degrees",
            )
        else:
            indices, _ = get_median_plane(
                hrtf=self,
                azimuth=0.0,
                angle_unit="degrees",
            )
            real_plane_elevation = 0.0
        if indices.size == 0:
            raise ValueError("Selected plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]

        if plane_key == "horizontal":
            plane_axis_values = np.asarray(spherical_positions[:, 0], dtype=float)
        else:
            lateral_polar_positions = spherical_to_lateral_polar(
                spherical_positions,
                angle_unit="degrees",
            )
            plane_axis_values = np.asarray(lateral_polar_positions[:, 1], dtype=float)

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        frequency_axis = (
            FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
        )
        resolved_frequency_axis = frequency_axis.build(
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            options=heatmap_frequency_axis_options,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        tf_magnitude = self.TF.magnitude
        plane_values = np.asarray(tf_magnitude[indices][..., frequency_mask], dtype=float)
        if plane_values.ndim == 2:
            plane_values = plane_values[:, np.newaxis, :]
        if plane_values.ndim != 3:
            raise ValueError("TF values for spectrum must have shape (M, E, F)")
        if unit == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                reference_values = plane_values
                if ear != "both":
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[1] <= ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in TF data"
                        )
                    reference_values = reference_values[:, ear_index, :]
                plot_reference = float(np.max(reference_values))
                plane_values = magnitude_to_db(plane_values, reference=plot_reference)
            else:
                plane_values = magnitude_to_db(plane_values, reference=reference)
        if ear == "both":
            if plane_values.shape[1] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            spectrum_matrices = [plane_values[:, 0, :], plane_values[:, 1, :]]
            panel_positions = ["left", "right"]
            default_panel_titles = ["Left Ear", "Right Ear"]
        else:
            if plane_values.shape[1] == 1:
                ear_index = 0
            else:
                ear_index = 0 if ear == "left" else 1
                if plane_values.shape[1] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in TF data"
                    )
            spectrum_matrices = [plane_values[:, ear_index, :]]
            panel_positions = ["main"]
            default_panel_titles = [f"{ear.capitalize()} Ear"]

        vmin = min(float(np.min(matrix)) for matrix in spectrum_matrices)
        vmax = max(float(np.max(matrix)) for matrix in spectrum_matrices)
        colorbar_label = (
            Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        )
        heatmap_colormap = "jet" if heatmap_options.cmap is None else str(heatmap_options.cmap)
        if heatmap_colormap not in ColorBar.colormaps:
            raise ValueError(
                f"heatmap cmap accepts: {', '.join(ColorBar.colormaps)}"
            )

        for panel_index, (panel_position, spectrum_matrix, default_panel_title) in enumerate(
            zip(panel_positions, spectrum_matrices, default_panel_titles)
        ):
            ax = layout.get_axis(panel_position)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(panel_index))
            panel_plane_axis_values = (
                AzimuthAnglesAxis.transform_values(
                    values=plane_axis_values,
                    options=resolved_axis_options,
                )
                if plane_key == "horizontal"
                else np.asarray(plane_axis_values, dtype=float)
            )
            panel_sort_indices = np.argsort(panel_plane_axis_values)
            sorted_panel_plane_axis_values = panel_plane_axis_values[panel_sort_indices]
            sorted_spectrum_matrix = spectrum_matrix[panel_sort_indices, :]
            mesh = ax.pcolormesh(
                frequency_khz,
                sorted_panel_plane_axis_values,
                sorted_spectrum_matrix,
                shading="auto",
                cmap=heatmap_colormap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.margins(x=0.0, y=0.0)
            frequency_label = (
                Labels.frequency
                if resolved_axis_options.xlabel is None
                else resolved_axis_options.xlabel
            )
            frequency_axis.apply(
                ax=ax,
                axis="x",
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            if plane_key == "horizontal":
                AzimuthAnglesAxis.apply(
                    ax=ax,
                    axis="y",
                    values=sorted_panel_plane_axis_values,
                    options=resolved_axis_options,
                )
            else:
                PolarAnglesAxis.apply(
                    ax=ax,
                    axis="y",
                    values=sorted_panel_plane_axis_values,
                    options=resolved_axis_options,
                )
            resolved_title = (
                default_panel_title
                if resolved_axis_options.title is None
                else resolved_axis_options.title
            )
            if not titles and resolved_axis_options.title is None:
                resolved_title = ""
            Titles.create_subplots_titles(ax=ax, title=resolved_title)
            grid_enabled = (
                False if resolved_axis_options.grid is None else resolved_axis_options.grid
            )
            if grid_enabled:
                ax.grid(True)
            ColorBar.create(
                fig=layout.fig,
                ax=ax,
                mesh=mesh,
                label=colorbar_label,
                options=heatmap_options,
                colormap=heatmap_colormap,
            )
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_plane_title(
                    plane=plane_key,
                    elevation_angle=real_plane_elevation,
                ),
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_elevation_spectrum(
        self: "HRTF",
        azimuth: float | str = 0.0,
        x_axis: str = "linear",
        unit: str = "db",
        ear: str = "both",
        reference: float | str = "max",
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot a fixed-azimuth elevation spectrum heatmap.

        This method selects the nearest azimuth slice in the current source
        grid and displays a frequency-versus-elevation heatmap for that slice.
        Numeric azimuths and the standard position aliases are accepted.

        Parameters
        ----------
        azimuth : float | str, default=0.0
            Azimuth used to select the elevation slice. Named aliases such as
            ``"front"``, ``"back"``, ``"left"``, and ``"right"`` are accepted.
            The nearest available azimuth in the source grid is used.
        x_axis : {"linear", "log"}, default="linear"
            Frequency scale used on the x axis.
        unit : {"db", "linear"}, default="db"
            Magnitude representation used for the heatmap values.
        ear : {"left", "right", "both"}, default="both"
            Ear channel to display. When ``"both"`` is selected, a separate panel
            is created for each ear.
        reference : float | {"max"}, default="max"
            Reference used when ``unit="db"``. ``"max"`` normalizes the plotted
            slice to its maximum value.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, heatmap, and panel overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress generated default panel and figure titles.
            Explicit titles provided through figure, axis, or panel options are
            still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect how magnitude changes with elevation at a fixed azimuth.
        - Compare left and right ear spectral structure along one azimuth slice.
        - Create elevation-spectrum heatmaps without showing them immediately.

        Examples
        --------
        Load a measured HRTF and inspect how the front spectrum changes with elevation:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_elevation_spectrum(
        ...     azimuth="front",
        ...     ear="both",
        ...     show=False,
        ... )

        Inspect the left-side elevation slice for one ear on a log-frequency axis:

        >>> hrtf.plot_elevation_spectrum(
        ...     azimuth="left",
        ...     ear="left",
        ...     x_axis="log",
        ...     show=False,
        ... )
        """
        if unit not in {"db", "linear"}:
            raise AttributeError(
                "unit accepts : db or linear"
            )
        if x_axis not in {"log", "linear"}:
            raise AttributeError(
                "x_axis accepts log or linear"
            )
        if ear not in {"left", "right", "both"}:
            raise AttributeError(
                "ear accepts left, right or both"
            )
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )
        heatmap_options = HeatmapOptions(cmap="jet").merge(plot_options.heatmap)
        heatmap_frequency_axis_options = (
            FrequencyAxisOptions()
            if axis_options.frequency_axis is None
            else axis_options.frequency_axis
        ).merge(FrequencyAxisOptions(margin_ratio=0.0))

        if self.TF.values is None or self.TF.frequency_bins is None:
            raise ValueError("TF data is not available")

        if isinstance(azimuth, str):
            azimuth_key = str(azimuth).strip().lower()
            named_positions = get_named_positions(angle_unit="degrees")
            if azimuth_key not in named_positions:
                raise ValueError("azimuth accepts a finite value or: front, back, left, right")
            resolved_azimuth = float(named_positions[azimuth_key][0])
        else:
            if isinstance(azimuth, bool):
                raise ValueError("azimuth must be a finite value")
            resolved_azimuth = float(azimuth)
            if not np.isfinite(resolved_azimuth):
                raise ValueError("azimuth must be a finite value")

        if ear == "both":
            resolved_layout = Layout_2Horizontal(
                figsize=figure_options.figsize or Layout_2Horizontal().figsize,
                margins=resolved_margins,
            )
        else:
            resolved_layout = Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        layout = Figure(resolved_layout)
        panel_axis_options = layout.get_panel_axis_options(plot_options)

        spherical_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )

        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        elevation_values = np.asarray(spherical_positions[:, 1], dtype=float)
        available_azimuths = np.unique(azimuth_values)
        azimuth_deltas = np.mod(available_azimuths - resolved_azimuth + 180.0, 360.0) - 180.0
        real_azimuth = float(available_azimuths[int(np.argmin(np.abs(azimuth_deltas)))])
        selected = np.isclose(
            np.mod(azimuth_values - real_azimuth + 180.0, 360.0) - 180.0,
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
        indices = np.where(selected)[0]
        if indices.size == 0:
            raise ValueError("Selected elevation spectrum does not contain any source positions")

        slice_elevation_values = elevation_values[indices]
        sort_indices = np.argsort(slice_elevation_values)
        sorted_elevation_values = slice_elevation_values[sort_indices]

        frequency_bins_hz = np.asarray(self.TF.frequency_bins, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        frequency_axis = (
            FrequencyLogAxis if x_axis == "log" else FrequencyLinearAxis
        )
        resolved_frequency_axis = frequency_axis.build(
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            options=heatmap_frequency_axis_options,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        tf_magnitude = self.TF.magnitude
        slice_values = np.asarray(tf_magnitude[indices][..., frequency_mask], dtype=float)
        if slice_values.ndim == 2:
            slice_values = slice_values[:, np.newaxis, :]
        if slice_values.ndim != 3:
            raise ValueError("TF values for vertical slice spectrum must have shape (M, E, F)")
        if unit == "db":
            if isinstance(reference, str) and str(reference).strip().lower() == "max":
                reference_values = slice_values
                if ear != "both":
                    ear_index = 0 if ear == "left" else 1
                    if reference_values.shape[1] <= ear_index:
                        raise ValueError(
                            f"Requested ear '{ear}' is not available in TF data"
                        )
                    reference_values = reference_values[:, ear_index, :]
                plot_reference = float(np.max(reference_values))
                slice_values = magnitude_to_db(slice_values, reference=plot_reference)
            else:
                slice_values = magnitude_to_db(slice_values, reference=reference)
        slice_values = slice_values[sort_indices]

        if ear == "both":
            if slice_values.shape[1] < 2:
                raise ValueError(
                    "Both ears requested but TF data does not contain two ear channels"
                )
            spectrum_matrices = [slice_values[:, 0, :], slice_values[:, 1, :]]
            panel_positions = ["left", "right"]
            default_panel_titles = ["Left Ear", "Right Ear"]
        else:
            if slice_values.shape[1] == 1:
                ear_index = 0
            else:
                ear_index = 0 if ear == "left" else 1
                if slice_values.shape[1] <= ear_index:
                    raise ValueError(
                        f"Requested ear '{ear}' is not available in TF data"
                    )
            spectrum_matrices = [slice_values[:, ear_index, :]]
            panel_positions = ["main"]
            default_panel_titles = [f"{ear.capitalize()} Ear"]

        vmin = min(float(np.min(matrix)) for matrix in spectrum_matrices)
        vmax = max(float(np.max(matrix)) for matrix in spectrum_matrices)
        colorbar_label = (
            Labels.magnitude_db if unit == "db" else Labels.magnitude_linear
        )
        heatmap_colormap = "jet" if heatmap_options.cmap is None else str(heatmap_options.cmap)
        if heatmap_colormap not in ColorBar.colormaps:
            raise ValueError(
                f"heatmap cmap accepts: {', '.join(ColorBar.colormaps)}"
            )

        for panel_index, (panel_position, spectrum_matrix, default_panel_title) in enumerate(
            zip(panel_positions, spectrum_matrices, default_panel_titles)
        ):
            ax = layout.get_axis(panel_position)
            resolved_axis_options = axis_options.merge(panel_axis_options.get(panel_index))
            mesh = ax.pcolormesh(
                frequency_khz,
                sorted_elevation_values,
                spectrum_matrix,
                shading="auto",
                cmap=heatmap_colormap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.margins(x=0.0, y=0.0)
            frequency_label = (
                Labels.frequency
                if resolved_axis_options.xlabel is None
                else resolved_axis_options.xlabel
            )
            frequency_axis.apply(
                ax=ax,
                axis="x",
                label=frequency_label,
                options=resolved_frequency_axis,
            )
            ElevationAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_elevation_values,
                options=resolved_axis_options,
            )
            resolved_title = (
                default_panel_title
                if resolved_axis_options.title is None
                else resolved_axis_options.title
            )
            if not titles and resolved_axis_options.title is None:
                resolved_title = ""
            Titles.create_subplots_titles(ax=ax, title=resolved_title)
            grid_enabled = (
                False if resolved_axis_options.grid is None else resolved_axis_options.grid
            )
            if grid_enabled:
                ax.grid(True)
            ColorBar.create(
                fig=layout.fig,
                ax=ax,
                mesh=mesh,
                label=colorbar_label,
                options=heatmap_options,
                colormap=heatmap_colormap,
            )
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_elevation_spectrum_title(real_azimuth=real_azimuth),
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_itd_curve(
        self: "HRTF",
        elevation_angle: float = 0.0,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot signed ITD over a horizontal plane as azimuth versus time delay.

        This method selects the nearest horizontal plane to the requested
        elevation and plots signed interaural time difference values in
        seconds against azimuth. The azimuth axis uses the signed
        ``-180 .. 180`` convention by default, where positive azimuth values
        correspond to the left side and negative azimuth values correspond to
        the right side.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        options : PlotOptions | None, default=None
            Optional figure, axis, and margin overrides. The azimuth axis is
            configured by default to use the signed ``-180 .. 180`` range.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect how signed ITD changes around the horizontal plane.
        - Compare left-side and right-side timing cues using a signed azimuth axis.
        - Generate a horizontal-plane ITD curve for later display with ``show=False``.

        Examples
        --------
        Load a measured HRTF and inspect its horizontal-plane ITD trend:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_itd_curve(show=False)

        Remove ITD from the dataset and compare the compensated curve:

        >>> aligned = hrtf.transform.delete_itd()
        >>> aligned.plot_itd_curve(show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = AxisOptions(
            azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ).merge(plot_options.axis)

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        itd_values = np.asarray(
            calculate_itd(
                self.IR,
                output="seconds",
            ),
            dtype=float,
        )
        if itd_values.ndim != 1:
            itd_values = itd_values.reshape(-1)
        indices, real_elevation = get_horizontal_plane(
            hrtf=self,
            elevation=elevation_angle,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Selected horizontal plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
            values=azimuth_values,
            options=axis_options,
        )
        if itd_values.shape[0] != self.Sources.get_positions(angle_unit="degrees").shape[0]:
            raise ValueError("ITD values must match the number of source positions")
        horizontal_itd_values = itd_values[indices]
        sort_indices = np.argsort(transformed_azimuth_values)
        sorted_azimuth_values = transformed_azimuth_values[sort_indices]
        sorted_itd_values = horizontal_itd_values[sort_indices]

        layout = Figure(
            Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        )
        ax = layout.get_axis("main")
        ax.plot(
            sorted_azimuth_values,
            sorted_itd_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.margins(x=0.0)
        AzimuthAnglesAxis.apply(
            ax=ax,
            axis="x",
            values=sorted_azimuth_values,
            options=axis_options,
        )
        Axis.apply_label(
            ax=ax,
            axis="y",
            default_label=Labels.itd,
            options=axis_options,
        )
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        if grid_enabled:
            ax.grid(True)
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_absolute_itd(
        self: "HRTF",
        elevation_angle: float = 0.0,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot absolute ITD over a horizontal plane in polar coordinates.

        This method selects the nearest horizontal plane to the requested
        elevation, computes absolute interaural time differences in seconds,
        and displays the result in a polar plot. Azimuth is represented on the
        angular axis and absolute ITD is represented on the radial axis.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        options : PlotOptions | None, default=None
            Optional figure, axis, and margin overrides. ``options.axis.ylabel``
            controls the radial-axis label shown at the top of the polar subplot.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title. The
            radial-axis label remains controlled by ``options.axis.ylabel`` or
            the method default label.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the azimuth-dependent ITD pattern in a horizontal plane.
        - Visualize binaural timing cues using a compact polar representation.
        - Generate an ITD figure for later display with ``show=False``.

        Examples
        --------
        Load a measured HRTF and visualize absolute ITD around the horizontal plane:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_absolute_itd(show=False)

        After ITD compensation, inspect the same polar summary again:

        >>> aligned = hrtf.transform.delete_itd()
        >>> aligned.plot_absolute_itd(show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        itd_values = np.abs(
            np.asarray(
                calculate_itd(
                    self.IR,
                    output="seconds",
                ),
                dtype=float,
            )
        )
        theta_values, radial_values, sorted_itd_values, real_elevation = Polar.create_horizontal_plane_curve(
            hrtf=self,
            values=itd_values,
            elevation=elevation_angle,
        )

        layout = Figure(
            Layout_1(
                figsize=(6, 7) if figure_options.figsize is None else figure_options.figsize,
                margins=resolved_margins,
                projection="polar",
            )
        )
        ax = layout.get_axis("main")

        ax.plot(
            theta_values,
            radial_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.set_theta_zero_location("N")
        theta_ticks = np.arange(0.0, 360.0, Polar.theta_tick_step, dtype=float)
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(tick)}°" for tick in theta_ticks])
        radial_max = float(np.max(sorted_itd_values)) if sorted_itd_values.size > 0 else 0.0
        radial_tick_step = 2e-4
        if np.isclose(radial_max, 0.0):
            resolved_radial_max = radial_tick_step
        else:
            resolved_radial_max = (
                np.ceil((radial_max * 1.1) / radial_tick_step) * radial_tick_step
            )
        radial_ticks = np.arange(
            radial_tick_step,
            resolved_radial_max + (0.5 * radial_tick_step),
            radial_tick_step,
            dtype=float,
        )
        ax.set_ylim(0.0, resolved_radial_max)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(
            [f"{tick:0.4f}".replace(".", ",") for tick in radial_ticks]
        )
        ax.set_rlabel_position(350.0)
        if axis_options.ylabel is not None:
            resolved_radial_label = axis_options.ylabel
        else:
            resolved_radial_label = Labels.itd_seconds
        ax.set_ylabel(resolved_radial_label, rotation=0)
        ax.yaxis.set_label_coords(0.5, ax.title.get_position()[1], transform=ax.transAxes)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("bottom")
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_ild_plane(
        self: "HRTF",
        plane: str = "horizontal",
        elevation_angle: float = 0.0,
        freq_min: float | None = None,
        freq_max: float | None = None,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot a frequency-dependent ILD heatmap for an HRTF plane.

        This method plots a heatmap where frequency is shown on the horizontal
        axis and the plane angle is shown on the vertical axis. The horizontal
        plane may be selected at another elevation through ``elevation_angle``.
        The median plane remains the canonical sagittal path.

        Parameters
        ----------
        plane : {"horizontal", "median"}, default="horizontal"
            Plane to visualize. ``"horizontal"`` uses a horizontal plane
            selected by elevation. ``"median"`` uses the canonical median
            plane defined by the front-back sagittal path.
        elevation_angle : float, default=0.0
            Target elevation used when ``plane="horizontal"``. The nearest
            available horizontal plane in the grid is selected. This parameter
            is not used for the median plane.
        freq_min : float | None, default=None
            Minimum frequency in Hz included in the plot.
        freq_max : float | None, default=None
            Maximum frequency in Hz included in the plot.
        options : PlotOptions | None, default=None
            Optional figure, axis, heatmap, and margin overrides. For the
            horizontal plane, the azimuth axis is configured by default to use
            the signed ``-180 .. 180`` convention, where positive values are
            on the left side and negative values are on the right side.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect frequency-dependent ILD over the horizontal plane.
        - Inspect frequency-dependent ILD over the canonical median plane.
        - Generate an ILD heatmap without showing it immediately.

        Examples
        --------
        Load a measured HRTF and inspect frequency-dependent ILD on the horizontal plane:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_ild_plane(
        ...     plane="horizontal",
        ...     freq_max=8000.0,
        ...     show=False,
        ... )

        Inspect the canonical median plane when you want sagittal ILD structure:

        >>> hrtf.plot_ild_plane(plane="median", show=False)
        """
        if plane not in ("horizontal", "median"):
            raise AttributeError("plot_ild_plane plane accepts horizontal or median")
        if isinstance(elevation_angle, bool):
            raise AttributeError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise AttributeError("elevation_angle must be a finite value")

        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = AxisOptions(
            azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ).merge(plot_options.axis)
        heatmap_options = HeatmapOptions(cmap="jet").merge(plot_options.heatmap)
        heatmap_frequency_axis_options = (
            FrequencyAxisOptions()
            if axis_options.frequency_axis is None
            else axis_options.frequency_axis
        ).merge(FrequencyAxisOptions(margin_ratio=0.0))

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")

        plane_key = str(plane).strip().lower()
        if plane_key != "horizontal" and not np.isclose(
            elevation_angle,
            0.0,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError("elevation_angle only applies when plane='horizontal'")

        layout = Figure(
            Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        )

        if plane_key == "horizontal":
            indices, real_plane_elevation = get_horizontal_plane(
                hrtf=self,
                elevation=elevation_angle,
                angle_unit="degrees",
            )
        else:
            indices, _ = get_median_plane(
                hrtf=self,
                azimuth=0.0,
                angle_unit="degrees",
            )
            real_plane_elevation = 0.0
        if indices.size == 0:
            raise ValueError("Selected plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        if plane_key == "horizontal":
            plane_axis_values = np.asarray(spherical_positions[:, 0], dtype=float)
        else:
            lateral_polar_positions = spherical_to_lateral_polar(
                spherical_positions,
                angle_unit="degrees",
            )
            plane_axis_values = np.asarray(lateral_polar_positions[:, 1], dtype=float)

        _, frequency_bins_hz, _ = tf_from_ir(
            np.asarray(self.IR.values, dtype=float),
            sample_rate=self.IR.sample_rate,
            fft_length=self.fft_length,
        )
        frequency_bins_hz = np.asarray(frequency_bins_hz, dtype=float)
        if frequency_bins_hz.ndim != 1 or frequency_bins_hz.size == 0:
            raise ValueError("TF frequency bins must be a non-empty 1D array")
        resolved_frequency_axis = FrequencyLinearAxis.build(
            frequency_bins=frequency_bins_hz,
            freq_min=freq_min,
            freq_max=freq_max,
            options=heatmap_frequency_axis_options,
        )
        frequency_mask = (
            (frequency_bins_hz >= float(resolved_frequency_axis.freq_min))
            & (frequency_bins_hz <= float(resolved_frequency_axis.freq_max))
        )
        if not np.any(frequency_mask):
            raise ValueError("Selected frequency range produced no TF bins")
        frequency_khz = frequency_bins_hz[frequency_mask] / 1000.0

        ild_values = np.asarray(
            calculate_ild(
                self.IR,
                sample_rate=self.IR.sample_rate,
                fft_length=self.fft_length,
                mode="frequency-dependent",
                output="db",
            ),
            dtype=float,
        )
        plane_matrix = np.asarray(ild_values[indices][..., frequency_mask], dtype=float)
        if plane_matrix.ndim != 2:
            raise ValueError("Frequency-dependent ILD values must have shape (M, F)")

        ax = layout.get_axis("main")
        resolved_axis_options = axis_options
        panel_plane_axis_values = (
            AzimuthAnglesAxis.transform_values(
                values=plane_axis_values,
                options=resolved_axis_options,
            )
            if plane_key == "horizontal"
            else np.asarray(plane_axis_values, dtype=float)
        )
        panel_sort_indices = np.argsort(panel_plane_axis_values)
        sorted_panel_plane_axis_values = panel_plane_axis_values[panel_sort_indices]
        sorted_plane_matrix = plane_matrix[panel_sort_indices, :]
        mesh = ax.pcolormesh(
            frequency_khz,
            sorted_panel_plane_axis_values,
            sorted_plane_matrix,
            shading="auto",
            cmap=(
                ColorBar.colormaps[
                    "jet" if heatmap_options.cmap is None else str(heatmap_options.cmap)
                ]
            ),
            vmin=float(np.min(sorted_plane_matrix)),
            vmax=float(np.max(sorted_plane_matrix)),
        )
        ax.margins(x=0.0, y=0.0)
        frequency_label = (
            Labels.frequency
            if resolved_axis_options.xlabel is None
            else resolved_axis_options.xlabel
        )
        FrequencyLinearAxis.apply(
            ax=ax,
            axis="x",
            label=frequency_label,
            options=resolved_frequency_axis,
        )
        if plane_key == "horizontal":
            AzimuthAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_panel_plane_axis_values,
                options=resolved_axis_options,
            )
        else:
            PolarAnglesAxis.apply(
                ax=ax,
                axis="y",
                values=sorted_panel_plane_axis_values,
                options=resolved_axis_options,
            )
        grid_enabled = (
            False if resolved_axis_options.grid is None else resolved_axis_options.grid
        )
        if grid_enabled:
            ax.grid(True)
        ColorBar.create(
            fig=layout.fig,
            ax=ax,
            mesh=mesh,
            label=Labels.ild,
            options=heatmap_options,
            colormap="jet" if heatmap_options.cmap is None else str(heatmap_options.cmap),
        )
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_plane_title(
                    plane=plane_key,
                    elevation_angle=real_plane_elevation,
                ),
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_ild_curve(
        self: "HRTF",
        elevation_angle: float = 0.0,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot signed ILD over a horizontal plane as azimuth versus level difference.

        This method selects the nearest horizontal plane to the requested
        elevation and plots signed broad-band interaural level difference
        values in decibels against azimuth. The azimuth axis uses the signed
        ``-180 .. 180`` convention by default, where positive azimuth values
        correspond to the left side and negative azimuth values correspond to
        the right side.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        options : PlotOptions | None, default=None
            Optional figure, axis, and margin overrides. The azimuth axis is
            configured by default to use the signed ``-180 .. 180`` range.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect how signed ILD changes around the horizontal plane.
        - Compare left-side and right-side level cues using a signed azimuth axis.
        - Generate a horizontal-plane ILD curve for later display with ``show=False``.

        Examples
        --------
        Load a measured HRTF and inspect broad-band ILD across the horizontal plane:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_ild_curve(show=False)

        Focus on another measured elevation when you want an off-horizontal slice:

        >>> hrtf.plot_ild_curve(elevation_angle=10.0, show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = AxisOptions(
            azimuth_axis=AzimuthAxisOptions(range_mode="-180-180")
        ).merge(plot_options.axis)

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        ild_values = np.asarray(
            calculate_ild(
                self.IR,
                output="db",
                mode="broad-band",
            ),
            dtype=float,
        )
        if ild_values.ndim != 1:
            ild_values = ild_values.reshape(-1)
        indices, real_elevation = get_horizontal_plane(
            hrtf=self,
            elevation=elevation_angle,
            angle_unit="degrees",
        )
        if indices.size == 0:
            raise ValueError("Selected horizontal plane does not contain any source positions")

        spherical_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="spherical",
            angle_unit="degrees",
        )[indices]
        azimuth_values = np.asarray(spherical_positions[:, 0], dtype=float)
        transformed_azimuth_values = AzimuthAnglesAxis.transform_values(
            values=azimuth_values,
            options=axis_options,
        )
        if ild_values.shape[0] != self.Sources.get_positions(angle_unit="degrees").shape[0]:
            raise ValueError("ILD values must match the number of source positions")
        horizontal_ild_values = ild_values[indices]
        sort_indices = np.argsort(transformed_azimuth_values)
        sorted_azimuth_values = transformed_azimuth_values[sort_indices]
        sorted_ild_values = horizontal_ild_values[sort_indices]

        layout = Figure(
            Layout_1(
                figsize=figure_options.figsize or Layout_1().figsize,
                margins=resolved_margins,
            )
        )
        ax = layout.get_axis("main")
        ax.plot(
            sorted_azimuth_values,
            sorted_ild_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.margins(x=0.0)
        AzimuthAnglesAxis.apply(
            ax=ax,
            axis="x",
            values=sorted_azimuth_values,
            options=axis_options,
        )
        Axis.apply_label(
            ax=ax,
            axis="y",
            default_label=Labels.ild,
            options=axis_options,
        )
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        if grid_enabled:
            ax.grid(True)
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_absolute_ild(
        self: "HRTF",
        elevation_angle: float = 0.0,
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot absolute ILD over a horizontal plane in polar coordinates.

        This method selects the nearest horizontal plane to the requested
        elevation, computes absolute interaural level differences in decibels,
        and displays the result in a polar plot. Azimuth is represented on the
        angular axis and absolute ILD is represented on the radial axis.

        Parameters
        ----------
        elevation_angle : float, default=0.0
            Target elevation used to select the horizontal plane. The nearest
            available elevation in the grid is used.
        options : PlotOptions | None, default=None
            Optional figure, axis, and margin overrides. ``options.axis.ylabel``
            controls the radial-axis label shown at the top of the polar subplot.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title. The
            radial-axis label remains controlled by ``options.axis.ylabel`` or
            the method default label.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the azimuth-dependent ILD pattern in a horizontal plane.
        - Visualize binaural level cues using a compact polar representation.
        - Generate an ILD figure for later display with ``show=False``.

        Examples
        --------
        Load a measured HRTF and summarize absolute ILD in a polar view:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_absolute_ild(show=False)

        Inspect a different elevation when you want a polar view away from the canonical plane:

        >>> hrtf.plot_absolute_ild(elevation_angle=10.0, show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        if self.IR.values is None:
            raise ValueError("IR data is not available")
        if self.IR.sample_rate is None:
            raise ValueError("IR sample_rate is required")
        if isinstance(elevation_angle, bool):
            raise ValueError("elevation_angle must be a finite value")
        elevation_angle = float(elevation_angle)
        if not np.isfinite(elevation_angle):
            raise ValueError("elevation_angle must be a finite value")

        ild_values = np.abs(
            np.asarray(
                calculate_ild(
                    self.IR,
                    output="db",
                    mode="broad-band",
                ),
                dtype=float,
            )
        )
        theta_values, radial_values, sorted_ild_values, real_elevation = Polar.create_horizontal_plane_curve(
            hrtf=self,
            values=ild_values,
            elevation=elevation_angle,
        )

        layout = Figure(
            Layout_1(
                figsize=(6, 7) if figure_options.figsize is None else figure_options.figsize,
                margins=resolved_margins,
                projection="polar",
            )
        )
        ax = layout.get_axis("main")

        ax.plot(
            theta_values,
            radial_values,
            color="steelblue",
            linewidth=2.0,
        )
        ax.set_theta_zero_location("N")
        theta_ticks = np.arange(0.0, 360.0, Polar.theta_tick_step, dtype=float)
        ax.set_xticks(np.deg2rad(theta_ticks))
        ax.set_xticklabels([f"{int(tick)}°" for tick in theta_ticks])
        radial_max = float(np.max(sorted_ild_values)) if sorted_ild_values.size > 0 else 0.0
        radial_tick_step = 5.0
        if np.isclose(radial_max, 0.0):
            resolved_radial_max = radial_tick_step
        else:
            resolved_radial_max = (
                np.ceil((radial_max * 1.1) / radial_tick_step) * radial_tick_step
            )
        radial_ticks = np.arange(
            radial_tick_step,
            resolved_radial_max + (0.5 * radial_tick_step),
            radial_tick_step,
            dtype=float,
        )
        ax.set_ylim(0.0, resolved_radial_max)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels(
            [f"{int(np.rint(tick))}" for tick in radial_ticks]
        )
        ax.set_rlabel_position(350.0)
        if axis_options.ylabel is not None:
            resolved_radial_label = axis_options.ylabel
        else:
            resolved_radial_label = Labels().ild_db
        ax.set_ylabel(resolved_radial_label, rotation=0)
        ax.yaxis.set_label_coords(0.5, ax.title.get_position()[1], transform=ax.transAxes)
        ax.yaxis.label.set_horizontalalignment("center")
        ax.yaxis.label.set_verticalalignment("bottom")
        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                Titles.create_plane_title(
                    plane="horizontal",
                    elevation_angle=real_elevation,
                ),
            )
        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)
        if show and plot_options.show:
            plt.show()
        return None

    def plot_source_grid(
        self: "HRTF",
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot the source grid as an interactive three-dimensional scatter.

        The method reads the current source positions from the HRTF instance,
        converts them to Cartesian coordinates when necessary, and renders the
        grid in a 3D Matplotlib axis. Direction arrows for front, right, and up
        are added to make the spatial orientation easier to interpret in the
        default camera view.

        Parameters
        ----------
        options : PlotOptions | None, default=None
            Optional figure, axis, and margin overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title. Explicit
            figure titles provided through ``options.figure.title`` are still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the spatial sampling pattern of a source grid.
        - Check how dense or sparse a dataset is across directions.
        - Visualize the currently selected subset of sources after spatial
          selection or transformation.

        Examples
        --------
        Load a measured HRTF and inspect the full source grid:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_source_grid(show=False)

        Plot only the selected horizontal plane to verify a spatial subset:

        >>> horizontal = hrtf.select(plane="horizontal", plane_angle=0.0)
        >>> horizontal.plot_source_grid(show=False)
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )

        cartesian_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="cartesian",
            angle_unit="degrees",
        )
        layout = Figure(
            Layout_1(
                figsize=(6, 7) if figure_options.figsize is None else figure_options.figsize,
                margins=resolved_margins,
                projection="3d",
            )
        )
        ax = layout.get_axis("main")

        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)
        ax.scatter(
            x_values,
            y_values,
            z_values,
            s=28.0,
            color="steelblue",
            edgecolors="black",
            linewidths=0.4,
            depthshade=True,
        )
        axis_half_span = ThreeDimensional1.configure_axis(
            ax=ax,
            cartesian_positions=cartesian_positions,
        )
        ThreeDimensional1.create_direction_markers(
            ax=ax,
            sources=self.Sources,
            axis_half_span=axis_half_span,
        )

        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)

        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                "Source Grid",
            )
        if show and plot_options.show:
            plt.show()
        return None

    def plot_plane_grid(
        self: "HRTF",
        plane: str | list[str] | tuple[str, ...] = "horizontal",
        options: PlotOptions | None = None,
        show: bool = True,
        titles: bool = True,
    ) -> None:
        """Plot the source grid and highlight canonical spatial planes in 3D.

        The full source grid is displayed as a light background scatter, while
        the selected canonical plane or planes are overlaid with stronger
        colors. The supported planes are the horizontal plane, the median plane,
        and the frontal plane, using the canonical definitions already provided
        by the spatial plane-selection logic in the library.

        Parameters
        ----------
        plane : str | list[str] | tuple[str, ...], default="horizontal"
            Plane or planes to highlight. Accepted values are ``"horizontal"``,
            ``"median"``, and ``"frontal"``. A single string highlights one
            plane, while a list or tuple highlights multiple planes in the same
            figure.
        options : PlotOptions | None, default=None
            Optional figure, axis, legend, and margin overrides.
        show : bool, default=True
            If ``True``, call ``matplotlib.pyplot.show()`` before returning.
        titles : bool, default=True
            If ``False``, suppress the generated default figure title. Explicit
            figure titles provided through ``options.figure.title`` are still shown.

        Returns
        -------
        None

        Use Cases
        ---------
        - Inspect the geometry of the canonical horizontal, median, and frontal
          planes in a dataset.
        - Verify whether a dataset contains the expected plane coverage.
        - Compare several canonical planes in one spatial grid view.

        Examples
        --------
        Load a measured HRTF and highlight the median plane in the full grid:

        >>> from hrtfpykit import HRTF
        >>> hrtf = HRTF.load_hrtf("my_hrtf.sofa")
        >>> hrtf.plot_plane_grid(plane="median", show=False)

        Compare the three canonical planes in one 3D view:

        >>> hrtf.plot_plane_grid(
        ...     plane=["horizontal", "median", "frontal"],
        ...     show=False,
        ... )
        """
        plot_options = PlotOptions() if options is None else options
        figure_options = (
            plot_options.figure if plot_options.figure is not None else FigureOptions()
        )
        resolved_margins = (
            figure_options.margins if figure_options.margins is not None else Margins()
        )
        axis_options = (
            plot_options.axis if plot_options.axis is not None else AxisOptions()
        )
        legend_options = (
            LegendOptions() if axis_options.legend is None else axis_options.legend
        )

        raw_planes = [plane] if isinstance(plane, str) else list(plane)
        if len(raw_planes) == 0:
            raise ValueError("plane must contain at least one value")

        resolved_planes: list[str] = []
        for raw_plane in raw_planes:
            plane_key = str(raw_plane).strip().lower()
            if plane_key not in {"horizontal", "median", "frontal"}:
                raise ValueError("plane accepts: horizontal, median, frontal")
            if plane_key not in resolved_planes:
                resolved_planes.append(plane_key)

        cartesian_positions = get_source_positions(
            sources=self.Sources,
            coordinate_system="cartesian",
            angle_unit="degrees",
        )
        layout = Figure(
            Layout_1(
                figsize=(6, 7) if figure_options.figsize is None else figure_options.figsize,
                margins=resolved_margins,
                projection="3d",
            )
        )
        ax = layout.get_axis("main")

        x_values = np.asarray(cartesian_positions[:, 0], dtype=float)
        y_values = np.asarray(cartesian_positions[:, 1], dtype=float)
        z_values = np.asarray(cartesian_positions[:, 2], dtype=float)

        ax.scatter(
            x_values,
            y_values,
            z_values,
            s=18.0,
            color="#9ecae1",
            edgecolors="none",
            depthshade=True,
            alpha=0.55,
            label="Source Grid",
        )

        plane_colors = {
            "horizontal": "green",
            "median": "red",
            "frontal": "blue",
        }
        plane_titles = {
            "horizontal": "Horizontal Plane Grid",
            "median": "Median Plane Grid",
            "frontal": "Frontal Plane Grid",
        }
        plane_labels = {
            "horizontal": "Horizontal Plane",
            "median": "Median Plane",
            "frontal": "Frontal Plane",
        }

        for plane_key in resolved_planes:
            if plane_key == "horizontal":
                indices, _ = get_horizontal_plane(
                    hrtf=self,
                    elevation=0.0,
                    angle_unit="degrees",
                )
            elif plane_key == "median":
                indices, _ = get_median_plane(
                    hrtf=self,
                    azimuth=0.0,
                    angle_unit="degrees",
                )
            else:
                indices, _ = get_frontal_plane(
                    hrtf=self,
                    azimuth=90.0,
                    angle_unit="degrees",
                )
            plane_positions = np.asarray(cartesian_positions[indices], dtype=float)
            ax.scatter(
                plane_positions[:, 0],
                plane_positions[:, 1],
                plane_positions[:, 2],
                s=34.0,
                color=plane_colors[plane_key],
                edgecolors="black",
                linewidths=0.35,
                depthshade=True,
                label=plane_labels[plane_key],
            )

        axis_half_span = ThreeDimensional1.configure_axis(
            ax=ax,
            cartesian_positions=cartesian_positions,
        )
        ThreeDimensional1.create_direction_markers(
            ax=ax,
            sources=self.Sources,
            axis_half_span=axis_half_span,
        )

        grid_enabled = True if axis_options.grid is None else axis_options.grid
        ax.grid(grid_enabled)

        legend_enabled = True if legend_options.enabled is None else legend_options.enabled
        if legend_enabled:
            resolved_legend_location = (
                "upper right"
                if legend_options.location is None
                else legend_options.location
            )
            ax.legend(loc=resolved_legend_location)

        if figure_options.title is not None:
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                figure_options.title,
            )
        elif titles:
            if len(resolved_planes) == 1:
                resolved_figure_title = plane_titles[resolved_planes[0]]
            else:
                resolved_figure_title = "Plane Grid"
            Titles.create_figure_title(
                layout.fig,
                layout.axes,
                layout.figure_title_y,
                resolved_figure_title,
            )
        if show and plot_options.show:
            plt.show()
        return None
