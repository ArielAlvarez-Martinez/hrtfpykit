from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .default import FigureSize, RC
from .layouts import Layout
from .options import AxisOptions, PlotOptions
from .types import Heatmap, ThreeDimension, TwoDimension


class Figure:
    """Figure wrapper that centralizes layout creation and plot primitives."""

    shared_x_visible: bool = True

    def __init__(self, layout: Layout, projection: str | None = None):
        """Initialize a figure using a layout and optional projection.

        Parameters
        ----------
        layout : Layout
            Layout definition containing rows, columns, margins, and positions.
        projection : str | None, default=None
            Optional Matplotlib projection passed to subplot creation
            (for example ``"polar"`` or ``"3d"``).

        Returns
        -------
        None

        Use Cases
        ---------
        - Build a 2D, 3D, or polar figure with consistent layout handling.
        - Reuse one figure abstraction across plot methods.

        Examples
        --------
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> figure = Figure(Layout_1())
        """
        self.layout = layout.code
        self.positions = layout.positions
        self.projection = projection
        self.figure_title_y = min(
            layout.margins.top + layout.figure_title_offset,
            0.98,
        )
        self.fig, self.axes = self.create(layout, projection=projection)

    @staticmethod
    def configure_rc() -> None:
        """Apply default Matplotlib rcParams used by plot rendering.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Use Cases
        ---------
        - Keep typography and figure styling consistent across plots.

        Examples
        --------
        >>> Figure.configure_rc()
        """
        rc = RC()
        plt.rcParams.update(
            {
                "font.size": rc.default,
                "axes.titlesize": rc.axis_title,
                "axes.labelsize": rc.axis_labels,
                "xtick.labelsize": rc.ticks,
                "ytick.labelsize": rc.ticks,
                "legend.fontsize": rc.legend,
                "legend.title_fontsize": rc.legend_title,
                "figure.titlesize": rc.fig_title,
                "figure.titleweight": "bold",
            }
        )

    @staticmethod
    def create(
        layout: Layout,
        projection: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create Matplotlib figure and flattened axes from a layout.

        Parameters
        ----------
        layout : Layout
            Layout definition with grid size, margins, and sharing settings.
        projection : str | None, default=None
            Optional subplot projection (for example ``"polar"`` or ``"3d"``).

        Returns
        -------
        tuple[plt.Figure, np.ndarray]
            Matplotlib figure and flattened object array of subplot axes.

        Use Cases
        ---------
        - Create plotting canvases for single or multi-subplot layouts.
        - Create projected axes while keeping one common layout pipeline.

        Examples
        --------
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> fig, axes = Figure.create(Layout_1())
        >>> axes.size
        1
        """
        Figure.configure_rc()
        if isinstance(layout.figsize, FigureSize):
            resolved_figsize = (layout.figsize.width, layout.figsize.height)
        else:
            resolved_figsize = layout.figsize
        subplot_kwargs: dict[str, object] = {}
        if projection is not None:
            subplot_kwargs["subplot_kw"] = {"projection": projection}
        fig, axes = plt.subplots(
            layout.rows,
            layout.cols,
            figsize=resolved_figsize,
            sharex=layout.sharex,
            sharey=layout.sharey,
            squeeze=False,
            **subplot_kwargs,
        )
        fig.subplots_adjust(
            left=layout.margins.left,
            bottom=layout.margins.bottom,
            right=layout.margins.right,
            top=layout.margins.top,
            wspace=layout.margins.wspace,
            hspace=layout.margins.hspace,
        )
        reshaped_axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax in reshaped_axes:
            setattr(ax, "hrtfpykit_subplot_title_y", 1.0)
            setattr(
                ax,
                "hrtfpykit_subplot_title_y_with_figure_title",
                layout.subplot_title_y,
            )
        return fig, reshaped_axes

    def get_ax(self, position: int | str = 0) -> plt.Axes:
        """Return one subplot axis by index or named position.

        Parameters
        ----------
        position : int | str, default=0
            Subplot index or position key defined by the layout.

        Returns
        -------
        plt.Axes
            Selected subplot axis.

        Use Cases
        ---------
        - Access subplot axes using stable position names like ``"main"``.
        - Access subplot axes by numeric index in loops.

        Examples
        --------
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> figure = Figure(Layout_1())
        >>> _ = figure.get_ax("main")
        """
        if isinstance(position, str):
            if position not in self.positions:
                raise ValueError(
                    f"position must be one of: {', '.join(self.positions)}"
                )
            axis_index = self.positions.index(position)
        else:
            axis_index = int(position)
            if axis_index < 0 or axis_index >= self.axes.size:
                raise ValueError(
                    f"position index must be between 0 and {self.axes.size - 1}"
                )
        return self.axes[axis_index]

    def hide_unused_axes(self, used_axes: int) -> None:
        """Hide subplot axes beyond the number of used panels.

        Parameters
        ----------
        used_axes : int
            Number of axes that remain visible from the beginning of
            the flattened axes array.

        Returns
        -------
        None

        Use Cases
        ---------
        - Hide unused panels when plotting fewer items than layout capacity.

        Examples
        --------
        >>> from hrtfpykit.plots.layouts import Layout_2Vertical
        >>> figure = Figure(Layout_2Vertical())
        >>> figure.hide_unused_axes(1)
        """
        if used_axes < 0:
            raise ValueError("used_axes must be non-negative")
        for ax in self.axes[used_axes:]:
            ax.set_visible(False)

    def get_subplots_axis_options(
        self,
        plot_options: PlotOptions,
    ) -> dict[int, AxisOptions]:
        """Resolve subplot-specific axis options indexed by subplot position.

        Parameters
        ----------
        plot_options : PlotOptions
            Plot options that may contain subplot overrides in ``subplots``.

        Returns
        -------
        dict[int, AxisOptions]
            Mapping from subplot index to axis options.

        Use Cases
        ---------
        - Apply per-subplot labels, titles, grids, and legends.
        - Convert named subplot keys into concrete indices.

        Examples
        --------
        >>> from hrtfpykit.plots.options import PlotOptions
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> figure = Figure(Layout_1())
        >>> figure.get_subplots_axis_options(PlotOptions())
        {}
        """
        subplot_axis_options: dict[int, AxisOptions] = {}
        if plot_options.subplots is None:
            return subplot_axis_options
        for subplot, subplot_options in plot_options.subplots.items():
            if isinstance(subplot, str):
                if subplot not in self.positions:
                    raise ValueError(
                        f"subplot accepts: {', '.join(self.positions)}"
                    )
                subplot_index = self.positions.index(subplot)
            else:
                subplot_index = int(subplot)
                if subplot_index < 0 or subplot_index >= self.axes.size:
                    raise ValueError(
                        f"subplot index must be between 0 and {self.axes.size - 1}"
                    )
            if subplot_index in subplot_axis_options:
                raise ValueError(
                    f"subplot override for subplot {subplot_index} is duplicated"
                )
            subplot_axis_options[subplot_index] = subplot_options
        return subplot_axis_options

    def create_two_dimension(self, ax: plt.Axes, x, y, **kwargs):
        """Create a 2D line plot on the provided axis.

        Parameters
        ----------
        ax : plt.Axes
            Target subplot axis.
        x : array-like
            X-axis values.
        y : array-like
            Y-axis values.
        **kwargs
            Extra Matplotlib line arguments.

        Returns
        -------
        object
            Matplotlib artist returned by ``TwoDimension.create``.

        Use Cases
        ---------
        - Draw waveform and curve plots through one shared wrapper.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> figure = Figure(Layout_1())
        >>> ax = figure.get_ax("main")
        >>> _ = figure.create_two_dimension(ax=ax, x=np.array([0, 1]), y=np.array([0, 1]))
        """
        return TwoDimension.create(
            ax=ax,
            x=x,
            y=y,
            **kwargs,
        )

    def create_heatmap(
        self,
        ax: plt.Axes,
        x,
        y,
        values,
        label: str | None = None,
        options=None,
        colormap: str | None = None,
        **kwargs,
    ):
        """Create a heatmap on the provided axis.

        Parameters
        ----------
        ax : plt.Axes
            Target subplot axis.
        x : array-like
            X-axis coordinates.
        y : array-like
            Y-axis coordinates.
        values : array-like
            Heatmap values.
        label : str | None, default=None
            Colorbar label.
        options : object, default=None
            Heatmap options object passed to ``Heatmap.create``.
        colormap : str | None, default=None
            Colormap name.
        **kwargs
            Extra Matplotlib pcolormesh arguments.

        Returns
        -------
        object
            Matplotlib mesh artist returned by ``Heatmap.create``.

        Use Cases
        ---------
        - Render spectrum and plane heatmaps with shared figure integration.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> figure = Figure(Layout_1())
        >>> ax = figure.get_ax("main")
        >>> _ = figure.create_heatmap(ax=ax, x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]), values=np.array([[0.0, 1.0], [1.0, 0.0]]))
        """
        return Heatmap.create(
            ax=ax,
            x=x,
            y=y,
            values=values,
            fig=self.fig,
            label=label,
            options=options,
            colormap=colormap,
            **kwargs,
        )

    def create_three_dimension(
        self,
        ax: plt.Axes,
        x,
        y,
        z,
        s: float = 28.0,
        color: str = "steelblue",
        edgecolors: str = "black",
        linewidths: float = 0.4,
        depthshade: bool = True,
        **kwargs,
    ):
        """Create a 3D scatter plot on the provided axis.

        Parameters
        ----------
        ax : plt.Axes
            Target 3D subplot axis.
        x : array-like
            X coordinates.
        y : array-like
            Y coordinates.
        z : array-like
            Z coordinates.
        s : float, default=28.0
            Marker size.
        color : str, default="steelblue"
            Marker face color.
        edgecolors : str, default="black"
            Marker edge color.
        linewidths : float, default=0.4
            Marker edge width.
        depthshade : bool, default=True
            Whether to apply depth shading.
        **kwargs
            Extra Matplotlib scatter arguments.

        Returns
        -------
        object
            Matplotlib artist returned by ``ThreeDimension.create``.

        Use Cases
        ---------
        - Render source-grid and plane-grid 3D scatter views.

        Examples
        --------
        >>> import numpy as np
        >>> from hrtfpykit.plots.layouts import Layout_1
        >>> figure = Figure(Layout_1(), projection="3d")
        >>> ax = figure.get_ax("main")
        >>> _ = figure.create_three_dimension(ax=ax, x=np.array([0.0]), y=np.array([0.0]), z=np.array([1.0]))
        """
        return ThreeDimension.create(
            ax=ax,
            x=x,
            y=y,
            z=z,
            s=s,
            color=color,
            edgecolors=edgecolors,
            linewidths=linewidths,
            depthshade=depthshade,
            **kwargs,
        )
