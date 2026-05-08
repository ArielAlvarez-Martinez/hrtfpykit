from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .default import FigureSize, RC
from .layouts import Layout
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
        colormap: str | None = None,
        colorbar: bool = True,
        colorbar_location: str | None = None,
        colorbar_fraction: float | None = None,
        colorbar_pad: float | None = None,
        colorbar_label: str | None = None,
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
        colormap : str | None, default=None
            Colormap name.
        colorbar : bool, default=True
            Whether to render the heatmap colorbar.
        colorbar_location : str | None, default=None
            Colorbar location.
        colorbar_fraction : float | None, default=None
            Colorbar width fraction.
        colorbar_pad : float | None, default=None
            Colorbar padding.
        colorbar_label : str | None, default=None
            Colorbar label override.
        **kwargs
            Extra Matplotlib pcolormesh arguments.

        Returns
        -------
        object
            Matplotlib mesh artist returned by ``Heatmap.create``.

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
            colormap=colormap,
            colorbar=colorbar,
            colorbar_location=colorbar_location,
            colorbar_fraction=colorbar_fraction,
            colorbar_pad=colorbar_pad,
            colorbar_label=colorbar_label,
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
