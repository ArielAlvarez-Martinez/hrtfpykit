from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class Legends(ABC):
    """Base interface for Matplotlib legend strategies used by plot helpers.

    :class:`~hrtfpykit.plots.legends.Legends` defines the small contract shared
    by concrete legend helpers in :mod:`~hrtfpykit.plots`. Implementations
    receive a Matplotlib axis and the strategy-specific context needed to label
    the traces already drawn on that axis. The class keeps legend creation
    centralized so plot functions can reuse consistent default locations, label
    validation, and Matplotlib legend calls.
    """

    @staticmethod
    @abstractmethod
    def apply(ax: plt.Axes, *args, **kwargs) -> None:
        """Render a legend on an axis.

        Concrete strategies implement this method to map plot-specific context
        to matplotlib.axes.Axes.legend. The base signature is intentionally
        generic because ear-channel legends and subject-comparison legends need
        different arguments.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis that already contains the plotted artists to label.
        *args
            Positional arguments required by a concrete legend strategy.
        **kwargs
            Keyword arguments required by a concrete legend strategy.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always raised by the abstract base implementation.
        """
        raise NotImplementedError


class Ear(Legends):
    """Legend strategy for ear-channel traces in single-HRTF plots.

    :class:`~hrtfpykit.plots.legends.Ear` labels traces by receiver channel
    rather than by subject. It is used by HRTF magnitude and impulse-response
    plotting code where one axis can display the left ear, the right ear, or
    both ears from the same :class:`~hrtfpykit.hrtf.hrtf.HRTF` object.
    The strategy validates that the requested labels match the selected ear mode
    before delegating to Matplotlib.

    Attributes
    ----------
    location : str
        Default Matplotlib legend location used when apply receives
        location=None.
    """

    location: str = "upper left"

    @staticmethod
    def apply(
        ax: plt.Axes,
        ear: str,
        location: str | None = None,
        labels: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Render an ear-channel legend for traces already drawn on an axis.

        The legend entries are derived from ear unless explicit labels are
        provided. ``both`` requires two labels and maps to left/right channel
        traces in plotting order; ``left`` and ``right`` each require one
        label. The method does not inspect artists on the axis, so callers must
        draw traces in the same order as the resolved labels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis that contains the ear-channel traces to label.
        ear : {``left``, ``right``, ``both``}
            Ear selection mode used by the plot. ``both`` creates labels for
            the left and right channel traces.
        location : str | None, default=None
            Matplotlib legend location. When None, Ear.location is used.
        labels : tuple[str, ...] | list[str] | None, default=None
            Custom legend labels. The number of labels must match the selected
            ear mode: two labels for ``both`` and one label for
            ``left`` or ``right``.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ear is not one of ``left``, ``right``, or ``both``, or
            if labels has the wrong number of entries.

        Notes
        -----
        The method calls ax.legend(labels=..., loc=...). It supplies labels
        by position instead of binding labels to individual artists, so it is
        best used immediately after drawing the corresponding ear traces.

        """
        default_labels_by_ear = {
            "both": ["Left Ear", "Right Ear"],
            "left": ["Left Ear"],
            "right": ["Right Ear"],
        }
        if ear not in default_labels_by_ear:
            raise ValueError("ear accepts left, right, or both")
        default_labels = default_labels_by_ear[ear]
        expected_label_count = len(default_labels)
        resolved_labels = (
            default_labels if labels is None else [str(label) for label in labels]
        )
        if len(resolved_labels) != expected_label_count:
            raise ValueError(
                f"legend labels must contain {expected_label_count} entries for ear='{ear}'"
            )
        resolved_location = Ear.location if location is None else location
        ax.legend(labels=resolved_labels, loc=resolved_location)


class Subjects(Legends):
    """Legend strategy for traces that compare multiple HRTF subjects.

    :class:`~hrtfpykit.plots.legends.Subjects` labels one trace per
    :class:`~hrtfpykit.hrtf.hrtf.HRTF` object in comparison plots. It is used by
    magnitude, impulse-response, ITD, ILD, and LSD comparison helpers where each
    plotted line represents a different subject, dataset entry, or processing
    pipeline. The strategy accepts an optional bounding-box anchor so plot
    functions can place legends outside dense Cartesian or polar axes.

    Attributes
    ----------
    location : str
        Default Matplotlib legend location used when apply receives
        location=None.
    bbox_to_anchor : tuple[float, float] | None
        Default Matplotlib legend anchor. None leaves anchoring to
        Matplotlib's normal legend placement.
    """

    location: str = "upper right"
    bbox_to_anchor: tuple[float, float] | None = None

    @staticmethod
    def create_default_labels(
        count: int,
    ) -> list[str]:
        """Create deterministic subject labels for comparison plots.

        Labels are generated as ``subject_1`` through ``subject_n`` and are used
        when comparison helpers receive multiple
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` objects without explicit legend names.
        The function validates count before creating labels so downstream legend
        code always receives a non-empty label sequence.

        Parameters
        ----------
        count : int
            Number of subject labels to create. Must be greater than zero.

        Returns
        -------
        list[str]
            Generated labels in plotting order.

        Raises
        ------
        ValueError
            If count is less than or equal to zero.

        """
        if count <= 0:
            raise ValueError("count must be positive")
        return [f"subject_{index + 1}" for index in range(count)]

    @staticmethod
    def apply(
        ax: plt.Axes,
        labels: tuple[str, ...] | list[str],
        location: str | None = None,
        bbox_to_anchor: tuple[float, float] | None = None,
    ) -> None:
        """Render a subject-comparison legend on an axis.

        Each label corresponds to one trace already drawn for a subject or HRTF
        instance. The method normalizes labels to strings, validates that at
        least one label is available, resolves class defaults, and then calls
        matplotlib.axes.Axes.legend. bbox_to_anchor is passed only when a
        concrete anchor is supplied or configured as a class default.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis that contains the subject traces to label.
        labels : tuple[str, ...] | list[str]
            Subject, dataset, or pipeline labels in the same order as the
            plotted traces.
        location : str | None, default=None
            Matplotlib legend location. When None, Subjects.location is
            used.
        bbox_to_anchor : tuple[float, float] | None, default=None
            Optional Matplotlib legend anchor (x, y). When None,
            Subjects.bbox_to_anchor is used.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If labels is empty.

        Notes
        -----
        The method does not verify the number of plotted artists. Callers should
        validate that the label count matches the number of
        :class:`~hrtfpykit.hrtf.hrtf.HRTF` objects or traces before rendering
        the legend.

        """
        resolved_labels = [str(label) for label in labels]
        if len(resolved_labels) == 0:
            raise ValueError("legend labels must contain at least one entry")
        resolved_location = Subjects.location if location is None else str(location)
        resolved_bbox = (
            Subjects.bbox_to_anchor
            if bbox_to_anchor is None
            else (float(bbox_to_anchor[0]), float(bbox_to_anchor[1]))
        )
        if resolved_bbox is None:
            ax.legend(labels=resolved_labels, loc=resolved_location)
            return
        ax.legend(labels=resolved_labels, loc=resolved_location, bbox_to_anchor=resolved_bbox)
