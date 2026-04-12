from __future__ import annotations

"""Legend helpers used by plot methods."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class Legends(ABC):
    """Abstract base class for legend strategies."""

    @staticmethod
    @abstractmethod
    def apply(ax: plt.Axes, *args, **kwargs) -> None:
        raise NotImplementedError


class Ear(Legends):
    """Legend strategy for left/right/both ear plot traces."""

    location: str = "upper left"

    @staticmethod
    def apply(
        ax: plt.Axes,
        ear: str,
        location: str | None = None,
        labels: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Apply ear legend labels to the target axis.

        Parameters
        ----------
        ax : plt.Axes
            Target axis where the legend is rendered.
        ear : str
            Ear selection mode: ``"left"``, ``"right"``, or ``"both"``.
        location : str | None, default=None
            Legend location string. Uses class default when omitted.
        labels : tuple[str, ...] | list[str] | None, default=None
            Optional custom labels. Label count must match the selected
            ``ear`` mode.

        Returns
        -------
        None

        Use Cases
        ---------
        - Add consistent ear legends for waveform and magnitude plots.
        - Override default left/right labels in custom figures.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([0, 1], [0, 1])
        >>> ax.plot([0, 1], [1, 0])
        >>> Ear.apply(ax=ax, ear="both")
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
