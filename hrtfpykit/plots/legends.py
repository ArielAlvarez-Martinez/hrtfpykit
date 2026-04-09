from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class Legends(ABC):
    @staticmethod
    @abstractmethod
    def apply(ax: plt.Axes, *args, **kwargs) -> None:
        raise NotImplementedError


class Ear(Legends):
    location: str = "upper left"

    @staticmethod
    def apply(
        ax: plt.Axes,
        ear: str,
        location: str | None = None,
        labels: tuple[str, ...] | list[str] | None = None,
    ) -> None:
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
