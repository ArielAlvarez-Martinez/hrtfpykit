import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hrtfpykit.plots import (
    Axis,
    AxisOptions,
    FrequencyAxisOptions,
    HRTFPlots,
    PlotOptions,
)


class DummySources:
    @staticmethod
    def get_position_queries(
        positions: np.ndarray | list | tuple | str,
    ) -> list[np.ndarray | str]:
        if isinstance(positions, str):
            return [positions]
        raw_positions = np.asarray(positions, dtype=object)
        if raw_positions.ndim == 1:
            return [np.asarray(raw_positions, dtype=float)]
        return [np.asarray(position, dtype=float) for position in raw_positions]

    def get_position_index(
        self,
        position: str | list[float] | np.ndarray,
        coordinate_system: str = "spherical",
    ) -> tuple[int, np.ndarray]:
        if isinstance(position, str):
            named_positions = {
                "front": np.array([0.0, 0.0], dtype=float),
                "left": np.array([90.0, 0.0], dtype=float),
                "right": np.array([270.0, 0.0], dtype=float),
                "back": np.array([180.0, 0.0], dtype=float),
            }
            return 0, named_positions[position]
        return 0, np.asarray(position, dtype=float)


class DummyTF:
    def __init__(
        self,
        magnitude_db: np.ndarray,
        magnitude: np.ndarray,
        frequency_bins: np.ndarray,
    ) -> None:
        self.values = np.asarray(magnitude, dtype=float)
        self._magnitude_db = np.asarray(magnitude_db, dtype=float)
        self.magnitude = np.asarray(magnitude, dtype=float)
        self.frequency_bins = np.asarray(frequency_bins, dtype=float)

    def get_magnitude_db(self, reference: float = 1.0) -> np.ndarray:
        if reference != 1.0:
            raise ValueError("DummyTF only supports reference=1.0 in tests")
        return self._magnitude_db
<<<<<<< HEAD

=======
>>>>>>> dev


class DummyPlotHRTF(HRTFPlots):
    def __init__(
        self,
        magnitude_db: np.ndarray,
        magnitude: np.ndarray,
        frequency_bins: np.ndarray,
    ) -> None:
        self.TF = DummyTF(magnitude_db, magnitude, frequency_bins)
        self.Sources = DummySources()


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


@pytest.fixture
def dummy_hrtf() -> DummyPlotHRTF:
    frequency_bins = np.array([0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0], dtype=float)
    magnitude = np.array(
        [[[1.0, 1.2, 1.4, 1.6, 1.8], [0.9, 1.1, 1.3, 1.5, 1.7]]],
        dtype=float,
    )
    magnitude_db = np.array(
        [[[0.0, 1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0, 3.0]]],
        dtype=float,
    )
    return DummyPlotHRTF(magnitude_db, magnitude, frequency_bins)


def test_create_frequency_axis_linear_sets_scale_ticks_and_label() -> None:
    fig, ax = plt.subplots()

    resolved = Axis.create_frequency_axis(
        ax=ax,
        axis="x",
        x_axis="linear",
        frequency_bins=np.array([0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0]),
        freq_min=250.0,
        freq_max=20_000.0,
        label="Frequency(kHz)",
    )

    assert resolved.freq_min == 250.0
    assert resolved.freq_max == 20_000.0
    assert ax.get_xscale() == "linear"
    assert ax.get_xlabel() == "Frequency(kHz)"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "2",
        "4",
        "6",
        "8",
        "10",
        "12",
        "14",
        "16",
        "18",
        "20",
    ]


def test_create_frequency_axis_db_uses_first_positive_bin_when_freq_min_is_missing() -> None:
    resolved = Axis.create_frequency_axis(
        ax=None,
        axis="x",
        x_axis="log",
        frequency_bins=np.array([0.0, 250.0, 500.0, 1_000.0]),
        freq_max=1_000.0,
    )

    assert resolved.freq_min == 250.0
    assert resolved.freq_max == 1_000.0
    assert resolved.ticks == (250.0, 500.0, 1_000.0)
    assert resolved.labels == ("0.25", "0.5", "1")


def test_create_magnitude_axis_sets_ylabel_title_and_legend() -> None:
    fig, ax = plt.subplots()

    ax.plot([1.0, 2.0], [0.0, 1.0])
    ax.plot([1.0, 2.0], [1.0, 2.0])

    Axis.create_magnitude_axis(
        ax=ax,
        axis="y",
        unit="db",
        selected_positions=np.array([30.0, 10.0]),
        position_coordinate_system="spherical",
        ear="both",
        options=AxisOptions(),
    )

    assert ax.get_ylabel() == "Magnitude (dB)"
    assert ax.get_title() == "Position : [Azimuth= 30.0°, Elevation= 10.0°]"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Left Ear",
        "Right Ear",
    ]


def test_plot_magnitude_single_position_returns_layout_and_lines(
    dummy_hrtf: DummyPlotHRTF,
) -> None:
    result = dummy_hrtf.plot_magnitude(
        [0.0, 0.0],
        unit="linear",
        ear="both",
        x_axis="linear",
        freq_min=250.0,
        freq_max=20_000.0,
        show=False,
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    assert result is None
    assert len(fig.axes) == 1
    assert len(ax.lines) == 2
    assert ax.get_xscale() == "linear"
    assert ax.get_xlabel() == "Frequency(kHz)"
    assert ax.get_ylabel() == "Magnitude"
    assert ax.get_title() == "Front : [Azimuth= 0.0°, Elevation= 0.0°]"


def test_plot_magnitude_panel_overrides_apply_by_index_and_name(
    dummy_hrtf: DummyPlotHRTF,
) -> None:
    result = dummy_hrtf.plot_magnitude(
        [[0.0, 0.0], [30.0, 0.0]],
        unit="linear",
        ear="both",
        x_axis="linear",
        freq_min=250.0,
        freq_max=20_000.0,
        options=PlotOptions(
            axis=AxisOptions(title="Common Title"),
            panels={
                0: AxisOptions(title="Top Panel", ylabel="Panel 1 Magnitude"),
                "bottom": AxisOptions(
                    title="Bottom Panel",
                    xlabel="Bottom Frequency(kHz)",
                    frequency_axis=FrequencyAxisOptions(
                        freq_min=5_000.0,
                        freq_max=15_000.0,
                        ticks=(5_000.0, 10_000.0, 15_000.0),
                        labels=("5", "10", "15"),
                    ),
                ),
            },
        ),
        show=False,
    )

    fig = plt.gcf()
    top_axis = fig.axes[0]
    bottom_axis = fig.axes[1]

    assert result is None
    assert top_axis.get_title() == "Top Panel"
    assert top_axis.get_ylabel() == "Panel 1 Magnitude"
    assert bottom_axis.get_title() == "Bottom Panel"
    assert bottom_axis.get_xlabel() == "Bottom Frequency(kHz)"
    assert [tick.get_text() for tick in bottom_axis.get_xticklabels()] == ["5", "10", "15"]


def test_plot_magnitude_rejects_invalid_panel_name(dummy_hrtf: DummyPlotHRTF) -> None:
    with pytest.raises(ValueError, match="panel accepts"):
        dummy_hrtf.plot_magnitude(
            [[0.0, 0.0], [30.0, 0.0]],
            unit="linear",
            ear="both",
            options=PlotOptions(
                panels={"invalid_panel": AxisOptions(title="Wrong")},
            ),
            show=False,
        )


def test_plot_magnitude_hides_unused_axis_for_three_positions(
    dummy_hrtf: DummyPlotHRTF,
) -> None:
    result = dummy_hrtf.plot_magnitude(
        [[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]],
        unit="linear",
        ear="both",
        x_axis="linear",
        freq_min=250.0,
        freq_max=20_000.0,
        show=False,
    )

<<<<<<< HEAD
    assert layout.layout == 3
    assert layout.get_axis("bottom_right").get_visible() is False
=======
    fig = plt.gcf()

    assert result is None
    assert len(fig.axes) == 4
    assert fig.axes[3].get_visible() is False
>>>>>>> dev
