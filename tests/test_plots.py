import os
from pathlib import Path

import matplotlib

if os.getenv("HRTFPYKIT_TEST_SHOW_PLOTS", "") != "1":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hrtfpykit.hrtf.hrtf import HRTF
from hrtfpykit.hrtf import load_hrtf
from hrtfpykit.plots.compare import (
    compare_absolute_ild,
    compare_absolute_itd,
    compare_amplitude,
    compare_ild_curve,
    compare_ild_difference,
    compare_itd_curve,
    compare_itd_difference,
    compare_lsd,
    compare_lsd_plane,
    compare_magnitude,
)
from hrtfpykit.plots.sh import (
    plot_sht_reconstruction_comparison,
    plot_sht_reconstruction_error,
)


FIXTURE_SOFA_PATH = Path(__file__).parent / "pp1_HRIRs_measured.sofa"
SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
if SOFA_PATH == "" and FIXTURE_SOFA_PATH.exists():
    SOFA_PATH = str(FIXTURE_SOFA_PATH)
SHOW_PLOTS = os.getenv("HRTFPYKIT_TEST_SHOW_PLOTS", "") == "1"
COMPARE_SOFA_PATHS = [
    path
    for path in os.getenv("HRTFPYKIT_TEST_COMPARE_SOFA_PATHS", "").split(os.pathsep)
    if path.strip() != ""
]
if len(COMPARE_SOFA_PATHS) == 0 and SOFA_PATH != "":
    COMPARE_SOFA_PATHS = [SOFA_PATH, SOFA_PATH]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


@pytest.fixture
def real_hrtf() -> HRTF:
    if SOFA_PATH == "" or not os.path.exists(SOFA_PATH):
        pytest.skip("Required local SOFA file is not available")
    return load_hrtf(SOFA_PATH)


@pytest.fixture
def comparison_hrtfs() -> list[HRTF]:
    if len(COMPARE_SOFA_PATHS) < 2:
        pytest.skip("Two real compare SOFA files must be passed with --compare-sofa-paths")

    missing_paths = [
        path
        for path in COMPARE_SOFA_PATHS[:2]
        if not os.path.exists(path)
    ]
    if len(missing_paths) > 0:
        pytest.skip(
            "Required compare SOFA files are not available: "
            + ", ".join(missing_paths)
        )

    return [
        load_hrtf(COMPARE_SOFA_PATHS[0]),
        load_hrtf(COMPARE_SOFA_PATHS[1]),
    ]


def assert_current_matplotlib_figure_has_axes() -> None:
    figure_numbers = plt.get_fignums()
    assert len(figure_numbers) >= 1
    figure = plt.figure(figure_numbers[-1])
    assert len(figure.axes) >= 1


def test_plot_magnitude_accepts_real_hrtf_file(real_hrtf: HRTF) -> None:
    result = real_hrtf.plot_magnitude(
        positions="front",
        ear="both",
        show=False,
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    assert result is None
    assert len(fig.axes) == 1
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Frequency(kHz)"
    assert ax.get_ylabel() == "Magnitude (dB)"


@pytest.mark.parametrize(
    ("name", "plot_call"),
    [
        (
            "plot_magnitude",
            lambda hrtf: hrtf.plot_magnitude(
                positions="front",
                ear="both",
                show=False,
            ),
        ),
        (
            "plot_amplitude",
            lambda hrtf: hrtf.plot_amplitude(
                positions="front",
                ear="both",
                show=False,
            ),
        ),
        (
            "plot_amplitude_and_magnitude",
            lambda hrtf: hrtf.plot_amplitude_and_magnitude(
                position="front",
                ear="both",
                show=False,
            ),
        ),
        (
            "plot_spectrum_plane",
            lambda hrtf: hrtf.plot_spectrum_plane(
                plane="horizontal",
                ear="left",
                show=False,
            ),
        ),
        (
            "plot_elevation_spectrum",
            lambda hrtf: hrtf.plot_elevation_spectrum(
                azimuth="front",
                ear="left",
                show=False,
            ),
        ),
        (
            "plot_itd_curve",
            lambda hrtf: hrtf.plot_itd_curve(
                show=False,
            ),
        ),
        (
            "plot_absolute_itd",
            lambda hrtf: hrtf.plot_absolute_itd(
                show=False,
            ),
        ),
        (
            "plot_ild_plane",
            lambda hrtf: hrtf.plot_ild_plane(
                plane="horizontal",
                show=False,
            ),
        ),
        (
            "plot_ild_curve",
            lambda hrtf: hrtf.plot_ild_curve(
                show=False,
            ),
        ),
        (
            "plot_absolute_ild",
            lambda hrtf: hrtf.plot_absolute_ild(
                show=False,
            ),
        ),
        (
            "plot_source_grid",
            lambda hrtf: hrtf.plot_source_grid(
                show=False,
            ),
        ),
        (
            "plot_plane_grid",
            lambda hrtf: hrtf.plot_plane_grid(
                plane=["horizontal", "median", "frontal"],
                show=False,
            ),
        ),
    ],
    ids=[
        "plot_magnitude",
        "plot_amplitude",
        "plot_amplitude_and_magnitude",
        "plot_spectrum_plane",
        "plot_elevation_spectrum",
        "plot_itd_curve",
        "plot_absolute_itd",
        "plot_ild_plane",
        "plot_ild_curve",
        "plot_absolute_ild",
        "plot_source_grid",
        "plot_plane_grid",
    ],
)
def test_real_hrtf_plot_methods_create_matplotlib_figures(
    real_hrtf: HRTF,
    name: str,
    plot_call,
) -> None:
    result = plot_call(real_hrtf)

    assert result is None
    assert_current_matplotlib_figure_has_axes()


@pytest.mark.parametrize(
    ("name", "plot_call"),
    [
        (
            "compare_magnitude",
            lambda hrtfs: compare_magnitude(
                hrtfs,
                positions="front",
                ear="left",
                show=False,
            ),
        ),
        (
            "compare_amplitude",
            lambda hrtfs: compare_amplitude(
                hrtfs,
                positions="front",
                ear="left",
                show=False,
            ),
        ),
        (
            "compare_absolute_itd",
            lambda hrtfs: compare_absolute_itd(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_absolute_ild",
            lambda hrtfs: compare_absolute_ild(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_itd_curve",
            lambda hrtfs: compare_itd_curve(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_ild_curve",
            lambda hrtfs: compare_ild_curve(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_itd_difference",
            lambda hrtfs: compare_itd_difference(
                hrtfs[0],
                hrtfs[1],
                method="maxiacce",
                show=False,
            ),
        ),
        (
            "compare_ild_difference",
            lambda hrtfs: compare_ild_difference(
                hrtfs[0],
                hrtfs[1],
                show=False,
            ),
        ),
        (
            "compare_lsd",
            lambda hrtfs: compare_lsd(
                hrtfs[0],
                hrtfs[1],
                ear="left",
                show=False,
            ),
        ),
        (
            "compare_lsd_plane",
            lambda hrtfs: compare_lsd_plane(
                hrtfs[0],
                hrtfs[1],
                plane="horizontal",
                ear="left",
                show=False,
            ),
        ),
    ],
    ids=[
        "compare_magnitude",
        "compare_amplitude",
        "compare_absolute_itd",
        "compare_absolute_ild",
        "compare_itd_curve",
        "compare_ild_curve",
        "compare_itd_difference",
        "compare_ild_difference",
        "compare_lsd",
        "compare_lsd_plane",
    ],
)
def test_real_hrtf_compare_plot_functions_create_matplotlib_figures(
    comparison_hrtfs: list[HRTF],
    name: str,
    plot_call,
) -> None:
    result = plot_call(comparison_hrtfs)

    assert result is None
    assert_current_matplotlib_figure_has_axes()


@pytest.mark.parametrize(
    ("name", "plot_call"),
    [
        (
            "plot_sht_reconstruction_comparison",
            lambda hrtf, reconstructed_magnitude: plot_sht_reconstruction_comparison(
                hrtf=hrtf,
                reconstructed_magnitude=reconstructed_magnitude,
                position="front",
                ear="left",
                show=False,
            ),
        ),
        (
            "plot_sht_reconstruction_error",
            lambda hrtf, reconstructed_magnitude: plot_sht_reconstruction_error(
                hrtf=hrtf,
                reconstructed_magnitude=reconstructed_magnitude,
                position="front",
                ear="left",
                show=False,
            ),
        ),
    ],
    ids=[
        "plot_sht_reconstruction_comparison",
        "plot_sht_reconstruction_error",
    ],
)
def test_real_hrtf_sh_plot_functions_create_matplotlib_figures(
    real_hrtf: HRTF,
    name: str,
    plot_call,
) -> None:
    reconstructed_magnitude = np.asarray(real_hrtf.TF.magnitude, dtype=float)
    result = plot_call(real_hrtf, reconstructed_magnitude)

    assert result is None
    assert_current_matplotlib_figure_has_axes()
