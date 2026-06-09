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
from hrtfpykit.plots import (
    compare_abs_bb_ild,
    compare_abs_itd,
    compare_amplitude,
    compare_signed_bb_ild,
    compare_signed_itd,
    plot_abs_bb_ild,
    plot_abs_bb_ild_diff,
    plot_abs_itd,
    plot_abs_itd_diff,
    plot_amplitude,
    plot_elevation_spectrum,
    plot_etc,
    plot_etc_plane,
    plot_lsd,
    plot_magnitude,
    plot_plane_grid,
    plot_signed_bb_ild,
    plot_signed_fd_ild,
    plot_signed_itd,
    plot_source_grid,
    plot_spectrum_plane,
    compare_magnitude,
)
from hrtfpykit.plots.sh import (
    sht_reconstruction_comparison,
    sht_reconstruction_error,
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
    plot_magnitude(
        real_hrtf,
        positions="front",
        ear="both",
        show=False,
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    assert len(fig.axes) == 1
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Frequency(kHz)"
    assert ax.get_ylabel() == "Magnitude (dB)"


@pytest.mark.parametrize(
    ("name", "plot_call"),
    [
        (
            "plot_magnitude",
            lambda hrtf: plot_magnitude(hrtf, 
                positions="front",
                ear="both",
                show=False,
            ),
        ),
        (
            "plot_amplitude",
            lambda hrtf: plot_amplitude(hrtf, 
                positions="front",
                ear="both",
                show=False,
            ),
        ),
        (
            "plot_etc",
            lambda hrtf: plot_etc(hrtf, 
                positions="front",
                ear="both",
                x_axis="samples",
                reference="max",
                show=False,
            ),
        ),
        (
            "plot_etc_plane",
            lambda hrtf: plot_etc_plane(hrtf, 
                plane="horizontal",
                ear="left",
                x_axis="samples",
                reference="max",
                show=False,
            ),
        ),
        (
            "plot_spectrum_plane",
            lambda hrtf: plot_spectrum_plane(hrtf, 
                plane="horizontal",
                ear="left",
                show=False,
            ),
        ),
        (
            "plot_elevation_spectrum",
            lambda hrtf: plot_elevation_spectrum(hrtf, 
                azimuth="front",
                ear="left",
                show=False,
            ),
        ),
        (
            "plot_signed_itd",
            lambda hrtf: plot_signed_itd(hrtf, 
                show=False,
            ),
        ),
        (
            "plot_abs_itd",
            lambda hrtf: plot_abs_itd(hrtf, 
                show=False,
            ),
        ),
        (
            "plot_signed_fd_ild",
            lambda hrtf: plot_signed_fd_ild(hrtf, 
                plane="horizontal",
                show=False,
            ),
        ),
        (
            "plot_signed_bb_ild",
            lambda hrtf: plot_signed_bb_ild(hrtf, 
                show=False,
            ),
        ),
        (
            "plot_abs_bb_ild",
            lambda hrtf: plot_abs_bb_ild(hrtf, 
                show=False,
            ),
        ),
        (
            "plot_source_grid",
            lambda hrtf: plot_source_grid(hrtf, 
                show=False,
            ),
        ),
        (
            "plot_plane_grid",
            lambda hrtf: plot_plane_grid(hrtf, 
                plane=["horizontal", "median", "frontal"],
                show=False,
            ),
        ),
    ],
    ids=[
        "plot_magnitude",
        "plot_amplitude",
        "plot_etc",
        "plot_etc_plane",
        "plot_spectrum_plane",
        "plot_elevation_spectrum",
        "plot_signed_itd",
        "plot_abs_itd",
        "plot_signed_fd_ild",
        "plot_signed_bb_ild",
        "plot_abs_bb_ild",
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
            "compare_abs_itd",
            lambda hrtfs: compare_abs_itd(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_abs_bb_ild",
            lambda hrtfs: compare_abs_bb_ild(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_signed_itd",
            lambda hrtfs: compare_signed_itd(
                hrtfs,
                show=False,
            ),
        ),
        (
            "compare_signed_bb_ild",
            lambda hrtfs: compare_signed_bb_ild(
                hrtfs,
                show=False,
            ),
        ),
        (
            "plot_abs_itd_diff",
            lambda hrtfs: plot_abs_itd_diff(
                hrtfs[0],
                hrtfs[1],
                method="maxiacce",
                show=False,
            ),
        ),
        (
            "plot_abs_bb_ild_diff",
            lambda hrtfs: plot_abs_bb_ild_diff(
                hrtfs[0],
                hrtfs[1],
                show=False,
            ),
        ),
        (
            "plot_lsd",
            lambda hrtfs: plot_lsd(
                hrtfs[0],
                hrtfs[1],
                ear="both",
                show=False,
            ),
        ),
    ],
    ids=[
        "compare_magnitude",
        "compare_amplitude",
        "compare_abs_itd",
        "compare_abs_bb_ild",
        "compare_signed_itd",
        "compare_signed_bb_ild",
        "plot_abs_itd_diff",
        "plot_abs_bb_ild_diff",
        "plot_lsd",
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
            "sht_reconstruction_comparison",
            lambda hrtf, reconstructed_magnitude: sht_reconstruction_comparison(
                hrtf=hrtf,
                reconstructed_magnitude=reconstructed_magnitude,
                position="front",
                ear="left",
                show=False,
            ),
        ),
        (
            "sht_reconstruction_error",
            lambda hrtf, reconstructed_magnitude: sht_reconstruction_error(
                hrtf=hrtf,
                reconstructed_magnitude=reconstructed_magnitude,
                position="front",
                ear="left",
                show=False,
            ),
        ),
    ],
    ids=[
        "sht_reconstruction_comparison",
        "sht_reconstruction_error",
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
