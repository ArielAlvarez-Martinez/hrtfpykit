from __future__ import annotations

import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hutubs-root",
        action="store",
        default=os.getenv("HUTUBS_ROOT", ""),
        help="Path to local HUTUBS dataset root",
    )
    parser.addoption(
        "--sonicom-root",
        action="store",
        default=os.getenv("SONICOM_ROOT", ""),
        help="Path to local SONICOM dataset root",
    )
    parser.addoption(
        "--image-path",
        action="store",
        default=os.getenv("HUTUBS_IMAGE_PATH", os.getenv("HUTUBS_IMAGE_ROOT", "")),
        help="Path to local image/video dataset root",
    )
    parser.addoption(
        "--sofa-path",
        action="store",
        default=os.getenv("HRTFPYKIT_TEST_SOFA_PATH", os.getenv("HRTFPYKIT_SOFA_PATH", "")),
        help="Path to a local SOFA/HRTF file used by HRTF, SOFA, and plot tests",
    )
    parser.addoption(
        "--compare-sofa-paths",
        action="store",
        nargs="+",
        default=None,
        help="Two or more SOFA/HRTF files used by compare plot tests",
    )
    parser.addoption(
        "--subjects",
        action="store",
        default=os.getenv("HUTUBS_TEST_SUBJECT_LIMIT", ""),
        help="Number of subjects to use for HUTUBS tests",
    )
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Run the full dataset test matrices (overrides default smoke mode)",
    )
    parser.addoption(
        "--sonicom-download",
        action="store_true",
        default=False,
        help="Run SONICOM network download tests",
    )
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="Show plots during plot tests",
    )
    parser.addoption(
        "--visual",
        action="store_true",
        default=False,
        help="Run curated visual plot checks using real SOFA files",
    )


def pytest_configure(config: pytest.Config) -> None:
    hutubs_root = str(config.getoption("hutubs_root") or os.getenv("HUTUBS_TEST_HUTUBS_ROOT", "")).strip()
    sonicom_root = str(config.getoption("sonicom_root") or os.getenv("SONICOM_TEST_ROOT", "")).strip()
    image_path = str(config.getoption("image_path") or os.getenv("HUTUBS_TEST_IMAGE_PATH", os.getenv("HUTUBS_IMAGE_PATH", os.getenv("HUTUBS_IMAGE_ROOT", "")))).strip()
    sofa_path = str(config.getoption("sofa_path") or os.getenv("HRTFPYKIT_TEST_SOFA_PATH", os.getenv("HRTFPYKIT_SOFA_PATH", ""))).strip()
    compare_sofa_paths_option = config.getoption("compare_sofa_paths")
    if compare_sofa_paths_option is None:
        compare_sofa_paths = os.getenv("HRTFPYKIT_TEST_COMPARE_SOFA_PATHS", "").strip()
    else:
        compare_sofa_paths = os.pathsep.join(
            str(path).strip()
            for path in compare_sofa_paths_option
            if str(path).strip() != ""
        )
    subjects = str(config.getoption("subjects") or os.getenv("HUTUBS_TEST_SUBJECT_LIMIT", "")).strip()
    full = bool(config.getoption("full"))
    sonicom_download = bool(config.getoption("sonicom_download"))
    show = bool(config.getoption("show"))
    visual = bool(config.getoption("visual"))
    if hutubs_root != "":
        os.environ["HUTUBS_TEST_HUTUBS_ROOT"] = hutubs_root
        os.environ["HUTUBS_ROOT"] = hutubs_root

    if sonicom_root != "":
        os.environ["SONICOM_TEST_ROOT"] = sonicom_root
        os.environ["SONICOM_ROOT"] = sonicom_root

    if image_path != "":
        os.environ["HUTUBS_TEST_IMAGE_PATH"] = image_path
        os.environ["HUTUBS_IMAGE_PATH"] = image_path
        os.environ["HUTUBS_IMAGE_ROOT"] = image_path

    if sofa_path != "":
        os.environ["HRTFPYKIT_TEST_SOFA_PATH"] = sofa_path
        os.environ["HRTFPYKIT_SOFA_PATH"] = sofa_path

    if compare_sofa_paths != "":
        os.environ["HRTFPYKIT_TEST_COMPARE_SOFA_PATHS"] = compare_sofa_paths

    if subjects != "":
        os.environ["HUTUBS_TEST_SUBJECT_LIMIT"] = subjects
        os.environ["SONICOM_TEST_SUBJECT_LIMIT"] = subjects

    if full:
        os.environ["HUTUBS_TEST_FULL"] = "1"
        os.environ["SONICOM_TEST_FULL"] = "1"
    else:
        os.environ.pop("HUTUBS_TEST_FULL", None)
        os.environ.pop("SONICOM_TEST_FULL", None)

    if sonicom_download:
        os.environ["SONICOM_TEST_DOWNLOAD"] = "1"

    if show or visual:
        os.environ["HRTFPYKIT_TEST_SHOW_PLOTS"] = "1"
    else:
        os.environ.pop("HRTFPYKIT_TEST_SHOW_PLOTS", None)

    if visual:
        os.environ["HRTFPYKIT_TEST_VISUAL_PLOTS"] = "1"
    else:
        os.environ.pop("HRTFPYKIT_TEST_VISUAL_PLOTS", None)
