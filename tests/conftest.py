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
        "--image-path",
        action="store",
        default=os.getenv("HUTUBS_IMAGE_PATH", os.getenv("HUTUBS_IMAGE_ROOT", "")),
        help="Path to local image/video dataset root",
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
        help="Run the full HUTUBS test matrix (overrides default smoke mode)",
    )


def pytest_configure(config: pytest.Config) -> None:
    hutubs_root = str(config.getoption("hutubs_root") or os.getenv("HUTUBS_TEST_HUTUBS_ROOT", "")).strip()
    image_path = str(config.getoption("image_path") or os.getenv("HUTUBS_TEST_IMAGE_PATH", os.getenv("HUTUBS_IMAGE_PATH", os.getenv("HUTUBS_IMAGE_ROOT", "")))).strip()
    subjects = str(config.getoption("subjects") or os.getenv("HUTUBS_TEST_SUBJECT_LIMIT", "")).strip()
    full = bool(config.getoption("full"))
    if hutubs_root != "":
        os.environ["HUTUBS_TEST_HUTUBS_ROOT"] = hutubs_root
        os.environ["HUTUBS_ROOT"] = hutubs_root

    if image_path != "":
        os.environ["HUTUBS_TEST_IMAGE_PATH"] = image_path
        os.environ["HUTUBS_IMAGE_PATH"] = image_path
        os.environ["HUTUBS_IMAGE_ROOT"] = image_path

    if subjects != "":
        os.environ["HUTUBS_TEST_SUBJECT_LIMIT"] = subjects

    if full:
        os.environ["HUTUBS_TEST_FULL"] = "1"
    else:
        os.environ.pop("HUTUBS_TEST_FULL", None)
