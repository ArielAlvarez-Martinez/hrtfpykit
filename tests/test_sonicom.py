import os
from pathlib import Path

import pytest

from hrtfpykit.datasets import SONICOM
from hrtfpykit.datasets.checksums import SONICOM_CHECKSUMS
from hrtfpykit.datasets.config import SONICOMConfig
from hrtfpykit.datasets.download import BaseDownload
from hrtfpykit.datasets.specs import MetadataSpec


SONICOM_ROOT = os.getenv("SONICOM_TEST_ROOT") or os.getenv("SONICOM_ROOT")
RUN_SONICOM_DOWNLOAD_TESTS = os.getenv("SONICOM_TEST_DOWNLOAD", "").strip() == "1"


def test_sonicom_config_subject_exclusions() -> None:
    assert len(SONICOMConfig.subject_ids) == 400
    assert SONICOMConfig.excluded_subject_ids == (
        "P0253",
        "P0258",
        "P0270",
        "P0272",
        "P0275",
        "P0396",
    )


def test_sonicom_metadata_download_plan(tmp_path: Path) -> None:
    jobs = BaseDownload(
        config=SONICOMConfig,
        root=tmp_path,
    ).build_download_plan(
        download_resources="metadata",
    )

    assert len(jobs) == 1
    assert jobs[0]["resource"] == "metadata"
    assert jobs[0]["relative_path"] == "metadata_and_readme/metadata.csv"
    assert jobs[0]["checksum"] == SONICOM_CHECKSUMS["metadata"]["metadata_and_readme/metadata.csv"]


def test_sonicom_default_windowed_hrtf_download_plan(tmp_path: Path) -> None:
    jobs = BaseDownload(
        config=SONICOMConfig,
        root=tmp_path,
    ).build_download_plan(
        download_resources="hrtf",
        download_hrtf_type="measured",
        download_hrtf_sample_rate=44100,
        download_hrtf_version="Windowed",
    )

    relative_paths = {str(job["relative_path"]) for job in jobs}
    assert len(jobs) == 394
    assert "P0001/HRTF/HRTF/44kHz/P0001_Windowed_44kHz.sofa" in relative_paths
    assert "P0253/HRTF/HRTF/44kHz/P0253_Windowed_44kHz.sofa" not in relative_paths
    assert all(job["checksum"] is not None for job in jobs)


def test_sonicom_windowed_checksums_cover_default_sample_rates() -> None:
    windowed = SONICOM_CHECKSUMS["hrtf"]["measured"]["Windowed"]

    assert set(windowed) == {44100, 48000, 96000}
    assert len(windowed[44100]) == 394
    assert len(windowed[48000]) == 394
    assert len(windowed[96000]) == 394


def test_sonicom_metadata_only_construction_requires_local_root() -> None:
    if SONICOM_ROOT is None or not Path(SONICOM_ROOT).expanduser().exists():
        pytest.skip(reason="SONICOM local dataset root is not available")

    dataset = SONICOM(
        root=SONICOM_ROOT,
        inputs=MetadataSpec(),
        target=None,
        verbose=False,
    )

    assert len(dataset.available_subjects) > 0
    assert dataset.inputs[0].name is None


@pytest.mark.skipif(
    not RUN_SONICOM_DOWNLOAD_TESTS,
    reason="Set SONICOM_TEST_DOWNLOAD=1 to run network download tests",
)
def test_sonicom_metadata_download(tmp_path: Path) -> None:
    dataset = SONICOM(
        root=tmp_path,
        download=True,
        download_resources="metadata",
        inputs=None,
        target=None,
        verbose=False,
    )

    assert (tmp_path / "metadata_and_readme" / "metadata.csv").is_file()
    assert len(dataset.available_subjects) == 394
