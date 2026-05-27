from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import replace
import io
import os
import re
from pathlib import Path
from typing import cast

import pytest

from hrtfpykit.datasets import ARI
from hrtfpykit.datasets.checksums import ARI_CHECKSUMS
from hrtfpykit.datasets.config import ARIConfig, DownloadConfig
from hrtfpykit.datasets.download import BaseDownload
from hrtfpykit.datasets.specs import AnthropometrySpec, MetadataSpec
from hrtfpykit.datasets.specs_workflow import DatasetSpecWorkflow


ARI_ROOT = os.getenv("ARI_TEST_ROOT") or os.getenv("ARI_ROOT")
RUN_ARI_DOWNLOAD_TESTS = os.getenv("ARI_TEST_DOWNLOAD", "").strip() == "1"
ALL_ARI_SUBJECT_IDS = tuple(ARIConfig.subject_ids)
_SUBJECT_LIMIT_OPTION = os.getenv("ARI_TEST_SUBJECT_LIMIT", "").strip()
_SAFE_SUBJECT_LIMIT_ENV = _SUBJECT_LIMIT_OPTION if _SUBJECT_LIMIT_OPTION != "" else "3"
try:
    _SAFE_SUBJECT_LIMIT = int(_SAFE_SUBJECT_LIMIT_ENV or "3")
except ValueError:
    _SAFE_SUBJECT_LIMIT = 3
if _SAFE_SUBJECT_LIMIT < 1:
    _SAFE_SUBJECT_LIMIT = 1
if _SAFE_SUBJECT_LIMIT > len(ALL_ARI_SUBJECT_IDS):
    _SAFE_SUBJECT_LIMIT = len(ALL_ARI_SUBJECT_IDS)
_TEST_SUBJECT_IDS = tuple(ALL_ARI_SUBJECT_IDS[:_SAFE_SUBJECT_LIMIT])
_TEST_SUBJECT_ID_SET = set(_TEST_SUBJECT_IDS)
_DOWNLOAD_EXCLUDED_SUBJECT_IDS = tuple(
    subject_id
    for subject_id in ALL_ARI_SUBJECT_IDS
    if subject_id not in _TEST_SUBJECT_ID_SET
)


def _sort_subject_ids(subject_ids: Sequence[str]) -> list[str]:
    def _sort_key(value: str) -> tuple[int, str]:
        value_str = str(value)
        match = re.search(r"(\d+)$", value_str)
        if match is None:
            return (0, value_str.lower())
        return (int(match.group(1)), value_str.lower())

    return sorted(subject_ids, key=_sort_key)


def _excluded_subject_ids_for(subject_ids: set[str]) -> tuple[str, ...]:
    return tuple(
        subject_id
        for subject_id in ALL_ARI_SUBJECT_IDS
        if subject_id not in subject_ids
    )


def _write_ari_anthropometry_csv(root: Path) -> None:
    (root / "anthro.csv").write_text(
        "\n".join(
            (
                "SubjectID,x1,x2,L_a1,L_d1,R_a1,R_d1",
                "nh2,1.0,2.0,3.0,4.0,5.0,6.0",
            )
        ),
        encoding="utf-8",
    )


def test_ari_config_subject_ids_are_valid() -> None:
    subject_ids = tuple(ARIConfig.subject_ids)

    assert ARIConfig.name == "ARI"
    assert len(subject_ids) == 263
    assert len(ARIConfig.hrtf_paths) == len(subject_ids)
    assert len(ARI_CHECKSUMS["hrtf"]) == len(subject_ids)
    assert len(set(subject_ids)) == len(subject_ids)
    assert all(isinstance(subject_id, str) for subject_id in subject_ids)
    assert all(subject_id.strip() != "" for subject_id in subject_ids)
    assert list(subject_ids) == _sort_subject_ids(subject_ids)


def test_ari_config_subject_exclusions() -> None:
    assert ARIConfig.excluded_subject_ids == ()


def test_ari_hrtf_download_plan(tmp_path: Path) -> None:
    jobs = BaseDownload(config=ARIConfig, root=tmp_path).build_download_plan(
        download_resources="hrtf",
        download_hrtf_variant=None,
    )

    relative_paths = {str(job["relative_path"]) for job in jobs}
    assert len(jobs) == len(ALL_ARI_SUBJECT_IDS)
    assert "hrtf b_nh2.sofa" in relative_paths
    assert "hrtf c_nh831.sofa" in relative_paths
    assert "hrtf d_nh1059.sofa" in relative_paths
    assert all(job["resource"] == "hrtf" for job in jobs)
    assert all(job["hrtf_variant"] == "hrtf" for job in jobs)
    assert all(job["checksum"] is not None for job in jobs)


def test_ari_download_plan_follows_subject_limit(tmp_path: Path) -> None:
    jobs = BaseDownload(
        config=ARIConfig,
        root=tmp_path,
        excluded_subject_ids=_DOWNLOAD_EXCLUDED_SUBJECT_IDS,
    ).build_download_plan(
        download_resources="all",
        download_hrtf_variant=None,
        download_mesh_variant=None,
    )
    subject_jobs = [job for job in jobs if job["subject_id"] is not None]
    root = tmp_path.resolve()

    assert {job["resource"] for job in jobs} == {"anthropometry", "metadata", "hrtf"}
    assert {job["subject_id"] for job in subject_jobs}.issubset(_TEST_SUBJECT_ID_SET)
    assert len(subject_jobs) == len(_TEST_SUBJECT_IDS)
    assert all(job["checksum"] is not None for job in jobs)
    assert all(str(job["url"]).startswith("https://") for job in jobs)
    assert all(Path(str(job["destination"])).resolve().is_relative_to(root) for job in jobs)


def test_ari_missing_checksum_fails_download_plan(tmp_path: Path) -> None:
    download_config = cast(DownloadConfig, ARIConfig.download)
    config = replace(
        ARIConfig(),
        download=replace(
            download_config,
            checksums={"hrtf": {}},
        ),
    )

    with pytest.raises(ValueError, match="missing"):
        BaseDownload(config=config, root=tmp_path).build_download_plan(
            download_resources="hrtf",
            download_hrtf_variant=None,
        )


def test_ari_checksum_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "anthro.csv"
    path.write_text("bad-data", encoding="utf-8")
    downloader = BaseDownload(config=ARIConfig, root=tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.verify_checksum(path, "0" * 64)


def test_ari_invalid_download_variant_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported download_hrtf_variant keys"):
        BaseDownload(config=ARIConfig, root=tmp_path).build_download_plan(
            download_resources="hrtf",
            download_hrtf_variant={"type": "hrtf", "bad": "value"},
        )

    with pytest.raises(ValueError, match=r"Unsupported download_hrtf_variant\['type'\]"):
        BaseDownload(config=ARIConfig, root=tmp_path).build_download_plan(
            download_resources="hrtf",
            download_hrtf_variant={"type": "bad"},
        )


def test_ari_spec_workflow_does_not_mutate_spec_objects() -> None:
    anthropometry_spec = AnthropometrySpec(
        accessed_by="ROW",
        grouped_by="subject-ear",
        ear="LEFT",
    )
    metadata_spec = MetadataSpec(accessed_by="ROW", name="subject_meta")

    DatasetSpecWorkflow.build(
        config=ARIConfig,
        inputs=(anthropometry_spec, metadata_spec),
        target=(),
    )

    assert anthropometry_spec.accessed_by == "ROW"
    assert anthropometry_spec.grouped_by == "subject-ear"
    assert anthropometry_spec.ear == "LEFT"
    assert metadata_spec.accessed_by == "ROW"
    assert metadata_spec.name == "subject_meta"


def test_ari_anthropometry_download_plan(tmp_path: Path) -> None:
    jobs = BaseDownload(config=ARIConfig, root=tmp_path).build_download_plan(
        download_resources="anthropometry",
    )

    assert len(jobs) == 1
    assert jobs[0]["resource"] == "anthropometry"
    assert jobs[0]["relative_path"] == "anthro.csv"
    assert jobs[0]["url"] == (
        "https://raw.githubusercontent.com/"
        "ArielAlvarez-Martinez/ari_anthropometry_and_metadata/v1.0/anthro.csv"
    )
    assert jobs[0]["checksum"] == ARI_CHECKSUMS["anthropometry"]["anthro.csv"]


def test_ari_metadata_download_plan(tmp_path: Path) -> None:
    jobs = BaseDownload(config=ARIConfig, root=tmp_path).build_download_plan(
        download_resources="metadata",
    )

    assert len(jobs) == 1
    assert jobs[0]["resource"] == "metadata"
    assert jobs[0]["relative_path"] == "metadata.csv"
    assert jobs[0]["url"] == (
        "https://raw.githubusercontent.com/"
        "ArielAlvarez-Martinez/ari_anthropometry_and_metadata/v1.0/metadata.csv"
    )
    assert jobs[0]["checksum"] == ARI_CHECKSUMS["metadata"]["metadata.csv"]


@pytest.mark.parametrize(
    ("ear", "expected_keys"),
    [
        ("left", {"x1", "x2", "L_a1", "L_d1"}),
        ("right", {"x1", "x2", "R_a1", "R_d1"}),
        ("both", {"x1", "x2", "L_a1", "L_d1", "R_a1", "R_d1"}),
        (None, {"x1", "x2", "L_a1", "L_d1", "R_a1", "R_d1"}),
    ],
)
def test_ari_anthropometry_ear_selection(
    tmp_path: Path,
    ear: str | None,
    expected_keys: set[str],
) -> None:
    _write_ari_anthropometry_csv(tmp_path)

    dataset = ARI(
        root=tmp_path,
        inputs=AnthropometrySpec(name="anthro", ear=ear),
        exclude_subject_ids=_excluded_subject_ids_for({"nh2"}),
    )
    sample = dataset[0]
    inputs = cast(dict[str, object], sample["inputs"])
    value = inputs["anthro"]

    assert isinstance(value, dict)
    assert set(value) == expected_keys


def test_ari_constructor_verbose_false_is_quiet(tmp_path: Path) -> None:
    _write_ari_anthropometry_csv(tmp_path)

    output = io.StringIO()
    with redirect_stdout(output):
        dataset = ARI(
            root=tmp_path,
            inputs=AnthropometrySpec(name="anthro"),
            exclude_subject_ids=_excluded_subject_ids_for({"nh2"}),
            verbose=False,
        )

    assert output.getvalue() == ""
    assert dataset.resources_summary() != ""
    assert dataset.dataset_summary() != ""


@pytest.mark.skipif(
    not RUN_ARI_DOWNLOAD_TESTS,
    reason="Set ARI_TEST_DOWNLOAD=1 to run ARI network download tests",
)
def test_ari_download_resources_follow_subject_limit(tmp_path: Path) -> None:
    download_root = Path(ARI_ROOT).expanduser() if ARI_ROOT else tmp_path
    download_cases = (
        ("anthropometry", {"anthropometry"}, 0),
        ("metadata", {"metadata"}, 0),
        ("hrtf", {"hrtf"}, len(_TEST_SUBJECT_IDS)),
        ("all", {"anthropometry", "metadata", "hrtf"}, len(_TEST_SUBJECT_IDS)),
    )

    for download_resources, expected_resources, expected_subject_jobs in download_cases:
        with redirect_stdout(io.StringIO()):
            dataset = ARI(
                root=download_root,
                download=True,
                download_resources=download_resources,
                exclude_subject_ids=_DOWNLOAD_EXCLUDED_SUBJECT_IDS,
                inputs=None,
                target=None,
                verbose=False,
            )

        jobs = BaseDownload(
            config=ARIConfig,
            root=download_root,
            excluded_subject_ids=_DOWNLOAD_EXCLUDED_SUBJECT_IDS,
        ).build_download_plan(
            download_resources=download_resources,
            download_hrtf_variant=None,
            download_mesh_variant=None,
        )
        subject_jobs = [job for job in jobs if job["subject_id"] is not None]

        assert {job["resource"] for job in jobs} == expected_resources
        assert {job["subject_id"] for job in subject_jobs}.issubset(_TEST_SUBJECT_ID_SET)
        assert len(subject_jobs) == expected_subject_jobs
        assert dataset.available_subjects == list(_TEST_SUBJECT_IDS)
        assert dataset.selected_subjects == list(_TEST_SUBJECT_IDS)
        assert all(Path(str(job["destination"])).is_file() for job in jobs)
