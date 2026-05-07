from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import replace
from itertools import product
import io
import os
import re
from pathlib import Path

import numpy as np
import pytest

from hrtfpykit.datasets import SONICOM
from hrtfpykit.datasets.checksums import SONICOM_CHECKSUMS
from hrtfpykit.datasets.config import SONICOMConfig
from hrtfpykit.datasets.download import BaseDownload
from hrtfpykit.datasets.specs import (
    HRTFSpec,
    ILDSpec,
    ITDSpec,
    MeshSpec,
    MetadataSpec,
    SHSpec,
)
from hrtfpykit.datasets.specs_workflow import DatasetSpecWorkflow
from hrtfpykit.datasets.split import DatasetSplitPlanner


SONICOM_ROOT = os.getenv("SONICOM_TEST_ROOT") or os.getenv("SONICOM_ROOT")
RUN_SONICOM_DOWNLOAD_TESTS = os.getenv("SONICOM_TEST_DOWNLOAD", "").strip() == "1"
_RUN_FULL_SONICOM_TESTS = os.getenv("SONICOM_TEST_FULL", "").strip() == "1"
ALL_SONICOM_SUBJECT_IDS = tuple(
    subject_id
    for subject_id in SONICOMConfig.subject_ids
    if subject_id not in set(SONICOMConfig.excluded_subject_ids)
)
_SUBJECT_LIMIT_OPTION = os.getenv("SONICOM_TEST_SUBJECT_LIMIT", "").strip()
_SAFE_SUBJECT_LIMIT_ENV = _SUBJECT_LIMIT_OPTION if _SUBJECT_LIMIT_OPTION != "" else "3"
try:
    _SAFE_SUBJECT_LIMIT = int(_SAFE_SUBJECT_LIMIT_ENV or "3")
except ValueError:
    _SAFE_SUBJECT_LIMIT = 3
if _SAFE_SUBJECT_LIMIT < 1:
    _SAFE_SUBJECT_LIMIT = 1
if _SAFE_SUBJECT_LIMIT > len(ALL_SONICOM_SUBJECT_IDS):
    _SAFE_SUBJECT_LIMIT = len(ALL_SONICOM_SUBJECT_IDS)
_TEST_SUBJECT_IDS = tuple(ALL_SONICOM_SUBJECT_IDS[:_SAFE_SUBJECT_LIMIT])

_DEFAULT_HRTF_VARIANT = {
    "type": "measured",
    "sample_rate": 44100,
    "version": "FreeFieldComp",
}
_DEFAULT_MESH_VARIANT = {
    "type": "scanned",
    "version": "watertight",
}
_VARIANT_VALUES: tuple[dict[str, object], ...] = (_DEFAULT_HRTF_VARIANT,)
if _RUN_FULL_SONICOM_TESTS:
    _VARIANT_VALUES = (
        _DEFAULT_HRTF_VARIANT,
        {"type": "measured", "sample_rate": 44100, "version": "Windowed"},
        {"type": "synthetic", "sample_rate": 44100, "version": "generic"},
    )
_SPLIT_VALUES = ("all", "train", "validation", "test")
if not _RUN_FULL_SONICOM_TESTS:
    _SPLIT_VALUES = ("all",)


def _sort_subject_ids(subject_ids: Sequence[str]) -> list[str]:
    def _sort_key(value: str) -> tuple[int, str]:
        value_str = str(value)
        match = re.search(r"(\d+)$", value_str)
        if match is None:
            return (0, value_str.lower())
        return (int(match.group(1)), value_str.lower())

    return sorted(subject_ids, key=_sort_key)


def _path_exists(path: str | Path | None) -> bool:
    if path is None:
        return False
    candidate = Path(path).expanduser()
    return candidate.exists()


def _subject_numbers() -> dict[str, int]:
    return DatasetSplitPlanner.build_subject_number_map(
        DatasetSplitPlanner.sort_subject_ids(tuple(SONICOMConfig.subject_ids))
    )


def _format_hrtf_path(subject_id: str, variant: Mapping[str, object]) -> Path:
    hrtf_type = str(variant["type"])
    sample_rate = variant.get("sample_rate")
    version = variant.get("version")
    hrtf_type_config = SONICOMConfig.hrtf.types[hrtf_type]
    sample_rate_label = None if sample_rate is None else str(sample_rate)
    if hrtf_type_config.sample_rate_labels is not None and sample_rate is not None:
        sample_rate_label = hrtf_type_config.sample_rate_labels.get(sample_rate, sample_rate_label)
    version_label = None if version is None else str(version)
    if hrtf_type_config.version_labels is not None and version is not None:
        version_label = hrtf_type_config.version_labels.get(str(version), version_label)
    relative_path = hrtf_type_config.path_pattern.format(
        subject_id=subject_id,
        subject_number=_subject_numbers()[subject_id],
        type=hrtf_type,
        hrtf_type=hrtf_type,
        sample_rate=sample_rate,
        hrtf_sample_rate=sample_rate,
        sample_rate_label=sample_rate_label,
        version=version,
        hrtf_version=version,
        version_label=version_label,
        hrtf_version_label=version_label,
        variant=hrtf_type,
    )
    return Path(SONICOM_ROOT).expanduser() / relative_path


def _format_mesh_path(subject_id: str, variant: Mapping[str, object]) -> Path:
    mesh_type = str(variant["type"])
    version = variant.get("version")
    mesh_type_config = SONICOMConfig.mesh.types[mesh_type]
    version_label = None if version is None else str(version)
    if mesh_type_config.version_labels is not None and version is not None:
        version_label = mesh_type_config.version_labels.get(str(version), version_label)
    relative_path = mesh_type_config.path_pattern.format(
        subject_id=subject_id,
        subject_number=_subject_numbers()[subject_id],
        type=mesh_type,
        mesh_type=mesh_type,
        version=version,
        mesh_version=version,
        version_label=version_label,
        mesh_version_label=version_label,
    )
    return Path(SONICOM_ROOT).expanduser() / relative_path


def _metadata_path() -> Path:
    return Path(SONICOM_ROOT).expanduser() / SONICOMConfig.metadata.path


def _requires_acoustic_specs(specs: Sequence[object]) -> bool:
    return any(isinstance(spec, (HRTFSpec, ITDSpec, ILDSpec, SHSpec)) for spec in specs)


def _requires_metadata_specs(specs: Sequence[object]) -> bool:
    return any(isinstance(spec, MetadataSpec) for spec in specs)


def _requires_mesh_specs(specs: Sequence[object]) -> bool:
    return any(isinstance(spec, MeshSpec) for spec in specs)


def _subjects_with_hrtf(variant: Mapping[str, object]) -> set[str]:
    if not _path_exists(SONICOM_ROOT):
        return set()
    return {
        subject_id
        for subject_id in _TEST_SUBJECT_IDS
        if _format_hrtf_path(subject_id, variant).is_file()
    }


def _subjects_with_mesh(variant: Mapping[str, object]) -> set[str]:
    if not _path_exists(SONICOM_ROOT):
        return set()
    return {
        subject_id
        for subject_id in _TEST_SUBJECT_IDS
        if _format_mesh_path(subject_id, variant).is_file()
    }


def _selected_subject_ids(
    inputs: Sequence[object],
    target: Sequence[object],
    dataset_hrtf_variant: Mapping[str, object],
    dataset_mesh_variant: Mapping[str, object] = _DEFAULT_MESH_VARIANT,
) -> tuple[str, ...]:
    selected = set(_TEST_SUBJECT_IDS)
    specs = tuple(inputs) + tuple(target)
    if _requires_acoustic_specs(specs):
        selected = selected.intersection(_subjects_with_hrtf(dataset_hrtf_variant))
    if _requires_metadata_specs(specs):
        selected = selected if _metadata_path().is_file() else set()
    if _requires_mesh_specs(specs):
        selected = selected.intersection(_subjects_with_mesh(dataset_mesh_variant))
    return tuple(_sort_subject_ids(selected))


def _paths_available(
    inputs: Sequence[object],
    target: Sequence[object],
    dataset_hrtf_variant: Mapping[str, object],
) -> bool:
    if not _path_exists(SONICOM_ROOT):
        return False
    return len(_selected_subject_ids(inputs, target, dataset_hrtf_variant)) > 0


def _spec_key_names(specs: Sequence[object]) -> tuple[str, ...]:
    return tuple(DatasetSpecWorkflow.get_spec_name(spec) for spec in specs)


def _identity(value: object) -> object:
    return value


def _to_array(value: object) -> np.ndarray:
    return np.asarray(value)


def _filename_only(value: object) -> object:
    return value if not isinstance(value, str) else str(Path(value).name)


def _bad_hrtf_transform(value: object) -> np.ndarray:
    return np.asarray(value)


def _hrtf_transform_name(transform: Callable | None) -> str:
    if transform is None:
        return "none"
    if transform is _identity:
        return "identity"
    if transform is _bad_hrtf_transform:
        return "bad"
    return transform.__name__


_ALL_COMBINATIONS: list[tuple[tuple[object, ...], tuple[object, ...]]] = [
    ((HRTFSpec(index_by=("subject",)),), ()),
    ((HRTFSpec(index_by=("subject", "position"), position_index=True, position_one_hot=True),), ()),
    (
        (
            HRTFSpec(
                index_by=("subject", "frequency"),
                domain="frequency",
                signal="tf_magnitude",
                frequency_index=True,
                frequency_one_hot=True,
            ),
        ),
        (),
    ),
    (
        (
            HRTFSpec(
                index_by=("subject", "position"),
                positions="all",
                plane=("horizontal", 0, "degrees"),
                position_index=True,
                position_one_hot=True,
                transform=_identity,
            ),
        ),
        (),
    ),
    (
        (
            HRTFSpec(
                index_by=("subject", "frequency", "position"),
                domain="frequency",
                signal="tf_magnitude_db",
                positions="all",
                plane=("median", 0.0),
                frequency_index=True,
                transform=_identity,
            ),
        ),
        (),
    ),
    ((HRTFSpec(index_by=("subject", "samples"), sample_index=True, sample_one_hot=True),), ()),
    (
        (
            HRTFSpec(
                index_by=("subject", "ear", "position"),
                ears=("left", "right"),
                positions=(0, 4, 8, 12),
                ear_index=True,
                ear_one_hot=True,
                position_index=True,
                position_one_hot=True,
            ),
        ),
        (HRTFSpec(index_by=("subject", "ear", "position"), ears=("left", "right"), positions=(0, 4, 8, 12)),),
    ),
    ((ITDSpec(index_by=("subject", "position"), positions="all", plane=("horizontal", 0.0), transform=_to_array),), ()),
    (
        (
            ILDSpec(
                index_by=("subject", "frequency", "position"),
                mode="frequency-dependent",
                positions="all",
                plane=("frontal", 90.0),
                frequency_index=True,
                frequency_one_hot=True,
                transform=_to_array,
            ),
        ),
        (),
    ),
    ((SHSpec(sh_order=3, index_by=("subject",), transform=_to_array),), ()),
    ((SHSpec(sh_order=2, index_by=("subject", "ear"), ears=("left",), ear_index=True, ear_one_hot=True, transform=_to_array),), ()),
    ((ITDSpec(index_by=("subject",), plane=("horizontal", 0, "degrees"), output="samples", transform=_to_array),), ()),
    ((ILDSpec(index_by=("subject",), mode="broad-band", output="db", transform=_to_array),), ()),
    ((MetadataSpec(transform=_to_array), MeshSpec(transform=_filename_only)), ()),
    ((HRTFSpec(index_by=("subject",), transform=_identity),), (ITDSpec(index_by=("subject",), transform=_to_array),)),
    ((HRTFSpec(index_by=("subject",), name="hrtf_input"),), (HRTFSpec(index_by=("subject",), name="hrtf_target"),)),
    ((ILDSpec(index_by=("subject",), name="ild_input"),), (ILDSpec(index_by=("subject",), name="ild_target"),)),
    ((MetadataSpec(),), (MeshSpec(),)),
    (
        (
            HRTFSpec(
                index_by=("subject", "position"),
                domain="frequency",
                signal="tf_real",
                transform=_identity,
                positions=(0, 1, 2),
                position_index=True,
                position_one_hot=True,
            ),
            ITDSpec(index_by=("subject", "position"), positions=(0, 1, 2), position_index=True, transform=_to_array),
        ),
        (ILDSpec(index_by=("subject", "position"), mode="broad-band", positions=(0, 1, 2), position_index=True, transform=_to_array),),
    ),
]

_COMBINATION_IDS = [
    "sub",
    "sub_pos",
    "sub_freq",
    "sub_pos_plane",
    "sub_freqplane",
    "sub_samples",
    "sub_ear_pos",
    "itd_planesel",
    "ild_planesel",
    "sh",
    "sh_ear_left",
    "itd_plain",
    "ild_plain",
    "metadata_mesh_transformed",
    "input_target_itd",
    "same_spec_input_target",
    "same_spec_ild_input_target",
    "metadata_input_mesh_target",
    "audio_multiinput_target",
]

COMBINATIONS = _ALL_COMBINATIONS if _RUN_FULL_SONICOM_TESTS else _ALL_COMBINATIONS[:1]
if not _RUN_FULL_SONICOM_TESTS:
    _COMBINATION_IDS = _COMBINATION_IDS[:1]

HRTF_INDEX_BY_GRID = (
    ("subject",),
    ("subject", "position"),
    ("subject", "ear", "position"),
    ("subject", "samples"),
    ("subject", "frequency"),
    ("position",),
    ("subject", "frequency", "position"),
)
HRTF_POSITION_GRID: tuple[object, ...] = ("all", (0, 1, 2))
HRTF_PLANE_GRID: tuple[object, ...] = (None, ("horizontal", 0.0), ("frontal", 90.0), "invalid-plane")
HRTF_TRANSFORM_GRID: tuple[Callable | None, ...] = (None, _identity, _bad_hrtf_transform)


def _hrtf_grid_expected_failure(
    index_by: tuple[str, ...],
    positions: object,
    plane: object,
    transform: Callable | None,
) -> tuple[str | None, str | None]:
    if len(index_by) == 0 or index_by[0] != "subject":
        return "index_by must start with 'subject'", None
    if "frequency" in index_by:
        return "unsupported axes", None
    if plane is not None and (not isinstance(positions, str) or positions != "all"):
        return None, "plane selection cannot be combined with custom positions"
    if isinstance(plane, str) and plane.strip().lower() not in {"horizontal", "median", "frontal"}:
        return None, "plane selection must be|plane must be"
    if isinstance(plane, tuple) and (len(plane) not in {2, 3} or not isinstance(plane[0], str)):
        return None, "plane selection must be|plane must be"
    if transform is _bad_hrtf_transform:
        return None, None
    return None, None


HRTF_GRID_CASES = [
    pytest.param(
        index_by,
        positions,
        plane,
        transform,
        *_hrtf_grid_expected_failure(index_by, positions, plane, transform),
        id=f"index_by={'-'.join(index_by)}|pos={positions}|plane={plane}|transform={_hrtf_transform_name(transform)}",
    )
    for index_by, positions, plane, transform in product(
        HRTF_INDEX_BY_GRID,
        HRTF_POSITION_GRID,
        HRTF_PLANE_GRID,
        HRTF_TRANSFORM_GRID,
    )
]

if not _RUN_FULL_SONICOM_TESTS:
    HRTF_GRID_CASES = [
        pytest.param(("subject",), "all", None, None, None, None, id="smoke-index-by-subject"),
        pytest.param(("position",), "all", None, None, *_hrtf_grid_expected_failure(("position",), "all", None, None), id="smoke-bad-index-by-position"),
        pytest.param(("subject", "frequency"), "all", None, None, *_hrtf_grid_expected_failure(("subject", "frequency"), "all", None, None), id="smoke-bad-unsupported-axis"),
        pytest.param(("subject", "position"), (0, 1, 2), ("horizontal", 0.0), None, *_hrtf_grid_expected_failure(("subject", "position"), (0, 1, 2), ("horizontal", 0.0), None), id="smoke-bad-plane-with-positions"),
        pytest.param(("subject",), "all", None, _bad_hrtf_transform, *_hrtf_grid_expected_failure(("subject",), "all", None, _bad_hrtf_transform), id="smoke-bad-transform"),
    ]


def _build_dataset(
    dataset_hrtf_variant: Mapping[str, object],
    split: str,
    inputs: tuple[object, ...],
    target: tuple[object, ...],
) -> SONICOM:
    if not _paths_available(inputs, target, dataset_hrtf_variant):
        pytest.skip(reason="Required local SONICOM resources are not available")

    with redirect_stdout(io.StringIO()):
        selected_subject_ids = _selected_subject_ids(inputs, target, dataset_hrtf_variant)
        if len(selected_subject_ids) == 0:
            pytest.skip(reason="No dataset subjects matched requested specs and local paths")
        excluded_subject_ids = tuple(
            subject_id
            for subject_id in SONICOMConfig.subject_ids
            if subject_id not in set(selected_subject_ids)
        )
        return SONICOM(
            root=SONICOM_ROOT,
            dataset_hrtf_variant=dataset_hrtf_variant,
            dataset_mesh_variant=_DEFAULT_MESH_VARIANT,
            inputs=inputs,
            target=target,
            split=split,
            split_ratio=(0.8, 0.1, 0.1),
            split_seed=0,
            exclude_subject_ids=excluded_subject_ids,
            verbose=False,
        )


def _expected_available_subjects(
    inputs: Sequence[object],
    target: Sequence[object],
    dataset_hrtf_variant: Mapping[str, object],
) -> list[str]:
    return list(_selected_subject_ids(inputs, target, dataset_hrtf_variant))


def _expected_selected_subjects(
    inputs: Sequence[object],
    target: Sequence[object],
    dataset_hrtf_variant: Mapping[str, object],
    split: str,
) -> list[str]:
    return list(
        DatasetSplitPlanner.split_subject_ids(
            _selected_subject_ids(inputs, target, dataset_hrtf_variant),
            split=split,
            split_ratio=(0.8, 0.1, 0.1),
            split_seed=0,
        )
    )


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
    jobs = BaseDownload(config=SONICOMConfig, root=tmp_path).build_download_plan(download_resources="metadata")

    assert len(jobs) == 1
    assert jobs[0]["resource"] == "metadata"
    assert jobs[0]["relative_path"] == "metadata_and_readme/metadata.csv"
    assert jobs[0]["checksum"] == SONICOM_CHECKSUMS["metadata"]["metadata_and_readme/metadata.csv"]


def test_sonicom_default_windowed_hrtf_download_plan(tmp_path: Path) -> None:
    jobs = BaseDownload(config=SONICOMConfig, root=tmp_path).build_download_plan(
        download_resources="hrtf",
        download_hrtf_variant={"type": "measured", "sample_rate": 44100, "version": "Windowed"},
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


def test_sonicom_missing_checksum_fails_download_plan(tmp_path: Path) -> None:
    config = replace(
        SONICOMConfig(),
        download=replace(
            SONICOMConfig.download,
            checksums={"metadata": {}},
        ),
    )

    with pytest.raises(ValueError, match="missing a checksum"):
        BaseDownload(config=config, root=tmp_path).build_download_plan(download_resources="metadata")


def test_sonicom_checksum_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("bad-data")
    downloader = BaseDownload(config=SONICOMConfig, root=tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader.verify_checksum(path, "0" * 64)


@pytest.mark.parametrize(
    "index_by,positions,plane,transform,expected_workflow_error,expected_dataset_error",
    HRTF_GRID_CASES,
)
def test_sonicom_hrtf_spec_grid(
    index_by: tuple[str, ...],
    positions: object,
    plane: object,
    transform: Callable | None,
    expected_workflow_error: str | None,
    expected_dataset_error: str | None,
) -> None:
    if not _path_exists(SONICOM_ROOT):
        pytest.skip(reason="Required local SONICOM dataset is not available")

    spec_kwargs = {}
    if "position" in index_by:
        spec_kwargs["position_index"] = True
        spec_kwargs["position_one_hot"] = True
    if "ear" in index_by:
        spec_kwargs["ear_index"] = True
        spec_kwargs["ear_one_hot"] = True
    if "samples" in index_by:
        spec_kwargs["sample_index"] = True
        spec_kwargs["sample_one_hot"] = True

    spec = HRTFSpec(
        index_by=index_by,
        positions=positions,
        plane=plane,
        transform=transform,
        **spec_kwargs,
    )

    if expected_workflow_error is not None:
        with pytest.raises((ValueError, TypeError), match=expected_workflow_error):
            DatasetSpecWorkflow.build(config=SONICOMConfig, inputs=(spec,), target=())
        return

    if expected_dataset_error is not None:
        with pytest.raises((ValueError, TypeError), match=expected_dataset_error):
            _build_dataset(_DEFAULT_HRTF_VARIANT, "all", (spec,), ())
        return

    if transform is _bad_hrtf_transform:
        dataset = _build_dataset(_DEFAULT_HRTF_VARIANT, "all", (spec,), ())
        with pytest.raises(AttributeError, match="IR"):
            _ = dataset[0]
        return

    dataset = _build_dataset(_DEFAULT_HRTF_VARIANT, "all", (spec,), ())
    sample = dataset[0]
    assert sample["inputs"] is not None
    hrtf_value = sample["inputs"]["hrtf"]
    assert isinstance(hrtf_value, np.ndarray)
    assert hrtf_value.size > 0


@pytest.mark.parametrize("split", _SPLIT_VALUES)
@pytest.mark.parametrize("dataset_hrtf_variant", _VARIANT_VALUES)
@pytest.mark.parametrize("specs", COMBINATIONS, ids=_COMBINATION_IDS)
def test_sonicom_real_dataset_all_combinations(
    dataset_hrtf_variant: Mapping[str, object],
    split: str,
    specs: tuple[tuple[object, ...], tuple[object, ...]],
) -> None:
    inputs, target = specs
    if not _paths_available(inputs, target, dataset_hrtf_variant):
        pytest.skip(reason="Required local SONICOM resources are not available")

    try:
        dataset = _build_dataset(dataset_hrtf_variant, split, inputs, target)
    except ValueError as exc:
        if "Split" in str(exc) and "produced an empty dataset" in str(exc):
            pytest.skip(reason=str(exc))
        raise

    assert dataset.dataset_hrtf_variant == dict(dataset_hrtf_variant)
    assert dataset.available_subjects == _expected_available_subjects(inputs, target, dataset_hrtf_variant)
    assert dataset.selected_subjects == _expected_selected_subjects(inputs, target, dataset_hrtf_variant, split)
    assert len(dataset) > 0

    uses_acoustic = _requires_acoustic_specs(inputs + target)
    if uses_acoustic:
        assert isinstance(dataset.sample_rate, float)
        assert dataset.positions is not None
    else:
        assert dataset.sample_rate is None
        assert dataset.positions is None

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "inputs" in sample
    assert "target" in sample

    expected_input_keys = _spec_key_names(inputs)
    expected_target_keys = _spec_key_names(target)
    if len(inputs) == 0:
        assert sample["inputs"] is None
    else:
        assert isinstance(sample["inputs"], dict)
        assert set(expected_input_keys).issubset(set(sample["inputs"].keys()))
        for value in sample["inputs"].values():
            assert value is not None
            if isinstance(value, np.ndarray):
                assert value.size > 0
    if len(target) == 0:
        assert sample["target"] is None
    else:
        assert isinstance(sample["target"], dict)
        assert set(sample["target"].keys()) == set(expected_target_keys)
        for value in sample["target"].values():
            assert value is not None
            if isinstance(value, np.ndarray):
                assert value.size > 0


def test_sonicom_get_subject_hrtf_uses_selected_subject() -> None:
    inputs = (HRTFSpec(index_by=("subject", "position")),)
    if not _paths_available(inputs, (), _DEFAULT_HRTF_VARIANT):
        pytest.skip(reason="Required local SONICOM resources are not available")

    dataset = _build_dataset(_DEFAULT_HRTF_VARIANT, "all", inputs, ())
    expected_subject = dataset.selected_subjects[0]
    hrtf = dataset.get_subject_hrtf(expected_subject)
    assert hrtf.IR.values.size > 0
    assert hrtf.Sources is not None


def test_sonicom_index_by_error_messages_are_actionable() -> None:
    if not _path_exists(SONICOM_ROOT):
        pytest.skip(reason="Required local SONICOM dataset is not available")

    with pytest.raises(
        ValueError,
        match=r"ILDSpec index_by=\('subject', 'ear'\).*Supported index_by combinations for ILDSpec: \('subject',\), \('subject', 'position'\)",
    ):
        _build_dataset(_DEFAULT_HRTF_VARIANT, "all", (ILDSpec(index_by=("subject", "ear"), mode="broad-band"),), ())

    with pytest.raises(
        ValueError,
        match=r"SHSpec.ear_one_hot requires index_by to include 'ear'.*Supported index_by combinations for SHSpec:",
    ):
        _build_dataset(_DEFAULT_HRTF_VARIANT, "all", (SHSpec(sh_order=2, index_by=("subject", "frequency"), ear_one_hot=True),), ())


def test_sonicom_len_matches_subject_count_for_ear_indexed_rows() -> None:
    inputs = (HRTFSpec(index_by=("subject", "ear"), ear_one_hot=True),)
    if not _paths_available(inputs, (), _DEFAULT_HRTF_VARIANT):
        pytest.skip(reason="Required local SONICOM resources are not available")

    dataset = _build_dataset(_DEFAULT_HRTF_VARIANT, "all", inputs, ())
    assert len(dataset) == len(dataset.selected_subjects) * 2


def test_sonicom_summary_reports_available_and_selected_subjects() -> None:
    inputs = (HRTFSpec(index_by=("subject",)),)
    if not _paths_available(inputs, (), _DEFAULT_HRTF_VARIANT):
        pytest.skip(reason="Required local SONICOM resources are not available")

    dataset = _build_dataset(_DEFAULT_HRTF_VARIANT, "train", inputs, ())
    summary = dataset.dataset_summary()
    assert f"available_subjects: {len(dataset.available_subjects)}" in summary
    assert f"selected_subjects: {len(dataset.selected_subjects)}" in summary
    assert "hrtf_variant: type=measured, sample_rate=44100, version=FreeFieldComp" in summary
    assert len(dataset.available_subjects) >= len(dataset.selected_subjects)


def test_sonicom_constructor_verbose_false_is_quiet() -> None:
    inputs = (HRTFSpec(index_by=("subject",)),)
    if not _paths_available(inputs, (), _DEFAULT_HRTF_VARIANT):
        pytest.skip(reason="Required local SONICOM resources are not available")

    output = io.StringIO()
    with redirect_stdout(output):
        dataset = _build_dataset(_DEFAULT_HRTF_VARIANT, "all", inputs, ())

    assert output.getvalue() == ""
    assert dataset.resources_summary() != ""
    assert dataset.dataset_summary() != ""


def test_sonicom_invalid_variant_keys_are_rejected() -> None:
    if not _path_exists(SONICOM_ROOT):
        pytest.skip(reason="Required local SONICOM dataset is not available")

    with pytest.raises(ValueError, match="Unsupported dataset_hrtf_variant keys"):
        SONICOM(root=SONICOM_ROOT, dataset_hrtf_variant={"type": "measured", "bad": "value"}, inputs=None, target=None)

    with pytest.raises(ValueError, match="Unsupported dataset_mesh_variant keys"):
        SONICOM(root=SONICOM_ROOT, dataset_mesh_variant={"type": "scanned", "bad": "value"}, inputs=None, target=None)


@pytest.mark.skipif(
    not RUN_SONICOM_DOWNLOAD_TESTS,
    reason="Set SONICOM_TEST_DOWNLOAD=1 or pass --sonicom-download to run network download tests",
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
