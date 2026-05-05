from collections.abc import Callable, Sequence
import io
import os
import re
from contextlib import redirect_stdout
from pathlib import Path
from itertools import product

import numpy as np
import pytest

from hrtfpykit.datasets import HUTUBS
from hrtfpykit.datasets.config import HUTUBSConfig
from hrtfpykit.datasets.specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ITDSpec,
    ILDSpec,
    MeshSpec,
    VideoSpec,
    SHSpec,
)
from hrtfpykit.datasets.specs_workflow import DatasetSpecWorkflow
from hrtfpykit.datasets.split import DatasetSubjectSplitPlanner


HUTUBS_ROOT = os.getenv("HUTUBS_TEST_HUTUBS_ROOT") or os.getenv("HUTUBS_ROOT")
IMAGE_ROOT = (
    os.getenv("HUTUBS_TEST_IMAGE_PATH")
    or os.getenv("HUTUBS_IMAGE_PATH")
    or os.getenv("HUTUBS_IMAGE_ROOT")
)
ALL_HUTUBS_SUBJECT_IDS = tuple(HUTUBSConfig.subject_ids)
_RUN_FULL_HUTUBS_TESTS = (
    os.getenv("HUTUBS_TEST_FULL", "").strip() == "1"
)
_SUBJECT_LIMIT_OPTION = os.getenv("HUTUBS_TEST_SUBJECT_LIMIT", "").strip()
_SAFE_SUBJECT_LIMIT_ENV = _SUBJECT_LIMIT_OPTION if _SUBJECT_LIMIT_OPTION != "" else "3"
try:
    _SAFE_SUBJECT_LIMIT = int(_SAFE_SUBJECT_LIMIT_ENV or "3")
except ValueError:
    _SAFE_SUBJECT_LIMIT = 3
if _SAFE_SUBJECT_LIMIT < 1:
    _SAFE_SUBJECT_LIMIT = 1
if _SAFE_SUBJECT_LIMIT > len(ALL_HUTUBS_SUBJECT_IDS):
    _SAFE_SUBJECT_LIMIT = len(ALL_HUTUBS_SUBJECT_IDS)
_TEST_SUBJECT_IDS = tuple(ALL_HUTUBS_SUBJECT_IDS[:_SAFE_SUBJECT_LIMIT])


def _sort_subject_ids(subject_ids: Sequence[str]) -> list[str]:
    def _sort_key(value: str) -> tuple[int, str]:
        value_str = str(value)
        match = re.search(r"(\d+)$", value_str)
        if match is None:
            return (0, value_str.lower())
        return (int(match.group(1)), value_str.lower())

    return sorted(subject_ids, key=_sort_key)


def _normalize_media_subject_id(name: str) -> str | None:
    subject_name = str(name).strip().lower()
    if subject_name == "":
        return None

    match = re.search(r"(\d+)$", subject_name)
    if match is None:
        return None

    normalized = match.group(1).lstrip("0")
    if normalized == "":
        normalized = "0"
    return f"pp{int(normalized)}"


def _collect_available_media_subject_ids(path: str | Path | None) -> set[str]:
    if path is None or path == "":
        return set()
    candidate_root = Path(path).expanduser()
    if not candidate_root.exists():
        return set()

    available: set[str] = set()
    for entry in candidate_root.iterdir():
        if not entry.is_dir():
            continue
        normalized = _normalize_media_subject_id(entry.name)
        if normalized is not None:
            available.add(normalized)
    return available


def _image_subject_ids() -> set[str]:
    return _collect_available_media_subject_ids(IMAGE_ROOT)


_video_subject_ids = _image_subject_ids


def _selected_subject_ids(
    inputs: Sequence[object],
    target: Sequence[object],
) -> tuple[str, ...]:
    selected = set(_TEST_SUBJECT_IDS)

    if _requires_media_path(inputs) or _requires_media_path(target):
        media_subject_ids = _image_subject_ids() | _video_subject_ids()
        if len(media_subject_ids) > 0:
            selected = selected.intersection(media_subject_ids)
        else:
            selected = set()

    return tuple(_sort_subject_ids(selected))


def _path_exists(path: str | Path) -> bool:
    if path is None:
        return False
    if isinstance(path, str):
        candidate = path.strip()
    else:
        candidate = str(path).strip()
    if candidate == "":
        return False
    return Path(candidate).exists()


def _requires_media_path(specs: Sequence[object]) -> bool:
    return any(isinstance(spec, (ImageSpec, VideoSpec)) for spec in specs)


def _paths_available(
    inputs: Sequence[object],
    target: Sequence[object],
) -> bool:
    if not _path_exists(HUTUBS_ROOT):
        return False
    if _requires_media_path(inputs) or _requires_media_path(target):
        if not _path_exists(IMAGE_ROOT):
            return False
    return True


def _spec_key_names(specs: Sequence[object]) -> tuple[str, ...]:
    return tuple(DatasetSpecWorkflow.get_spec_name(spec) for spec in specs)


def _uses_acoustic_specs(
    inputs: Sequence[object], target: Sequence[object]
) -> bool:
    acoustic_specs = (HRTFSpec, ITDSpec, ILDSpec, SHSpec)
    return any(isinstance(spec, acoustic_specs) for spec in inputs + target)


def _identity(value: object) -> object:
    return value


def _filename_only(value: object) -> object:
    return value if not isinstance(value, str) else str(Path(value).name)


def _to_array(value: object) -> np.ndarray:
    return np.asarray(value)


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
    (
        (HRTFSpec(index_by=("subject",)),),
        (),
    ),
    (
        (HRTFSpec(index_by=("subject", "position"), position_index=True, position_one_hot=True),),
        (),
    ),
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
    (
        (
            HRTFSpec(
                index_by=("subject", "samples"),
                sample_index=True,
                sample_one_hot=True,
                transform=_identity,
            ),
        ),
        (),
    ),
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
            ImageSpec(path=IMAGE_ROOT, grouped_by=("subject", "ear")),
        ),
        (
            HRTFSpec(
                index_by=("subject", "ear", "position"),
                ears=("left", "right"),
                positions=(0, 4, 8, 12),
                ear_index=True,
                position_index=True,
            ),
        ),
    ),
    (
        (
            ITDSpec(
                index_by=("subject", "position"),
                positions="all",
                plane=("horizontal", 0.0),
                position_index=True,
                position_one_hot=True,
                transform=_to_array,
            ),
        ),
        (),
    ),
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
    (
        (
            SHSpec(
                sh_order=3,
                index_by=("subject",),
                transform=_to_array,
            ),
        ),
        (),
    ),
    (
        (
            SHSpec(
                sh_order=2,
                index_by=("subject", "ear"),
                ears=("left",),
                ear_index=True,
                ear_one_hot=True,
                transform=_to_array,
            ),
        ),
        (),
    ),
    (
        (
            ITDSpec(
                index_by=("subject",),
                plane=("horizontal", 0, "degrees"),
                output="samples",
                transform=_to_array,
            ),
        ),
        (),
    ),
    (
        (
            ILDSpec(
                index_by=("subject",),
                mode="broad-band",
                output="db",
                transform=_to_array,
            ),
         ),
        (),
    ),
    (
        (
            ImageSpec(path=IMAGE_ROOT, grouped_by="subject", transform=_filename_only),
            MeshSpec(transform=_filename_only),
            AnthropometrySpec(transform=_to_array),
        ),
        (),
    ),
    (
        (HRTFSpec(index_by=("subject",), transform=_identity),),
        (
            ITDSpec(index_by=("subject",), transform=_to_array),
        ),
    ),
    (
        (HRTFSpec(index_by=("subject",), name="hrtf_input"),),
        (HRTFSpec(index_by=("subject",), name="hrtf_target"),),
    ),
    (
        (ILDSpec(index_by=("subject",), name="ild_input"),),
        (ILDSpec(index_by=("subject",), name="ild_target"),),
    ),
    (
        (ImageSpec(path=IMAGE_ROOT, grouped_by="subject"),),
        (AnthropometrySpec(),),
    ),
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
            ITDSpec(
                index_by=("subject", "position"),
                positions=(0, 1, 2),
                position_index=True,
                transform=_to_array,
            ),
        ),
        (
            ILDSpec(
                index_by=("subject", "position"),
                mode="broad-band",
                positions=(0, 1, 2),
                position_index=True,
                transform=_to_array,
            ),
        ),
    ),
]

_COMBINATION_IDS: list[str] = [
    "sub",
    "sub_pos",
    "sub_freq",
    "sub_freq_onehot",
    "sub_pos_plane",
    "sub_freqplane",
    "sub_samples",
    "sub_ear_pos_media",
    "ild_planesel",
    "sh",
    "sh_ear_left",
    "itd_plain",
    "ild_plain",
    "media_transformed",
    "input_target_itd",
    "same_spec_input_target",
    "same_spec_ild_input_target",
    "media_input_target",
    "audio_multiinput_target",
]

COMBINATIONS: list[tuple[tuple[object, ...], tuple[object, ...]]] = (
    _ALL_COMBINATIONS if _RUN_FULL_HUTUBS_TESTS else _ALL_COMBINATIONS[:1]
)

if not _RUN_FULL_HUTUBS_TESTS:
    _COMBINATION_IDS = _COMBINATION_IDS[:1]


HRTF_INDEX_BY_GRID: tuple[tuple[str, ...], ...] = (
    ("subject",),
    ("subject", "position"),
    ("subject", "ear", "position"),
    ("subject", "samples"),
    ("subject", "frequency"),
    ("position",),
    ("subject", "frequency", "position"),
)

HRTF_POSITION_GRID: tuple[object, ...] = (
    "all",
    (0, 1, 2),
)

HRTF_PLANE_GRID: tuple[object, ...] = (
    None,
    ("horizontal", 0.0),
    ("frontal", 90.0),
    "invalid-plane",
)

HRTF_TRANSFORM_GRID: tuple[Callable | None, ...] = (
    None,
    _identity,
    _bad_hrtf_transform,
)


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
    if isinstance(plane, str):
        if str(plane).strip().lower() not in {"horizontal", "median", "frontal"}:
            return None, "plane selection must be|plane must be"
    if isinstance(plane, tuple):
        if len(plane) not in {2, 3} or not isinstance(plane[0], str):
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
        id=(
            f"index_by={'-'.join(index_by)}|pos={positions}|"
            f"plane={plane}|transform={_hrtf_transform_name(transform)}"
        ),
    )
    for index_by, positions, plane, transform in product(
        HRTF_INDEX_BY_GRID,
        HRTF_POSITION_GRID,
        HRTF_PLANE_GRID,
        HRTF_TRANSFORM_GRID,
    )
]

if not _RUN_FULL_HUTUBS_TESTS:
    HRTF_GRID_CASES = [
        pytest.param(
            ("subject",),
            "all",
            None,
            None,
            None,
            None,
            id="smoke-index-by-subject",
        ),
        pytest.param(
            ("position",),
            "all",
            None,
            None,
            *_hrtf_grid_expected_failure(("position",), "all", None, None),
            id="smoke-bad-index-by-position",
        ),
        pytest.param(
            ("subject", "frequency"),
            "all",
            None,
            None,
            *_hrtf_grid_expected_failure(("subject", "frequency"), "all", None, None),
            id="smoke-bad-unsupported-axis",
        ),
        pytest.param(
            ("subject", "position"),
            (0, 1, 2),
            ("horizontal", 0.0),
            None,
            *_hrtf_grid_expected_failure(
                ("subject", "position"),
                (0, 1, 2),
                ("horizontal", 0.0),
                None,
            ),
            id="smoke-bad-plane-with-positions",
        ),
        pytest.param(
            ("subject",),
            "all",
            None,
            _bad_hrtf_transform,
            *_hrtf_grid_expected_failure(("subject",), "all", None, _bad_hrtf_transform),
            id="smoke-bad-transform",
        ),
    ]

_SPLIT_VALUES = ("all", "train", "validation", "test")
_VARIANT_VALUES = ("measured", "simulated")
if not _RUN_FULL_HUTUBS_TESTS:
    _SPLIT_VALUES = ("all",)
    _VARIANT_VALUES = ("measured",)


def _build_dataset(
    dataset_hrtf_variant: str,
    split: str,
    inputs: tuple[object, ...],
    target: tuple[object, ...],
) -> HUTUBS:
    if not _paths_available(inputs, target):
        pytest.skip(reason="Required local datasets are not available")

    with redirect_stdout(io.StringIO()):
        selected_subject_ids = _selected_subject_ids(inputs, target)
        if len(selected_subject_ids) == 0:
            pytest.skip(
                reason="No dataset subjects matched requested specs and local paths"
            )
        excluded_subject_ids = tuple(
            subject_id
            for subject_id in ALL_HUTUBS_SUBJECT_IDS
            if subject_id not in set(selected_subject_ids)
        )

        return HUTUBS(
            root=HUTUBS_ROOT,
            dataset_hrtf_variant=dataset_hrtf_variant,
            inputs=inputs,
            target=target,
            split=split,
            split_ratio=(0.8, 0.1, 0.1),
            split_seed=0,
            exclude_subject_ids=excluded_subject_ids,
            verbose=False,
        )


@pytest.mark.parametrize(
    "index_by,positions,plane,transform,expected_workflow_error,expected_dataset_error",
    HRTF_GRID_CASES,
)
def test_hutubs_hrtf_spec_grid(
    index_by: tuple[str, ...],
    positions: object,
    plane: object,
    transform: Callable | None,
    expected_workflow_error: str | None,
    expected_dataset_error: str | None,
) -> None:
    if not _path_exists(HUTUBS_ROOT):
        pytest.skip(reason="Required local datasets are not available")

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
            DatasetSpecWorkflow.build(config=HUTUBSConfig, inputs=(spec,), target=())
        return

    if expected_dataset_error is not None:
        with pytest.raises((ValueError, TypeError), match=expected_dataset_error):
            _build_dataset(
                dataset_hrtf_variant="measured",
                split="all",
                inputs=(spec,),
                target=(),
            )
        return

    if transform is _bad_hrtf_transform:
        dataset = _build_dataset(
            dataset_hrtf_variant="measured",
            split="all",
            inputs=(spec,),
            target=(),
        )
        with pytest.raises(
            AttributeError,
            match="IR",
        ):
            _ = dataset[0]
        return

    dataset = _build_dataset(
        dataset_hrtf_variant="measured",
        split="all",
        inputs=(spec,),
        target=(),
    )
    sample = dataset[0]
    assert "inputs" in sample
    assert sample["inputs"] is not None
    hrtf_value = sample["inputs"]["hrtf"]
    assert isinstance(hrtf_value, np.ndarray)
    assert hrtf_value.size > 0
    return


def _expected_available_subjects(
    inputs: Sequence[object],
    target: Sequence[object],
) -> list[str]:
    return list(_selected_subject_ids(inputs, target))


def _expected_selected_subjects(
    inputs: Sequence[object],
    target: Sequence[object],
    split: str,
) -> list[str]:
    return list(
        DatasetSubjectSplitPlanner.split_subject_ids(
            _selected_subject_ids(inputs, target),
            split=split,
            split_ratio=(0.8, 0.1, 0.1),
            split_seed=0,
        )
    )


@pytest.mark.parametrize("split", _SPLIT_VALUES)
@pytest.mark.parametrize("dataset_hrtf_variant", _VARIANT_VALUES)
@pytest.mark.parametrize(
    "specs",
    COMBINATIONS,
    ids=_COMBINATION_IDS,
)
def test_hutubs_real_dataset_all_combinations(
    dataset_hrtf_variant: str,
    split: str,
    specs: tuple[tuple[object, ...], tuple[object, ...]],
) -> None:
    inputs, target = specs
    if not _paths_available(inputs, target):
        pytest.skip(reason="Required local datasets are not available")

    try:
        dataset = _build_dataset(
            dataset_hrtf_variant=dataset_hrtf_variant,
            split=split,
            inputs=inputs,
            target=target,
        )
    except ValueError as exc:
        if "Split" in str(exc) and "produced an empty dataset" in str(exc):
            pytest.skip(reason=str(exc))
        raise

    assert dataset.variant == dataset_hrtf_variant
    assert dataset.available_subjects == _expected_available_subjects(
        inputs,
        target,
    )
    assert dataset.selected_subjects == _expected_selected_subjects(
        inputs,
        target,
        split=split,
    )
    assert len(dataset) > 0
    uses_acoustic = _uses_acoustic_specs(inputs, target)
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


def test_hutubs_get_subject_hrtf_uses_selected_subject() -> None:
    if not _paths_available((HRTFSpec(index_by=("subject", "position")),), ()):
        pytest.skip(reason="Required local datasets are not available")

    dataset = _build_dataset(
        dataset_hrtf_variant="measured",
        split="all",
        inputs=(HRTFSpec(index_by=("subject", "position")),),
        target=(),
    )
    expected_subject = dataset.selected_subjects[0]
    hrtf = dataset.get_subject_hrtf(expected_subject)
    assert hrtf.IR.values.size > 0
    assert hrtf.Sources is not None


def test_hutubs_index_by_error_messages_are_actionable() -> None:
    if not _path_exists(HUTUBS_ROOT):
        pytest.skip(reason="Required local datasets are not available")

    with pytest.raises(
        ValueError,
        match=r"ILDSpec index_by=\('subject', 'ear'\).*Supported index_by combinations for ILDSpec: \('subject',\), \('subject', 'position'\)",
    ):
        _build_dataset(
            dataset_hrtf_variant="measured",
            split="all",
            inputs=(ILDSpec(index_by=("subject", "ear"), mode="broad-band"),),
            target=(),
        )

    with pytest.raises(
        ValueError,
        match=r"SHSpec.ear_one_hot requires index_by to include 'ear'.*Supported index_by combinations for SHSpec:",
    ):
        _build_dataset(
            dataset_hrtf_variant="measured",
            split="all",
            inputs=(SHSpec(sh_order=2, index_by=("subject", "frequency"), ear_one_hot=True),),
            target=(),
        )


def test_hutubs_len_matches_subject_count_for_ear_indexed_rows() -> None:
    if not _path_exists(HUTUBS_ROOT):
        pytest.skip(reason="Required HUTUBS local dataset is not available")

    inputs = (
        AnthropometrySpec(grouped_by=("subject", "ear"), ear_one_hot=True),
    )
    target = (
        HRTFSpec(
            name="targetHrtf",
            plane="horizontal",
            index_by=("subject", "ear"),
            domain="frequency",
            ear_one_hot=True,
        ),
    )
    selected_subject_ids = _selected_subject_ids(inputs, target)
    if len(selected_subject_ids) == 0:
        pytest.skip(reason="No subjects matched requested scope")
    excluded_subject_ids = tuple(
        subject_id
        for subject_id in ALL_HUTUBS_SUBJECT_IDS
        if subject_id not in set(selected_subject_ids)
    )

    dataset = HUTUBS(
        root=HUTUBS_ROOT,
        dataset_hrtf_variant="measured",
        inputs=inputs,
        target=target,
        exclude_subject_ids=excluded_subject_ids,
        split="all",
        verbose=False,
    )

    expected_subjects = list(selected_subject_ids)
    assert dataset.available_subjects == expected_subjects
    assert dataset.selected_subjects == expected_subjects
    assert len(dataset) == len(expected_subjects) * 2


def test_hutubs_summary_reports_available_and_selected_subjects() -> None:
    if not _path_exists(HUTUBS_ROOT):
        pytest.skip(reason="Required HUTUBS local dataset is not available")

    inputs = (HRTFSpec(index_by=("subject",)),)
    selected_subject_ids = _selected_subject_ids(inputs, ())
    if len(selected_subject_ids) == 0:
        pytest.skip(reason="No subjects matched requested scope")
    excluded_subject_ids = tuple(
        subject_id
        for subject_id in ALL_HUTUBS_SUBJECT_IDS
        if subject_id not in set(selected_subject_ids)
    )

    dataset = HUTUBS(
        root=HUTUBS_ROOT,
        dataset_hrtf_variant="measured",
        inputs=inputs,
        target=(),
        exclude_subject_ids=excluded_subject_ids,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        split_seed=0,
        verbose=False,
    )

    summary = dataset.dataset_summary()
    assert f"available_subjects: {len(dataset.available_subjects)}" in summary
    assert f"selected_subjects: {len(dataset.selected_subjects)}" in summary
    assert len(dataset.available_subjects) >= len(dataset.selected_subjects)


def test_hutubs_constructor_verbose_false_is_quiet() -> None:
    if not _path_exists(HUTUBS_ROOT):
        pytest.skip(reason="Required HUTUBS local dataset is not available")

    inputs = (HRTFSpec(index_by=("subject",)),)
    selected_subject_ids = _selected_subject_ids(inputs, ())
    if len(selected_subject_ids) == 0:
        pytest.skip(reason="No subjects matched requested scope")
    excluded_subject_ids = tuple(
        subject_id
        for subject_id in ALL_HUTUBS_SUBJECT_IDS
        if subject_id not in set(selected_subject_ids)
    )

    output = io.StringIO()
    with redirect_stdout(output):
        dataset = HUTUBS(
            root=HUTUBS_ROOT,
            dataset_hrtf_variant="measured",
            inputs=inputs,
            target=(),
            exclude_subject_ids=excluded_subject_ids,
            split="all",
            verbose=False,
        )

    assert output.getvalue() == ""
    assert dataset.resources_summary() != ""
    assert dataset.dataset_summary() != ""


def test_spec_workflow_does_not_mutate_grouped_spec_objects() -> None:
    image_spec = ImageSpec(path=IMAGE_ROOT, grouped_by="subject")
    anthropometry_spec = AnthropometrySpec(accessed_by="ROW", grouped_by="subject-ear", ear="LEFT")

    DatasetSpecWorkflow.build(
        config=HUTUBSConfig,
        inputs=(image_spec, anthropometry_spec),
        target=(),
    )

    assert image_spec.grouped_by == "subject"
    assert anthropometry_spec.accessed_by == "ROW"
    assert anthropometry_spec.grouped_by == "subject-ear"
    assert anthropometry_spec.ear == "LEFT"
