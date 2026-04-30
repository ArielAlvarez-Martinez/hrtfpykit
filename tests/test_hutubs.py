from collections.abc import Callable, Sequence
from pathlib import Path
from itertools import product

import numpy as np
import pytest

from hrtfpykit.datasets import HUTUBS
from hrtfpykit.datasets.specs import (
    AnthropometrySpec,
    HRTFSpec,
    ImageSpec,
    ITDSpec,
    ILDSpec,
    MeshSpec,
    SHSpec,
    get_spec_name,
)


HUTUBS_ROOT = ""
IMAGE_ROOT = ""
SUBJECT_IDS = (1, 2, 3, 4, 5, 6, 8, 9, 10)
EXCLUDED_SUBJECT_IDS = tuple(i for i in range(1, 97) if i not in SUBJECT_IDS)


def _path_exists(path: str | Path) -> bool:
    if isinstance(path, Path):
        return path.exists()
    return Path(path).exists()


def _spec_key_names(specs: Sequence[object]) -> tuple[str, ...]:
    return tuple(get_spec_name(spec) for spec in specs)


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


COMBINATIONS: list[tuple[tuple[object, ...], tuple[object, ...]]] = [
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
) -> str | None:
    if len(index_by) == 0 or index_by[0] != "subject":
        return "index_by must start with 'subject'"
    if "frequency" in index_by:
        return "unsupported axes"
    if plane is not None and (not isinstance(positions, str) or positions != "all"):
        return "plane selection cannot be combined with custom positions"
    if isinstance(plane, str):
        if str(plane).strip().lower() not in {"horizontal", "median", "frontal"}:
            return "Plane selection must be|plane must be"
    if isinstance(plane, tuple):
        if len(plane) not in {2, 3} or not isinstance(plane[0], str):
            return "Plane selection must be|plane must be"
    if transform is _bad_hrtf_transform:
        return "runtime-bad-transform"
    return None


HRTF_GRID_CASES = [
    pytest.param(
        index_by,
        positions,
        plane,
        transform,
        _hrtf_grid_expected_failure(index_by, positions, plane, transform),
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


def _build_dataset(
    variant: str,
    split: str,
    inputs: tuple[object, ...],
    target: tuple[object, ...],
) -> HUTUBS:
    if not _path_exists(HUTUBS_ROOT) or not _path_exists(IMAGE_ROOT):
        raise pytest.Skip(reason="Required local datasets are not available")

    return HUTUBS(
        root=HUTUBS_ROOT,
        variant=variant,
        inputs=inputs,
        target=target,
        split=split,
        split_ratio=(0.8, 0.1, 0.1),
        split_seed=0,
        exclude_subject_ids=EXCLUDED_SUBJECT_IDS,
    )


@pytest.mark.parametrize(
    "index_by,positions,plane,transform,expected_error",
    HRTF_GRID_CASES,
)
def test_hutubs_hrtf_spec_grid(
    index_by: tuple[str, ...],
    positions: object,
    plane: object,
    transform: Callable | None,
    expected_error: str | None,
) -> None:
    if not _path_exists(HUTUBS_ROOT) or not _path_exists(IMAGE_ROOT):
        raise pytest.Skip(reason="Required local datasets are not available")

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

    if expected_error is None:
        dataset = _build_dataset(
            variant="measured",
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

    if expected_error == "runtime-bad-transform":
        dataset = _build_dataset(
            variant="measured",
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

    with pytest.raises((ValueError, TypeError), match=expected_error):
        _build_dataset(
            variant="measured",
            split="all",
            inputs=(spec,),
            target=(),
        )


@pytest.mark.parametrize("split", ["all", "train", "validation", "test"])
@pytest.mark.parametrize("variant", ["measured", "simulated"])
@pytest.mark.parametrize(
    "specs",
    COMBINATIONS,
    ids=[
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
    ],
)
def test_hutubs_real_dataset_all_combinations(
    variant: str,
    split: str,
    specs: tuple[tuple[object, ...], tuple[object, ...]],
) -> None:
    inputs, target = specs
    dataset = _build_dataset(variant=variant, split=split, inputs=inputs, target=target)

    assert dataset.variant == variant
    assert dataset.available_subjects == [f"pp{subject_id}" for subject_id in SUBJECT_IDS]
    assert len(dataset) > 0
    uses_acoustic = _uses_acoustic_specs(inputs, target)
    if uses_acoustic:
        assert isinstance(dataset.dataset_sample_rate, float)
        assert dataset.dataset_positions is not None
    else:
        assert dataset.dataset_sample_rate is None
        assert dataset.dataset_positions is None

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
    if not _path_exists(HUTUBS_ROOT) or not _path_exists(IMAGE_ROOT):
        raise pytest.Skip(reason="Required local datasets are not available")

    dataset = _build_dataset(
        variant="measured",
        split="all",
        inputs=(HRTFSpec(index_by=("subject", "position")),),
        target=(),
    )
    expected_subject = dataset.available_subjects[0]
    hrtf = dataset.get_subject_hrtf(expected_subject)
    assert hrtf.IR.values.size > 0
    assert hrtf.Sources is not None


def test_hutubs_index_by_error_messages_are_actionable() -> None:
    if not _path_exists(HUTUBS_ROOT) or not _path_exists(IMAGE_ROOT):
        raise pytest.Skip(reason="Required local datasets are not available")

    with pytest.raises(
        ValueError,
        match=r"ILDSpec index_by=\('subject', 'ear'\).*Supported index_by combinations for ILDSpec: \('subject',\), \('subject', 'position'\)",
    ):
        _build_dataset(
            variant="measured",
            split="all",
            inputs=(ILDSpec(index_by=("subject", "ear"), mode="broad-band"),),
            target=(),
        )

    with pytest.raises(
        ValueError,
        match=r"SHSpec.ear_one_hot requires index_by to include 'ear'.*Supported index_by combinations for SHSpec:",
    ):
        _build_dataset(
            variant="measured",
            split="all",
            inputs=(SHSpec(sh_order=2, index_by=("subject", "frequency"), ear_one_hot=True),),
            target=(),
        )


def test_hutubs_len_matches_subject_count_for_ear_indexed_rows() -> None:
    if not _path_exists(HUTUBS_ROOT):
        raise pytest.Skip(reason="Required HUTUBS local dataset is not available")

    dataset = HUTUBS(
        root=HUTUBS_ROOT,
        variant="measured",
        inputs=(
            AnthropometrySpec(grouped_by=("subject", "ear"), ear_one_hot=True),
        ),
        target=(
            HRTFSpec(
                name="targetHrtf",
                plane="horizontal",
                index_by=("subject", "ear"),
                domain="frequency",
                ear_one_hot=True,
            ),
        ),
        exclude_subject_ids=np.arange(6, 97),
        split="all",
    )

    excluded_subjects = set(int(value) for value in np.arange(6, 97))
    expected_subjects = [
        f"pp{subject_id}"
        for subject_id in SUBJECT_IDS
        if subject_id not in excluded_subjects
    ]
    assert dataset.available_subjects == expected_subjects
    assert len(dataset) == len(expected_subjects) * 2
