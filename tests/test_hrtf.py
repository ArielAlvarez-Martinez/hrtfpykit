import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from hrtfpykit.datasets import HRTFTransform
from hrtfpykit.hrtf.hrtf import HRTF
from hrtfpykit.hrtf import hrtf_difference, ild, load_hrtf, rms, sht, sht_inverse
from hrtfpykit.utils.metrics import itd
from hrtfpykit.sofa import load_sofa


FIXTURE_SOFA_PATH = Path(__file__).parent / "pp1_HRIRs_measured.sofa"
SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
if SOFA_PATH == "" and FIXTURE_SOFA_PATH.exists():
    SOFA_PATH = str(FIXTURE_SOFA_PATH)
pytestmark = pytest.mark.skipif(
    SOFA_PATH == "" or not os.path.exists(SOFA_PATH),
    reason="Required SOFA fixture is not available",
)


def _ir_values(hrtf: HRTF) -> np.ndarray:
    values = hrtf.IR.values
    assert values is not None
    return values


def _tf_values(hrtf: HRTF) -> np.ndarray:
    values = hrtf.TF.values
    assert values is not None
    return values


def _frequency_bins(hrtf: HRTF) -> np.ndarray:
    values = hrtf.TF.frequency_bins
    assert values is not None
    return values


def _sample_rate(hrtf: HRTF) -> float:
    sample_rate = hrtf.IR.sample_rate
    assert sample_rate is not None
    return float(sample_rate)


def _source_coordinate_system(hrtf: HRTF) -> str:
    coordinate_system = hrtf.Sources.source_coordinate_system
    assert coordinate_system is not None
    return coordinate_system


def _sofa(hrtf: HRTF) -> Any:
    sofa = hrtf.Sofa
    assert sofa is not None
    return sofa


def _wrap_value(container: Any, name: str) -> Any:
    wrapper = container.get(name)
    assert wrapper is not None
    return wrapper.value


def _sofa_dataset_close(sofa: Any) -> None:
    dataset = sofa.netCDF4_dataset
    assert dataset is not None
    dataset.close()


def transform_apply_window(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_window("hann")


def transform_apply_padding(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_padding(padding_length=8, location="end")


def transform_apply_fir_filter(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_fir_filter(
        filter="lowpass",
        cutoff=3000.0,
        num_taps=31,
    )


def transform_apply_iir_filter(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_iir_filter(
        filter="lowpass",
        cutoff=3000.0,
        order=4,
    )


def transform_minimum_phase(hrtf: HRTF) -> HRTF:
    return hrtf.transform.minimum_phase()


def transform_to_ctf(hrtf: HRTF) -> HRTF:
    return hrtf.transform.to_ctf(weights=False)


def transform_to_dtf(hrtf: HRTF) -> HRTF:
    return hrtf.transform.to_dtf(weights=False)


def transform_modify_ir(hrtf: HRTF) -> HRTF:
    ir = np.array(_ir_values(hrtf), copy=True)
    return hrtf.transform.modify_ir(ir)


def transform_modify_phase(hrtf: HRTF) -> HRTF:
    phase = np.array(hrtf.TF.phase, copy=True)
    phase[..., 0] = phase[..., 0] + 1.0
    return hrtf.transform.modify_phase(phase, unit="degrees")


def transform_modify_tf(hrtf: HRTF) -> HRTF:
    tf = np.array(_tf_values(hrtf), copy=True)
    return hrtf.transform.modify_tf(tf)


def transform_modify_magnitude(hrtf: HRTF) -> HRTF:
    magnitude = np.array(hrtf.TF.magnitude, copy=True)
    magnitude = magnitude * 0.99
    return hrtf.transform.modify_magnitude(magnitude, scale="linear")


def transform_apply_gain(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_gain(-1.0, scale="db")


def transform_modify_fft_length(hrtf: HRTF) -> HRTF:
    return hrtf.transform.modify_fft_length(int(_ir_values(hrtf).shape[-1]) + 32)


def transform_add_itd(hrtf: HRTF) -> HRTF:
    return hrtf.transform.add_itd(1, unit="samples")


def transform_delete_itd(hrtf: HRTF) -> HRTF:
    return hrtf.transform.delete_itd()


def transform_add_ild(hrtf: HRTF) -> HRTF:
    return hrtf.transform.add_ild(2.0)


def transform_delete_ild(hrtf: HRTF) -> HRTF:
    return hrtf.transform.delete_ild()


TRANSFORM_CASES = [
    ("apply_window", transform_apply_window, True),
    ("apply_padding", transform_apply_padding, False),
    ("apply_fir_filter", transform_apply_fir_filter, True),
    ("apply_iir_filter", transform_apply_iir_filter, True),
    ("minimum_phase", transform_minimum_phase, True),
    ("to_ctf", transform_to_ctf, True),
    ("to_dtf", transform_to_dtf, True),
    ("modify_ir", transform_modify_ir, True),
    ("modify_phase", transform_modify_phase, True),
    ("modify_tf", transform_modify_tf, True),
    ("modify_magnitude", transform_modify_magnitude, True),
    ("apply_gain", transform_apply_gain, True),
    ("modify_fft_length", transform_modify_fft_length, True),
    ("add_itd", transform_add_itd, True),
    ("delete_itd", transform_delete_itd, True),
    ("add_ild", transform_add_ild, True),
    ("delete_ild", transform_delete_ild, True),
]


@pytest.fixture
def real_hrtf() -> HRTF:
    return load_hrtf(SOFA_PATH)


def backed_sofa_convention(hrtf: HRTF) -> str:
    return _wrap_value(_sofa(hrtf).GlobalAttributes, "SOFAConventions")


def can_update_without_dimension_changes(hrtf: HRTF, default_expected: bool) -> bool:
    if not default_expected:
        return False
    if backed_sofa_convention(hrtf) != "SimpleFreeFieldHRTF":
        return True
    dataset = _sofa(hrtf).netCDF4_dataset
    if dataset is None:
        raise ValueError("SOFA dataset is not loaded")
    if _tf_values(hrtf) is None or _frequency_bins(hrtf) is None:
        return False
    required_variables = ("Data.Real", "Data.Imag", "N")
    if any(variable_name not in dataset.variables for variable_name in required_variables):
        return False
    tf_shape = np.asarray(_tf_values(hrtf)).shape
    frequency_shape = np.asarray(_frequency_bins(hrtf)).shape
    return (
        tuple(dataset.variables["Data.Real"].shape) == tf_shape
        and tuple(dataset.variables["Data.Imag"].shape) == tf_shape
        and tuple(dataset.variables["N"].shape) == frequency_shape
    )


def test_real_hrtf_load_populates_domains_and_sources(real_hrtf: HRTF) -> None:
    assert real_hrtf.SOFAConventions in {
        "SimpleFreeFieldHRIR",
        "SimpleFreeFieldHRTF",
    }
    assert real_hrtf.Sofa is not None
    assert _sofa(real_hrtf).netCDF4_dataset is not None
    assert real_hrtf.is_transformed() is False

    ir_values = np.asarray(_ir_values(real_hrtf))
    tf_values = np.asarray(_tf_values(real_hrtf))
    frequency_bins = np.asarray(_frequency_bins(real_hrtf), dtype=float)
    source_positions = real_hrtf.Sources.get_positions()
    cartesian_positions = real_hrtf.Sources.get_positions(coordinate_system="cartesian")
    lateral_polar_positions = real_hrtf.Sources.get_positions(coordinate_system="lateral-polar")
    sample_rate = float(_sample_rate(real_hrtf))

    assert ir_values.ndim == 3
    assert tf_values.ndim == 3
    assert ir_values.shape[:-1] == tf_values.shape[:-1]
    assert tf_values.shape[-1] == frequency_bins.size
    assert source_positions.shape == (ir_values.shape[0], 3)
    assert cartesian_positions.shape == source_positions.shape
    assert lateral_polar_positions.shape == source_positions.shape
    assert _source_coordinate_system(real_hrtf) == "spherical"
    assert np.isfinite(ir_values).all()
    assert np.isfinite(np.real(tf_values)).all()
    assert np.isfinite(np.imag(tf_values)).all()
    assert np.isfinite(sample_rate)
    assert sample_rate > 0.0
    assert frequency_bins.ndim == 1
    assert frequency_bins[0] == pytest.approx(0.0)
    assert np.all(np.diff(frequency_bins) > 0.0)
    assert real_hrtf.fft_length is not None
    assert np.allclose(
        frequency_bins,
        np.fft.rfftfreq(int(real_hrtf.fft_length), 1.0 / sample_rate),
    )


def test_load_hrtf_can_close_and_reopen_backing_sofa() -> None:
    hrtf = load_hrtf(SOFA_PATH, sofa_open=False)

    assert hrtf.Sofa is not None
    assert hrtf.Sofa.is_open() is False
    assert _ir_values(hrtf).size > 0
    assert _tf_values(hrtf).size > 0
    assert hrtf.Sources.get_positions().size > 0
    with pytest.raises(ValueError, match="SOFA dataset is closed"):
        _ = hrtf.Sofa.Variables

    hrtf.Sofa.open(check_sofa_against_conventions=False)
    assert hrtf.Sofa.is_open() is True
    hrtf.Sofa.close()
    assert hrtf.Sofa.is_open() is False


def test_transform_returns_independent_hrtf_without_mutating_source(
    real_hrtf: HRTF,
) -> None:
    original_ir = np.array(_ir_values(real_hrtf), copy=True)
    original_tf = np.array(_tf_values(real_hrtf), copy=True)

    transformed_hrtf = real_hrtf.transform.apply_gain(-3.0, scale="db")

    assert transformed_hrtf is not real_hrtf
    assert transformed_hrtf.is_transformed() is True
    assert real_hrtf.is_transformed() is False
    assert _ir_values(transformed_hrtf).shape == original_ir.shape
    assert _tf_values(transformed_hrtf).shape == original_tf.shape
    assert not np.shares_memory(_ir_values(transformed_hrtf), _ir_values(real_hrtf))
    assert not np.shares_memory(_tf_values(transformed_hrtf), _tf_values(real_hrtf))
    assert not np.allclose(_tf_values(transformed_hrtf), original_tf)
    assert np.allclose(_ir_values(real_hrtf), original_ir)
    assert np.allclose(_tf_values(real_hrtf), original_tf)


def test_transform_ear_parameter_targets_selected_channel(real_hrtf: HRTF) -> None:
    original_ir = np.array(_ir_values(real_hrtf), copy=True)
    original_tf = np.array(_tf_values(real_hrtf), copy=True)
    sample_count = int(original_ir.shape[-1])
    window_end = min(32, sample_count)
    crop_start = 4
    crop_end = min(8, sample_count)
    if window_end <= 2 or crop_end <= crop_start:
        pytest.skip("Ear-selective transform test requires at least eight IR samples")

    windowed = real_hrtf.transform.apply_window(
        "hann",
        start_sample=0,
        end_sample=window_end,
        ear="right",
    )
    np.testing.assert_allclose(_ir_values(windowed)[..., 0, :], original_ir[..., 0, :])
    assert not np.allclose(
        _ir_values(windowed)[..., 1, :window_end],
        original_ir[..., 1, :window_end],
    )

    crop_length = crop_end - crop_start
    zero_tail = np.zeros(original_ir[..., 0, :crop_length].shape, dtype=original_ir.dtype)
    expected_left_crop = np.concatenate(
        (
            original_ir[..., 0, :crop_start],
            original_ir[..., 0, crop_end:],
            zero_tail,
        ),
        axis=-1,
    )
    cropped = real_hrtf.transform.apply_crop(crop_start, crop_end, ear="left")
    np.testing.assert_allclose(_ir_values(cropped)[..., 0, :], expected_left_crop)
    np.testing.assert_allclose(_ir_values(cropped)[..., 1, :], original_ir[..., 1, :])

    fir_filtered = real_hrtf.transform.apply_fir_filter(
        filter="lowpass",
        cutoff=3000.0,
        num_taps=31,
        ear="left",
    )
    assert not np.allclose(_ir_values(fir_filtered)[..., 0, :], original_ir[..., 0, :])
    np.testing.assert_allclose(_ir_values(fir_filtered)[..., 1, :], original_ir[..., 1, :])

    iir_filtered = real_hrtf.transform.apply_iir_filter(
        filter="lowpass",
        cutoff=3000.0,
        order=4,
        ear="right",
    )
    np.testing.assert_allclose(_ir_values(iir_filtered)[..., 0, :], original_ir[..., 0, :])
    assert not np.allclose(_ir_values(iir_filtered)[..., 1, :], original_ir[..., 1, :])

    gained = real_hrtf.transform.apply_gain(6.0, scale="db", ear="left")
    assert not np.allclose(_tf_values(gained)[..., 0, :], original_tf[..., 0, :])
    np.testing.assert_allclose(_tf_values(gained)[..., 1, :], original_tf[..., 1, :])

    with pytest.raises(ValueError, match="ear must be one of"):
        real_hrtf.transform.apply_gain(1.0, ear="center")


def test_transform_apply_padding_ear_and_preserve_length(real_hrtf: HRTF) -> None:
    original_ir = np.array(_ir_values(real_hrtf), copy=True)
    sample_count = int(original_ir.shape[-1])
    padding_length = 3
    if sample_count <= padding_length:
        pytest.skip("Padding transform test requires more samples than padding length")

    left_start = real_hrtf.transform.apply_padding(
        padding_length,
        location="start",
        ear="left",
    )
    assert _ir_values(left_start).shape[-1] == sample_count + padding_length
    np.testing.assert_allclose(_ir_values(left_start)[..., 0, :padding_length], 0.0)
    np.testing.assert_allclose(_ir_values(left_start)[..., 0, padding_length:], original_ir[..., 0, :])
    np.testing.assert_allclose(_ir_values(left_start)[..., 1, :sample_count], original_ir[..., 1, :])
    np.testing.assert_allclose(_ir_values(left_start)[..., 1, sample_count:], 0.0)

    left_end = real_hrtf.transform.apply_padding(
        2,
        location="end",
        value=5.0,
        ear="left",
    )
    assert _ir_values(left_end).shape[-1] == sample_count + 2
    np.testing.assert_allclose(_ir_values(left_end)[..., 0, :sample_count], original_ir[..., 0, :])
    np.testing.assert_allclose(_ir_values(left_end)[..., 0, sample_count:], 5.0)
    np.testing.assert_allclose(_ir_values(left_end)[..., 1, :sample_count], original_ir[..., 1, :])
    np.testing.assert_allclose(_ir_values(left_end)[..., 1, sample_count:], 0.0)

    left_preserved = real_hrtf.transform.apply_padding(
        padding_length,
        location="start",
        preserve_length=True,
        ear="left",
    )
    assert _ir_values(left_preserved).shape == original_ir.shape
    np.testing.assert_allclose(_ir_values(left_preserved)[..., 0, :padding_length], 0.0)
    np.testing.assert_allclose(
        _ir_values(left_preserved)[..., 0, padding_length:],
        original_ir[..., 0, :-padding_length],
    )
    np.testing.assert_allclose(_ir_values(left_preserved)[..., 1, :], original_ir[..., 1, :])

    both_preserved = real_hrtf.transform.apply_padding(
        padding_length,
        location="start",
        preserve_length=True,
        ear="both",
    )
    assert _ir_values(both_preserved).shape == original_ir.shape
    np.testing.assert_allclose(_ir_values(both_preserved)[..., :padding_length], 0.0)
    np.testing.assert_allclose(
        _ir_values(both_preserved)[..., padding_length:],
        original_ir[..., :-padding_length],
    )

    with pytest.raises(ValueError, match="preserve_length=True"):
        real_hrtf.transform.apply_padding(
            padding_length,
            location="end",
            preserve_length=True,
        )


def test_dataset_transform_wrappers_forward_ear_arguments(real_hrtf: HRTF) -> None:
    sample_count = int(_ir_values(real_hrtf).shape[-1])
    window_end = min(32, sample_count)
    crop_start = 4
    crop_end = min(8, sample_count)
    if window_end <= 2 or crop_end <= crop_start:
        pytest.skip("Dataset transform wrapper test requires at least eight IR samples")

    wrapper_cases = (
        (
            HRTFTransform.apply_window("hann", end_sample=window_end, ear="left"),
            lambda hrtf: hrtf.transform.apply_window(
                "hann",
                end_sample=window_end,
                ear="left",
            ),
        ),
        (
            HRTFTransform.apply_crop(crop_start, crop_end, ear="right"),
            lambda hrtf: hrtf.transform.apply_crop(crop_start, crop_end, ear="right"),
        ),
        (
            HRTFTransform.apply_padding(
                3,
                location="start",
                preserve_length=True,
                ear="left",
            ),
            lambda hrtf: hrtf.transform.apply_padding(
                3,
                location="start",
                preserve_length=True,
                ear="left",
            ),
        ),
        (
            HRTFTransform.apply_fir_filter(
                "lowpass",
                cutoff=3000.0,
                num_taps=31,
                ear="left",
            ),
            lambda hrtf: hrtf.transform.apply_fir_filter(
                "lowpass",
                cutoff=3000.0,
                num_taps=31,
                ear="left",
            ),
        ),
        (
            HRTFTransform.apply_iir_filter(
                "lowpass",
                cutoff=3000.0,
                order=4,
                ear="right",
            ),
            lambda hrtf: hrtf.transform.apply_iir_filter(
                "lowpass",
                cutoff=3000.0,
                order=4,
                ear="right",
            ),
        ),
        (
            HRTFTransform.apply_gain(3.0, scale="db", ear="right"),
            lambda hrtf: hrtf.transform.apply_gain(3.0, scale="db", ear="right"),
        ),
        (
            HRTFTransform.add_ild(2.0),
            lambda hrtf: hrtf.transform.add_ild(2.0),
        ),
        (
            HRTFTransform.delete_ild(),
            lambda hrtf: hrtf.transform.delete_ild(),
        ),
    )

    for wrapper, direct_transform in wrapper_cases:
        wrapped_hrtf = cast(HRTF, wrapper(real_hrtf))
        direct_hrtf = cast(HRTF, direct_transform(real_hrtf))
        np.testing.assert_allclose(_ir_values(wrapped_hrtf), _ir_values(direct_hrtf))
        np.testing.assert_allclose(_tf_values(wrapped_hrtf), _tf_values(direct_hrtf))


def test_transform_add_and_delete_ild_update_tf_and_ir(real_hrtf: HRTF) -> None:
    original_tf = np.array(_tf_values(real_hrtf), copy=True)
    source_count = int(original_tf.shape[0])
    frequency_count = int(original_tf.shape[-1])

    scalar_ild = 6.0
    scalar_modified = real_hrtf.transform.add_ild(scalar_ild)
    left_gain = 10 ** (scalar_ild / 40.0)
    right_gain = 10 ** (-scalar_ild / 40.0)

    assert scalar_modified.is_transformed() is True
    assert _tf_values(scalar_modified).shape == original_tf.shape
    assert _ir_values(scalar_modified).shape == _ir_values(real_hrtf).shape
    np.testing.assert_allclose(
        _tf_values(scalar_modified)[..., 0, :],
        original_tf[..., 0, :] * left_gain,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _tf_values(scalar_modified)[..., 1, :],
        original_tf[..., 1, :] * right_gain,
        rtol=1e-12,
        atol=1e-12,
    )

    source_ild = np.linspace(-3.0, 3.0, source_count)
    source_modified = real_hrtf.transform.add_ild(source_ild)
    source_left_gain = 10 ** (source_ild[:, np.newaxis] / 40.0)
    source_right_gain = 10 ** (-source_ild[:, np.newaxis] / 40.0)
    np.testing.assert_allclose(
        _tf_values(source_modified)[:, 0, :],
        original_tf[:, 0, :] * source_left_gain,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _tf_values(source_modified)[:, 1, :],
        original_tf[:, 1, :] * source_right_gain,
        rtol=1e-12,
        atol=1e-12,
    )

    frequency_ild = np.broadcast_to(
        np.linspace(-1.0, 1.0, frequency_count),
        (source_count, frequency_count),
    )
    frequency_modified = real_hrtf.transform.add_ild(frequency_ild)
    frequency_left_gain = 10 ** (frequency_ild / 40.0)
    frequency_right_gain = 10 ** (-frequency_ild / 40.0)
    np.testing.assert_allclose(
        _tf_values(frequency_modified)[:, 0, :],
        original_tf[:, 0, :] * frequency_left_gain,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _tf_values(frequency_modified)[:, 1, :],
        original_tf[:, 1, :] * frequency_right_gain,
        rtol=1e-12,
        atol=1e-12,
    )

    no_ild = scalar_modified.transform.delete_ild()
    np.testing.assert_allclose(
        np.abs(_tf_values(no_ild)[..., 0, :]),
        np.abs(_tf_values(no_ild)[..., 1, :]),
        rtol=1e-8,
        atol=1e-9,
    )

    with pytest.raises(ValueError, match="finite dB"):
        real_hrtf.transform.add_ild(np.nan)


def test_reset_restores_selected_transformed_hrtf_to_backed_state(
    real_hrtf: HRTF,
) -> None:
    source_positions = real_hrtf.Sources.get_positions()
    selected_count = min(2, source_positions.shape[0])
    selected_hrtf = real_hrtf.select(
        positions=source_positions[:selected_count],
        position_coordinate_system=_source_coordinate_system(real_hrtf),
    )
    transformed_hrtf = selected_hrtf.transform.apply_gain(-3.0, scale="db")

    assert _ir_values(transformed_hrtf).shape[0] == selected_count
    assert transformed_hrtf.Sources.get_positions().shape[0] == selected_count
    assert transformed_hrtf.is_transformed() is True

    restored_hrtf = transformed_hrtf.reset()

    assert restored_hrtf is transformed_hrtf
    assert restored_hrtf.is_transformed() is False
    assert _ir_values(restored_hrtf).shape == _ir_values(real_hrtf).shape
    assert _tf_values(restored_hrtf).shape == _tf_values(real_hrtf).shape
    assert (
        restored_hrtf.Sources.get_positions().shape
        == real_hrtf.Sources.get_positions().shape
    )
    assert np.allclose(_ir_values(restored_hrtf), _ir_values(real_hrtf))
    assert np.allclose(_tf_values(restored_hrtf), _tf_values(real_hrtf))


@pytest.mark.parametrize(
    ("name", "transform_fn", "expect_update_without_resize"),
    TRANSFORM_CASES,
    ids=[case[0] for case in TRANSFORM_CASES],
)
def test_update_sofa_all_transform_methods(
    real_hrtf: HRTF,
    name: str,
    transform_fn,
    expect_update_without_resize: bool,
) -> None:
    transformed_hrtf = transform_fn(real_hrtf)

    assert transformed_hrtf.is_transformed() is True

    if can_update_without_dimension_changes(
        transformed_hrtf,
        expect_update_without_resize,
    ):
        transformed_hrtf.update_sofa(change_sofa_dimensions=False)
    else:
        with pytest.raises(
            ValueError,
            match="Set change_sofa_dimensions=True",
        ):
            transformed_hrtf.update_sofa(change_sofa_dimensions=False)

    transformed_hrtf.update_sofa(change_sofa_dimensions=True)


def test_update_sofa_no_transform_is_quiet(real_hrtf: HRTF, capsys) -> None:
    assert real_hrtf.is_transformed() is False
    expect_noop = can_update_without_dimension_changes(real_hrtf, True)

    real_hrtf.update_sofa()
    captured = capsys.readouterr()

    if expect_noop:
        assert captured.out == ""
    else:
        dataset = _sofa(real_hrtf).netCDF4_dataset
        assert dataset is not None
        assert tuple(dataset.variables["Data.Real"].shape) == np.asarray(_tf_values(real_hrtf)).shape
        assert tuple(dataset.variables["Data.Imag"].shape) == np.asarray(_tf_values(real_hrtf)).shape
        assert tuple(dataset.variables["N"].shape) == np.asarray(_frequency_bins(real_hrtf)).shape
    assert real_hrtf.is_transformed() is False


def test_save_runs_after_update_sofa(real_hrtf: HRTF, tmp_path) -> None:
    transformed_hrtf = real_hrtf.transform.apply_window("hann")
    destination = tmp_path / "hrtf_saved.sofa"

    saved_path = transformed_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=not can_update_without_dimension_changes(transformed_hrtf, True),
    )

    assert saved_path == destination
    assert destination.exists()


@pytest.mark.parametrize(
    "sofa_convention",
    [
        "same",
        "SimpleFreeFieldHRIR",
        "SimpleFreeFieldHRTF",
    ],
)
def test_save_sofa_convention_with_selected_positions(
    real_hrtf: HRTF,
    tmp_path,
    sofa_convention: str,
) -> None:
    selected_hrtf = real_hrtf.select(positions=cast(Any, ["front", "left", "right"]))
    destination = tmp_path / f"selected_{sofa_convention}.sofa"

    saved_path = selected_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=True,
        sofa_convention=sofa_convention,
    )
    assert saved_path == destination
    assert destination.exists()

    saved_sofa = load_sofa(destination)
    saved_variables = set(saved_sofa.Variables.get_names())

    resolved_expected_convention = (
        backed_sofa_convention(real_hrtf) if sofa_convention == "same" else sofa_convention
    )
    present_variables: tuple[str, ...]
    absent_variables: tuple[str, ...]
    if resolved_expected_convention == "SimpleFreeFieldHRIR":
        expected_data_type = "FIR"
        present_variables = ("Data.IR", "Data.SamplingRate")
        absent_variables = ("Data.Real", "Data.Imag", "N")
    else:
        expected_data_type = "TF"
        present_variables = ("Data.Real", "Data.Imag", "N")
        absent_variables = ("Data.IR", "Data.SamplingRate")
    assert (
        _wrap_value(saved_sofa.GlobalAttributes, "SOFAConventions")
        == resolved_expected_convention
    )
    assert _wrap_value(saved_sofa.GlobalAttributes, "DataType") == expected_data_type

    for variable_name in present_variables:
        assert variable_name in saved_variables
    for variable_name in absent_variables:
        assert variable_name not in saved_variables

    source_position = np.asarray(_wrap_value(saved_sofa.Variables, "SourcePosition"))
    assert source_position.shape[0] == 3

    if "Data.IR" in present_variables:
        data_ir = np.asarray(_wrap_value(saved_sofa.Variables, "Data.IR"))
        assert data_ir.shape[0] == 3
    if "Data.Real" in present_variables:
        data_real = np.asarray(_wrap_value(saved_sofa.Variables, "Data.Real"))
        assert data_real.shape[0] == 3

    _sofa_dataset_close(saved_sofa)


def test_save_rejects_ear_selected_hrtf(real_hrtf: HRTF, tmp_path) -> None:
    selected_hrtf = real_hrtf.select(ear="left")

    with pytest.raises(ValueError, match="ear-selected HRTF"):
        selected_hrtf.update_sofa(change_sofa_dimensions=True)
    with pytest.raises(ValueError, match="ear-selected HRTF"):
        selected_hrtf.save(
            path=tmp_path / "left_only.sofa",
            overwrite=True,
            change_sofa_dimensions=True,
        )


def test_real_hrtf_selects_positions_and_ears(real_hrtf: HRTF) -> None:
    selected_hrtf = real_hrtf.select(
        positions=cast(Any, ["front", "left", "right"]),
        ear="left",
    )

    assert _ir_values(selected_hrtf).shape[0] == 3
    assert _ir_values(selected_hrtf).ndim == 2
    assert _tf_values(selected_hrtf) is not None


def test_real_hrtf_selects_numeric_positions_crop_and_right_ear(
    real_hrtf: HRTF,
) -> None:
    source_positions = real_hrtf.Sources.get_positions()
    selected_count = min(2, source_positions.shape[0])
    crop_start = 1
    crop_end = min(17, _ir_values(real_hrtf).shape[-1])
    original_ir = np.array(_ir_values(real_hrtf), copy=True)
    original_tf = np.array(_tf_values(real_hrtf), copy=True)

    selected_hrtf = real_hrtf.select(
        positions=source_positions[:selected_count],
        position_coordinate_system=_source_coordinate_system(real_hrtf),
        ear="right",
        start_sample=crop_start,
        end_sample=crop_end,
    )

    assert crop_end > crop_start
    crop_length = crop_end - crop_start
    assert _ir_values(selected_hrtf).shape == (selected_count, crop_end - crop_start)
    assert _tf_values(selected_hrtf).shape[0] == selected_count
    assert _tf_values(selected_hrtf).ndim == 2
    assert _tf_values(selected_hrtf).shape[-1] == np.fft.rfftfreq(
        crop_length,
        d=1.0 / _sample_rate(selected_hrtf),
    ).shape[0]
    assert selected_hrtf.fft_length == crop_length
    assert selected_hrtf.Sources.get_positions().shape == (selected_count, 3)
    assert np.allclose(_ir_values(real_hrtf), original_ir)
    assert np.allclose(_tf_values(real_hrtf), original_tf)


def test_real_hrtf_metric_itd_runs_on_loaded_file(real_hrtf: HRTF) -> None:
    values = itd(real_hrtf)

    assert np.asarray(values).shape[0] == _ir_values(real_hrtf).shape[0]
    assert np.all(np.isfinite(values))


def test_real_hrtf_metric_ild_runs_on_loaded_file(real_hrtf: HRTF) -> None:
    broad_band_values = ild(real_hrtf, mode="broad-band")
    frequency_dependent_values = ild(
        real_hrtf,
        mode="frequency-dependent",
    )
    absolute_values = ild(real_hrtf, mode="broad-band", absolute=True)

    assert broad_band_values.shape == _ir_values(real_hrtf).shape[:-2]
    assert frequency_dependent_values.shape == (
        _tf_values(real_hrtf).shape[:-2] + (_tf_values(real_hrtf).shape[-1],)
    )
    assert np.all(np.isfinite(broad_band_values))
    assert np.all(np.isfinite(frequency_dependent_values))
    np.testing.assert_allclose(absolute_values, np.abs(broad_band_values))


def test_real_hrtf_metric_rms_matches_sample_axis_calculation(real_hrtf: HRTF) -> None:
    expected = np.sqrt(np.mean(np.square(np.asarray(_ir_values(real_hrtf), dtype=float)), axis=-1))

    linear_values = rms(real_hrtf, output="linear")
    db_values = rms(real_hrtf, output="db", reference="max")
    source_mean = rms(real_hrtf, output="linear", reduction_axis="source")
    ear_rms = rms(
        real_hrtf,
        output="linear",
        reduction_axis="ear",
        reduction_method="rms",
    )
    global_rms = rms(
        real_hrtf,
        output="linear",
        reduction_axis="global",
        reduction_method="rms",
    )

    np.testing.assert_allclose(linear_values, expected)
    assert db_values.shape == expected.shape
    assert np.max(db_values) == pytest.approx(0.0)
    np.testing.assert_allclose(source_mean, np.mean(expected, axis=0))
    np.testing.assert_allclose(ear_rms, np.sqrt(np.mean(np.square(expected), axis=-1)))
    np.testing.assert_allclose(global_rms, np.sqrt(np.mean(np.square(expected))))

    with pytest.raises(ValueError, match="hrtf must be an HRTF"):
        rms(real_hrtf.IR)  # type: ignore[arg-type]


def test_real_hrtf_sht_requires_hrtf_object(real_hrtf: HRTF) -> None:
    sh = sht(real_hrtf, sh_order=1, ear="both")
    coefficients = sh.get_coefficients()
    reconstructed = sht_inverse(sh)

    assert coefficients.shape == (4, 2, _tf_values(real_hrtf).shape[-1])
    assert sh.Y.shape == (_tf_values(real_hrtf).shape[0], 4)
    assert reconstructed.shape == _tf_values(real_hrtf).shape
    assert np.all(np.isfinite(coefficients))
    assert np.all(np.isfinite(reconstructed))

    with pytest.raises(ValueError, match="hrtf must be an HRTF instance"):
        sht(real_hrtf.TF, sh_order=1)  # type: ignore[arg-type]


def test_real_hrtf_metric_ir_errors_match_manual_calculation(real_hrtf: HRTF) -> None:
    processed = real_hrtf.transform.apply_gain(-1.0, scale="db")
    reference_ir = np.asarray(_ir_values(real_hrtf), dtype=float)
    compared_ir = np.asarray(processed.IR.values, dtype=float)
    error = compared_ir - reference_ir

    expected_rmse = np.sqrt(np.mean(np.square(error), axis=-1))
    expected_mae = np.mean(np.abs(error), axis=-1)
    expected_nrmse = 20.0 * np.log10(
        np.maximum(
            np.sqrt(
                np.sum(np.square(error), axis=-1)
                / np.maximum(np.sum(np.square(reference_ir), axis=-1), 1e-12)
            ),
            1e-12,
        )
    )

    np.testing.assert_allclose(hrtf_difference(real_hrtf, processed, metric="rmse"), expected_rmse)
    np.testing.assert_allclose(hrtf_difference(real_hrtf, processed, metric="mae"), expected_mae)
    np.testing.assert_allclose(hrtf_difference(real_hrtf, processed, metric="nrmse"), expected_nrmse)

    left_rmse = hrtf_difference(real_hrtf, processed, metric="rmse", ear="left")
    left_mae = hrtf_difference(real_hrtf, processed, metric="mae", ear="left")
    left_nrmse = hrtf_difference(real_hrtf, processed, metric="nrmse", ear="left")

    assert np.asarray(left_rmse).shape == reference_ir.shape[:1]
    assert np.asarray(left_mae).shape == reference_ir.shape[:1]
    assert np.asarray(left_nrmse).shape == reference_ir.shape[:1]
    np.testing.assert_allclose(left_rmse, expected_rmse[:, 0])
    np.testing.assert_allclose(left_mae, expected_mae[:, 0])
    np.testing.assert_allclose(left_nrmse, expected_nrmse[:, 0])


def test_real_hrtf_metric_ir_errors_reduce_like_lsd(real_hrtf: HRTF) -> None:
    processed = real_hrtf.transform.apply_gain(-1.0, scale="db")
    quieter = real_hrtf.transform.apply_gain(-2.0, scale="db")
    source_count = int(_ir_values(real_hrtf).shape[0])

    values = hrtf_difference(real_hrtf, [processed, quieter], metric="rmse", ear="both")
    assert np.asarray(values).shape == (2, source_count, 2)

    compared_reduced = hrtf_difference(
        real_hrtf,
        [processed, quieter],
        metric="rmse",
        reduction_axis="differences",
    )
    assert np.asarray(compared_reduced).shape == (source_count, 2)

    source_reduced = hrtf_difference(
        real_hrtf,
        [processed, quieter],
        metric="rmse",
        reduction_axis="sources",
    )
    assert np.asarray(source_reduced).shape == (2, 2)

    ear_reduced = hrtf_difference(real_hrtf, processed, metric="rmse", reduction_axis="ears")
    assert np.asarray(ear_reduced).shape == (source_count,)

    per_hrtf_scores = hrtf_difference(
        real_hrtf,
        [processed, quieter],
        metric="rmse",
        reduction_axis=("sources", "ears"),
        reduction_method="rms",
    )
    assert np.asarray(per_hrtf_scores).shape == (2,)

    global_score = hrtf_difference(
        real_hrtf,
        [processed, quieter],
        metric="rmse",
        reduction_axis="global",
        reduction_method="rms",
    )
    assert isinstance(global_score, float)

    mae_global = hrtf_difference(real_hrtf, processed, metric="mae", reduction_axis="global")
    assert isinstance(mae_global, float)

    nrmse_score = hrtf_difference(
        real_hrtf,
        processed,
        metric="nrmse",
        reduction_axis=("sources", "ears"),
        reduction_method="rms",
    )
    assert isinstance(nrmse_score, float)

    with pytest.raises(ValueError, match="can only be used when ear='both'"):
        hrtf_difference(real_hrtf, processed, metric="rmse", ear="left", reduction_axis="ears")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        hrtf_difference(real_hrtf, processed, metric="nrmse", output="db")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="metric must be one of"):
        hrtf_difference(real_hrtf, processed, metric="power")


def test_real_hrtf_metric_ir_errors_validate_sample_rate(real_hrtf: HRTF) -> None:
    processed = real_hrtf.transform.apply_gain(-1.0, scale="db")
    mismatched = processed.clone()
    mismatched.IR.sample_rate = float(_sample_rate(real_hrtf)) + 1.0

    with pytest.raises(ValueError, match="same IR sample_rate"):
        hrtf_difference(real_hrtf, mismatched, metric="rmse")


def test_real_hrtf_metric_lsd_accepts_frequency_bands(real_hrtf: HRTF) -> None:
    processed = real_hrtf.transform.apply_gain(-1.0, scale="db")
    frequency_bins = np.asarray(_frequency_bins(real_hrtf), dtype=float)
    positive_bins = frequency_bins[frequency_bins >= 20.0]
    if positive_bins.size < 4:
        pytest.skip("LSD frequency band test requires at least four positive frequency bins")
    band = (float(positive_bins[0]), float(positive_bins[min(3, positive_bins.size - 1)]))
    expected_indices = np.where((frequency_bins >= band[0]) & (frequency_bins <= band[1]))[0]

    band_values = hrtf_difference(
        real_hrtf,
        processed,
        metric="lsd",
        ear="left",
        positions="front",
        frequency_bands=band,
    )
    explicit_values = hrtf_difference(
        real_hrtf,
        processed,
        metric="lsd",
        ear="left",
        positions="front",
        frequencies=frequency_bins[expected_indices],
    )
    both_ear_values = hrtf_difference(
        real_hrtf,
        processed,
        metric="lsd",
        ear="both",
        positions="front",
        frequency_bands=band,
    )
    source_reduced = hrtf_difference(
        real_hrtf,
        processed,
        metric="lsd",
        ear="both",
        positions="front",
        frequency_bands=band,
        reduction_axis="sources",
    )
    ears_reduced = hrtf_difference(
        real_hrtf,
        processed,
        metric="lsd",
        ear="both",
        positions="front",
        frequency_bands=band,
        reduction_axis="ears",
    )
    left_source_reduced = hrtf_difference(
        real_hrtf,
        processed,
        metric="lsd",
        ear="left",
        positions="front",
        frequency_bands=band,
        reduction_axis="sources",
    )

    assert np.asarray(band_values).shape == np.asarray(explicit_values).shape
    assert np.asarray(band_values).shape == (1,)
    assert np.asarray(both_ear_values).shape == (1, 2)
    assert np.asarray(source_reduced).shape == (2,)
    assert np.asarray(ears_reduced).shape == (1,)
    assert isinstance(left_source_reduced, float)
    np.testing.assert_allclose(band_values, explicit_values)

    with pytest.raises(ValueError, match="can only be used when ear='both'"):
        hrtf_difference(
            real_hrtf,
            processed,
            metric="lsd",
            ear="left",
            frequency_bands=band,
            reduction_axis="ears",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        hrtf_difference(
            real_hrtf,
            processed,
            metric="lsd",
            frequencies=[1000.0],
            frequency_bands=band,
        )
    with pytest.raises(ValueError, match="minimum must not exceed maximum"):
        hrtf_difference(real_hrtf, processed, metric="lsd", frequency_bands=(band[1], band[0]))


@pytest.mark.parametrize(
    ("select_kwargs", "error_match"),
    [
        ({"ear": "center"}, "ear must be one of"),
        ({"plane": "diagonal"}, "plane must be one of"),
        ({"start_sample": 4, "end_sample": 4}, "Crop end must be greater than crop start"),
        ({"start_sample": True}, "start_sample must be an integer"),
        ({"positions": ["not-a-direction"]}, "named position accepts"),
    ],
)
def test_real_hrtf_select_rejects_invalid_arguments(
    real_hrtf: HRTF,
    select_kwargs: dict[str, Any],
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        real_hrtf.select(**select_kwargs)


def test_save_refuses_overwrite_and_reloads_selected_hrtf(
    real_hrtf: HRTF,
    tmp_path,
) -> None:
    source_positions = real_hrtf.Sources.get_positions()
    selected_count = min(2, source_positions.shape[0])
    selected_hrtf = real_hrtf.select(
        positions=source_positions[:selected_count],
        position_coordinate_system=_source_coordinate_system(real_hrtf),
    )
    destination = tmp_path / "selected_saved.sofa"

    saved_path = selected_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=True,
    )

    assert saved_path == destination
    assert destination.exists()
    with pytest.raises(FileExistsError):
        selected_hrtf.save(
            path=destination,
            change_sofa_dimensions=True,
        )

    overwritten_path = selected_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=True,
    )
    reloaded_hrtf = load_hrtf(overwritten_path)
    try:
        assert overwritten_path == destination
        assert reloaded_hrtf.SOFAConventions == backed_sofa_convention(real_hrtf)
        assert _ir_values(reloaded_hrtf).shape[0] == selected_count
        assert _tf_values(reloaded_hrtf).shape[0] == selected_count
        assert reloaded_hrtf.Sources.get_positions().shape == (selected_count, 3)
        assert np.isfinite(_ir_values(reloaded_hrtf)).all()
        assert np.isfinite(np.real(_tf_values(reloaded_hrtf))).all()
        assert np.isfinite(np.imag(_tf_values(reloaded_hrtf))).all()
    finally:
        _sofa(reloaded_hrtf).netCDF4_dataset.close()
