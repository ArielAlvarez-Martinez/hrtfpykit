import numpy as np
import pytest

from hrtfpykit.dsp import (
    apply_fir_filter,
    apply_iir_filter,
    calculate_itd,
    minimum_phase,
    modify_magnitude,
    modify_phase,
)
from hrtfpykit.hrtf import HRTF


def test_transform_apply_window_unsupported_keeps_values() -> None:
    hrtf = HRTF()
    original = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.values = original.copy()
    hrtf.IR.sample_rate = 48_000.0

    with pytest.raises(ValueError, match="Unsupported window"):
        hrtf.transform.apply_window("unsupported_window")

    assert np.array_equal(hrtf.IR.values, original)
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None


def test_transform_apply_window_supported_updates_values_and_tf() -> None:
    hrtf = HRTF()
    original = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.values = original.copy()
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.apply_window("hann")

    assert transformed_hrtf is not hrtf
    assert np.array_equal(hrtf.IR.values, original)
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert not np.array_equal(transformed_hrtf.IR.values, original)
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_transform_apply_crop_time_by_seconds_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.arange(8, dtype=float).reshape(1, -1)
    hrtf.IR.sample_rate = 4.0

    transformed_hrtf = hrtf.transform.apply_crop(
        domain="time",
        start_seconds=0.5,
        end_seconds=1.5,
    )

    assert hrtf.IR.values.shape[-1] == 8
    assert np.array_equal(hrtf.IR.values, np.arange(8, dtype=float).reshape(1, -1))
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.IR.values.shape[-1] == 4
    assert np.array_equal(transformed_hrtf.IR.values, np.array([[2.0, 3.0, 4.0, 5.0]]))
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_transform_apply_padding_in_ir_domain_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.apply_padding(padding_length=2, location="end")

    assert hrtf.IR.values.shape[-1] == 4
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.IR.values.shape[-1] == 6
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_transform_apply_fir_filter_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.apply_fir_filter(
        filter="lowpass",
        cutoff=3_000.0,
        num_taps=5,
    )

    assert hrtf.IR.values.shape[-1] == 7
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.IR.values.shape[-1] == 7
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_transform_apply_iir_filter_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.apply_iir_filter(
        filter="lowpass",
        cutoff=3_000.0,
        order=4,
    )

    assert transformed_hrtf is not hrtf
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.IR.values.shape[-1] == 7
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_dsp_apply_fir_and_iir_filter_accept_ndarray_and_ir() -> None:
    ir_values = np.zeros((2, 32), dtype=float)
    ir_values[:, 4] = 1.0

    fir_filtered = apply_fir_filter(
        ir_values,
        filter="lowpass",
        sample_rate=48_000.0,
        cutoff=3_000.0,
        num_taps=11,
    )
    assert fir_filtered.shape == ir_values.shape
    assert np.all(np.isfinite(fir_filtered))

    iir_filtered = apply_iir_filter(
        ir_values,
        filter="lowpass",
        sample_rate=48_000.0,
        cutoff=3_000.0,
        order=4,
    )
    assert iir_filtered.shape == ir_values.shape
    assert np.all(np.isfinite(iir_filtered))

    hrtf = HRTF()
    hrtf.IR.values = ir_values.copy()
    hrtf.IR.sample_rate = 48_000.0
    fir_from_ir = apply_fir_filter(
        hrtf.IR,
        filter="lowpass",
        sample_rate=48_000.0,
        cutoff=3_000.0,
        num_taps=11,
    )
    iir_from_ir = apply_iir_filter(
        hrtf.IR,
        filter="lowpass",
        sample_rate=48_000.0,
        cutoff=3_000.0,
        order=4,
    )
    assert fir_from_ir.shape == ir_values.shape
    assert iir_from_ir.shape == ir_values.shape

def test_dsp_minimum_phase_accepts_ndarray_and_ir() -> None:
    ir_values = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)

    transformed_array = minimum_phase(ir_values)
    assert transformed_array.shape == ir_values.shape
    assert np.all(np.isfinite(transformed_array))

    hrtf = HRTF()
    hrtf.IR.values = ir_values.copy()
    transformed_ir = minimum_phase(hrtf.IR)
    assert transformed_ir.shape == ir_values.shape
    assert np.all(np.isfinite(transformed_ir))
    assert np.array_equal(hrtf.IR.values, ir_values)
    assert np.allclose(transformed_array, transformed_ir)


def test_dsp_minimum_phase_rejects_tf_input() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf_with_tf = hrtf.transform.modify_fft_length(8)

    with pytest.raises(ValueError, match="NumPy array or an IR instance"):
        minimum_phase(hrtf_with_tf.TF)


def test_transform_minimum_phase_updates_ir_and_tf_immutably() -> None:
    hrtf = HRTF()
    original_ir = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.values = original_ir.copy()
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.minimum_phase()

    assert transformed_hrtf is not hrtf
    assert np.array_equal(hrtf.IR.values, original_ir)
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.IR.values.shape == original_ir.shape
    assert np.all(np.isfinite(transformed_hrtf.IR.values))
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_dsp_minimum_phase_cepstrum_method_runs() -> None:
    ir_values = np.array([[0.0, 1.0, 0.25, -0.1, 0.05, 0.0, 0.0, 0.0]], dtype=float)

    transformed_values = minimum_phase(
        ir_values,
        method="cepstrum",
        fft_length=16,
    )

    assert transformed_values.shape == ir_values.shape
    assert np.all(np.isfinite(transformed_values))


def test_dsp_modify_phase_accepts_ndarray_and_tf() -> None:
    tf_values = np.array([[1.0 + 1.0j, 0.5 - 0.5j]], dtype=complex)
    new_phase_degrees = np.array([[0.0, 180.0]], dtype=float)

    modified_array = modify_phase(tf_values, new_phase_degrees, unit="degrees")
    assert modified_array.shape == tf_values.shape
    assert np.allclose(np.abs(modified_array), np.abs(tf_values))
    assert np.allclose(np.angle(modified_array, deg=True), new_phase_degrees)

    hrtf = HRTF()
    hrtf.TF.values = tf_values.copy()
    modified_tf_object = modify_phase(hrtf.TF, new_phase_degrees, unit="degrees")
    assert modified_tf_object.shape == tf_values.shape
    assert np.allclose(np.abs(modified_tf_object), np.abs(tf_values))
    assert np.array_equal(hrtf.TF.values, tf_values)


def test_dsp_modify_magnitude_accepts_ndarray_and_tf() -> None:
    tf_values = np.array([[1.0 + 1.0j, 0.5 - 0.5j]], dtype=complex)
    new_magnitude_linear = np.array([[0.25, 0.75]], dtype=float)

    modified_linear = modify_magnitude(tf_values, new_magnitude_linear, scale="linear")
    assert modified_linear.shape == tf_values.shape
    assert np.allclose(np.abs(modified_linear), new_magnitude_linear)
    assert np.allclose(np.angle(modified_linear), np.angle(tf_values))

    new_magnitude_db = np.array([[-6.0, 0.0]], dtype=float)
    modified_db = modify_magnitude(tf_values, new_magnitude_db, scale="db")
    assert modified_db.shape == tf_values.shape
    assert np.allclose(np.abs(modified_db), 10.0 ** (new_magnitude_db / 20.0))
    assert np.allclose(np.angle(modified_db), np.angle(tf_values))

    hrtf = HRTF()
    hrtf.TF.values = tf_values.copy()
    modified_tf_object = modify_magnitude(hrtf.TF, new_magnitude_linear, scale="lineal")
    assert modified_tf_object.shape == tf_values.shape
    assert np.allclose(np.abs(modified_tf_object), new_magnitude_linear)
    assert np.array_equal(hrtf.TF.values, tf_values)


def test_dsp_modify_phase_and_magnitude_validation() -> None:
    tf_values = np.array([[1.0 + 0.0j, 0.5 + 0.0j]], dtype=complex)
    wrong_shape = np.array([[1.0]], dtype=float)

    with pytest.raises(ValueError, match="new_phase must match TF shape"):
        modify_phase(tf_values, wrong_shape)

    with pytest.raises(ValueError, match="unit must be one of: degrees, radians"):
        modify_phase(tf_values, np.zeros_like(tf_values, dtype=float), unit="grad")

    with pytest.raises(ValueError, match="new_magnitude must match TF shape"):
        modify_magnitude(tf_values, wrong_shape)

    with pytest.raises(ValueError, match="scale must be one of: linear, lineal, db"):
        modify_magnitude(tf_values, np.ones_like(tf_values, dtype=float), scale="log")

    with pytest.raises(ValueError, match="new_magnitude must be non-negative"):
        modify_magnitude(tf_values, np.array([[-1.0, 0.5]], dtype=float), scale="linear")


def test_dsp_calculate_itd_accepts_ndarray_and_ir() -> None:
    sample_rate = 48_000.0
    ir_values = np.zeros((3, 2, 128), dtype=float)

    ir_values[0, 0, 30] = 1.0
    ir_values[0, 1, 32] = 1.0
    ir_values[1, 0, 50] = 1.0
    ir_values[1, 1, 48] = 1.0
    ir_values[2, 0, 70] = 1.0
    ir_values[2, 1, 70] = 1.0

    expected_samples = np.array([-2, 2, 0], dtype=int)
    expected_seconds = expected_samples / sample_rate

    itd_array = calculate_itd(
        ir_values,
        sample_rate=sample_rate,
        method="maxiacce",
        output="seconds",
    )
    assert itd_array.shape == (3,)
    assert np.allclose(itd_array, expected_seconds)

    itd_array_samples = calculate_itd(
        ir_values,
        sample_rate=sample_rate,
        method="maxiacce",
        output="samples",
    )
    assert np.issubdtype(itd_array_samples.dtype, np.integer)
    assert np.allclose(itd_array_samples, expected_samples)

    hrtf = HRTF()
    hrtf.IR.values = ir_values.copy()
    hrtf.IR.sample_rate = sample_rate

    itd_ir = calculate_itd(hrtf.IR, method="maxiacce", output="seconds")
    assert itd_ir.shape == (3,)
    assert np.allclose(itd_ir, expected_seconds)
    assert np.array_equal(hrtf.IR.values, ir_values)


def test_dsp_calculate_itd_threshold_method() -> None:
    sample_rate = 48_000.0
    ir_values = np.zeros((2, 2, 128), dtype=float)
    ir_values[0, 0, 20] = 1.0
    ir_values[0, 1, 24] = 1.0
    ir_values[1, 0, 40] = 1.0
    ir_values[1, 1, 36] = 1.0

    expected = np.array([-4.0 / sample_rate, 4.0 / sample_rate], dtype=float)
    itd_threshold = calculate_itd(
        ir_values,
        method="threshold",
        sample_rate=sample_rate,
        thresh_level=-10.0,
        upper_cut_freq=3_000.0,
        filter_order=11,
    )
    assert itd_threshold.shape == (2,)
    assert np.allclose(itd_threshold, expected)


def test_dsp_calculate_itd_validation() -> None:
    with pytest.raises(ValueError, match="sample_rate is required"):
        calculate_itd(np.zeros((2, 16), dtype=float))

    with pytest.raises(ValueError, match="at least two channels"):
        calculate_itd(
            np.zeros((1, 16), dtype=float),
            sample_rate=48_000.0,
        )

    with pytest.raises(ValueError, match="method must be one of: threshold, maxiacce"):
        calculate_itd(
            np.zeros((2, 16), dtype=float),
            sample_rate=48_000.0,
            method="xcorr",
        )

    with pytest.raises(ValueError, match="output must be one of: seconds, samples"):
        calculate_itd(
            np.zeros((2, 2, 16), dtype=float),
            sample_rate=48_000.0,
            output="ms",
        )


def test_transform_modify_phase_updates_tf_and_ir_immutably() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf_with_tf = hrtf.transform.modify_fft_length(8)

    original_tf_values = np.array(hrtf_with_tf.TF.values, copy=True)
    new_phase_degrees = np.full(original_tf_values.shape, 45.0, dtype=float)
    transformed_hrtf = hrtf_with_tf.transform.modify_phase(
        new_phase=new_phase_degrees,
        unit="degrees",
    )

    assert transformed_hrtf is not hrtf_with_tf
    assert np.array_equal(hrtf_with_tf.TF.values, original_tf_values)
    assert np.allclose(np.abs(transformed_hrtf.TF.values), np.abs(original_tf_values))
    nonzero_mask = np.abs(transformed_hrtf.TF.values) > 1e-8
    assert np.allclose(
        np.angle(transformed_hrtf.TF.values, deg=True)[nonzero_mask],
        new_phase_degrees[nonzero_mask],
    )
    assert transformed_hrtf.IR.values is not None
    assert transformed_hrtf.IR.sample_rate is not None


def test_transform_modify_magnitude_updates_tf_and_ir_immutably() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf_with_tf = hrtf.transform.modify_fft_length(8)

    original_tf_values = np.array(hrtf_with_tf.TF.values, copy=True)
    new_magnitude = np.full(original_tf_values.shape, 0.5, dtype=float)
    transformed_linear = hrtf_with_tf.transform.modify_magnitude(
        new_magnitude=new_magnitude,
        scale="linear",
    )

    assert transformed_linear is not hrtf_with_tf
    assert np.array_equal(hrtf_with_tf.TF.values, original_tf_values)
    assert np.allclose(np.abs(transformed_linear.TF.values), new_magnitude)
    phase_mask = np.abs(original_tf_values) > 1e-8
    assert np.allclose(
        np.angle(transformed_linear.TF.values)[phase_mask],
        np.angle(original_tf_values)[phase_mask],
    )
    assert transformed_linear.IR.values is not None
    assert transformed_linear.IR.sample_rate is not None

    new_magnitude_db = np.full(original_tf_values.shape, -6.0, dtype=float)
    transformed_db = hrtf_with_tf.transform.modify_magnitude(
        new_magnitude=new_magnitude_db,
        scale="db",
    )
    assert np.allclose(np.abs(transformed_db.TF.values), 10.0 ** (new_magnitude_db / 20.0))


def test_transform_apply_crop_frequency_updates_ir() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf_with_fft = hrtf.transform.modify_fft_length(8)

    original_ir = np.array(hrtf_with_fft.IR.values, copy=True)
    cropped_hrtf = hrtf_with_fft.transform.apply_crop(
        domain="frequency",
        start=1,
        end=3,
    )

    assert hrtf.fft_length is None
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert np.allclose(hrtf_with_fft.IR.values, original_ir)
    assert cropped_hrtf.TF.values is not None
    assert cropped_hrtf.IR.values is not None
    assert np.allclose(cropped_hrtf.TF.values[..., 0], 0.0)
    assert np.allclose(cropped_hrtf.TF.values[..., 3:], 0.0)
    assert not np.allclose(cropped_hrtf.IR.values, original_ir)


def test_transform_resampling_updates_sample_rate_and_syncs_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    upsampled_hrtf = hrtf.transform.upsampling(96_000.0)
    assert hrtf.IR.sample_rate == 48_000.0
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert upsampled_hrtf.IR.sample_rate == 96_000.0
    assert upsampled_hrtf.TF.values is not None
    assert upsampled_hrtf.TF.frequency_bins is not None

    downsampled_hrtf = upsampled_hrtf.transform.downsampling(48_000.0)
    assert upsampled_hrtf.IR.sample_rate == 96_000.0
    assert upsampled_hrtf.TF.values is not None
    assert upsampled_hrtf.TF.frequency_bins is not None
    assert downsampled_hrtf is not upsampled_hrtf
    assert downsampled_hrtf.IR.sample_rate == 48_000.0
    assert downsampled_hrtf.TF.values is not None
    assert downsampled_hrtf.TF.frequency_bins is not None


def test_clone_returns_independent_hrtf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf = hrtf.transform.modify_fft_length(8)

    cloned = hrtf.clone()
    assert cloned is not hrtf
    assert np.array_equal(cloned.IR.values, hrtf.IR.values)
    assert np.array_equal(cloned.TF.values, hrtf.TF.values)
    assert np.array_equal(cloned.TF.frequency_bins, hrtf.TF.frequency_bins)

    cloned.IR.values[..., 0] = 999.0
    assert not np.array_equal(cloned.IR.values, hrtf.IR.values)
    assert np.all(hrtf.IR.values[..., 0] != 999.0)

    cloned.TF.values[..., 0] = 0.0
    assert not np.array_equal(cloned.TF.values, hrtf.TF.values)
    assert np.any(hrtf.TF.values[..., 0] != 0.0)

    cloned.TF.frequency_bins[..., 0] = -1.0
    assert not np.array_equal(cloned.TF.frequency_bins, hrtf.TF.frequency_bins)


def test_transform_apply_padding_in_tf_domain_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf_with_tf = hrtf.transform.modify_fft_length(8)
    original_bins = np.array(hrtf_with_tf.TF.frequency_bins, copy=True)

    padded_hrtf = hrtf_with_tf.transform.apply_padding(
        padding_length=2,
        location="end",
        domain="frequency",
    )

    assert padded_hrtf is not hrtf_with_tf
    assert hrtf_with_tf.TF.values.shape[-1] == 5
    assert hrtf_with_tf.TF.frequency_bins.shape[-1] == 5
    assert padded_hrtf.TF.values.shape[-1] == 7
    assert padded_hrtf.TF.frequency_bins.shape[-1] == 7
    assert padded_hrtf.IR.values is not None
    assert padded_hrtf.IR.sample_rate is not None
    assert np.array_equal(hrtf_with_tf.TF.frequency_bins, original_bins)


def test_transform_apply_padding_invalid_domain_raises() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    with pytest.raises(ValueError, match="domain must be 'time' or 'frequency'"):
        hrtf.transform.apply_padding(
            padding_length=2,
            location="end",
            domain="other",
        )

    assert hrtf.IR.values.shape[-1] == 4
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None


def test_transform_modify_fft_length_without_ir_raises_and_does_not_mutate() -> None:
    hrtf = HRTF()
    with pytest.raises(ValueError, match="IR data is not available"):
        hrtf.transform.modify_fft_length(8)

    assert hrtf.fft_length is None
    assert hrtf.IR.values is None
    assert hrtf.TF.values is None


def test_transform_apply_window_returns_independent_instance() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.apply_window("hann")
    transformed_hrtf.IR.values[..., 0] = 123.0

    assert np.all(hrtf.IR.values[..., 0] != 123.0)
    assert hrtf.TF.values is None


def test_transform_resampling_chain_preserves_previous_instances() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    upsampled_hrtf = hrtf.transform.upsampling(96_000.0)
    downsampled_hrtf = upsampled_hrtf.transform.downsampling(48_000.0)

    assert hrtf.IR.sample_rate == 48_000.0
    assert upsampled_hrtf.IR.sample_rate == 96_000.0
    assert downsampled_hrtf.IR.sample_rate == 48_000.0
    assert upsampled_hrtf is not downsampled_hrtf
    assert hrtf is not upsampled_hrtf
    assert hrtf is not downsampled_hrtf


def test_transform_property_is_cached() -> None:
    hrtf = HRTF()
    assert hrtf.transform is hrtf.transform


def test_modify_fft_length_delegates_to_transform() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.modify_fft_length(8)

    assert hrtf.fft_length is None
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.fft_length == 8
    assert transformed_hrtf.TF.values.shape[-1] == 5
    assert transformed_hrtf.TF.frequency_bins.shape[-1] == 5


def test_ir_length_without_values_raises_by_design() -> None:
    hrtf = HRTF()
    with pytest.raises((AttributeError, TypeError, ValueError)):
        _ = hrtf.IR.ir_length


def test_tf_length_without_values_raises_by_design() -> None:
    hrtf = HRTF()
    with pytest.raises((AttributeError, TypeError, ValueError)):
        _ = hrtf.TF.tf_length


@pytest.mark.parametrize(
    "property_name",
    ["frequency_bins_step", "min_frequency_bin", "max_frequency_bin"],
)
def test_tf_frequency_metadata_without_bins_raises_by_design(property_name: str) -> None:
    hrtf = HRTF()
    hrtf.TF.values = np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=complex)

    with pytest.raises((AttributeError, TypeError, ValueError)):
        _ = getattr(hrtf.TF, property_name)
