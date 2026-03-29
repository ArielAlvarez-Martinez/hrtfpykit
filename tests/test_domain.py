import numpy as np
import pytest

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


def test_transform_apply_ir_crop_by_seconds_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.arange(8, dtype=float).reshape(1, -1)
    hrtf.IR.sample_rate = 4.0

    transformed_hrtf = hrtf.transform.apply_ir_crop(start_seconds=0.5, end_seconds=1.5)

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


def test_transform_apply_filter_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    transformed_hrtf = hrtf.transform.apply_filter(filter="lowpass", cutoff=3_000.0, num_taps=5)

    assert hrtf.IR.values.shape[-1] == 7
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None
    assert transformed_hrtf.IR.values.shape[-1] == 7
    assert transformed_hrtf.TF.values is not None
    assert transformed_hrtf.TF.frequency_bins is not None


def test_transform_apply_tf_crop_updates_ir() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf_with_fft = hrtf.transform.modify_fft_length(8)

    original_ir = np.array(hrtf_with_fft.IR.values, copy=True)
    cropped_hrtf = hrtf_with_fft.transform.apply_tf_crop(start=1, end=3)

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
        domain="tf",
    )

    assert padded_hrtf is not hrtf_with_tf
    assert hrtf_with_tf.TF.values.shape[-1] == 5
    assert hrtf_with_tf.TF.frequency_bins.shape[-1] == 5
    assert padded_hrtf.TF.values.shape[-1] == 7
    assert padded_hrtf.TF.frequency_bins.shape[-1] == 7
    assert padded_hrtf.IR.values is not None
    assert padded_hrtf.IR.sample_rate is not None
    assert np.array_equal(hrtf_with_tf.TF.frequency_bins, original_bins)


def test_transform_modify_positions_reference_returns_new_hrtf() -> None:
    hrtf = HRTF()
    transformed_hrtf = hrtf.transform.modify_positions_reference()

    assert transformed_hrtf is not hrtf
    assert transformed_hrtf.Sofa is hrtf.Sofa
    assert transformed_hrtf.SOFAConventions == hrtf.SOFAConventions


def test_transform_apply_padding_invalid_domain_raises() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    with pytest.raises(ValueError, match="domain must be 'ir' or 'tf'"):
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
