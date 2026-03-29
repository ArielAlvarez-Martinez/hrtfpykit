import numpy as np
import pytest

from hrtfpykit.hrtf import HRTF


def test_transform_apply_window_unsupported_keeps_values() -> None:
    hrtf = HRTF()
    original = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.values = original.copy()
    hrtf.IR.sample_rate = 48_000.0

    with pytest.raises(ValueError, match="Unsupported window"):
        hrtf.IR.transform.apply_window("unsupported_window")

    assert np.array_equal(hrtf.IR.values, original)
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None


def test_transform_apply_window_supported_updates_values_and_tf() -> None:
    hrtf = HRTF()
    original = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.values = original.copy()
    hrtf.IR.sample_rate = 48_000.0

    hrtf.IR.transform.apply_window("hann")

    assert not np.array_equal(hrtf.IR.values, original)
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None


def test_transform_apply_ir_crop_by_seconds_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.arange(8, dtype=float).reshape(1, -1)
    hrtf.IR.sample_rate = 4.0

    hrtf.IR.transform.apply_ir_crop(start_seconds=0.5, end_seconds=1.5)

    assert hrtf.IR.values.shape[-1] == 4
    assert np.array_equal(hrtf.IR.values, np.array([[2.0, 3.0, 4.0, 5.0]]))
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None


def test_transform_apply_padding_in_ir_domain_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    hrtf.IR.transform.apply_padding(padding_length=2, location="end")

    assert hrtf.IR.values.shape[-1] == 6
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None


def test_transform_apply_filter_updates_ir_and_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    hrtf.IR.transform.apply_filter(filter="lowpass", cutoff=3_000.0, num_taps=5)

    assert hrtf.IR.values.shape[-1] == 7
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None


def test_transform_apply_tf_crop_updates_ir() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0
    hrtf.IR.transform.modify_fft_length(8)

    original_ir = np.array(hrtf.IR.values, copy=True)
    hrtf.TF.transform.apply_tf_crop(start=1, end=3)

    assert hrtf.TF.values is not None
    assert hrtf.IR.values is not None
    assert np.allclose(hrtf.TF.values[..., 0], 0.0)
    assert np.allclose(hrtf.TF.values[..., 3:], 0.0)
    assert not np.allclose(hrtf.IR.values, original_ir)


def test_transform_resampling_updates_sample_rate_and_syncs_tf() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    hrtf.IR.transform.upsampling(96_000.0)
    assert hrtf.IR.sample_rate == 96_000.0
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None

    hrtf.IR.transform.downsampling(48_000.0)
    assert hrtf.IR.sample_rate == 48_000.0
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None


def test_transform_property_is_cached() -> None:
    hrtf = HRTF()
    assert hrtf.IR.transform is hrtf.IR.transform
    assert hrtf.TF.transform is hrtf.TF.transform
    assert hrtf.Sources.transform is hrtf.Sources.transform


def test_sources_transform_placeholder_method_returns_none() -> None:
    hrtf = HRTF()
    assert hrtf.Sources.transform.modify_positions_reference() is None


def test_modify_fft_length_delegates_to_transform() -> None:
    hrtf = HRTF()
    hrtf.IR.values = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    hrtf.IR.sample_rate = 48_000.0

    hrtf.modify_fft_length(8)

    assert hrtf.fft_length == 8
    assert hrtf.TF.values.shape[-1] == 5
    assert hrtf.TF.frequency_bins.shape[-1] == 5


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
