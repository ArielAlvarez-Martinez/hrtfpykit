import numpy as np
import pytest

from hrtfpykit.hrtf import HRTF


def test_ir_apply_window_unsupported_keeps_values() -> None:
    hrtf = HRTF()
    original = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.values = original.copy()
    hrtf.IR.sample_rate = 48_000.0

    with pytest.raises(ValueError, match="Unsupported window"):
        hrtf.IR.apply_window("unsupported_window")

    assert np.array_equal(hrtf.IR.values, original)
    assert hrtf.TF.values is None
    assert hrtf.TF.frequency_bins is None


def test_ir_apply_window_supported_updates_values_and_tf() -> None:
    hrtf = HRTF()
    original = np.array([[1.0, 0.5, 0.25, 0.0]], dtype=float)
    hrtf.IR.values = original.copy()
    hrtf.IR.sample_rate = 48_000.0

    hrtf.IR.apply_window("hann")

    assert not np.array_equal(hrtf.IR.values, original)
    assert hrtf.TF.values is not None
    assert hrtf.TF.frequency_bins is not None


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
