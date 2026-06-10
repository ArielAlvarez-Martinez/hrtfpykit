import os
from pathlib import Path

import numpy as np
import pytest

from hrtfpykit.hrtf.hrtf import HRTF
from hrtfpykit.hrtf import ild, load_hrtf
from hrtfpykit.utils.metrics import itd, lsd
from hrtfpykit.sofa import load_sofa


FIXTURE_SOFA_PATH = Path(__file__).parent / "pp1_HRIRs_measured.sofa"
SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
if SOFA_PATH == "" and FIXTURE_SOFA_PATH.exists():
    SOFA_PATH = str(FIXTURE_SOFA_PATH)
pytestmark = pytest.mark.skipif(
    SOFA_PATH == "" or not os.path.exists(SOFA_PATH),
    reason="Required SOFA fixture is not available",
)


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
    ir = np.array(hrtf.IR.values, copy=True)
    return hrtf.transform.modify_ir(ir)


def transform_modify_phase(hrtf: HRTF) -> HRTF:
    phase = np.array(hrtf.TF.phase, copy=True)
    phase[..., 0] = phase[..., 0] + 1.0
    return hrtf.transform.modify_phase(phase, unit="degrees")


def transform_modify_tf(hrtf: HRTF) -> HRTF:
    tf = np.array(hrtf.TF.values, copy=True)
    return hrtf.transform.modify_tf(tf)


def transform_modify_magnitude(hrtf: HRTF) -> HRTF:
    magnitude = np.array(hrtf.TF.magnitude, copy=True)
    magnitude = magnitude * 0.99
    return hrtf.transform.modify_magnitude(magnitude, scale="linear")


def transform_apply_gain(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_gain(-1.0, scale="db")


def transform_modify_fft_length(hrtf: HRTF) -> HRTF:
    return hrtf.transform.modify_fft_length(int(hrtf.IR.values.shape[-1]) + 32)


def transform_add_itd(hrtf: HRTF) -> HRTF:
    return hrtf.transform.add_itd(1, unit="samples")


def transform_delete_itd(hrtf: HRTF) -> HRTF:
    return hrtf.transform.delete_itd()


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
]


@pytest.fixture
def real_hrtf() -> HRTF:
    return load_hrtf(SOFA_PATH)


def backed_sofa_convention(hrtf: HRTF) -> str:
    return hrtf.Sofa.GlobalAttributes.get("SOFAConventions").value


def can_update_without_dimension_changes(hrtf: HRTF, default_expected: bool) -> bool:
    if not default_expected:
        return False
    if backed_sofa_convention(hrtf) != "SimpleFreeFieldHRTF":
        return True
    dataset = hrtf.Sofa.netCDF4_dataset
    if dataset is None:
        raise ValueError("SOFA dataset is not loaded")
    if hrtf.TF.values is None or hrtf.TF.frequency_bins is None:
        return False
    required_variables = ("Data.Real", "Data.Imag", "N")
    if any(variable_name not in dataset.variables for variable_name in required_variables):
        return False
    tf_shape = np.asarray(hrtf.TF.values).shape
    frequency_shape = np.asarray(hrtf.TF.frequency_bins).shape
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
    assert real_hrtf.Sofa.netCDF4_dataset is not None
    assert real_hrtf.is_transformed() is False

    ir_values = np.asarray(real_hrtf.IR.values)
    tf_values = np.asarray(real_hrtf.TF.values)
    frequency_bins = np.asarray(real_hrtf.TF.frequency_bins, dtype=float)
    source_positions = real_hrtf.Sources.get_positions()
    cartesian_positions = real_hrtf.Sources.get_positions(coordinate_system="cartesian")
    lateral_polar_positions = real_hrtf.Sources.get_positions(coordinate_system="lateral-polar")
    sample_rate = float(real_hrtf.IR.sample_rate)

    assert ir_values.ndim == 3
    assert tf_values.ndim == 3
    assert ir_values.shape[:-1] == tf_values.shape[:-1]
    assert tf_values.shape[-1] == frequency_bins.size
    assert source_positions.shape == (ir_values.shape[0], 3)
    assert cartesian_positions.shape == source_positions.shape
    assert lateral_polar_positions.shape == source_positions.shape
    assert real_hrtf.Sources.source_coordinate_system == "spherical"
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


def test_transform_returns_independent_hrtf_without_mutating_source(
    real_hrtf: HRTF,
) -> None:
    original_ir = np.array(real_hrtf.IR.values, copy=True)
    original_tf = np.array(real_hrtf.TF.values, copy=True)

    transformed_hrtf = real_hrtf.transform.apply_gain(-3.0, scale="db")

    assert transformed_hrtf is not real_hrtf
    assert transformed_hrtf.is_transformed() is True
    assert real_hrtf.is_transformed() is False
    assert transformed_hrtf.IR.values.shape == original_ir.shape
    assert transformed_hrtf.TF.values.shape == original_tf.shape
    assert not np.shares_memory(transformed_hrtf.IR.values, real_hrtf.IR.values)
    assert not np.shares_memory(transformed_hrtf.TF.values, real_hrtf.TF.values)
    assert not np.allclose(transformed_hrtf.TF.values, original_tf)
    assert np.allclose(real_hrtf.IR.values, original_ir)
    assert np.allclose(real_hrtf.TF.values, original_tf)


def test_reset_restores_selected_transformed_hrtf_to_backed_state(
    real_hrtf: HRTF,
) -> None:
    source_positions = real_hrtf.Sources.get_positions()
    selected_count = min(2, source_positions.shape[0])
    selected_hrtf = real_hrtf.select(
        positions=source_positions[:selected_count],
        position_coordinate_system=real_hrtf.Sources.source_coordinate_system,
    )
    transformed_hrtf = selected_hrtf.transform.apply_gain(-3.0, scale="db")

    assert transformed_hrtf.IR.values.shape[0] == selected_count
    assert transformed_hrtf.Sources.get_positions().shape[0] == selected_count
    assert transformed_hrtf.is_transformed() is True

    restored_hrtf = transformed_hrtf.reset()

    assert restored_hrtf is transformed_hrtf
    assert restored_hrtf.is_transformed() is False
    assert restored_hrtf.IR.values.shape == real_hrtf.IR.values.shape
    assert restored_hrtf.TF.values.shape == real_hrtf.TF.values.shape
    assert (
        restored_hrtf.Sources.get_positions().shape
        == real_hrtf.Sources.get_positions().shape
    )
    assert np.allclose(restored_hrtf.IR.values, real_hrtf.IR.values)
    assert np.allclose(restored_hrtf.TF.values, real_hrtf.TF.values)


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


def test_update_sofa_no_transform_prints_message(real_hrtf: HRTF, capsys) -> None:
    assert real_hrtf.is_transformed() is False
    expect_noop = can_update_without_dimension_changes(real_hrtf, True)

    real_hrtf.update_sofa()
    captured = capsys.readouterr()

    if expect_noop:
        assert "already up to date" in captured.out
    else:
        dataset = real_hrtf.Sofa.netCDF4_dataset
        assert dataset is not None
        assert tuple(dataset.variables["Data.Real"].shape) == np.asarray(real_hrtf.TF.values).shape
        assert tuple(dataset.variables["Data.Imag"].shape) == np.asarray(real_hrtf.TF.values).shape
        assert tuple(dataset.variables["N"].shape) == np.asarray(real_hrtf.TF.frequency_bins).shape
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
    selected_hrtf = real_hrtf.select(positions=["front", "left", "right"])
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
        saved_sofa.GlobalAttributes.get("SOFAConventions").value
        == resolved_expected_convention
    )
    assert saved_sofa.GlobalAttributes.get("DataType").value == expected_data_type

    for variable_name in present_variables:
        assert variable_name in saved_variables
    for variable_name in absent_variables:
        assert variable_name not in saved_variables

    source_position = np.asarray(saved_sofa.Variables.get("SourcePosition").value)
    assert source_position.shape[0] == 3

    if "Data.IR" in present_variables:
        data_ir = np.asarray(saved_sofa.Variables.get("Data.IR").value)
        assert data_ir.shape[0] == 3
    if "Data.Real" in present_variables:
        data_real = np.asarray(saved_sofa.Variables.get("Data.Real").value)
        assert data_real.shape[0] == 3

    saved_sofa.netCDF4_dataset.close()


def test_real_hrtf_selects_positions_and_ears(real_hrtf: HRTF) -> None:
    selected_hrtf = real_hrtf.select(
        positions=["front", "left", "right"],
        ear="left",
    )

    assert selected_hrtf.IR.values.shape[0] == 3
    assert selected_hrtf.IR.values.ndim == 2
    assert selected_hrtf.TF.values is not None


def test_real_hrtf_selects_numeric_positions_crop_and_right_ear(
    real_hrtf: HRTF,
) -> None:
    source_positions = real_hrtf.Sources.get_positions()
    selected_count = min(2, source_positions.shape[0])
    crop_start = 1
    crop_end = min(17, real_hrtf.IR.values.shape[-1])
    original_ir = np.array(real_hrtf.IR.values, copy=True)
    original_tf = np.array(real_hrtf.TF.values, copy=True)

    selected_hrtf = real_hrtf.select(
        positions=source_positions[:selected_count],
        position_coordinate_system=real_hrtf.Sources.source_coordinate_system,
        ear="right",
        start_sample=crop_start,
        end_sample=crop_end,
    )

    assert crop_end > crop_start
    crop_length = crop_end - crop_start
    assert selected_hrtf.IR.values.shape == (selected_count, crop_end - crop_start)
    assert selected_hrtf.TF.values.shape[0] == selected_count
    assert selected_hrtf.TF.values.ndim == 2
    assert selected_hrtf.TF.values.shape[-1] == np.fft.rfftfreq(
        crop_length,
        d=1.0 / selected_hrtf.IR.sample_rate,
    ).shape[0]
    assert selected_hrtf.fft_length == crop_length
    assert selected_hrtf.Sources.get_positions().shape == (selected_count, 3)
    assert np.allclose(real_hrtf.IR.values, original_ir)
    assert np.allclose(real_hrtf.TF.values, original_tf)


def test_real_hrtf_metric_itd_runs_on_loaded_file(real_hrtf: HRTF) -> None:
    values = itd(real_hrtf)

    assert np.asarray(values).shape[0] == real_hrtf.IR.values.shape[0]
    assert np.all(np.isfinite(values))


def test_real_hrtf_metric_ild_runs_on_loaded_file(real_hrtf: HRTF) -> None:
    broad_band_values = ild(real_hrtf, mode="broad-band")
    frequency_dependent_values = ild(
        real_hrtf,
        mode="frequency-dependent",
    )
    absolute_values = ild(real_hrtf, mode="broad-band", absolute=True)

    assert broad_band_values.shape == real_hrtf.IR.values.shape[:-2]
    assert frequency_dependent_values.shape == (
        real_hrtf.TF.values.shape[:-2] + (real_hrtf.TF.values.shape[-1],)
    )
    assert np.all(np.isfinite(broad_band_values))
    assert np.all(np.isfinite(frequency_dependent_values))
    np.testing.assert_allclose(absolute_values, np.abs(broad_band_values))


def test_real_hrtf_metric_lsd_accepts_frequency_bands(real_hrtf: HRTF) -> None:
    processed = real_hrtf.transform.apply_gain(-1.0, scale="db")
    frequency_bins = np.asarray(real_hrtf.TF.frequency_bins, dtype=float)
    positive_bins = frequency_bins[frequency_bins >= 20.0]
    if positive_bins.size < 4:
        pytest.skip("LSD frequency band test requires at least four positive frequency bins")
    band = (float(positive_bins[0]), float(positive_bins[min(3, positive_bins.size - 1)]))
    expected_indices = np.where((frequency_bins >= band[0]) & (frequency_bins <= band[1]))[0]

    band_values = lsd(
        real_hrtf,
        processed,
        ear="left",
        positions="front",
        frequency_bands=band,
    )
    explicit_values = lsd(
        real_hrtf,
        processed,
        ear="left",
        positions="front",
        frequencies=frequency_bins[expected_indices],
    )
    both_ear_values = lsd(
        real_hrtf,
        processed,
        ear="both",
        positions="front",
        frequency_bands=band,
    )
    source_reduced = lsd(
        real_hrtf,
        processed,
        ear="both",
        positions="front",
        frequency_bands=band,
        reduction_axis="sources",
    )
    ears_reduced = lsd(
        real_hrtf,
        processed,
        ear="both",
        positions="front",
        frequency_bands=band,
        reduction_axis="ears",
    )
    left_source_reduced = lsd(
        real_hrtf,
        processed,
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
        lsd(
            real_hrtf,
            processed,
            ear="left",
            frequency_bands=band,
            reduction_axis="ears",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        lsd(
            real_hrtf,
            processed,
            frequencies=[1000.0],
            frequency_bands=band,
        )
    with pytest.raises(ValueError, match="minimum must not exceed maximum"):
        lsd(real_hrtf, processed, frequency_bands=(band[1], band[0]))


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
    select_kwargs: dict,
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
        position_coordinate_system=real_hrtf.Sources.source_coordinate_system,
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
        assert reloaded_hrtf.IR.values.shape[0] == selected_count
        assert reloaded_hrtf.TF.values.shape[0] == selected_count
        assert reloaded_hrtf.Sources.get_positions().shape == (selected_count, 3)
        assert np.isfinite(reloaded_hrtf.IR.values).all()
        assert np.isfinite(np.real(reloaded_hrtf.TF.values)).all()
        assert np.isfinite(np.imag(reloaded_hrtf.TF.values)).all()
    finally:
        reloaded_hrtf.Sofa.netCDF4_dataset.close()
