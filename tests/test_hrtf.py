<<<<<<< HEAD
import numpy as np
import pytest

from hrtfpykit import load_hrtf, load_sofa
from hrtfpykit.hrtf.hrtf import HRTF


SOFA_PATH = "hrtfs/P0001_FreeFieldComp_44kHz.sofa"


def _transform_apply_window(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_window("hann")


def _transform_apply_padding(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_padding(padding_length=8, location="end")


def _transform_apply_fir_filter(hrtf: HRTF) -> HRTF:
=======
import os

import numpy as np
import pytest

from hrtfpykit.hrtf.hrtf import HRTF
from hrtfpykit.hrtf import load_hrtf
from hrtfpykit.hrtf.metrics import itd
from hrtfpykit.sofa import load_sofa


SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
pytestmark = pytest.mark.skipif(
    SOFA_PATH == "" or not os.path.exists(SOFA_PATH),
    reason="Required local SOFA file is not available",
)


def transform_apply_window(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_window("hann")


def transform_apply_padding(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_padding(padding_length=8, location="end")


def transform_apply_fir_filter(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    return hrtf.transform.apply_fir_filter(
        filter="lowpass",
        cutoff=3000.0,
        num_taps=31,
    )


<<<<<<< HEAD
def _transform_apply_iir_filter(hrtf: HRTF) -> HRTF:
=======
def transform_apply_iir_filter(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    return hrtf.transform.apply_iir_filter(
        filter="lowpass",
        cutoff=3000.0,
        order=4,
    )


<<<<<<< HEAD
def _transform_minimum_phase(hrtf: HRTF) -> HRTF:
    return hrtf.transform.minimum_phase()


def _transform_to_ctf(hrtf: HRTF) -> HRTF:
    return hrtf.transform.to_ctf(weights=False)


def _transform_to_dtf(hrtf: HRTF) -> HRTF:
    return hrtf.transform.to_dtf(weights=False)


def _transform_modify_ir(hrtf: HRTF) -> HRTF:
=======
def transform_minimum_phase(hrtf: HRTF) -> HRTF:
    return hrtf.transform.minimum_phase()


def transform_to_ctf(hrtf: HRTF) -> HRTF:
    return hrtf.transform.to_ctf(weights=False)


def transform_to_dtf(hrtf: HRTF) -> HRTF:
    return hrtf.transform.to_dtf(weights=False)


def transform_modify_ir(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    ir = np.array(hrtf.IR.values, copy=True)
    return hrtf.transform.modify_ir(ir)


<<<<<<< HEAD
def _transform_modify_phase(hrtf: HRTF) -> HRTF:
=======
def transform_modify_phase(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    phase = np.array(hrtf.TF.phase, copy=True)
    phase[..., 0] = phase[..., 0] + 1.0
    return hrtf.transform.modify_phase(phase, unit="degrees")


<<<<<<< HEAD
def _transform_modify_tf(hrtf: HRTF) -> HRTF:
=======
def transform_modify_tf(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    tf = np.array(hrtf.TF.values, copy=True)
    return hrtf.transform.modify_tf(tf)


<<<<<<< HEAD
def _transform_modify_magnitude(hrtf: HRTF) -> HRTF:
=======
def transform_modify_magnitude(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    magnitude = np.array(hrtf.TF.magnitude, copy=True)
    magnitude = magnitude * 0.99
    return hrtf.transform.modify_magnitude(magnitude, scale="linear")


<<<<<<< HEAD
def _transform_apply_gain(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_gain(-1.0, scale="db")


def _transform_modify_fft_length(hrtf: HRTF) -> HRTF:
    return hrtf.transform.modify_fft_length(int(hrtf.IR.values.shape[-1]) + 32)


def _transform_modify_source_coordinate_system(hrtf: HRTF) -> HRTF:
    return hrtf.transform.modify_source_coordinate_system("cartesian")


def _transform_add_itd(hrtf: HRTF) -> HRTF:
    return hrtf.transform.add_itd(1, unit="samples")


def _transform_delete_itd(hrtf: HRTF) -> HRTF:
=======
def transform_apply_gain(hrtf: HRTF) -> HRTF:
    return hrtf.transform.apply_gain(-1.0, scale="db")


def transform_modify_fft_length(hrtf: HRTF) -> HRTF:
    return hrtf.transform.modify_fft_length(int(hrtf.IR.values.shape[-1]) + 32)


def transform_modify_source_coordinate_system(hrtf: HRTF) -> HRTF:
    return hrtf.transform.modify_source_coordinate_system("cartesian")


def transform_add_itd(hrtf: HRTF) -> HRTF:
    return hrtf.transform.add_itd(1, unit="samples")


def transform_delete_itd(hrtf: HRTF) -> HRTF:
>>>>>>> dev
    return hrtf.transform.delete_itd()


TRANSFORM_CASES = [
<<<<<<< HEAD
    ("apply_window", _transform_apply_window, True),
    ("apply_padding", _transform_apply_padding, False),
    ("apply_fir_filter", _transform_apply_fir_filter, True),
    ("apply_iir_filter", _transform_apply_iir_filter, True),
    ("minimum_phase", _transform_minimum_phase, True),
    ("to_ctf", _transform_to_ctf, True),
    ("to_dtf", _transform_to_dtf, True),
    ("modify_ir", _transform_modify_ir, True),
    ("modify_phase", _transform_modify_phase, True),
    ("modify_tf", _transform_modify_tf, True),
    ("modify_magnitude", _transform_modify_magnitude, True),
    ("apply_gain", _transform_apply_gain, True),
    ("modify_fft_length", _transform_modify_fft_length, True),
    ("modify_source_coordinate_system", _transform_modify_source_coordinate_system, True),
    ("add_itd", _transform_add_itd, True),
    ("delete_itd", _transform_delete_itd, True),
]


=======
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
    ("modify_source_coordinate_system", transform_modify_source_coordinate_system, True),
    ("add_itd", transform_add_itd, True),
    ("delete_itd", transform_delete_itd, True),
]


@pytest.fixture
def real_hrtf() -> HRTF:
    return load_hrtf(SOFA_PATH)


>>>>>>> dev
@pytest.mark.parametrize(
    ("name", "transform_fn", "expect_update_without_resize"),
    TRANSFORM_CASES,
    ids=[case[0] for case in TRANSFORM_CASES],
)
def test_update_sofa_all_transform_methods(
<<<<<<< HEAD
=======
    real_hrtf: HRTF,
>>>>>>> dev
    name: str,
    transform_fn,
    expect_update_without_resize: bool,
) -> None:
<<<<<<< HEAD
    hrtf = load_hrtf(SOFA_PATH)
    transformed_hrtf = transform_fn(hrtf)
=======
    transformed_hrtf = transform_fn(real_hrtf)
>>>>>>> dev

    assert transformed_hrtf.is_transformed() is True

    if expect_update_without_resize:
        transformed_hrtf.update_sofa(change_sofa_dimensions=False)
    else:
        with pytest.raises(
            ValueError,
            match="Set change_sofa_dimensions=True",
        ):
            transformed_hrtf.update_sofa(change_sofa_dimensions=False)

    transformed_hrtf.update_sofa(change_sofa_dimensions=True)


<<<<<<< HEAD
def test_update_sofa_no_transform_prints_message(capsys) -> None:
    hrtf = load_hrtf(SOFA_PATH)
    assert hrtf.is_transformed() is False

    hrtf.update_sofa()
    captured = capsys.readouterr()

    assert "already up to date" in captured.out
    assert hrtf.is_transformed() is False


def test_save_runs_after_update_sofa(tmp_path) -> None:
    hrtf = load_hrtf(SOFA_PATH)
    transformed_hrtf = hrtf.transform.apply_window("hann")
=======
def test_update_sofa_no_transform_prints_message(real_hrtf: HRTF, capsys) -> None:
    assert real_hrtf.is_transformed() is False

    real_hrtf.update_sofa()
    captured = capsys.readouterr()

    assert "already up to date" in captured.out
    assert real_hrtf.is_transformed() is False


def test_save_runs_after_update_sofa(real_hrtf: HRTF, tmp_path) -> None:
    transformed_hrtf = real_hrtf.transform.apply_window("hann")
>>>>>>> dev
    destination = tmp_path / "hrtf_saved.sofa"

    saved_path = transformed_hrtf.save(
        path=destination,
        overwrite=True,
        change_sofa_dimensions=False,
    )

    assert saved_path == destination
    assert destination.exists()


@pytest.mark.parametrize(
    ("sofa_convention", "expected_data_type", "present_variables", "absent_variables"),
    [
        (
            "same",
            "FIR",
            ("Data.IR", "Data.SamplingRate"),
            ("Data.Real", "Data.Imag", "N"),
        ),
        (
            "SimpleFreeFieldHRIR",
            "FIR",
            ("Data.IR", "Data.SamplingRate"),
            ("Data.Real", "Data.Imag", "N"),
        ),
        (
            "SimpleFreeFieldHRTF",
            "TF",
            ("Data.Real", "Data.Imag", "N"),
            ("Data.IR", "Data.SamplingRate"),
        ),
    ],
)
def test_save_sofa_convention_with_selected_positions(
<<<<<<< HEAD
=======
    real_hrtf: HRTF,
>>>>>>> dev
    tmp_path,
    sofa_convention: str,
    expected_data_type: str,
    present_variables: tuple[str, ...],
    absent_variables: tuple[str, ...],
) -> None:
<<<<<<< HEAD
    hrtf = load_hrtf(SOFA_PATH)
    selected_hrtf = hrtf.select(positions=["front", "left", "right"])
=======
    selected_hrtf = real_hrtf.select(positions=["front", "left", "right"])
>>>>>>> dev
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
        "SimpleFreeFieldHRIR" if sofa_convention == "same" else sofa_convention
    )
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
<<<<<<< HEAD
=======

    saved_sofa.netCDF4_dataset.close()


def test_real_hrtf_selects_positions_and_ears(real_hrtf: HRTF) -> None:
    selected_hrtf = real_hrtf.select(
        positions=["front", "left", "right"],
        ear="left",
    )

    assert selected_hrtf.IR.values.shape[0] == 3
    assert selected_hrtf.IR.values.ndim == 2
    assert selected_hrtf.TF.values is not None


def test_real_hrtf_metric_itd_runs_on_loaded_file(real_hrtf: HRTF) -> None:
    values = itd(real_hrtf.IR, sample_rate=real_hrtf.IR.sample_rate)

    assert np.asarray(values).shape[0] == real_hrtf.IR.values.shape[0]
    assert np.all(np.isfinite(values))
>>>>>>> dev
