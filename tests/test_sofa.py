import os
import warnings
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from hrtfpykit.utils.warnings import SOFAConventionWarning
from hrtfpykit.sofa.check import check_sofa_against_conventions, check_sofa_security
from hrtfpykit.sofa.sofa import SOFA
from hrtfpykit.sofa.sofa import load_sofa


FIXTURE_SOFA_PATH = Path(__file__).parent / "pp1_HRIRs_measured.sofa"
SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
if SOFA_PATH == "" and FIXTURE_SOFA_PATH.exists():
    SOFA_PATH = str(FIXTURE_SOFA_PATH)
pytestmark = pytest.mark.skipif(
    SOFA_PATH == "" or not os.path.exists(SOFA_PATH),
    reason="Required SOFA fixture is not available",
)


@pytest.fixture
def real_sofa() -> Generator[SOFA, None, None]:
    sofa = load_sofa(SOFA_PATH, check_sofa_against_conventions=True)
    try:
        yield sofa
    finally:
        sofa.netCDF4_dataset.close()


def test_real_sofa_file_loads(real_sofa: SOFA) -> None:
    assert real_sofa.Dimensions is not None
    assert real_sofa.GlobalAttributes is not None
    assert real_sofa.VariableAttributes is not None
    assert real_sofa.Variables is not None
    assert "SOFAConventions" in real_sofa.GlobalAttributes.get_names()


def test_real_sofa_summary_contains_main_sections(real_sofa: SOFA) -> None:
    summary = real_sofa.summary()

    assert "GLOBAL ATTRIBUTES" in summary
    assert "VARIABLES AND VARIABLES ATTRIBUTES" in summary
    assert "SOFAConventions" in summary


def test_real_sofa_dimensions_are_available(real_sofa: SOFA) -> None:
    dimensions = real_sofa.Dimensions

    assert dimensions is not None
    assert len(dimensions) > 0
    assert len(dimensions.get_names()) > 0
    assert dimensions.summary() != ""


def test_real_sofa_variables_are_available(real_sofa: SOFA) -> None:
    variables = real_sofa.Variables

    assert variables is not None
    assert len(variables.get_names()) > 0
    assert variables.summary() != ""

    if "Data.IR" in variables.get_names():
        data_ir = variables.get("Data.IR")
        assert data_ir is not None
        assert isinstance(data_ir.value, np.ndarray)
    if "Data.Real" in variables.get_names():
        data_real = variables.get("Data.Real")
        assert data_real is not None
        assert isinstance(data_real.value, np.ndarray)


def test_real_sofa_clone_mutates_variables_attributes_and_saves(real_sofa: SOFA, tmp_path: Path) -> None:
    editable = real_sofa.clone()
    try:
        editable.create_dimension("Q", 3)
        assert editable.Dimensions.get("Q").value == 3

        editable.create_global_attribute("IntegrationNote", "initial")
        editable.modify_global_attribute("IntegrationNote", "modified")
        assert editable.GlobalAttributes.get("IntegrationNote").value == "modified"

        editable.create_global_attribute("TemporaryNote", "delete-me")
        editable.delete_global_attribute("TemporaryNote")
        assert "TemporaryNote" not in editable.GlobalAttributes.get_names()

        editable.create_variable(
            "IntegrationVector",
            [1.0, 2.0, 3.0],
            ("Q",),
            attributes={"Units": "1"},
        )
        np.testing.assert_allclose(
            editable.Variables.get("IntegrationVector").value,
            np.array([1.0, 2.0, 3.0]),
        )
        assert editable.VariableAttributes.get("IntegrationVector:Units").value == "1"

        editable.modify_variable("IntegrationVector", [4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            editable.Variables.get("IntegrationVector").value,
            np.array([4.0, 5.0, 6.0]),
        )

        editable.create_variable_attribute("IntegrationVector:Description", "initial")
        editable.modify_variable_attribute("IntegrationVector:Description", "modified")
        assert editable.VariableAttributes.get("IntegrationVector:Description").value == "modified"

        editable.create_variable_attribute("IntegrationVector:Temporary", "delete-me")
        editable.delete_variable_attribute("IntegrationVector:Temporary")
        assert "IntegrationVector:Temporary" not in editable.VariableAttributes.get_names()

        editable.create_variable("TemporaryVector", [0.0, 0.0, 0.0], ("Q",))
        editable.delete_variable("TemporaryVector")
        assert "TemporaryVector" not in editable.Variables.get_names()

        destination = tmp_path / "edited.sofa"
        saved_path = editable.save(destination, overwrite=True)
    finally:
        if editable.netCDF4_dataset is not None:
            editable.netCDF4_dataset.close()

    assert saved_path == destination
    assert destination.exists()

    saved_sofa = load_sofa(destination, check_sofa_against_conventions=False)
    try:
        assert saved_sofa.Dimensions.get("Q").value == 3
        assert saved_sofa.GlobalAttributes.get("IntegrationNote").value == "modified"
        assert "TemporaryNote" not in saved_sofa.GlobalAttributes.get_names()
        np.testing.assert_allclose(
            saved_sofa.Variables.get("IntegrationVector").value,
            np.array([4.0, 5.0, 6.0]),
        )
        assert saved_sofa.VariableAttributes.get("IntegrationVector:Units").value == "1"
        assert saved_sofa.VariableAttributes.get("IntegrationVector:Description").value == "modified"
        assert "IntegrationVector:Temporary" not in saved_sofa.VariableAttributes.get_names()
        assert "TemporaryVector" not in saved_sofa.Variables.get_names()
    finally:
        saved_sofa.netCDF4_dataset.close()


def test_real_sofa_save_refuses_overwrite_unless_requested(real_sofa: SOFA, tmp_path: Path) -> None:
    editable = real_sofa.clone()
    destination = tmp_path / "copy.sofa"
    try:
        assert editable.save(destination) == destination
        with pytest.raises(FileExistsError, match="SOFA file already exists"):
            editable.save(destination)
        assert editable.save(destination, overwrite=True) == destination
    finally:
        if editable.netCDF4_dataset is not None:
            editable.netCDF4_dataset.close()


def test_real_sofa_convention_check_reports_file_context() -> None:
    sofa = load_sofa(SOFA_PATH, check_sofa_against_conventions=False)
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = check_sofa_against_conventions(sofa.netCDF4_dataset)

        assert result["convention"]["name"] == getattr(
            sofa.netCDF4_dataset,
            "SOFAConventions",
            None,
        )
        assert all(
            issubclass(warning.category, SOFAConventionWarning)
            for warning in record
        )
    finally:
        sofa.netCDF4_dataset.close()


def test_real_sofa_security_check_runs_on_file() -> None:
    report = check_sofa_security(
        target=SOFA_PATH,
        hdf5_version="1.14.4",
        print_report=False,
    )

    assert isinstance(report["passed"], bool)
    assert isinstance(report["checks"], list)
    assert len(report["checks"]) > 0
    assert all("name" in check and "passed" in check for check in report["checks"])

    failed_checks = [
        check["name"]
        for check in report["checks"]
        if check["passed"] is False
    ]
    assert report["failed"] == failed_checks
    assert report["passed"] is (len(failed_checks) == 0)
