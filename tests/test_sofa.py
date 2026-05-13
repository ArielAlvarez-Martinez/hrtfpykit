import os
import warnings
from collections.abc import Generator

import numpy as np
import pytest

from hrtfpykit._warnings import SOFAConventionWarning
from hrtfpykit.sofa.check import check_sofa_against_conventions, check_sofa_security
from hrtfpykit.sofa.sofa import SOFA
from hrtfpykit.sofa.sofa import load_sofa


SOFA_PATH = os.getenv("HRTFPYKIT_TEST_SOFA_PATH", "")
pytestmark = pytest.mark.skipif(
    SOFA_PATH == "" or not os.path.exists(SOFA_PATH),
    reason="Required local SOFA file is not available",
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
