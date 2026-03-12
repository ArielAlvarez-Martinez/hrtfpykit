import netCDF4 as ncdf
import pytest

from hrtfpykit.sofa.check import check_sofa_security


def _make_dataset() -> ncdf.Dataset:
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    ds.SOFAConventions = "SimpleFreeFieldHRIR"
    ds.SOFAConventionsVersion = "1.0"
    ds.createDimension("M", 1)
    ds.createDimension("R", 2)
    ds.createDimension("N", 4)
    ds.createVariable("Data.IR", "f8", ("M", "R", "N"))
    return ds


def test_security_passes_on_min_safe_version() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset()
    try:
        report = check_sofa_security(target=ds, hdf5_version="1.14.4")
    finally:
        ds.close()
    assert report["passed"] is True
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["hdf5_min_safe_version"]["passed"] is True


def test_security_fails_on_old_version() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset()
    try:
        report = check_sofa_security(target=ds, hdf5_version="1.14.3")
    finally:
        ds.close()
    assert report["passed"] is False
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["hdf5_min_safe_version"]["passed"] is False


def test_security_fails_on_external_link_in_attributes() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset()
    ds.Origin = "https://evil.example.com/payload"
    try:
        report = check_sofa_security(target=ds, hdf5_version="1.14.4")
    finally:
        ds.close()
    assert report["passed"] is False
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["risk_external_links_in_attributes"]["passed"] is False


def test_security_fails_on_suspicious_extension() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset()
    ds.Notes = "see report.pdf for details"
    try:
        report = check_sofa_security(target=ds, hdf5_version="1.14.4")
    finally:
        ds.close()
    assert report["passed"] is False
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["risk_suspicious_attribute_extensions"]["passed"] is False
