import warnings

import netCDF4 as ncdf
import numpy as np
import pytest

from hrtfpykit.sofa.check import check_sofa_against_conventions


def _make_dataset(convention: str | None, version: str | None) -> ncdf.Dataset:
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    if convention is not None:
        ds.SOFAConventions = convention
    if version is not None:
        ds.SOFAConventionsVersion = version
    return ds


def test_missing_conventions_warns() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset(None, None)
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_sofa_against_conventions(ds)
        messages = [str(w.message) for w in record]
        assert any("Missing SOFAConventions" in m for m in messages)
    finally:
        ds.close()


def test_unsupported_conventions_warns() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset("UnknownConvention", "0.0")
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_sofa_against_conventions(ds)
        messages = [str(w.message) for w in record]
        assert any("API may not behave as expected" in m for m in messages)
    finally:
        ds.close()


def test_dimension_and_default_checks_warn() -> None:
    pytest.importorskip("netCDF4")
    ds = _make_dataset("SimpleFreeFieldHRIR", "1.0")
    try:
        ds.DataType = "TF"
        ds.createDimension("M", 1)
        ds.createDimension("C", 3)
        ds.createDimension("N", 4)
        var = ds.createVariable("Data.IR", "f8", ("M", "C", "N"))
        var[:] = np.zeros((1, 3, 4))

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_sofa_against_conventions(ds)
        messages = [str(w.message) for w in record]
        assert any("Global attribute DataType should be" in m for m in messages)
        assert any("Variable Data.IR has dims" in m for m in messages)
    finally:
        ds.close()
