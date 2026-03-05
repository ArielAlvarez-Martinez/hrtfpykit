import warnings

import netCDF4 as ncdf

from hrtfpykit import check_hrtf


def _make_dataset(data_ir_dims):
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    ds.SOFAConventions = "SimpleFreeFieldHRIR"
    ds.SOFAConventionsVersion = "1.0"
    ds.DataType = "FIR"

    for dim in data_ir_dims:
        if dim not in ds.dimensions:
            ds.createDimension(dim, 1)

    var = ds.createVariable("Data.IR", "f8", data_ir_dims)
    var[:] = 0.0
    return ds


def test_data_ir_zero_values_no_mismatch_warning():
    ds = _make_dataset(("m", "R", "n"))
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_hrtf(ds)
        messages = [str(w.message) for w in record]
        assert not any("Variable Data.IR has dims" in m for m in messages)
        assert not any("Variable Data.IR does not match default value" in m for m in messages)
    finally:
        ds.close()


def test_data_ir_dimension_mismatch_warns():
    ds = _make_dataset(("M", "C", "N"))
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            check_hrtf(ds)
        messages = [str(w.message) for w in record]
        assert any("Variable Data.IR has dims" in m for m in messages)
    finally:
        ds.close()
