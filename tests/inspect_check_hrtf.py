import warnings

import netCDF4 as ncdf
import numpy as np

from hrtfpykit.sofa import check_hrtf


def _print_warnings(caught):
    if not caught:
        print("(no warnings)")
        return
    for w in caught:
        print(f"- {w.message}")


def _case(title, func):
    print(f"\n== {title} ==")
    try:
        func()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def case_unsupported_convention():
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    ds.SOFAConventions = "NotAConvention"
    ds.SOFAConventionsVersion = "1.0"
    ds.DataType = "FIR"
    try:
        check_hrtf(ds)
    finally:
        ds.close()


def case_wrong_datatype():
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    ds.SOFAConventions = "SimpleFreeFieldHRIR"
    ds.SOFAConventionsVersion = "1.0"
    ds.DataType = "TF"
    try:
        check_hrtf(ds)
    finally:
        ds.close()


def case_missing_mandatory_and_dim_mismatch():
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    ds.SOFAConventions = "SimpleFreeFieldHRIR"
    ds.SOFAConventionsVersion = "1.0"
    ds.DataType = "FIR"

    ds.createDimension("M", 1)
    ds.createDimension("C", 3)
    ds.createDimension("N", 1)
    ds.createVariable("Data.IR", "f8", ("M", "N", "N"))

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_hrtf(ds)
        _print_warnings(caught)
    finally:
        ds.close()


def case_values_and_dim_checks():
    ds = ncdf.Dataset("inmemory", mode="w", diskless=True, persist=False)
    ds.SOFAConventions = "SimpleFreeFieldHRIR"
    ds.SOFAConventionsVersion = "1.0"
    ds.DataType = "FIR"

    ds.createDimension("m", 2)
    ds.createDimension("R", 2)
    ds.createDimension("n", 4)
    ds.createDimension("M", 1)
    ds.createDimension("C", 3)
    ds.createDimension("N", 4)
    ds.createDimension("I", 1)

    ir = ds.createVariable("Data.IR", "f8", ("M", "C", "N"))
    ir[:] = 0.0

    sr = ds.createVariable("Data.SamplingRate", "f8", ("I",))
    sr[:] = 0.0

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_hrtf(ds)
        _print_warnings(caught)

        ir_data = np.array(ir[:])
        if ir_data.size == 0 or np.all(ir_data == 0) or np.any(np.isnan(ir_data)):
            print("- Data.IR values are missing/zero/NaN")

        sr_data = np.array(sr[:])
        if sr_data.size == 0 or np.all(sr_data == 0) or np.any(np.isnan(sr_data)):
            print("- Data.SamplingRate values are missing/zero/NaN")
    finally:
        ds.close()


if __name__ == "__main__":
    _case("Unsupported SOFAConventions", case_unsupported_convention)
    _case("Wrong DataType", case_wrong_datatype)
    _case("Missing mandatory + dim mismatch", case_missing_mandatory_and_dim_mismatch)
    _case("Values + dimension checks", case_values_and_dim_checks)


