from __future__ import annotations

import inspect
import pathlib
import sys
import warnings
from typing import Any, TextIO, cast


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
__all__ = [
    "HRTFPyKitWarning",
    "SOFAWarning",
    "SOFAConventionWarning",
    "SOFAShapeWarning",
    "warn_user",
]


class HRTFPyKitWarning(UserWarning):
    """Base warning category for hrtfpykit."""


class SOFAWarning(HRTFPyKitWarning):
    """Base warning category for SOFA-related issues."""


class SOFAConventionWarning(SOFAWarning):
    """Warning category for SOFA convention validation issues."""


class SOFAShapeWarning(SOFAWarning):
    """Warning category for SOFA dimension and shape mismatches."""


ORIGINAL_SHOWWARNING = getattr(
    warnings.showwarning,
    "_hrtfpykit_original",
    warnings.showwarning,
)


def showwarning(message, category, filename, lineno, file=None, line=None):
    """Render hrtfpykit warnings as concise message lines."""
    stream = sys.stderr if file is None else cast(TextIO, file)
    stream.write(f"{category.__name__}: {message}\n")


cast(Any, showwarning)._hrtfpykit_original = ORIGINAL_SHOWWARNING
warnings.showwarning = showwarning


def warn_user(
    message: str,
    category: type[Warning] = HRTFPyKitWarning,
) -> None:
    """Emit a warning from the first caller outside the hrtfpykit package."""
    stacklevel = 2
    fallback_stacklevel = 2
    current_frame = inspect.currentframe()
    frame = None if current_frame is None else current_frame.f_back
    try:
        while frame is not None:
            filename = frame.f_code.co_filename
            module_name = frame.f_globals.get("__name__", "")
            is_package_frame = False
            if not filename.startswith("<"):
                resolved_filename = pathlib.Path(filename).resolve()
                is_package_frame = PACKAGE_ROOT in resolved_filename.parents
            if module_name == "__main__":
                break
            if is_package_frame:
                fallback_stacklevel = stacklevel
            if not is_package_frame:
                break
            stacklevel += 1
            frame = frame.f_back
        if frame is None:
            stacklevel = fallback_stacklevel
        warnings.warn(message, category, stacklevel=stacklevel)
    finally:
        del current_frame
        del frame
