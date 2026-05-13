from __future__ import annotations

import atexit
import builtins
import inspect
import linecache
import pathlib
import sys
import warnings


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
ORIGINAL_PRINT = getattr(
    builtins.print,
    "_hrtfpykit_original",
    builtins.print,
)
PENDING_WARNING_CONTEXT: tuple[object, str | None, int | None, str | None] | None = None


def flush_pending_warning_context() -> None:
    global PENDING_WARNING_CONTEXT
    if PENDING_WARNING_CONTEXT is None:
        return
    stream, display_filename, lineno, rendered_line = PENDING_WARNING_CONTEXT
    if display_filename and lineno:
        stream.write(f"  {display_filename}, line {lineno}\n")
    if rendered_line:
        stream.write(f"  {rendered_line.strip()}\n")
    PENDING_WARNING_CONTEXT = None


def showwarning(message, category, filename, lineno, file=None, line=None):
    """Render warnings with message, location, and source line."""
    global PENDING_WARNING_CONTEXT
    rendered_line = line
    if rendered_line is None and filename and lineno:
        rendered_line = linecache.getline(filename, lineno)
    display_filename = filename
    if filename and not filename.startswith("<"):
        resolved_filename = pathlib.Path(filename).resolve()
        try:
            display_filename = str(resolved_filename.relative_to(pathlib.Path.cwd().resolve()))
        except ValueError:
            display_filename = str(resolved_filename)
    stream = sys.stderr if file is None else file
    warning_context = (
        stream,
        display_filename,
        lineno,
        rendered_line,
    )
    if PENDING_WARNING_CONTEXT is not None and PENDING_WARNING_CONTEXT != warning_context:
        flush_pending_warning_context()
    stream.write(f"{category.__name__}: {message}\n")
    PENDING_WARNING_CONTEXT = warning_context


showwarning._hrtfpykit_original = ORIGINAL_SHOWWARNING
warnings.showwarning = showwarning


def print(*args, **kwargs):
    """Flush pending warning context before normal printed output."""
    flush_pending_warning_context()
    return ORIGINAL_PRINT(*args, **kwargs)


print._hrtfpykit_original = ORIGINAL_PRINT
builtins.print = print


atexit.register(flush_pending_warning_context)


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
