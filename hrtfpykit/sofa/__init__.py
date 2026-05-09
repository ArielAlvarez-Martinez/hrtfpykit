__version__ = "1.0.0"

__all__ = [
    "__version__",
    "load_sofa",
    "check_sofa_against_conventions",
    "check_sofa_security",
]


def __getattr__(name: str):
    if name == "load_sofa":
        from .sofa import load_sofa
        return load_sofa
    if name == "check_sofa_against_conventions":
        from .check import check_sofa_against_conventions
        return check_sofa_against_conventions
    if name == "check_sofa_security":
        from .check import check_sofa_security
        return check_sofa_security
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
