__all__ = [
    "itd_difference",
    "ild_difference",
    "lsd",
    "SH",
    "sht",
    "sht_inverse",
    "sht_error",
    "load_hrtf"
]


def __getattr__(name: str):
    if name == "load_hrtf":
        from .hrtf import load_hrtf
        return load_hrtf
    if name == "itd_difference":
        from .metrics import itd_difference
        return itd_difference
    if name == "ild_difference":
        from .metrics import ild_difference
        return ild_difference
    if name == "lsd":
        from .metrics import lsd
        return lsd
    if name == "SH":
        from .sh import SH
        return SH
    if name == "sht":
        from .sh import sht
        return sht
    if name == "sht_inverse":
        from .sh import sht_inverse
        return sht_inverse
    if name == "sht_error":
        from .sh import sht_error
        return sht_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
