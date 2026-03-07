"""hrtfpykit package."""

from . import sofa
from .sofa.core import SOFA


def load(path, **kwargs):
    return SOFA.load(path, **kwargs)


__all__ = [
    "sofa",
    "SOFA",
    "load"
]
