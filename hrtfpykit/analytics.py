from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hrtf import HRTF


class Analytics:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf
