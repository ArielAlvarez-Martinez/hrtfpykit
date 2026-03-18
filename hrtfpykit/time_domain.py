from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .hrtf import HRTF


class TimeDomainWrapper:
    def __init__(self, hrtf: HRTF) -> None:
        self._hrtf = hrtf

    @property
    def ir(self) -> np.ndarray | None:
        return self._hrtf.ir
