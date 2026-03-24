import warnings
from pathlib import Path

import numpy as np
from .analytics import Analytics
from .sofa.core import SOFA
from .spatial import Sources
from .domain import TimeDomain, FrequencyDomain
from .utils import from_sofa


class HRTF:
    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        self.Sofa: SOFA | None = Sofa
        self.ir: np.ndarray | None = None
        self.tf: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.frequency_bins: np.ndarray | None = None
        self.sofa_convention: str | None = None
        self.fft_length: int | None = None

    @property
    def TimeDomain(self) -> "TimeDomain":
        return TimeDomain(self)

    @property
    def FrequencyDomain(self) -> "FrequencyDomain":
        return FrequencyDomain(self)

    @property
    def Sources(self) -> "Sources":
        return Sources(self)

    @property
    def Analytics(self) -> "Analytics":
        return Analytics(self)

    @classmethod
    def load_hrtf(
        cls,
        path: str | Path,
        mode: str = "r",
        parallel: bool = False,
        check_sofa_against_conventions: bool = True,
        SampleRate: int | None = None,
        FFT_length: int | None = None,
    ) -> "HRTF":
        
        Sofa = SOFA.load(
            path,
            mode=mode,
            parallel=parallel,
            check_sofa_against_conventions=check_sofa_against_conventions,
        )
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        global_attrs = Sofa.GlobalAttributes
        if global_attrs is None:
            message = "Loaded SOFA dataset is unavailable; cannot verify HRTF convention."
            warnings.warn(message, UserWarning)
        else:
            try:
                convention = global_attrs.get("SOFAConventions").value
            except ValueError:
                convention = None
            if convention not in allowed:
                message = (
                    "SOFAConventions is not an HRTF convention. "
                    f"Expected one of {sorted(allowed)}, got {convention!r} "
                    f"for {path!s}."
                )
                warnings.warn(message, UserWarning)
        return from_sofa(
            cls,
            Sofa,
            SampleRate_override=SampleRate,
            FFT_length=FFT_length,
        )

