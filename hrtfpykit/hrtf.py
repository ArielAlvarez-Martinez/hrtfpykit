from pathlib import Path
import warnings

from .sofa.core import SOFA

class HRTF():
 
    def __init__(self, Sofa : SOFA | None) -> None:
        self.Sofa: SOFA | None = Sofa

    @classmethod
    def load_hrtf(
        cls,
        path: str | Path,
        mode: str = "r",
        parallel: bool = False,
        check_sofa_against_conventions: bool = True,
    ) -> "HRTF":
        """Load an HRTF from a SOFA file.

        Parameters
        ----------
        path : str | Path
            Path to the SOFA file.
        mode : str, optional
            netCDF4 open mode (e.g., "r", "r+").
        parallel : bool, optional
            Whether to open in parallel mode.
        check_sofa_against_conventions : bool, optional
            If True, validates against SOFA conventions on open.

        Returns
        -------
        HRTF
            HRTF instance with a loaded SOFA object.

        Examples
        --------
        >>> hrtf = HRTF.load_hrtf("my.sofa")
        """

        Sofa = SOFA.load(
            path,
            mode=mode,
            parallel=parallel,
            check_sofa_against_conventions=check_sofa_against_conventions,
        )
        cls._warn_if_non_hrtf_convention(Sofa, path)
        return cls(Sofa=Sofa)

    @staticmethod
    def _warn_if_non_hrtf_convention(Sofa: SOFA, path: str | Path) -> None:
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        dataset = Sofa.netCDF4_dataset
        convention = getattr(dataset, "SOFAConventions", None)
        if convention not in allowed:
            warnings.warn(
                (
                    "SOFAConventions is not an HRTF convention. "
                    f"Expected one of {sorted(allowed)}, got {convention!r} "
                    f"for {path!s}."
                ),
                UserWarning,
            )

