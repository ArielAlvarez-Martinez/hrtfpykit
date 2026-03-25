import warnings
from pathlib import Path

import numpy as np
from .analytics import Analytics
from .dsp import compute_ir_from_tf, compute_tf_from_ir
from .sofa.core import SOFA
from .spatial import Sources
from .domain import TimeDomain, FrequencyDomain


class HRTF:
    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        self.Sofa: SOFA | None = Sofa
        self.ir: np.ndarray | None = None
        self.tf: np.ndarray | None = None
        self.sample_rate: float | None = None
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
        fft_length: int | None = None,
    ) -> "HRTF":
        """Load an HRTF from a SOFA file and populate time/frequency fields.

        This method loads a SOFA dataset and builds an ``HRTF`` object with
        its impulse response (IR), transfer function (TF), and related metadata.
        Both SimpleFreeFieldHRIR (time domain) and SimpleFreeFieldHRTF (frequency
        domain) conventions are supported.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the SOFA file.
        mode : str, optional
            File open mode passed to ``SOFA.load`` (e.g., "r", "r+").
        parallel : bool, optional
            Whether to enable parallel mode in ``netCDF4`` (if supported).
        check_sofa_against_conventions : bool, optional
            If True, validate the SOFA dataset against the conventions registry.
        fft_length : int, optional
            FFT length to use when converting between IR and TF.

        Returns
        -------
        HRTF
            Loaded HRTF object with ``ir``, ``tf``, ``sample_rate``,
            ``frequency_bins``, and ``fft_length`` populated when possible.

        Examples
        --------
        >>> hrtf = HRTF.load_hrtf("hrtf.sofa")
        >>> hrtf = HRTF.load_hrtf("hrtf.sofa", fft_length=512)

        Best Practices
        --------------
        - Prefer SOFA files that include ``Data.SamplingRate`` (HRIR) or ``N``
          ``frequency_bins`` (HRTF) to avoid ambiguity.
        - Use a consistent ``fft_length`` when comparing HRTFs.
        - Keep the file open mode read-only unless you intend to modify it.

        Notes
        -----
        - HRIR (time-domain) path:
          - ``sample_rate`` is taken from ``Data.SamplingRate`` when available;
            otherwise it remains unknown.
          - ``fft_length`` controls the FFT length for IR→TF conversion.
            If omitted, the IR length is used.
        - HRTF (frequency-domain) path:
          - If a valid, uniform ``N`` frequency_bins array is present, the FFT length
            is inferred from it and overrides any provided ``fft_length``.
          - ``sample_rate`` is inferred as ``(frequency_bins[1] - frequency_bins[0]) * fft_length``
            when frequency_bins are uniform; otherwise it remains unknown.
          - If frequency_bins are missing, ``fft_length`` is used to attempt
            TF→IR conversion, but frequency_bins remain unknown.

        Warnings / Errors
        -----------------
        - Raises ValueError if the SOFA dataset fails to load.
        - Warns if the SOFA convention is not HRTF-related or missing.
        - Warns when required metadata (sample_rate or frequency_bins) is
          unavailable to compute the other domain.
        """
        
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
        global_attrs = Sofa.GlobalAttributes
        variables = Sofa.Variables
        if global_attrs is None or variables is None:
            raise ValueError("SOFA dataset is not loaded")

        try:
            convention = global_attrs.get("SOFAConventions").value
        except ValueError:
            convention = None
        variable_names = set(variables.get_names())
        if convention == "SimpleFreeFieldHRIR" or "Data.IR" in variable_names:
            ir = np.asarray(variables.get("Data.IR").value)
            resolved_sample_rate = None
            if "Data.SamplingRate" in variable_names:
                data = np.asarray(
                    variables.get("Data.SamplingRate").value,
                    dtype=float,
                )
                if data.size > 0:
                    resolved_sample_rate = int(data.flat[0])
            tf = None
            freqs = None
            n_fft_used = None
            if resolved_sample_rate is None:
                message = "Missing Data.SamplingRate; cannot compute TF from IR."
                warnings.warn(message, UserWarning)
            else:
                tf, freqs, n_fft_used = compute_tf_from_ir(
                    ir,
                    resolved_sample_rate,
                    fft_length=fft_length,
                )
            hrtf = cls(Sofa)
            hrtf.ir = ir
            hrtf.tf = tf
            hrtf.sample_rate = resolved_sample_rate
            hrtf.frequency_bins = freqs
            hrtf.fft_length = fft_length
            if n_fft_used is not None:
                hrtf.fft_length = n_fft_used
            hrtf.sofa_convention = convention
            return hrtf

        if (
            convention == "SimpleFreeFieldHRTF"
            or ("Data.Real" in variable_names and "Data.Imag" in variable_names)
        ):
            real = np.asarray(variables.get("Data.Real").value, dtype=float)
            imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
            tf = real + 1j * imag
            freqs = None
            if "N" in variable_names:
                data = np.asarray(variables.get("N").value, dtype=float)
                if data.size > 0:
                    freqs = data
            if freqs is None:
                message = "Missing N frequency_bins; cannot compute IR from TF."
                warnings.warn(message, UserWarning)
            else:
                if freqs.ndim == 1 and freqs.size > 0:
                    if float(np.min(freqs)) >= 0.0 and not np.isclose(freqs[0], 0.0):
                        warnings.warn(
                            "Frequency axis should start at 0 Hz to compute IR from TF.",
                            UserWarning,
                        )
            tf_normalization = None
            if "Normalization" in variable_names:
                norm_data = np.asarray(variables.get("Normalization").value, dtype=float)
                if norm_data.size > 0:
                    tf_normalization = float(norm_data.flat[0])
            resolved_sample_rate = None
            n_fft_used = None
            if freqs is not None and freqs.ndim == 1 and freqs.size >= 2:
                diffs = np.diff(freqs)
                first = float(diffs[0])
                if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
                    if float(np.min(freqs)) < 0.0:
                        expected_n_fft = freqs.size
                    else:
                        expected_n_fft = 2 * (freqs.size - 1)
                    if fft_length is not None and fft_length != expected_n_fft:
                        warnings.warn(
                            "FFT length does not match the provided frequency bins; using inferred length.",
                            UserWarning,
                        )
                    n_fft_used = expected_n_fft
                    resolved_sample_rate = first * expected_n_fft
            ir = compute_ir_from_tf(
                tf,
                frequency_bins=freqs,
                fft_length=n_fft_used or fft_length,
                tf_normalization=tf_normalization,
                normalization_action="undo",
            )
            if n_fft_used is None:
                n_fft_used = fft_length or (2 * (tf.shape[-1] - 1))
            if ir is None:
                message = "Unable to compute IR from TF with the provided frequency_bins."
                warnings.warn(message, UserWarning)
            if resolved_sample_rate is None:
                warnings.warn("Unable to infer samplerate from frequency_bins.", UserWarning)
            hrtf = cls(Sofa)
            hrtf.ir = ir
            hrtf.tf = tf
            hrtf.sample_rate = resolved_sample_rate
            hrtf.frequency_bins = freqs
            hrtf.fft_length = n_fft_used
            hrtf.sofa_convention = convention
            return hrtf

        message = "Unable to determine HRTF domain from SOFA content."
        warnings.warn(message, UserWarning)
        hrtf = cls(Sofa)
        hrtf.fft_length = fft_length
        hrtf.sofa_convention = convention
        return hrtf

