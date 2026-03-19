import warnings
from pathlib import Path

import numpy as np
from .analytics import AnalyticsWrapper
from .frequency_domain import FrequencyDomainWrapper
from .sofa.core import SOFA
from .spatial import SpatialWrapper
from .time_domain import TimeDomainWrapper


class HRTF:
    def __init__(
        self,
        Sofa: SOFA | None = None,
    ) -> None:
        self.Sofa: SOFA | None = Sofa
        # TODO : create a better logic for IR in TimeDomainWrapper, e.g : IR["both", "left", "right"]
        self.IR: np.ndarray | None = None
        self.TF: np.ndarray | None = None
        self.SampleRate: int | None = None
        self.FrequencyBins: np.ndarray | None = None
        self.SOFAConvention: str | None = None
        self.FFT_length: int | None = None

    @property
    def TimeDomain(self) -> "TimeDomainWrapper":
        return TimeDomainWrapper(self)

    @property
    def FrequencyDomain(self) -> "FrequencyDomainWrapper":
        return FrequencyDomainWrapper(self)

    @property
    def Spatial(self) -> "SpatialWrapper":
        if self.Sofa is None:
            return SpatialWrapper()
        positions, pos_type, pos_units = self._extract_source_metadata(self.Sofa)
        return SpatialWrapper(
            positions=positions,
            position_type=pos_type,
            position_units=pos_units,
        )

    @property
    def Analytics(self) -> "AnalyticsWrapper":
        return AnalyticsWrapper(self)

    def with_crop(self, start: int | None = None, end: int | None = None) -> "HRTF":
        if self.IR is None:
            raise ValueError("IR data is not available")
        new_ir = self.IR[..., slice(start, end)]
        return self._clone_with_ir(new_ir)

    def with_window(self, window: str) -> "HRTF":
        if self.IR is None:
            raise ValueError("IR data is not available")
        window_values = self._window(window, self.IR.shape[-1])
        if window_values is None:
            raise ValueError(f"Unsupported window '{window}'")
        new_ir = self.IR * window_values
        return self._clone_with_ir(new_ir)

    def with_itd_shift(self, samples: int) -> "HRTF":
        if self.IR is None:
            raise ValueError("IR data is not available")
        new_ir = self._shift_ir(self.IR, samples)
        return self._clone_with_ir(new_ir)

    def with_filter(self, kernel: np.ndarray) -> "HRTF":
        if self.IR is None:
            raise ValueError("IR data is not available")
        kernel_arr = np.asarray(kernel)
        if kernel_arr.ndim != 1:
            raise ValueError("Filter kernel must be 1D")
        new_ir = np.apply_along_axis(
            lambda x: np.convolve(x, kernel_arr, mode="same"),
            axis=-1,
            arr=self.IR,
        )
        return self._clone_with_ir(new_ir)

    def with_fft_length(self, fft_length: int) -> "HRTF":
        if self.IR is None:
            raise ValueError("IR data is not available")
        new_hrtf = self._clone_with_ir(self.IR)
        new_hrtf.FFT_length = int(fft_length)
        new_hrtf._recompute_tf_from_ir(FFT_length=int(fft_length))
        return new_hrtf

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
        cls._warn_if_non_hrtf_convention(Sofa, path)
        return cls._from_sofa(
            Sofa,
            SampleRate_override=SampleRate,
            FFT_length=FFT_length,
        )

    @staticmethod
    def _warn_if_non_hrtf_convention(
        Sofa: SOFA,
        path: str | Path,
    ) -> None:
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        global_attrs = Sofa.GlobalAttributes
        if global_attrs is None:
            message = "Loaded SOFA dataset is unavailable; cannot verify HRTF convention."
            warnings.warn(message, UserWarning)
            return
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
            warnings.warn(
                (
                    message
                ),
                UserWarning,
            )

    @classmethod
    def _from_sofa(
        cls,
        Sofa: SOFA,
        SampleRate_override: float | None = None,
        FFT_length: int | None = None,
    ) -> "HRTF":
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
            SampleRate = SampleRate_override or cls._extract_sampling_rate(Sofa)
            tf = None
            freqs = None
            if SampleRate is None:
                message = "Missing Data.SamplingRate; cannot compute TF from IR."
                warnings.warn(message, UserWarning)
            else:
                tf, freqs, n_fft_used = cls._compute_tf_from_ir(
                    ir,
                    SampleRate,
                    FFT_length=FFT_length,
                )
            hrtf = cls(Sofa)
            hrtf.IR = ir
            hrtf.TF = tf
            hrtf.SampleRate = SampleRate
            hrtf.FrequencyBins = freqs
            hrtf.FFT_length = FFT_length
            if n_fft_used is not None:
                hrtf.FFT_length = n_fft_used
            hrtf.SOFAConvention = convention
            return hrtf

        if (
            convention == "SimpleFreeFieldHRTF"
            or ("Data.Real" in variable_names and "Data.Imag" in variable_names)
        ):
            real = np.asarray(variables.get("Data.Real").value, dtype=float)
            imag = np.asarray(variables.get("Data.Imag").value, dtype=float)
            tf = real + 1j * imag
            freqs = cls._extract_freqs(Sofa)
            if freqs is None:
                message = "Missing N frequency axis; cannot compute IR from TF."
                warnings.warn(message, UserWarning)
            else:
                if freqs.ndim == 1 and freqs.size > 0:
                    if float(np.min(freqs)) >= 0.0 and not np.isclose(freqs[0], 0.0):
                        warnings.warn(
                            "Frequency axis should start at 0 Hz to compute IR from TF.",
                            UserWarning,
                        )
            ir, SampleRate, n_fft_used = cls._compute_ir_from_tf(
                tf,
                freqs,
                FFT_length=FFT_length,
            )
            if SampleRate_override is not None:
                SampleRate = SampleRate_override
            if ir is None:
                message = "Unable to compute IR from TF with the provided frequency axis."
                warnings.warn(message, UserWarning)
            if SampleRate is None:
                warnings.warn("Unable to infer samplerate from frequency axis.", UserWarning)
            hrtf = cls(Sofa)
            hrtf.IR = ir
            hrtf.TF = tf
            hrtf.SampleRate = SampleRate
            hrtf.FrequencyBins = freqs
            hrtf.FFT_length = FFT_length
            if n_fft_used is not None:
                hrtf.FFT_length = n_fft_used
            hrtf.SOFAConvention = convention
            return hrtf

        message = "Unable to determine HRTF domain from SOFA content."
        warnings.warn(message, UserWarning)
        hrtf = cls(Sofa)
        hrtf.FFT_length = FFT_length
        hrtf.SOFAConvention = convention
        return hrtf

    def _clone_with_ir(self, ir: np.ndarray) -> "HRTF":
        new_hrtf = HRTF(self.Sofa)
        new_hrtf.IR = ir
        new_hrtf.TF = None
        new_hrtf.SampleRate = self.SampleRate
        new_hrtf.FrequencyBins = None
        new_hrtf.FFT_length = self.FFT_length
        new_hrtf.SOFAConvention = self.SOFAConvention
        new_hrtf._recompute_tf_from_ir()
        return new_hrtf

    def _recompute_tf_from_ir(
        self,
        FFT_length: int | None = None,
        window: str | None = None,
        normalize: bool | None = None,
    ) -> None:
        if self.IR is None:
            raise ValueError("IR data is not available")
        self._sync_sofa_ir()
        if self.SampleRate is None:
            warnings.warn("Missing samplerate; cannot compute TF from IR.", UserWarning)
            return
        fft_length_value = FFT_length if FFT_length is not None else self.FFT_length
        window_value = window if window is not None else None
        normalize_value = normalize if normalize is not None else False
        tf, freqs, n_fft_used = self._compute_tf_from_ir(
            self.IR,
            self.SampleRate,
            FFT_length=fft_length_value,
            window=window_value,
            normalize=normalize_value,
        )
        self.TF = tf
        self.FrequencyBins = freqs
        if n_fft_used is not None:
            self.FFT_length = n_fft_used
        self._sync_sofa_tf()
        self._sync_sofa_freqs()

    def _sync_sofa_ir(self) -> None:
        if self.Sofa is None or self.IR is None:
            return
        variables = self.Sofa.Variables
        if variables is None:
            return
        if "Data.IR" not in set(variables.get_names()):
            return
        self.Sofa.modify_variable("Data.IR", self.IR)

    def _sync_sofa_tf(self) -> None:
        if self.Sofa is None or self.TF is None:
            return
        variables = self.Sofa.Variables
        if variables is None:
            return
        variable_names = set(variables.get_names())
        if "Data.Real" in variable_names and "Data.Imag" in variable_names:
            self.Sofa.modify_variable("Data.Real", np.real(self.TF))
            self.Sofa.modify_variable("Data.Imag", np.imag(self.TF))
            return
        if "Data.TF" in variable_names:
            self.Sofa.modify_variable("Data.TF", self.TF)

    def _sync_sofa_freqs(self) -> None:
        if self.Sofa is None or self.FrequencyBins is None:
            return
        variables = self.Sofa.Variables
        if variables is None:
            return
        if "N" not in set(variables.get_names()):
            return
        self.Sofa.modify_variable("N", self.FrequencyBins)

    @staticmethod
    def _extract_convention(Sofa: SOFA | None) -> str | None:
        if Sofa is None:
            return None
        global_attrs = Sofa.GlobalAttributes
        if global_attrs is None:
            return None
        try:
            return str(global_attrs.get("SOFAConventions").value)
        except ValueError:
            return None

    @staticmethod
    def _shift_ir(ir: np.ndarray, samples: int) -> np.ndarray:
        if samples == 0:
            return ir.copy()
        out = np.zeros_like(ir)
        if samples > 0:
            out[..., samples:] = ir[..., :-samples]
        else:
            out[..., :samples] = ir[..., -samples:]
        return out

    @staticmethod
    def _extract_sampling_rate(Sofa: SOFA) -> float | None:
        variables = Sofa.Variables
        if variables is None:
            return None
        if "Data.SamplingRate" not in set(variables.get_names()):
            return None
        data = np.asarray(variables.get("Data.SamplingRate").value, dtype=float)
        if data.size == 0:
            return None
        return int(data.flat[0])

    @staticmethod
    def _extract_freqs(Sofa: SOFA) -> np.ndarray | None:
        variables = Sofa.Variables
        if variables is None:
            return None
        if "N" not in set(variables.get_names()):
            return None
        freqs = np.asarray(variables.get("N").value, dtype=float)
        if freqs.size == 0:
            return None
        return freqs

    @staticmethod
    def _extract_source_metadata(
        Sofa: SOFA,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        variables = Sofa.Variables
        if variables is None:
            return None, None, None
        if "SourcePosition" not in set(variables.get_names()):
            return None, None, None
        positions = np.asarray(variables.get("SourcePosition").value, dtype=float)
        var_attrs = Sofa.VariableAttributes
        pos_type = None
        pos_units = None
        if var_attrs is not None:
            try:
                pos_type = var_attrs.get("SourcePosition:Type").value
            except ValueError:
                pos_type = None
            try:
                pos_units = var_attrs.get("SourcePosition:Units").value
            except ValueError:
                pos_units = None
        return positions, pos_type, pos_units

    @staticmethod
    def _compute_tf_from_ir(
        ir: np.ndarray,
        SampleRate: float,
        FFT_length: int | None = None,
        window: str | None = None,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        n_fft = FFT_length if FFT_length is not None else ir.shape[-1]

        signal = ir
        if window:
            window_values = HRTF._window(window, ir.shape[-1])
            if window_values is not None:
                signal = ir * window_values

        tf = np.fft.rfft(signal, n=n_fft, axis=-1)
        if normalize and n_fft:
            tf = tf / float(n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / SampleRate)
        return tf, freqs, int(n_fft)

    @classmethod
    def _compute_ir_from_tf(
        cls,
        tf: np.ndarray,
        freqs: np.ndarray | None,
        FFT_length: int | None = None,
        normalize: bool = False,
    ) -> tuple[np.ndarray | None, float | None, int | None]:
        n_fft = FFT_length

        if tf.shape[-1] < 2:
            return None, None, None

        if freqs is not None and freqs.ndim == 1 and freqs.size == tf.shape[-1]:
            step = cls._uniform_step(freqs)
            if step is not None:
                if float(np.min(freqs)) < 0.0:
                    n_fft_used = n_fft or freqs.size
                    samplerate = step * n_fft_used
                    tf_used = tf * float(n_fft_used) if normalize else tf
                    ir = np.fft.ifft(tf_used, n=n_fft_used, axis=-1)
                    ir = np.real_if_close(ir, tol=1000)
                    return ir, float(samplerate), int(n_fft_used)
                n_fft_used = n_fft or (2 * (freqs.size - 1))
                samplerate = step * n_fft_used
                tf_used = tf * float(n_fft_used) if normalize else tf
                ir = np.fft.irfft(tf_used, n=n_fft_used, axis=-1)
                return ir, float(samplerate), int(n_fft_used)

        n_fft_used = n_fft or (2 * (tf.shape[-1] - 1))
        if n_fft_used <= 0:
            return None, None, None
        tf_used = tf * float(n_fft_used) if normalize else tf
        ir = np.fft.irfft(tf_used, n=n_fft_used, axis=-1)
        return ir, None, int(n_fft_used)

    @staticmethod
    def _window(name: str, length: int) -> np.ndarray | None:
        if length <= 0:
            return None
        key = name.strip().lower()
        if key in {"hann", "hanning"}:
            return np.hanning(length)
        if key == "hamming":
            return np.hamming(length)
        if key == "blackman":
            return np.blackman(length)
        warnings.warn(
            f"Unsupported window '{name}'; proceeding without windowing.",
            UserWarning,
        )
        return None

    @staticmethod
    def _uniform_step(freqs: np.ndarray) -> float | None:
        if freqs.size < 2:
            return None
        diffs = np.diff(freqs)
        first = float(diffs[0])
        if np.allclose(diffs, first, rtol=1e-5, atol=1e-8):
            return first
        return None
