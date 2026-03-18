import warnings
from pathlib import Path
from typing import Any

import numpy as np
from .frequency_domain import FrequencyDomainWrapper
from .analytics import AnalyticsWrapper
from .sofa.core import SOFA
from .source import SourceWrapper
from .time_domain import TimeDomainWrapper

class HRTF:
    def __init__(
        self,
        Sofa: SOFA | None = None,
        ir: np.ndarray | None = None,
        tf: np.ndarray | None = None,
        samplerate: float | None = None,
        freqs: np.ndarray | None = None,
        attrs: dict[str, Any] | None = None,
        native_domain: str | None = None,
        source_positions: np.ndarray | None = None,
        source_position_type: str | None = None,
        source_position_units: str | None = None,
    ) -> None:
        self.Sofa: SOFA | None = Sofa
        self.ir: np.ndarray | None = ir
        self.tf: np.ndarray | None = tf
        self.samplerate: float | None = samplerate
        self.freqs: np.ndarray | None = freqs
        self.attrs: dict[str, Any] = attrs or {}
        self.native_domain: str | None = native_domain
        self.source_positions: np.ndarray | None = source_positions
        self.source_position_type: str | None = source_position_type
        self.source_position_units: str | None = source_position_units
        self.attrs["fft"] = self._default_fft_config(
            self.attrs["fft"] if isinstance(self.attrs.get("fft"), dict) else None
        )

        self._time: TimeDomainWrapper | None = None
        self._freq: FrequencyDomainWrapper | None = None
        self._source: SourceWrapper | None = None
        self._analytics: AnalyticsWrapper | None = None

    @property
    def TimeDomain(self) -> "TimeDomainWrapper":
        if self._time is None:
            self._time = TimeDomainWrapper(self)
        return self._time

    @property
    def FrequencyDomain(self) -> "FrequencyDomainWrapper":
        if self._freq is None:
            self._freq = FrequencyDomainWrapper(self)
        return self._freq

    @property
    def Source(self) -> "SourceWrapper":
        if self._source is None:
            self._source = SourceWrapper(self)
        return self._source

    @property
    def Analytics(self) -> "AnalyticsWrapper":
        if self._analytics is None:
            self._analytics = AnalyticsWrapper(self)
        return self._analytics

    @classmethod
    def load_hrtf(
        cls,
        path: str | Path,
        mode: str = "r",
        parallel: bool = False,
        check_sofa_against_conventions: bool = True,
        strict: bool = False,
        samplerate: float | None = None,
        fft_length: int | None = None,
    ) -> "HRTF":
        
        Sofa = SOFA.load(
            path,
            mode=mode,
            parallel=parallel,
            check_sofa_against_conventions=check_sofa_against_conventions,
        )
        cls._warn_if_non_hrtf_convention(Sofa, path, strict=strict)
        return cls._from_sofa(
            Sofa,
            strict=strict,
            samplerate_override=samplerate,
            fft_length=fft_length,
        )

    @staticmethod
    def _warn_if_non_hrtf_convention(
        Sofa: SOFA,
        path: str | Path,
        *,
        strict: bool = False,
    ) -> None:
        allowed = {"SimpleFreeFieldHRIR", "SimpleFreeFieldHRTF"}
        dataset = Sofa.netCDF4_dataset
        if dataset is None:
            message = "Loaded SOFA dataset is unavailable; cannot verify HRTF convention."
            if strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning)
            return
        convention = getattr(dataset, "SOFAConventions", None)
        if convention not in allowed:
            message = (
                "SOFAConventions is not an HRTF convention. "
                f"Expected one of {sorted(allowed)}, got {convention!r} "
                f"for {path!s}."
            )
            if strict:
                raise ValueError(message)
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
        *,
        strict: bool = False,
        samplerate_override: float | None = None,
        fft_length: int | None = None,
    ) -> "HRTF":
        dataset = Sofa.netCDF4_dataset
        if dataset is None:
            raise ValueError("SOFA dataset is not loaded")

        convention = getattr(dataset, "SOFAConventions", None)
        variables = getattr(dataset, "variables", {})
        source_positions, source_position_type, source_position_units = cls._extract_source_metadata(dataset)
        fft_overrides = {"n_fft": fft_length} if fft_length is not None else None
        attrs = {"fft": cls._default_fft_config(fft_overrides, strict=strict)}

        if convention == "SimpleFreeFieldHRIR" or "Data.IR" in variables:
            ir = np.asarray(variables["Data.IR"][:])
            samplerate = samplerate_override or cls._extract_sampling_rate(dataset)
            tf = None
            freqs = None
            if samplerate is None:
                message = "Missing Data.SamplingRate; cannot compute TF from IR."
                if strict:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)
            else:
                tf, freqs, n_fft_used = cls._compute_tf_from_ir(
                    ir,
                    samplerate,
                    fft_length=attrs["fft"]["n_fft"],
                    window=attrs["fft"]["window"],
                    normalize=bool(attrs["fft"]["normalize"]),
                )
                if n_fft_used is not None:
                    attrs["fft"]["n_fft"] = n_fft_used
            return cls(
                Sofa=Sofa,
                ir=ir,
                tf=tf,
                samplerate=samplerate,
                freqs=freqs,
                attrs=attrs,
                native_domain="ir",
                source_positions=source_positions,
                source_position_type=source_position_type,
                source_position_units=source_position_units,
            )

        if (
            convention == "SimpleFreeFieldHRTF"
            or ("Data.Real" in variables and "Data.Imag" in variables)
        ):
            real = np.asarray(variables["Data.Real"][:], dtype=float)
            imag = np.asarray(variables["Data.Imag"][:], dtype=float)
            tf = real + 1j * imag
            freqs = cls._extract_freqs(dataset)
            if freqs is None:
                message = "Missing N frequency axis; cannot compute IR from TF."
                if strict and samplerate_override is None:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)
            else:
                if strict and freqs.ndim == 1 and freqs.size > 0:
                    if float(np.min(freqs)) >= 0.0 and not np.isclose(freqs[0], 0.0):
                        raise ValueError(
                            "Frequency axis must start at 0 Hz to compute IR from TF."
                        )
            ir, samplerate, n_fft_used = cls._compute_ir_from_tf(
                tf,
                freqs,
                fft_length=attrs["fft"]["n_fft"],
                normalize=bool(attrs["fft"]["normalize"]),
            )
            if n_fft_used is not None:
                attrs["fft"]["n_fft"] = n_fft_used
            if samplerate_override is not None:
                samplerate = samplerate_override
            if ir is None:
                message = "Unable to compute IR from TF with the provided frequency axis."
                if strict:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)
            if strict and samplerate is None:
                raise ValueError("Unable to infer samplerate from frequency axis.")
            return cls(
                Sofa=Sofa,
                ir=ir,
                tf=tf,
                samplerate=samplerate,
                freqs=freqs,
                attrs=attrs,
                native_domain="tf",
                source_positions=source_positions,
                source_position_type=source_position_type,
                source_position_units=source_position_units,
            )

        message = "Unable to determine HRTF domain from SOFA content."
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning)
        return cls(
            Sofa=Sofa,
            attrs=attrs,
            source_positions=source_positions,
            source_position_type=source_position_type,
            source_position_units=source_position_units,
        )

    @staticmethod
    def _extract_sampling_rate(dataset: Any) -> float | None:
        if "Data.SamplingRate" not in dataset.variables:
            return None
        data = np.asarray(dataset.variables["Data.SamplingRate"][:], dtype=float)
        if data.size == 0:
            return None
        return float(data.flat[0])

    @staticmethod
    def _extract_freqs(dataset: Any) -> np.ndarray | None:
        if "N" not in dataset.variables:
            return None
        freqs = np.asarray(dataset.variables["N"][:], dtype=float)
        if freqs.size == 0:
            return None
        return freqs

    @staticmethod
    def _extract_source_metadata(
        dataset: Any,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        if "SourcePosition" not in dataset.variables:
            return None, None, None
        var = dataset.variables["SourcePosition"]
        positions = np.asarray(var[:], dtype=float)
        pos_type = getattr(var, "Type", None)
        pos_units = getattr(var, "Units", None)
        return positions, pos_type, pos_units

    @staticmethod
    def _default_fft_config(
        overrides: dict[str, Any] | None = None,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        config = {
            "n_fft": None,
            "window": None,
            "normalize": False,
        }
        if overrides:
            unknown = set(overrides) - set(config)
            if unknown:
                message = f"Ignoring unsupported fft options: {sorted(unknown)}"
                if strict:
                    raise ValueError(message)
                warnings.warn(message, UserWarning)
            for key in config:
                if key in overrides:
                    config[key] = overrides[key]
        return config

    @staticmethod
    def _compute_tf_from_ir(
        ir: np.ndarray,
        samplerate: float,
        *,
        fft_length: int | None = None,
        window: str | None = None,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        n_fft = fft_length if fft_length is not None else ir.shape[-1]

        signal = ir
        if window:
            window_values = HRTF._window(window, ir.shape[-1])
            if window_values is not None:
                signal = ir * window_values

        tf = np.fft.rfft(signal, n=n_fft, axis=-1)
        if normalize and n_fft:
            tf = tf / float(n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / samplerate)
        return tf, freqs, int(n_fft)

    @classmethod
    def _compute_ir_from_tf(
        cls,
        tf: np.ndarray,
        freqs: np.ndarray | None,
        *,
        fft_length: int | None = None,
        normalize: bool = False,
    ) -> tuple[np.ndarray | None, float | None, int | None]:
        n_fft = fft_length

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
