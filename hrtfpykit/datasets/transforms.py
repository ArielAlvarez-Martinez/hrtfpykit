from collections.abc import Callable


class HRTFTransform:
    @staticmethod
    def build(method_name: str, *args, **kwargs) -> Callable:
        def transform(hrtf):
            if not hasattr(hrtf, "transform"):
                raise TypeError(
                    "HRTFTransform expects an HRTF-like object with a transform attribute"
                )
            method = getattr(hrtf.transform, method_name, None)
            if method is None or not callable(method):
                raise AttributeError(
                    f"HRTF transform {method_name!r} is not available on {type(hrtf)!r}"
                )
            return method(*args, **kwargs)

        transform.__hrtf_transform__ = True
        return transform

    @staticmethod
    def select(*args, **kwargs) -> Callable:
        def transform(hrtf):
            method = getattr(hrtf, "select", None)
            if method is None or not callable(method):
                raise AttributeError(
                    f"HRTF select is not available on {type(hrtf)!r}"
                )
            return method(*args, **kwargs)

        transform.__hrtf_transform__ = True
        return transform

    @staticmethod
    def apply_window(window_name: str) -> Callable:
        return HRTFTransform.build("apply_window", window_name)

    @staticmethod
    def apply_padding(
        padding_length: int,
        location: str = "end",
        value: float = 0,
    ) -> Callable:
        return HRTFTransform.build(
            "apply_padding",
            padding_length,
            location=location,
            value=value,
        )

    @staticmethod
    def upsampling(new_sample_rate: float) -> Callable:
        return HRTFTransform.build("upsampling", new_sample_rate)

    @staticmethod
    def downsampling(new_sample_rate: float) -> Callable:
        return HRTFTransform.build("downsampling", new_sample_rate)

    @staticmethod
    def apply_fir_filter(
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        num_taps: int = 101,
        window: str | None = None,
    ) -> Callable:
        return HRTFTransform.build(
            "apply_fir_filter",
            filter,
            cutoff=cutoff,
            num_taps=num_taps,
            window=window,
        )

    @staticmethod
    def apply_iir_filter(
        filter: str,
        cutoff: float | tuple[float, float] | None = None,
        order: int = 10,
    ) -> Callable:
        return HRTFTransform.build(
            "apply_iir_filter",
            filter,
            cutoff=cutoff,
            order=order,
        )

    @staticmethod
    def minimum_phase(
        method: str = "homomorphic",
        fft_length: int | None = None,
        epsilon: float = 1e-12,
    ) -> Callable:
        return HRTFTransform.build(
            "minimum_phase",
            method=method,
            fft_length=fft_length,
            epsilon=epsilon,
        )

    @staticmethod
    def to_ctf(
        weights: bool = False,
        magnitude_average: str = "log",
        attenuation: float | None = None,
    ) -> Callable:
        return HRTFTransform.build(
            "to_ctf",
            weights=weights,
            magnitude_average=magnitude_average,
            attenuation=attenuation,
        )

    @staticmethod
    def to_dtf(
        weights: bool = False,
        magnitude_average: str = "log",
        attenuation: float | None = None,
    ) -> Callable:
        return HRTFTransform.build(
            "to_dtf",
            weights=weights,
            magnitude_average=magnitude_average,
            attenuation=attenuation,
        )

    @staticmethod
    def modify_ir(new_ir) -> Callable:
        return HRTFTransform.build("modify_ir", new_ir)

    @staticmethod
    def modify_phase(
        new_phase,
        unit: str = "degrees",
    ) -> Callable:
        return HRTFTransform.build(
            "modify_phase",
            new_phase,
            unit=unit,
        )

    @staticmethod
    def modify_tf(new_tf) -> Callable:
        return HRTFTransform.build("modify_tf", new_tf)

    @staticmethod
    def modify_magnitude(
        new_magnitude,
        scale: str = "linear",
    ) -> Callable:
        return HRTFTransform.build(
            "modify_magnitude",
            new_magnitude,
            scale=scale,
        )

    @staticmethod
    def apply_gain(
        gain,
        scale: str = "db",
    ) -> Callable:
        return HRTFTransform.build(
            "apply_gain",
            gain,
            scale=scale,
        )

    @staticmethod
    def modify_fft_length(new_fft_length: int) -> Callable:
        return HRTFTransform.build("modify_fft_length", new_fft_length)

    @staticmethod
    def modify_source_coordinate_system(coordinate_system: str) -> Callable:
        return HRTFTransform.build(
            "modify_source_coordinate_system",
            coordinate_system,
        )

    @staticmethod
    def add_itd(
        itd: float,
        unit: str = "samples",
    ) -> Callable:
        return HRTFTransform.build(
            "add_itd",
            itd,
            unit=unit,
        )

    @staticmethod
    def delete_itd(
        method: str = "threshold",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
    ) -> Callable:
        return HRTFTransform.build(
            "delete_itd",
            method=method,
            thresh_level=thresh_level,
            upper_cut_freq=upper_cut_freq,
            filter_order=filter_order,
        )
