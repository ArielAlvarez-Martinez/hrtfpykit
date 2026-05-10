__all__ = [
    "compare_magnitude",
    "compare_amplitude",
    "compare_absolute_itd",
    "compare_absolute_ild",
    "compare_itd_curve",
    "compare_ild_curve",
    "compare_itd_difference",
    "compare_ild_difference",
    "compare_lsd",
    "compare_lsd_plane",
    "plot_sht_reconstruction_comparison",
    "plot_sht_reconstruction_error",
]


def __getattr__(name: str):
    if name == "compare_magnitude":
        from .compare import compare_magnitude
        return compare_magnitude
    if name == "compare_amplitude":
        from .compare import compare_amplitude
        return compare_amplitude
    if name == "compare_absolute_itd":
        from .compare import compare_absolute_itd
        return compare_absolute_itd
    if name == "compare_absolute_ild":
        from .compare import compare_absolute_ild
        return compare_absolute_ild
    if name == "compare_itd_curve":
        from .compare import compare_itd_curve
        return compare_itd_curve
    if name == "compare_ild_curve":
        from .compare import compare_ild_curve
        return compare_ild_curve
    if name == "compare_itd_difference":
        from .compare import compare_itd_difference
        return compare_itd_difference
    if name == "compare_ild_difference":
        from .compare import compare_ild_difference
        return compare_ild_difference
    if name == "compare_lsd":
        from .compare import compare_lsd
        return compare_lsd
    if name == "compare_lsd_plane":
        from .compare import compare_lsd_plane
        return compare_lsd_plane
    if name == "plot_sht_reconstruction_comparison":
        from .sh import plot_sht_reconstruction_comparison
        return plot_sht_reconstruction_comparison
    if name == "plot_sht_reconstruction_error":
        from .sh import plot_sht_reconstruction_error
        return plot_sht_reconstruction_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
