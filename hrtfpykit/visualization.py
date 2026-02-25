import numpy as np 
import matplotlib.pyplot as plt

"""
    HRTF VISUALIZATION AND COMPARISON 
"""

def _get_index_from_specific_source_position(source_directions, desired_az, desired_el) -> int:
    """
    source_directions   :   numpy array of shape (N, 3)
                    [azimuth, elevation, distance]
    desired_az  :   desired azimuth in degrees
    desired_el  :   desired elevation in degrees

    returns: index of closest position
    """
    # Compute angular distance
    diff = np.sqrt(
        (source_directions[:, 0] - desired_az)**2 +
        (source_directions[:, 1] - desired_el)**2
    )
    # closest index
    idx = int(np.argmin(diff))
    return idx

def plot_hrtf_magnitude(
    hrtf_mag,
    source_directions,
    frequency_vector,
    azimuth,
    elevation,
    ear=0,
    freq_limits=None
):
    """
    Plot HRTF magnitude for a given direction and ear.

    Parameters
    ----------
    hrtf_mag : ndarray, shape (N_dirs, N_ears, N_freqs)
        HRTF magnitude
    source_directions : ndarray, shape (N_dirs, 2)
        [azimuth, elevation] in degrees
    frequency_vector : ndarray, shape (N_freqs,)
        Frequency axis in Hz
    azimuth : float
        Desired azimuth (degrees)
    elevation : float
        Desired elevation (degress)
    ear : int
        Ear index (0 = left, 1 = right)
    freq_limits : tuple or None
        (fmin, fmax) in Hz
    """

    # --- find closest direction ---
    idx  = _get_index_from_specific_source_position(source_directions, azimuth, elevation)

    mag = hrtf_mag[idx, ear, :]

    # --- frequency limits ---
    if freq_limits is not None:
        fmin, fmax = freq_limits
        mask = (frequency_vector >= fmin) & (frequency_vector <= fmax)
        freqs_plot = frequency_vector[mask]
        mag = mag[mask]
    else:
        freqs_plot = frequency_vector

    # --- plot ---
    plt.figure(figsize=(10,5))
    plt.semilogx(freqs_plot, 20 * np.log10(mag + 1e-12))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(
        f"HRTF magnitude | az={azimuth:.2f} rad, "
        f"el={elevation:.2f} rad, ear={ear}"
    )
    plt.xticks(ticks =
               ([1000,5000,10000,20000]), labels=(['1K', '5K', '10K', '20k']))
    plt.grid(linestyle = '--')
    plt.show()

def plot_hrtf_magnitude_multiple_hrtfs(
    hrtf_mag_list,
    source_directions,
    frequency_vector,
    azimuth,
    elevation,
    ear=0,
    freq_limits=None,
    labels=None
):
    """
    Plot and compare HRTF magnitudes from multiple HRTFs at a given
    source direction and ear.

    Parameters
    ----------
    hrtf_mag_list : list of ndarray
        List of HRTF magnitude arrays.
        Each array must have shape (N_dirs, N_ears, N_freqs).
        Length: 1 to ~10.
    source_directions : ndarray, shape (N_dirs, 2)
        [azimuth, elevation] in degrees
    frequency_vector : ndarray, shape (N_freqs,)
        Frequency axis in Hz
    azimuth : float
        Desired azimuth (degrees)
    elevation : float
        Desired elevation (degrees)
    ear : int
        Ear index (0 = left, 1 = right)
    freq_limits : tuple or None
        (fmin, fmax) in Hz
    labels : list of str or None
        Labels for each HRTF (used in legend)
    """

    if labels is None:
        labels = [f"HRTF {i+1}" for i in range(len(hrtf_mag_list))]

    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']

    # --- find closest direction (shared for all HRTFs) ---
    idx = _get_index_from_specific_source_position(
        source_directions, azimuth, elevation
    )

    plt.figure(figsize=(10, 5))

    for i, hrtf_mag in enumerate(hrtf_mag_list):
        mag = hrtf_mag[idx, ear, :]

        # --- frequency limits ---
        if freq_limits is not None:
            fmin, fmax = freq_limits
            mask = (frequency_vector >= fmin) & (frequency_vector <= fmax)
            freqs_plot = frequency_vector[mask]
            mag_plot = mag[mask]
        else:
            freqs_plot = frequency_vector
            mag_plot = mag

        plt.semilogx(
            freqs_plot,
            20 * np.log10(mag_plot + 1e-12),
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            label=labels[i]
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(
        f"HRTF magnitude comparison | az={azimuth:.1f}°, "
        f"el={elevation:.1f}°, ear={ear}"
    )

    plt.xticks(
        ticks=[1000, 5000, 10000, 20000],
        labels=['1K', '5K', '10K', '20K']
    )

    plt.grid(linestyle='--')
    plt.legend()
    plt.show()


def plot_hrtf_magnitude_multiple_source_positions(
    hrtf_mag,
    source_directions,
    frequency_vector,
    positions,
    ear=0,
    freq_limits=None
):
    """
    Plot HRTF magnitude for multiple source positions in the same figure.

    Parameters
    ----------
    hrtf_mag : ndarray, shape (N_dirs, N_ears, N_freqs)
        HRTF magnitude
    source_directions : ndarray, shape (N_dirs, 3)
        [azimuth, elevation, distance] in degrees
    frequency_vector : ndarray, shape (N_freqs,)
        Frequency axis in Hz
    positions : list of tuples
        [(az1, el1), (az2, el2), ..., (azN, elN)] in degrees
    ear : int
        Ear index (0 = left, 1 = right)
    freq_limits : tuple or None
        (fmin, fmax) in Hz
    """

    plt.figure(figsize=(10, 5))

    for azimuth, elevation in positions:
        # --- find closest direction ---
        idx = _get_index_from_specific_source_position(
            source_directions, azimuth, elevation
        )

        mag = hrtf_mag[idx, ear, :]

        # --- frequency limits ---
        if freq_limits is not None:
            fmin, fmax = freq_limits
            mask = (frequency_vector >= fmin) & (frequency_vector <= fmax)
            freqs_plot = frequency_vector[mask]
            mag_plot = mag[mask]
        else:
            freqs_plot = frequency_vector
            mag_plot = mag

        plt.semilogx(
            freqs_plot,
            20 * np.log10(mag_plot + 1e-12),
            label=f"az={azimuth:.1f}°, el={elevation:.1f}°"
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"HRTF magnitude comparison | ear={ear}")

    plt.xticks(
        ticks=[1000, 5000, 10000, 20000],
        labels=['1K', '5K', '10K', '20K']
    )

    plt.grid(linestyle='--')
    plt.legend()
    plt.show()

def plot_hrtf_magnitude_both_ears(
    hrtf_mag,
    source_directions,
    frequency_vector,
    azimuth,
    elevation,
    freq_limits=None
):
    """
    Plot HRTF magnitude for both ears at a given source direction.

    Parameters
    ----------
    hrtf_mag : ndarray, shape (N_dirs, N_ears, N_freqs)
        HRTF magnitude
    source_directions : ndarray, shape (N_dirs, 3)
        [azimuth, elevation, distance] in degrees
    frequency_vector : ndarray, shape (N_freqs,)
        Frequency axis in Hz
    azimuth : float
        Desired azimuth (degrees)
    elevation : float
        Desired elevation (degrees)
    freq_limits : tuple or None
        (fmin, fmax) in Hz
    """

    # --- find closest direction ---
    idx = _get_index_from_specific_source_position(
        source_directions, azimuth, elevation
    )

    mag_left  = hrtf_mag[idx, 0, :]
    mag_right = hrtf_mag[idx, 1, :]

    # --- frequency limits ---
    if freq_limits is not None:
        fmin, fmax = freq_limits
        mask = (frequency_vector >= fmin) & (frequency_vector <= fmax)
        freqs_plot = frequency_vector[mask]
        mag_left = mag_left[mask]
        mag_right = mag_right[mask]
    else:
        freqs_plot = frequency_vector

    # --- plot ---
    plt.figure(figsize=(10, 5))
    plt.semilogx(freqs_plot, 20 * np.log10(mag_left + 1e-12), label="Left ear")
    plt.semilogx(freqs_plot, 20 * np.log10(mag_right + 1e-12), label="Right ear")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(
        f"HRTF magnitude | az={azimuth:.1f}°, "
        f"el={elevation:.1f}°"
    )

    plt.xticks(
        ticks=[1000, 5000, 10000, 20000],
        labels=['1K', '5K', '10K', '20K']
    )

    plt.grid(linestyle='--')
    plt.legend()
    plt.show()

def plot_hrtf_magnitude_main_directions(
    hrtf_mag,
    source_directions,
    frequency_vector,
    ear=0,
    freq_limits=None
):
    """
    Plot HRTF magnitude for four cardinal directions in one figure:
    Front (0,0), Back (180,0), Left (90,0), Right (270,0)

    Parameters
    ----------
    hrtf_mag : ndarray, shape (N_dirs, N_ears, N_freqs)
        HRTF magnitude
    source_directions : ndarray, shape (N_dirs, 3)
        [azimuth, elevation, distance] in degrees
    frequency_vector : ndarray, shape (N_freqs,)
        Frequency axis in Hz
    ear : int
        Ear index (0 = left, 1 = right)
    freq_limits : tuple or None
        (fmin, fmax) in Hz
    """

    directions = {
        "Front (0°, 0°)": (0, 0),
        "Back (180°, 0°)": (180, 0),
        "Left (90°, 0°)": (90, 0),
        "Right (270°, 0°)": (270, 0)
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (label, (az, el)) in zip(axes, directions.items()):
        print(source_directions, az, el)
        idx = _get_index_from_specific_source_position(
            source_directions, az, el
        )
        mag = hrtf_mag[idx, ear, :]
        if freq_limits is not None:
            fmin, fmax = freq_limits
            mask = (frequency_vector >= fmin) & (frequency_vector <= fmax)
            freqs_plot = frequency_vector[mask]
            mag_plot = mag[mask]
        else:
            freqs_plot = frequency_vector
            mag_plot = mag

        ax.semilogx(freqs_plot, 20 * np.log10(mag_plot + 1e-12))
        ax.set_title(label)
        ax.grid(linestyle='--')

        ax.set_xticks([1000, 5000, 10000, 20000])
        ax.set_xticklabels(['1K', '5K', '10K', '20K'])

    fig.suptitle(f"HRTF magnitude | ear={ear}", fontsize=14)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Magnitude (dB)")

    plt.tight_layout()
    plt.show()

def plot_hrtf_magnitude_main_directions_multiple_hrtf(
    hrtf_mag_list,
    source_directions,
    frequency_vector,
    ear=0,
    freq_limits=None,
    labels=None
):
    """
    Plot HRTF magnitude for four cardinal directions (Front, Back, Left, Right)
    comparing multiple HRTFs in the same figure.

    Parameters
    ----------
    hrtf_mag_list : list of ndarray
        Each ndarray has shape (N_dirs, N_ears, N_freqs)
        Length: 1 to ~10 HRTFs
    source_directions : ndarray, shape (N_dirs, 3)
        [azimuth, elevation, distance] in degrees
    frequency_vector : ndarray, shape (N_freqs,)
        Frequency axis in Hz
    ear : int
        Ear index (0 = left, 1 = right)
    freq_limits : tuple or None
        (fmin, fmax) in Hz
    labels : list of str or None
        Labels for each HRTF (for legend)
    """

    directions = {
        "Front (0°, 0°)": (0, 0),
        "Back (180°, 0°)": (180, 0),
        "Left (90°, 0°)": (90, 0),
        "Right (270°, 0°)": (270, 0)
    }

    if labels is None:
        labels = [f"HRTF {i+1}" for i in range(len(hrtf_mag_list))]

    # Color + linestyle cycles (safe up to ~10)
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (dir_label, (az, el)) in zip(axes, directions.items()):
        idx = _get_index_from_specific_source_position(
            source_directions, az, el
        )

        for i, hrtf_mag in enumerate(hrtf_mag_list):
            mag = hrtf_mag[idx, ear, :]

            if freq_limits is not None:
                fmin, fmax = freq_limits
                mask = (frequency_vector >= fmin) & (frequency_vector <= fmax)
                freqs_plot = frequency_vector[mask]
                mag_plot = mag[mask]
            else:
                freqs_plot = frequency_vector
                mag_plot = mag

            ax.semilogx(
                freqs_plot,
                20 * np.log10(mag_plot + 1e-12),
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                label=labels[i] if ax is axes[0] else None
            )

        ax.set_title(dir_label)
        ax.grid(linestyle='--')
        ax.set_xticks([1000, 5000, 10000, 20000])
        ax.set_xticklabels(['1K', '5K', '10K', '20K'])

    fig.suptitle(f"HRTF magnitude comparison | ear={ear}", fontsize=14)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Magnitude (dB)")

    # Single shared legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, legend_labels, loc='lower center', ncol=4)
    axes[0].legend(
    handles,
    legend_labels,
    loc='upper left',
    fontsize=9,
    frameon=True
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def plot_comparison_two_hrtf_magnitude_vectors(hrtf_mag_1, hrtf_mag_2, frequency_vector, labels=["HRTF_magnitude_1", "HRTF_magnitude_2"]):
    
    """
    Plot and visually compare two HRTF magnitude vectors on the same
    frequency axis.

    This function overlays the magnitude responses of two HRTFs in the
    frequency domain using a logarithmic frequency scale. It is intended
    for qualitative comparison between two HRTF magnitude representations,
    such as original vs reconstructed HRTFs or comparisons between
    different subjects or models.

    Parameters
    ----------
    hrtf_mag_1 : array_like, shape (N_freqs,)
        First HRTF magnitude vector. This is typically the reference HRTF
        (e.g., ground truth or measured HRTF).

    hrtf_mag_2 : array_like, shape (N_freqs,)
        Second HRTF magnitude vector. This is typically a reconstructed,
        predicted, or alternative HRTF to be compared against the reference.

    frequency_vector : array_like, shape (N_freqs,)
        Frequency axis in Hz corresponding to the HRTF magnitude vectors.
        Both HRTF magnitude vectors are assumed to be defined on the same
        frequency grid.

    labels : list of str, optional
        Labels used in the plot legend to identify each HRTF magnitude curve.
        The list must contain two strings, one for each HRTF.
        Default is ["HRTF_magnitude_1", "HRTF_magnitude_2"].
    """

    # Magnitude vector checks
    if hrtf_mag_1.shape != hrtf_mag_2.shape:
        raise ValueError("HRTF magnitude vectors must have the same shape")

    if hrtf_mag_1.shape != frequency_vector.shape or hrtf_mag_2.shape != frequency_vector.shape:
        raise ValueError("HRTF magnitude vectors and frequency_vector must have the same shape")

    plt.figure(figsize=(10,5))
    plt.semilogx(frequency_vector, 20 * np.log10(hrtf_mag_1 + 1e-12), label=labels[0], linestyle="solid")
    plt.semilogx(frequency_vector, 20 * np.log10(hrtf_mag_2 + 1e-12), label=labels[1], linestyle="dashed")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title( "HRTF magnitude comparison")
    plt.xticks(ticks =
               ([1000,5000,10000,20000]), labels=(['1K', '5K', '10K', '20k']))
    plt.grid(linestyle = '--')
    plt.legend()
    plt.show()

