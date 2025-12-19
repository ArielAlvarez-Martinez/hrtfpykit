import numpy as np
import sofa
from scipy.fft import rfft, rfftfreq
from scipy.special import sph_harm
import matplotlib.pyplot as plt

def _read_sofa(path):
    sofa_object = sofa.Database.open(path)
    hrir = np.array(sofa_object.Data.IR)         
    sampling_rate = int(np.array(sofa_object.Data.SamplingRate)[0])
    source_directions = np.array(sofa_object.Source.Position)
    source_direction_type = sofa_object.Source.Position.Type

    return hrir, sampling_rate, source_directions, source_direction_type

def _parse_directions(source_direction, source_direction_type):
    if source_direction_type.lower() != 'spherical':
        raise ValueError(
            "Unsupported SourcePositionType. Only 'spherical' is supported."
        )

    az = source_direction[:, 0]
    el = source_direction[:, 1]

    if np.max(np.abs(az)) > 2 * np.pi:
        az = np.deg2rad(az)
    if np.max(np.abs(el)) > np.pi:
        el = np.deg2rad(el)

    return np.column_stack((az, el))

def _hrir_to_hrtf_magnitude(hrir, sampling_rate):
    H = rfft(hrir, axis=-1, n=256)
    freqs = rfftfreq(256, d=1.0 / sampling_rate)
    return np.abs(H), freqs


def load_hrtf(sofa_path):
    """
    Load HRTF magnitude data from a SOFA file.

    Returns
    -------
    hrtf_mag : ndarray, shape (N_dirs, N_ears, N_freqs)
    source_directions     : ndarray, shape (N_dirs, 2) [az, el] in radians
    frequency_vector    : ndarray, shape (N_freqs,)
    sampling_rate       : float
    """
    hrir, sampling_rate, source_direction, source_direction_type = _read_sofa(sofa_path)
    source_directions = _parse_directions(source_direction, source_direction_type)
    hrtf_mag, frequency_vector = _hrir_to_hrtf_magnitude(hrir, sampling_rate)

    return hrtf_mag, source_directions, frequency_vector, sampling_rate

import os


def load_multiple_hrtfs_from_folder(folder_path):
    """
    Load multiple HRTFs from a folder containing SOFA files.

    Parameters
    ----------
    folder_path : str
        Path to folder containing SOFA files

    Returns
    -------
    hrtf_mag_list : list of ndarray
        Each element has shape (N_dirs, N_ears, N_freqs)
    source_directions : ndarray
        Source directions (shared across all HRTFs)
    frequency_vector : ndarray
        Frequency axis in Hz (shared)
    fs : float
        Sampling rate (shared)
    file_names : list of str
        Loaded SOFA file names (for labeling / tracking)
    """

    hrtf_mag_list = []
    file_names = []

    source_directions_ref = None
    frequency_vector_ref = None
    fs_ref = None

    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".sofa"):
            continue

        sofa_path = os.path.join(folder_path, fname)

        hrtf_mag, source_directions, frequency_vector, fs = load_hrtf(sofa_path)

        # --- consistency checks ---
        if source_directions_ref is None:
            source_directions_ref = source_directions
            frequency_vector_ref = frequency_vector
            fs_ref = fs
        else:
            if not np.allclose(source_directions, source_directions_ref):
                raise ValueError(f"Source directions mismatch in {fname}")

            if not np.allclose(frequency_vector, frequency_vector_ref):
                raise ValueError(f"Frequency vector mismatch in {fname}")

            if fs != fs_ref:
                raise ValueError(f"Sampling rate mismatch in {fname}")

        hrtf_mag_list.append(hrtf_mag)
        file_names.append(fname)

    if len(hrtf_mag_list) == 0:
        raise RuntimeError("No SOFA files found in the given folder.")

    return (
        hrtf_mag_list,
        source_directions_ref,
        frequency_vector_ref,
        fs_ref,
        file_names
    )



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

def plot_hrtf_magnitude_multiple_positions(
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

def reconstruction_error(f, f_recons):

    f = np.asarray(f).reshape(-1)
    f_recons = np.asarray(f_recons).reshape(-1)

    diff = f - f_recons

    abs_err = np.linalg.norm(diff)
    rel_err = abs_err / np.linalg.norm(f)
    rms_err = np.sqrt(np.mean(diff**2))

    return abs_err, rel_err, rms_err


def print_validation_report(f, f_recons, label=""):
    abs_err, rel_err, rms_err = reconstruction_error(f, f_recons)

    print("---- Validation Report", label, "----")
    print(f"Absolute error  : {abs_err:.8f}")
    print(f"Relative error  : {rel_err:.8f}")
    print(f"RMS error       : {rms_err:.8f}")
    print(f"Max |diff|      : {np.max(np.abs(f - f_recons)):.8f}")
    print("-----------------------------------")



hrtf_mag, source_directions, frequency_vector, sampling_rate = load_hrtf('hrtf.sofa')

source_directions_degree = np.rad2deg(source_directions)

hrtfs_folder_path = r'<local-projects>\sh_transformation'

#multi = load_multiple_hrtfs_from_folder(hrtfs_folder_path)

f = hrtf_mag[:,0,0]


from sh_transformation_tools import sht_core_from_scratch, sht_core
import hrtf_handling
import sh_transformation_tools

C, f_recons, Y = sht_core(f,source_directions,20)

sh_transformation_tools.plot_sht_reconstruction_comparison(f, f_recons)
