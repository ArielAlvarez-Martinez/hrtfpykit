import numpy as np
import sofa
from scipy.fft import rfft, rfftfreq
import os

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

def _hrir_to_hrtf_magnitude(hrir, sampling_rate, fft_length=256):
    H = rfft(hrir, axis=-1, n=fft_length)

    freqs = rfftfreq(fft_length, d=1.0 / sampling_rate)
    return np.abs(H), freqs


def load_hrtf(sofa_path, fft_length=256):
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
    hrtf_mag, frequency_vector = _hrir_to_hrtf_magnitude(hrir, sampling_rate, fft_length)

    return hrtf_mag, source_directions, frequency_vector, sampling_rate



def load_multiple_hrtfs_from_folder(folder_path, fft_length=256):
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
        hrtf_mag, source_directions, frequency_vector, fs = load_hrtf(sofa_path, fft_length)

        # --- consistency checks ---
        if source_directions_ref is None:
            source_directions_ref = source_directions
            frequency_vector_ref = frequency_vector
            fs_ref = fs
        else:
            if source_directions_ref.shape != source_directions.shape:
                raise ValueError(f'Source directions shape mismatch in {fname}')
            
            if frequency_vector_ref.shape != frequency_vector.shape:
                raise ValueError(f'Source directions shape mismatch in {fname}')    
            
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

