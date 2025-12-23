from hrtf_loader import load_hrtf
import numpy as np
import matplotlib.pyplot as plt

class HRTF:
    def __init__(self, hrtf_mag, source_directions, frequency_vector, sampling_rate):
        self.hrtf_mag = hrtf_mag
        self.source_directions_radians = source_directions
        self.source_directions_degrees = np.rad2deg(self.source_directions_radians)
        self.frequency_vector = frequency_vector
        self.sampling_rate = sampling_rate
        
    @classmethod
    def load_hrtf(cls, sofa_path, fft_length=256):
        hrtf_mag, source_directions, frequency_vector, sampling_rate = load_hrtf(sofa_path, fft_length)

        return cls(hrtf_mag=hrtf_mag,
                   source_directions=source_directions,
                   frequency_vector=frequency_vector,
                   sampling_rate=sampling_rate)
        
    """
    HRTF VISUALIZATION 
    """

    def _get_index_from_specific_source_position(self, desired_az, desired_el) -> int:
        """
        desired_az  :   desired azimuth in degrees
        desired_el  :   desired elevation in degrees

        returns: index of closest position
        """
        # Compute angular distance
        diff = np.sqrt(
            (self.source_directions_degrees[:, 0] - desired_az)**2 +
            (self.source_directions_degrees[:, 1] - desired_el)**2
        )
        # closest index
        idx = int(np.argmin(diff))
        return idx

    def plot_hrtf_magnitude(self,
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
        idx  = self._get_index_from_specific_source_position(azimuth, elevation)

        # Get real azimuth and elevation 
        azimuth, elevation = self.source_directions_degrees[idx]
        mag = self.hrtf_mag[idx, ear, :]

        # --- frequency limits ---
        if freq_limits is not None:
            fmin, fmax = freq_limits
            mask = (self.frequency_vector >= fmin) & (self.frequency_vector <= fmax)
            freqs_plot = self.frequency_vector[mask]
            mag = mag[mask]
        else:
            freqs_plot = self.frequency_vector

        # --- plot ---
        plt.figure(figsize=(10,5))
        plt.semilogx(freqs_plot, 20 * np.log10(mag + 1e-12))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        if ear == 0 :
            ear_title = 'left'
        else:
            ear_title = 'right'
        plt.title(
            f"HRTF magnitude | az={azimuth:.0f} rad, "
            f"el={elevation:.0f} rad, Ear={ear_title}"
        )
        plt.xticks(ticks =
                ([1000,5000,10000,20000]), labels=(['1K', '5K', '10K', '20k']))
        plt.grid(linestyle = '--')
        plt.show()



hrtf_1 = HRTF.load_hrtf("hrtf.sofa")

hrtf_1.plot_hrtf_magnitude(100,200)

print(hrtf_1.source_directions_degrees)