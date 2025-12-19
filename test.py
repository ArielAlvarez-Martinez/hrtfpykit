import numpy as np
import sofa
from scipy.fft import rfft, rfftfreq
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import os
from sh_transformation_tools import sht_core
import hrtf_handling
import sh_transformation_tools

hrtf_mag, source_directions, frequency_vector, sampling_rate = hrtf_handling.load_hrtf('hrtfs/hrtf.sofa')

source_directions_degree = np.rad2deg(source_directions)

hrtfs_folder_path = r'C:\Ariel\projects\sh_transformation'

#multi = load_multiple_hrtfs_from_folder(hrtfs_folder_path)

f = hrtf_mag[:,0,0]

print(type(f))



C, f_recons, Y = sht_core(f,source_directions,20)

sh_transformation_tools.plot_sht_reconstruction_comparison(f, f_recons)
