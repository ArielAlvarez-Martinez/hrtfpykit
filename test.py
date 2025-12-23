import numpy as np
import sofa
from scipy.fft import rfft, rfftfreq
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import os
from sh_transformation_tools import sht_core
import hrtf_loader, hrtf_visualization, sh_transformation_tools


hrtf_mag, source_directions, frequency_vector, sampling_rate = hrtf_loader.load_hrtf('hrtf.sofa')
hrtfs_list = hrtf_loader.load_multiple_hrtfs_from_folder('hrtfs/')

hrtfs_mag_list = hrtfs_list[0]

f = hrtf_mag[:,0,0]

hrtf_1 = hrtf_loader.load_hrtf('hrtfs/hrtf_24.sofa')
hrtf_2 = hrtf_loader.load_hrtf('hrtfs/hrtf_58.sofa')

hrtf_mag_1 = hrtf_1[0]
hrtf_mag_2 = hrtf_2[0]

hrtf_mag_specific_direction_1 = hrtf_mag_1[10,0,:]
hrtf_mag_specific_direction_2 = hrtf_mag_2[10,0,:]



#frequency_vector = frequency_vector[:120]

#hrtf_handling.plot_hrtf_magnitude(hrtf_mag_1, source_directions, frequency_vector, 0, 0,0)
#hrtf_visualization.plot_hrtf_magnitude_both_ears(hrtf_mag_2, source_directions, frequency_vector, 0, 0)
# hrtf_visualization.plot_hrtf_magnitude_multiple_hrtfs(
#     hrtf_mag_list=hrtfs_mag_list,
#     source_directions=source_directions,
#     frequency_vector=frequency_vector,
#     azimuth=0,
#     elevation=0, 
#     ear=0,
#     labels=["Subject A", "Reconstructed", "Reconstructed"]
# )

# C, f_recons, Y = sht_core(f,source_directions,20)

# hrtf_visualization.plot_comparison_two_hrtf_magnitude_vectors(hrtf_mag_specific_direction_1,hrtf_mag_specific_direction_2, frequency_vector=frequency_vector,
#                                                                labels=["Original", "Reconstructed"])


hrtf = hrtf_loader._read_sofa('hrtf.sofa')

print(hrtf[2][100])