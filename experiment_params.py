import numpy as np

cosmology = 'Planck15'  # must be an astropy compatible cosmology

FWHM = 4  # arcmin Haavard, Pullen
# FWHM = 6  # arcmin Li

n_pix_x = 22  # no of pixels
n_pix_y = 22

# should be calculated later
# sigma_T = 11.#2.75#1e9  # muK, noise Haavard
sigma_T = 11.0  # 11  # 41.5/np.sqrt(40)#23.25# MuK, Li,  2*11 = 1500 h
map_smoothing = True

halo_catalogue_folder = 'catalogues/'

min_mass = 2.5e10  # 1e12  # 2.5e10

# field of view in degrees, field size
fov_x = 1.5
fov_y = 1.5

n_nu_bins = 512  # 100 number of frequency bins realistically 2^r
nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
nu_i = 34.    # GHz
nu_f = 26.
