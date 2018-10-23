import numpy as np

cosmology = 'Planck15'  # must be an astropy compatible cosmology

map_smoothing = True
FWHM = 4  # arcmin Haavard, Pullen
resolution_factor = 4  # how much finer resolution to use for pixels in high-res map before smoothing
# FWHM = 6  # arcmin Li

lumfunc_bins = np.logspace(3.5, 7.5, 51)
luminosity = 0.5 * (lumfunc_bins[:-1] + lumfunc_bins[1:])
delta_lum = np.diff(lumfunc_bins)

n_pix_x = 22  # no of pixels
n_pix_y = 22

# should be calculated later
# sigma_T = 11.#2.75#1e9  # muK, noise Haavard
sigma_T = 11 * np.sqrt(2)  # 11.0  # 11  # 41.5/np.sqrt(40)#23.25# MuK, Li,  2*11 = 1500 h


halo_catalogue_folder = 'catalogues/'

min_mass = 2.5e10  # 1e12  # 2.5e10

# field of view in degrees, field size
fov_x = 1.5
fov_y = 1.5

n_nu_bins = 512  # 100 number of frequency bins realistically 2^r
nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
nu_i = 34.    # GHz
nu_f = 26.
