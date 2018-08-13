import numpy as np

# for now this just sets the dimensions of the map
# this should talk about FWHM, pixel size, field size
# and frequancy range and resolution etc.

# We should also have info about noise here.

cosmology = 'Planck15'  # must be an astropy compatible cosmology

#FWHM = 4  # arcmin Haavard
FWHM = 6  # arcmin Li

n_pix_x = 22  # no of pixels
n_pix_y = 22

# should be calculated later
#sigma_T = 11.#2.75#1e9  # muK, noise Haavard
sigma_T = 23.25# MuK, Li
map_smoothing = True

halo_catalogue_file = 'catalogues/peakpatchcatalogue_1pt4deg_z2pt4-3pt4.npz'
#halo_catalogue_file = 'full_catalogues/COMAP_z2.39-3.44_1140Mpc_seed_13579.npz'
#halo_catalogue_file = 'full_catalogues/COMAP_z2.39-3.44_1140Mpc_seed_13581.npz'
#halo_catalogue_file = 'full_catalogues/COMAP_z2.39-3.44_1140Mpc_seed_13583.npz'
#halo_catalogue_file = 'full_catalogues/COMAP_z2.39-3.44_1140Mpc_seed_13585.npz'

min_mass = 2.5e11

# field of view in degrees, field size
fov_x = 1.4
fov_y = 1.4

# Pullen
#n_nu_bins = 512  # number of frequency bins 2^r
#nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
#nu_i = 34.    # GHz
#nu_f = 26.

# Li
n_nu_bins = 100  # 100 number of frequency bins 2^r
nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
nu_i = 34.    # GHz
nu_f = 30.
