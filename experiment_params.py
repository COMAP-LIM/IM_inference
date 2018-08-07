import numpy as np

# for now this just sets the dimensions of the map
# this should talk about FWHM, pixel size, field size
# and frequancy range and resolution etc.

# We should also have info about noise here.

cosmology = 'Planck13'
#n = 30
#x = np.linspace(0, 1, n + 1)
#y = np.linspace(0, 1, n + 1)
#z = np.linspace(0, 1, n + 1)

FWHM = 4  # arcmin
n_pix_x = 22  # no of pixels
n_pix_y = 22

# should be calculated later
sigma_T = 1e6  # muK, noise

map_smoothing = True

halo_catalogue_file = 'catalogues/peakpatchcatalogue_1pt4deg_z2pt4-3pt4.npz'
min_mass = 2.5e11

# field of view in degrees, field size
fov_x = 1.4
fov_y = 1.4

n_nu_bins = 32  # number of frequency bins 2^r
nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
nu_i = 34.    # GHz
nu_f = 26.
