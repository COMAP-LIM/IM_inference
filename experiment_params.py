import numpy as np

# for now this just sets the dimensions of the map
# this should talk about FWHM, pixel size, field size
# and frequancy range and resolution etc.

# We should also have info about noise here.

n = 300
x = np.linspace(0, 1, n + 1)
y = np.linspace(0, 1, n + 1)
z = np.linspace(0, 1, n + 1)

sigma_T = 14.  # noise

sigma_x = 0.5  # for instrumental beam
sigma_y = 0.5

map_smoothing = True
#n_sigma = 5.

coeffs  = None  # specify None for default coeffs
halo_catalogue_file = 'catalogues/peakpatchcatalogue_1pt4deg_z2pt4-3pt4.npz'
min_mass = 2.5e10

# field of view in degrees
fov_x = 1.4
fov_y = 1.4

nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
nu_i    = 34.    # GHz
nu_f    = 26.
nmaps = 300
