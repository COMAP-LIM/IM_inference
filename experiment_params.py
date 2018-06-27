import numpy as np

# for now this just sets the dimensions of the map
# this should talk about FWHM, pixel size, field size
# and frequancy range and resolution etc.

# We should also have info about noise here.

n = 10
x = np.linspace(0, 1, n + 1)
y = np.linspace(0, 1, n + 1)
z = np.linspace(0, 1, n + 1)

sigma_T = 14. # noise

sigma_x = 0.5 # for instrumental beam
sigma_y = 0.5

map_smoothing = True
#n_sigma = 5.