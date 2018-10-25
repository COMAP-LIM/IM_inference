import numpy as np

cosmology = 'Planck15'  # must be an astropy compatible cosmology

map_smoothing = True
FWHM = 4  # arcmin Haavard, Pullen
resolution_factor = 4  # how much finer resolution to use for pixels in high-res map before smoothing
# FWHM = 6  # arcmin Li

lumfunc_bins = np.logspace(3.5, 7.5, 51)
luminosity = 0.5 * (lumfunc_bins[:-1] + lumfunc_bins[1:])
delta_lum = np.diff(lumfunc_bins)

# ps_kbins = np.logspace(1, 2, 10)
# vid_Tbins = np.logspace(2,3, 11)
ps_kbins = np.logspace(-1.5, 0.0, 21)  # (-1.5, -0.5, 10)#10)
vid_Tbins = np.logspace(1, 2, 26)
# vid_Tbins = np.logspace(5.7, 8, 10)  # Lco, 10x10x10

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

# model uset to make covariance matrices
# observables = ('ps',)
cov_observables = ('ps', 'vid')
# observables =('vid',)
cov_extra_observables = ('lum',)
# cov_model = 'wn_ps'
# cov_model = 'pl_ps'
# cov_model = 'Lco_Pullen'
# cov_model = 'Lco_Li'
cov_model = 'simp_Li'

model_params_true = dict()
model_params_true['wn_ps'] = [8.3]  # sigma_T for wn_ps
model_params_true['pl_ps'] = [8., 1.]  # A and alpha for pw_ps
model_params_true['Lco_Pullen'] = [-7.3]  # np.log10(1e6/5e11)]
model_params_true['Lco_Li'] = [0.0, 1.17, 0.21, 0.3, 0.3]  # [0.0, 1.37, -1.74, 0.3, 0.3]
model_params_true['simp_Li'] = [1.17, 0.21, 0.5]  # alpha, beta, sigma_tot

cov_full_fov = 1.5  # degrees

cov_catalogue_folder = 'catalogues/'
cov_output_dir = 'cov_output/'
