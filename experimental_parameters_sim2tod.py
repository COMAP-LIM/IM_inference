import numpy as np

cosmology = "Planck15"  # must be an astropy compatible cosmology

map_smoothing = True
FWHM = 4.5  # arcmin Haavard, Pullen
FWHM_nu = 32.0 * 1e-3  # in GHz
resolution_factor = 1  # 1  # how much finer resolution to use for pixels in high-res map before smoothing
# FWHM = 6  # arcmin Li
use_linewidth_bins = True

comap_beam = True
comap_beam_path = "/mn/stornext/d5/data/nilsoles/nils/im_inference_fork/tables"
comap_beam_file = "beam.txt"
comap_beam_radius_file = "beam_r.txt"

lumfunc_bins = np.logspace(3.5, 7.5, 51)
luminosity = 0.5 * (lumfunc_bins[:-1] + lumfunc_bins[1:])
delta_lum = np.diff(lumfunc_bins)

# ps_kbins = np.logspace(1, 2, 10)
# vid_Tbins = np.logspace(2,3, 11)
# ps_kbins = np.logspace(-1.5, 0.0, 21)  # (-1.5, -0.5, 10)#10)
ps_kbins = np.logspace(-2.0, np.log10(1.5), 21) # (-1.5, -0.5, 10)#10)
vid_Tbins = np.logspace(1, 2, 26)  # np.logspace(1, 2, 26)
# vid_Tbins = np.logspace(5.7, 8, 10)  # Lco, 10x10x10

n_pix_x = 150 * 1  # no of pixels
n_pix_y = 150 * 1

# should be calculated later
# sigma_T = 11.#2.75#1e9  # muK, noise Haavard
sigma_T = 18.35  # 11  # * np.sqrt(2)  # 11.0  # 11  # 41.5/np.sqrt(40)#23.25# MuK, Li,  2*11 = 1500 h


# halo_catalogue_folder = 'catalogues/'

min_mass = 2.5e10  # 1e12  # 2.5e10

# field of view in degrees, field size
fov_x = 5.0
fov_y = 5.0

n_nu_bins = 4096  # 256  # 100 number of frequency bins realistically 2^r
nu_rest = 115.27  # rest frame frequency of CO(1-0) transition in GHz
nu_i = 34.0  # GHz
nu_f = 26.0

# model uset to make covariance matrices
# observables = ('ps',)
cov_observables = ("ps", "vid")
# observables =('vid',)
cov_extra_observables = ("lum",)
# cov_model = 'wn_ps'
# cov_model = 'pl_ps'
# cov_model = 'Lco_Pullen'
#cov_model = 'Lco_Li'
# cov_model = 'simp_Li'
cov_model = "power_cov"

model_params_cov = dict()
model_params_cov["wn_ps"] = [8.3]  # sigma_T for wn_ps
model_params_cov["pl_ps"] = [8.0, 1.0]  # A and alpha for pw_ps
model_params_cov["Lco_Pullen"] = [-7.3]  # np.log10(1e6/5e11)]
model_params_cov["Lco_Li"] = [0.0, 1.17, 0.21, 0.3, 0.3]  # [0.0, 1.37, -1.74, 0.3, 0.3]
model_params_cov["simp_Li"] = [1.17, 0.21, 0.5]  # alpha, beta, sigma_tot
model_params_cov["power_cov"] = [-2.85, -0.42, 10.63, 12.3, 0.42]

cov_full_fov = 9.0  # 1.5  # degrees

cov_catalogue_folder = "../limlam_mocker/full_cita_catalogues/"
cov_output_dir = "cov_output/"
