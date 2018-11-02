import numpy as np

n_walkers = 48
n_steps = 2000
n_patches = 1

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 10

# likelihood = 'chi_squared'
likelihood = 'chi_squared_cov'
cov_output_folder = 'cov_output'
cov_id = '2'

# observables = ('ps',)
observables = ('ps', 'vid')
# observables =('vid',)
extra_observables = ('lum',)
# mcmc_model = 'wn_ps'
# mcmc_model = 'pl_ps'
# mcmc_model = 'Lco_Pullen'
# mcmc_model = 'Lco_Li'
mcmc_model = 'simp_Li'

prior_params = dict()

# Gaussian prior for white noise power spectrum
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]

# Gaussian prior for power law power spectrum
# [mean, stddev]
prior_params['pl_ps'] = [
    [7.0, 3.0],  # A
    [2.5, 1.7]  # alpha
]

# Gaussian prior for linear L_CO model
prior_params['Lco_Pullen'] = [
    [-6.0, 1.]  # A
]

prior_params['Lco_Li'] = [
    [0.0, 0.3], 	# logdelta_MF
    [1.17, 0.37],	 # alpha - log10 slope
    [0.21, 3.74],	 # beta - log10 intercept
    [0.3, 0.1],		# sigma_SFR
    [0.3, 0.1],		# sigma_Lco
]

prior_params['simp_Li'] = [
    [1.17, 0.37],	 # alpha - log10 slope
    [0.21, 3.74],	 # beta - log10 intercept
    [0.5, 0.3],	     # sigma_tot
]

# halo_catalogue_folder = 'catalogues/'

# min_mass = 2.5e10  # 1e12  # 2.5e10

model_params_true = dict()
model_params_true['wn_ps'] = [8.3]  # sigma_T for wn_ps
model_params_true['pl_ps'] = [8., 1.]  # A and alpha for pw_ps
model_params_true['Lco_Pullen'] = [-7.3]  # np.log10(1e6/5e11)]
model_params_true['Lco_Li'] = [0.0, 1.17, 0.21, 0.3, 0.3]  # [0.0, 1.37, -1.74, 0.3, 0.3]
model_params_true['simp_Li'] = [1.17, 0.21, 0.5]  # alpha, beta, sigma_tot

map_filename = 'trial4.npy'
samples_filename = 'samples_lco_test4.npy'
run_log_file = 'run_log.txt'

output_dir = 'testing_output'
generate_file = True
save_file = True

# Use MPI-pool?
pool = True
n_threads = 4
