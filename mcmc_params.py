import numpy as np

n_walkers = 10
n_steps = 200

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 1

likelihood = 'chi_squared'

#observables = ('ps')
observables = ('ps', 'vid')
#observables =('vid')
#model = 'wn_ps'
#model = 'pl_ps'
model = 'Lco_Pullen'

prior_params = dict()

#ps_kbins = np.logspace(1, 2, 10)
#vid_Tbins = np.logspace(2,3, 11)
ps_kbins = np.logspace(-1.5, -0.1, 10)
vid_Tbins = np.logspace(5.5, 6.7, 10)
#vid_Tbins = np.logspace(5.7, 8, 10)  # Lco, 10x10x10

# Gaussian prior for white noise power spectrum
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]

# Gaussian prior for power law power spectrum
prior_params['pl_ps'] = [
    [7.0, 3.0],  # A
    [2.5, 1.7]  # alpha
]

# Gaussian prior for linear L_CO model
prior_params['Lco_Pullen'] = [
    #[3, 2]
    [np.log10(4e-5), 1.7]  # A
]


model_params_true = dict()
model_params_true['wn_ps'] = [8.3]  # sigma_T for wn_ps
model_params_true['pl_ps'] = [8., 2.]  # A and alpha for pw_ps
model_params_true['Lco_Pullen'] = [np.log10(1e6/5e11)]  # linear model, specify None for default coeffs

map_filename = 'trial2.npy'
samples_filename = 'samples_lco_test2.npy'

output_dir = 'testing_output'
generate_file = True
save_file = True
