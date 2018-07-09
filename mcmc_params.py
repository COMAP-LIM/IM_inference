import numpy as np

n_walkers = 200
n_steps = 500

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 1

likelihood = 'chi_squared'

#observables = ('ps')
observables = ('ps', 'vid')
#observables =('vid')
#model = 'wn_ps'
#model = 'pl_ps'
model 	 = 'Lco_test'

prior_params = dict()

ps_kbins = np.logspace(1.0, 1.5, 10)
#vid_Tbins = np.logspace(2,3, 11)
#ps_kbins = np.logspace(1.0, 1.5, 10)
vid_Tbins = np.logspace(6,7, 11)

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
prior_params['Lco_test'] = [
    [1.0, 2.0]  # A
]


model_params_true = dict()
model_params_true['wn_ps'] = [8.3] #sigma_T for wn_ps
model_params_true['pl_ps'] = [8., 2.] # A and alpha for pw_ps
model_params_true['Lco_test'] = [2.] # linear model

map_filename = 'trial2.npy'
samples_filename = 'samples_lco_test2.npy'

output_dir = 'testing_output'
generate_file = True
save_file = True
