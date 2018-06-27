import numpy as np

n_walkers = 10
n_steps = 1000

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 10

likelihood = 'chi_squared'

#observables = ('ps')
#observables = ('ps', 'vid')
observables =('vid')
#model = 'wn_ps'
model = 'pl_ps'

prior_params = dict()

ps_kbins = np.logspace(1.0, 1.5, 10)
vid_Tbins = np.logspace(2,3, 11)

# Gaussian prior for white noise power spectrum
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]

# Gaussian prior for power law power spectrum
prior_params['pl_ps'] = [
    [7.0, 3.0],  # A
    [2.5, 1.7]  # alpha
]

model_params_true = dict()
model_params_true['wn_ps'] = [8.3] #sigma_T for wn_ps
model_params_true['pl_ps'] = [8., 2.] # A and alpha for pw_ps

filename = 'trial_of_generate2.npy'
generate_file = False #True
save_file = True