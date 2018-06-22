import numpy as np

n_walkers = 10
n_steps = 1000

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 10

likelihood = 'chi_squared'

observables = ('ps')

# model = 'wn_ps'
model = 'pl_ps'

prior_params = dict()

ps_kbins = np.logspace(1.0, 2.0, 10)

# Gaussian prior for white noise power spectrum
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]

# Gaussian prior for power law power spectrum
prior_params['pl_ps'] = [
    [7.0, 2.0],  # A
    [2.5, 1.0]  # alpha
]
