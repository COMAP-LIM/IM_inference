import numpy as np

n_walkers = 4
n_steps = 100

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 10

likelihood = 'chi_squared'

observables = ('ps')

#model = 'wn_ps'
model = 'pl_ps'

prior_params = dict()

ps_kbins = np.logspace(1,2, 10)

# Gaussian prior for white_noise power spectrum
#prior_params['wn_ps'] = [
#    [5.0, 3.0]  # sigma_T
#]
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]


prior_params['pl_ps'] = [
    [7.0, 2.0], # A
    [2.5, 1.0] # alpha
]
