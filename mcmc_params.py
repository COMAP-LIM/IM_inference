n_walkers = 2
n_steps = 400

# number of independent simulations of the model for each
# step in the MCMC
n_realizations = 10

likelihood = 'chi_squared'

observables = ('ps')

model = 'wn_ps'

prior_params = dict()

# Gaussian prior for white_noise power spectrum
prior_params['wn_ps'] = [
    [5.0, 3.0]  # sigma_T
]
