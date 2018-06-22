"""
Script to do MCMC inference from data.

The experiment setup is configured in the
"experiment_params.py" file, while the setup
for the mcmc-run is configured in the
"mcmc_params.py" file.

Basically, we take some input data (corresponding
to a set of observables) and constrain the model
parameters of some IM-model.
"""
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') # Ensure that the Tkinter backend is used for generating figures
from scipy.stats import norm
import matplotlib.pyplot as plt
import src.MapObj
import src.tools
import src.Model
import src.Observable
import src.likelihoods
import mcmc_params
import experiment_params
import emcee


# Perhaps move this function somewhere else ?
def set_up_mcmc(mcmc_params, exp_params):
    """
    Sets up the different objects for the mcmc-run.
    """

    observables = []
    # Add more if-statements as other observables are implemented.
    # At some point we should add some checks to make sure that a
    # valid model, and a set of observables are actually picked.
    if 'ps' in mcmc_params.observables:
        ps = src.Observable.Power_Spectrum()
        observables.append(ps)
    if (mcmc_params.model == 'wn_ps'):
        model = src.Model.WhiteNoisePowerSpectrum(exp_params)
    map_obj = src.MapObj.MapObj(exp_params)
    return model, observables, map_obj


def insert_data(data, observables):
    # Inserts data downloaded from file into the corresponding observable
    # objects.
    for observable in observables:
        # remove "item()"" here if data is dict (and not gotten from file)
        observable.data = data.item()[observable.label]


def lnprob(model_params, model, observables, map_obj):
    """
    Simulates the experiment for given model parameters to estimate the
    mean of the observables and uses this mean to estimate the
    likelihood.
    """

    # Simulate the required number of realizations in order
    # to estimate the mean value of the different observables
    # at the current model parameters.
    for i in range(mcmc_params.n_realizations):
        map_obj.map = model.generate_map(model_params)
        map_obj.calculate_observables(observables)
        for observable in observables:
            observable.add_observable_to_sum()
    for observable in observables:
        observable.calculate_mean(mcmc_params.n_realizations)

    # calculate the actual likelihoods
    ln_likelihood = 0.0
    if (mcmc_params.likelihood == 'chi_squared'):
        for observable in observables:
            ln_likelihood += \
                src.likelihoods.ln_chi_squared(
                    observable.data,
                    observable.mean,
                    observable.independent_var
                )
    # Implement priors in some good way (this is completely ad-hoc).
    prior = norm.logpdf(model_params[0], loc=5, scale=2)
    return prior + ln_likelihood


model, observables, map_obj = set_up_mcmc(mcmc_params, experiment_params)

# load mock data
data = np.load("ps_data.npy")

insert_data(data, observables)

sampler = emcee.EnsembleSampler(mcmc_params.n_walkers, model.n_params, lnprob,
                                args=(model, observables, map_obj))

# starting positions (when implementing priors properly,
# find a good way to draw the starting values from that prior.)
pos = 5 + np.random.randn(mcmc_params.n_walkers, model.n_params)

samples = np.zeros((mcmc_params.n_steps, len(pos), model.n_params))

i = 0
while i < mcmc_params.n_steps:
    print('undergoing iteration {0}'.format(i))
    for result in sampler.sample(pos, iterations=1, storechain=True):
        samples[i], _, blobs = result
        pos = samples[i]
        i += 1
samples = samples.reshape(mcmc_params.n_steps * len(pos), model.n_params)

np.save('samles', samples)

# maybe swich to corner?
n_cut = mcmc_params.n_steps // 5
plt.hist(samples[n_cut:, 0])
plt.show()
