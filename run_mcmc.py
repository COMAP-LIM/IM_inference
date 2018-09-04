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
import src.MapObj
import src.tools
import src.Model
import src.Observable
import src.likelihoods
import mcmc_params
import experiment_params as exp_params
import emcee

# from schwimmbad import MPIPool # In future emcee release
from emcee.utils import MPIPool
import sys
import os
import datetime

os.environ["OMP_NUM_THREADS"] = "1"
# Perhaps move this function somewhere else ?


def set_up_mcmc(mcmc_params, exp_params):
    """
    Sets up the different objects for the mcmc-run.
    """

    observables = []
    map_obj = src.MapObj.MapObj(exp_params)

    # Add more if-statements as other observables are implemented.
    # At some point we should add some checks to make sure that a
    # valid model, and a set of observables are actually picked.
    if 'ps' in mcmc_params.observables:
        ps = src.Observable.Power_Spectrum(mcmc_params)
        observables.append(ps)
    if 'vid' in mcmc_params.observables:
        vid = src.Observable.Voxel_Intensity_Distribution(mcmc_params)
        observables.append(vid)

    if (mcmc_params.model == 'wn_ps'):
        model = src.Model.WhiteNoisePowerSpectrum(exp_params)
    if (mcmc_params.model == 'pl_ps'):
        model = src.Model.PowerLawPowerSpectrum(exp_params, map_obj)

    if (mcmc_params.model == 'Lco_Pullen'):
        model = src.Model.Mhalo_to_Lco_Pullen(exp_params, map_obj)
    if (mcmc_params.model == 'Lco_Li'):
        model = src.Model.Mhalo_to_Lco_Li(exp_params, map_obj)

    model.set_up()
    return model, observables, map_obj


def insert_data(data, observables):
    # Inserts data downloaded from file into the corresponding observable
    # objects.
    if isinstance(data, dict):
        for observable in observables:
            # remove "item()"" here if data is dict (and not gotten from file)
            observable.data = data[observable.label]
    else:
        for observable in observables:
            # remove "item()"" here if data is dict (and not gotten from file)
            observable.data = data.item()[observable.label]


def lnprob(model_params, model, observables, map_obj):
    """
    Simulates the experiment for given model parameters to estimate the
    mean of the observables and uses this mean to estimate the
    likelihood.
    """
    # print('in lnprob')
    # Simulate the required number of realizations in order
    # to estimate the mean value of the different observables
    # at the current model parameters.
    ln_prior = 0.0
    ln_prior += model.ln_prior(model_params,
                               mcmc_params.prior_params[model.label])
    if not np.isfinite(ln_prior):
        return -np.infty
    for i in range(mcmc_params.n_realizations):
        map_obj.map = model.generate_map(
            model_params) + map_obj.generate_noise_map()
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
    if not np.isfinite(ln_likelihood):
        return -np.infty

    return ln_prior + ln_likelihood


def get_data(mcmc_params, exp_params, observables, model):

    if not mcmc_params.generate_file:
        print('opening map data file')
        map_obj.map = np.load(mcmc_params.map_filename)

    else:
        # print('in else')
        model_params = mcmc_params.model_params_true[model.label]
        map_obj.map = model.generate_map(
            model_params) + map_obj.generate_noise_map()
        if mcmc_params.save_file:
            print('saving map to file')
            np.save(mcmc_params.map_filename, map_obj.map)

    map_obj.calculate_observables(observables)

    data = dict()

    for observable in observables:
        data[observable.label] = observable.values
        print(observable.label, observable.values)
        if 0 in observable.values:
            print('some of data values are equal to 0')
            sys.exit()

    # print(data.item()['ps'])

    insert_data(data, observables)
    return data


start_time = datetime.datetime.now()
mcmc_chains_fp, mcmc_log_fp = src.tools.make_log_file_handles(
    mcmc_params.output_dir)
src.tools.make_picklable(exp_params, mcmc_params)

model, observables, map_obj = set_up_mcmc(mcmc_params, exp_params)
# print('loaded stuff')
# load mock data
get_data(mcmc_params, exp_params, observables,
         model)  # np.load("ps_data.npy")

# pool = MPIPool(loadbalance=True)
# if not pool.is_master():
#     pool.wait()
#     sys.exit(0)

sampler = emcee.EnsembleSampler(mcmc_params.n_walkers, model.n_params, lnprob,
                                args=(model, observables, map_obj), threads=4)
# pool=pool)


pos = model.mcmc_walker_initial_positions(
    mcmc_params.prior_params[model.label], mcmc_params.n_walkers)
samples = np.zeros((mcmc_params.n_steps,
                    mcmc_params.n_walkers,
                    model.n_params))

i = 0
with open(mcmc_chains_fp, 'w') as chains_file:
    while i < mcmc_params.n_steps:
        print('undergoing iteration {0}'.format(i))
        sys.stdout.flush()
        for result in sampler.sample(pos, iterations=1, storechain=True):
            samples[i], _, blobs = result
            pos = samples[i]
            chains_file.write('\n'.join([str(item) for sublist in pos
                                         for item in sublist]) + '\n')
            sys.stdout.flush()
            i += 1

# pool.close()
samples = samples.reshape(mcmc_params.n_steps * mcmc_params.n_walkers,
                          model.n_params)


np.save(mcmc_params.samples_filename, samples)

src.tools.write_log_file(mcmc_log_fp, start_time)
