# To run: screen -dm bash -c 'script -c "mpiexec -n 48 python run_mcmc.py" output.txt'

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
import h5py
import src.MapObj
import src.tools
import src.Model
import src.Observable
import src.likelihoods
import emcee

from schwimmbad import MPIPool  # current emcee release
# from emcee.utils import MPIPool  # old 
from multiprocessing import Pool
import sys
import os
import datetime
import importlib
import inspect

os.environ["OMP_NUM_THREADS"] = "1"

if len(sys.argv) < 2:
    import mcmc_params
else:
    mcmc_params = importlib.import_module(sys.argv[1][:-3])
mcmc_src = inspect.getsource(mcmc_params)

# Make this into a funciton maybe
if mcmc_params.likelihood == 'chi_squared':
    if len(sys.argv) < 3:
        import experiment_params as exp_params
    else:
        exp_params = importlib.import_module(sys.argv[2][:-3])
else:
    cov_folder = mcmc_params.cov_output_folder
    cov_id = int(mcmc_params.cov_id)
    exp_params_fp = (
        cov_folder + '.param'
        + '.experiment_params_id{0:d}'.format(cov_id)
    )
    data_fp = os.path.join(
        cov_folder, 'data',
        'data_id{0:d}.h5'.format(cov_id))
    exp_params = importlib.import_module(exp_params_fp)
    cov_data = h5py.File(data_fp, 'r')
    cov_mat_0 = np.array(cov_data['cov_mat'][:])
    var_ind_0 = np.array(cov_data['var_indep_0'][:]) 
    cov_data.close()
exp_src = inspect.getsource(exp_params)


def lnprob(model_params, model, observables, extra_observables, map_obj):
    """
    Simulates the experiment for given model parameters to estimate the
    mean of the observables and uses this mean to estimate the
    likelihood.
    """
    # Simulate the required number of realizations in order
    # to estimate the mean value of the different observables
    # at the current model parameters.
    ln_prior = 0.0
    ln_prior += model.ln_prior(model_params,
                               mcmc_params.prior_params[model.label])

    for i in range(mcmc_params.n_realizations):
        if exp_params.map_smoothing:
            if exp_params.use_linewidth_bins:
                map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map(
                    model, model_params
                )
            elif exp_params.FWHM_nu is None:
                map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map(
                    model, model_params
                )
            else:
                map_obj.map, map_obj.lum_func = src.tools.create_smoothed_map_3d(
                    model, model_params
                )
        else:
            map_obj.map, map_obj.lum_func = model.generate_map(
                model_params)
        map_obj.map += map_obj.generate_noise_map()
        map_obj.map -= np.mean(map_obj.map.flatten())
        map_obj.calculate_observables(observables)
        map_obj.calculate_observables(extra_observables)
        for observable in observables:
            observable.add_observable_to_sum()
        for observable in extra_observables:
            observable.add_observable_to_sum()
    for observable in observables:
        observable.calculate_mean(mcmc_params.n_realizations)
    for observable in extra_observables:
        observable.calculate_mean(mcmc_params.n_realizations)

    # calculate the actual likelihoods
    ln_likelihood = 0.0
    n_samp = mcmc_params.n_realizations / mcmc_params.n_patches
    if (mcmc_params.likelihood == 'chi_squared'):
        for observable in observables:
            ln_likelihood += \
                src.likelihoods.ln_chi_squared(
                    observable.data,
                    observable.mean,
                    (1 + n_samp) / n_samp * (
                        observable.independent_var
                        / mcmc_params.n_patches
                    )
                )
    elif (mcmc_params.likelihood == 'chi_squared_cov'):
        #n_data = len(var_ind_0)
        n_cov = len(var_ind_0)
        n_data = sum([len(observable.data) for observable in observables])
        i = 0
        data = np.zeros(n_data)
        mean = np.zeros(n_data)
        ind_var = np.zeros(n_data)
        for observable in observables:
            n_data_obs = len(observable.data)
            data[i:i + n_data_obs] = observable.data
            mean[i:i + n_data_obs] = observable.mean
            ind_var[i:i + n_data_obs] = observable.independent_var
            i += n_data_obs
        if (len(observables) > 1):
            var_ratio = np.sqrt(
                np.outer(ind_var, ind_var)
                / np.outer(var_ind_0[:], var_ind_0[:])
            )
            cov_mat = (1 + n_samp) / n_samp * (
                cov_mat_0[:] * var_ratio / mcmc_params.n_patches
            )
        elif (observables[0].label == 'ps'):
            var_ratio = np.sqrt(
                np.outer(ind_var, ind_var)
                / np.outer(var_ind_0[:n_data_obs], var_ind_0[:n_data_obs])
            )
            cov_mat = (1 + n_samp) / n_samp * (
                cov_mat_0[:n_data_obs, :n_data_obs] * var_ratio / mcmc_params.n_patches
            )
        else:
            var_ratio = np.sqrt(
                np.outer(ind_var, ind_var)
            / np.outer(var_ind_0[n_cov-n_data_obs:], var_ind_0[n_cov-n_data_obs:])
            )
            cov_mat = (1 + n_samp) / n_samp * (
                cov_mat_0[n_cov-n_data_obs:, n_cov-n_data_obs:] * var_ratio / mcmc_params.n_patches
            )
        # print(cov_mat)
        ln_likelihood += src.likelihoods.ln_chi_squared_cov(
            data, mean, cov_mat)

    ln_post = ln_prior + ln_likelihood
    if not np.isfinite(ln_post):
        ln_post = -np.infty
    

    return ln_post, (observables, extra_observables)


if __name__ == "__main__":
    src.tools.make_picklable((exp_params, mcmc_params))
    if mcmc_params.pool:
        pool = MPIPool()
        n_pool = pool.size + 1
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = Pool(mcmc_params.n_threads)
        n_pool = mcmc_params.n_threads
    start_time = datetime.datetime.now()
    mcmc_chains_fp, mcmc_log_fp, samples_log_fp, blob_fp, runid = \
        src.tools.set_up_log(mcmc_params.output_dir,
                             mcmc_params, mcmc_src,
                             exp_src, n_pool)

    model, observables, extra_observables, map_obj = src.tools.set_up_mcmc(
        mcmc_params, exp_params)

    src.tools.get_data(mcmc_params, exp_params, model,
                       observables, map_obj, runid)
    
    # filename = "test.h5"
    # backend = emcee.backends.HDFBackend(filename)
    # backend.reset(mcmc_params.n_walkers, model.n_params)
    
    if mcmc_params.pool:
        sampler = emcee.EnsembleSampler(
            mcmc_params.n_walkers, model.n_params, lnprob,
            args=(model, observables, extra_observables, map_obj),
            pool=pool)
    else:
        sampler = emcee.EnsembleSampler(
            mcmc_params.n_walkers, model.n_params, lnprob,
            args=(model, observables, extra_observables, map_obj),
            pool=pool)

    pos = model.mcmc_walker_initial_positions(
        mcmc_params.prior_params[model.label], mcmc_params.n_walkers)
    samples = np.zeros((mcmc_params.n_steps,
                        mcmc_params.n_walkers,
                        model.n_params))
        # for sample in sampler.sample(pos, iterations=mcmc_params.n_steps, progress=True):
        #     # Only check convergence every 100 steps
        #     if sampler.iteration % 10:
        #         continue
                
        #     # Compute the autocorrelation time so far
        #     # Using tol=0 means that we'll always get an estimate even
        #     # if it isn't trustworthy
        #     tau = sampler.get_autocorr_time(tol=0)
        #     autocorr[index] = np.mean(tau)
        #     index += 1
        #     print(tau)
        #     # Check convergence
        #     converged = np.all(tau * 100 < sampler.iteration)
        #     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        #     if converged:
        #         break
        #     old_tau = tau
                    
    i = 0

    while i < mcmc_params.n_steps:
        print('starting iteration %i out of %i of run %i' % (
            i + 1, mcmc_params.n_steps, runid),
              datetime.datetime.now() - start_time)
        sys.stdout.flush()
        for result in sampler.sample(pos, iterations=1, store=True):
            pos, _, _, blobs = result
            samples[i] = pos
            src.tools.write_state_to_file(pos, blobs, mcmc_chains_fp,
                                          blob_fp, mcmc_params, runid)
            i += 1
    #if mcmc_params.pool:
    pool.close()

    np.save(mcmc_params.samples_filename, samples)

    src.tools.write_log_file(mcmc_log_fp, samples_log_fp,
                             start_time, samples, mcmc_src,
                             n_pool)
