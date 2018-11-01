import numpy as np
# import matplotlib.pyplot as plt
import scipy
import src.MapObj
import src.tools
import src.Model
import src.Observable
import mcmc_params
import experiment_params as exp_params

sigma = exp_params.sigma_T

src.tools.make_picklable((exp_params, mcmc_params))
mcmc_params.observables = ('ps', 'vid')
model, observables, _, map_obj = src.tools.set_up_mcmc(
    mcmc_params, exp_params)

map_obj.map = map_obj.generate_noise_map()

for observable in observables:
    observable.calculate_observable(map_obj)


def test_ps():
    ps = observables[0].values
    k = observables[0].k
    observables[0].mean = k * 0 + map_obj.voxel_volume * sigma ** 2
    observables[0].independent_variance()
    df = len(k)
    chi2 = np.sum((ps - observables[0].mean) ** 2 / observables[0].independent_var)
    assert(scipy.stats.chi2.cdf(chi2, df=df) < 0.999)
    assert(scipy.stats.chi2.cdf(chi2, df=df) > 0.001)


def test_vid():
    vid = observables[1].values
    t = exp_params.vid_Tbins
    dt = np.diff(t)
    T = 0.5 * (t[1:] + t[:-1])
    assert((T == observables[1].T).all)
    B_noise = scipy.stats.norm.pdf(
        T, scale=exp_params.sigma_T) * dt * map_obj.n_vox
    observables[1].sum = T * 0 + B_noise
    observables[1].calculate_mean(n=1)
    observables[1].independent_variance()
    df = len(T)
    chi2 = np.sum((vid - observables[1].mean) ** 2 / observables[1].independent_var)
    assert(scipy.stats.chi2.cdf(chi2, df=df) < 0.999)
    assert(scipy.stats.chi2.cdf(chi2, df=df) > 0.001)
