import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
# import mcmc_params
import h5py
import importlib
import corner

from chainconsumer import ChainConsumer


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def insert_gaussian_priors_in_cornerplot_cov(cov, mu, color='k', fig=None):                                                                                 
    from matplotlib.patches import Ellipse                                                                                                      
    from scipy.stats import norm                                                                                                                

    sigma = np.sqrt(np.diag(cov))                                                      
    factor1 = np.sqrt(2.30) * 2  # relative width of 1 sigma ellipse, 0.6827                                                                    
    factor2 = np.sqrt(6.18) * 2  # relative width of 2 sigma ellipse, 0.9545                                                                    
    ndim = len(sigma)
    if fig is None:                                                           
        f, axes = plt.subplots(ndim, ndim, figsize=(10, 10))
    else: 
        f = fig
        axes = fig.subplots(ndim, ndim)                                                                     
    for i in range(ndim):                                                                                                                       
        x = np.linspace(-10 * sigma[i] + mu[i], 10 * sigma[i] + mu[i], 10000)                                                                   
        axes[i, i].plot(x, norm.pdf(x, scale=sigma[i], loc=mu[i]), color=color, linewidth=1)                                                   
        for j in range(i):
            cov_local = np.zeros((2, 2))
            cov_local[0, 0] = cov[j, j]
            cov_local[1, 1] = cov[i, i]
            cov_local[0, 1] = cov[j, i]
            cov_local[1, 0] = cov[i, j]
            vals, vecs = eigsorted(cov_local)
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = np.sqrt(vals)
            e = Ellipse(xy=[mu[j], mu[i]], width=factor1*w, height=factor1*h, angle=theta, fill=False, color=color, linewidth=1)              
            e2 = Ellipse(xy=[mu[j], mu[i]], width=factor2*w, height=factor2*h, angle=theta, fill=False, color=color, linewidth=1)
            axes[i, j].add_artist(e)                                                                                                            
            axes[i, j].add_artist(e2)
    return f, axes


output_dir = 'testing_output'

runid = int(sys.argv[1])

mcmc_params_fp = (
    output_dir + '.params'
    + '.mcmc_params_run{0:d}'.format(runid)
)
mcmc_params = importlib.import_module(mcmc_params_fp)

# filename_ps = os.path.join(output_dir, 'blob',
#                            'blob_ps_{0:d}.dat'.format(runid))

# with open(filename_ps) as my_file:
#     lines = [line.split() for line in my_file]

# ps = np.array(lines).astype(float)[:, 1:]

# filename_vid = os.path.join(output_dir, 'blob',
#                             'blob_vid_{0:d}.dat'.format(runid))

# with open(filename_vid) as my_file:
#     lines = [line.split() for line in my_file]

# vid = np.array(lines).astype(float)[:, 1:]

# filename_data = os.path.join(output_dir, 'blob',
#                              'data_run{0:d}.npy'.format(runid))
# data_ps = np.load(filename_data)[()]['ps']
# data_vid = np.load(filename_data)[()]['vid']

filename_samp = os.path.join(
    output_dir, 'chains', 
    'mcmc_chains_run{0:d}.dat'.format(runid)
)

with open(filename_samp) as my_file:
    lines = [line.split() for line in my_file]

    my_array = np.array(lines).astype(float)
    print(my_array.shape)

    n_walkers = int(my_array[:, 0].max() + 1)
    n_samples = int(len(my_array[:, 0]))
    n_steps = n_samples // n_walkers
    n_pos = int(len(my_array[0, :]) - 1)

    print("n_walkers = ", n_walkers)
    print("n_samples = ", n_samples)
    print("n_steps = ", n_steps)
    print("n_pos = ", n_pos)

    samples = my_array[:, 1:]

n_cut = n_walkers * 1

print(samples.shape)
data = samples[n_cut:]

model = mcmc_params.mcmc_model
# parameters = mcmc_params.labels[model]
parameters = mcmc_params.labels[model]
prior_params = mcmc_params.prior_params[model]
mu, cov = prior_params[0], np.array(prior_params[1])
sigma = np.sqrt(cov.diagonal())
print(parameters)
truth = mcmc_params.model_params_true[model]
c = ChainConsumer()
c.add_chain(data, walkers=n_walkers, parameters=parameters, name=model).configure(statistics="mean")
# fig = c.plotter.plot(figsize="page", truth=truth)

gelman_rubin_converged = c.diagnostic.gelman_rubin()
# And also using the Geweke metric
geweke_converged = c.diagnostic.geweke()

fig = c.plotter.plot_walks(convolve=50)

f, axes = insert_gaussian_priors_in_cornerplot_cov(cov, mu, 'r')


corner.corner(data, fig=f, show_titles=True, plot_datapoints=False,
              plot_density=False, levels=(0.6827, 0.9545), hist_kwargs={"normed":'True'})

factor3 = 5

for i in range(len(mu)):                                                                                                                                     
    axes[i, i].set_xlim(-factor3 * sigma[i] + mu[i], factor3 * sigma[i] + mu[i])                                                                             
    for j in range(i):                                                                                                                                       
        axes[i, j].set_xlim(-factor3 * sigma[j] + mu[j], factor3 * sigma[j] + mu[j])                                                                         
        axes[i, j].set_ylim(-factor3 * sigma[i] + mu[i], factor3 * sigma[i] + mu[i])

plt.show()