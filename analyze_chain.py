import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
# import mcmc_params
import h5py
import importlib

from chainconsumer import ChainConsumer

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
parameters = [r'$\log A$', r'$\sigma_z$'] #mcmc_params.labels[model]
print(parameters)
truth = mcmc_params.model_params_true[model]
c = ChainConsumer()
c.add_chain(data, walkers=n_walkers, parameters=parameters, name=model).configure(statistics="mean")
fig = c.plotter.plot(figsize="page", truth=truth)

gelman_rubin_converged = c.diagnostic.gelman_rubin()
# And also using the Geweke metric
geweke_converged = c.diagnostic.geweke()

fig = c.plotter.plot_walks(convolve=50)

plt.show()