import numpy as np
import matplotlib.pyplot as plt
import h5py

import src.MapObj
import src.tools
import src.Model
import src.Observable
import src.likelihoods

import sys
import os
import datetime
import importlib
import shutil
import inspect
from mpi4py import MPI

if len(sys.argv) < 2:
    import experiment_params as exp_params
else:
    exp_params = importlib.import_module(sys.argv[2][:-3])

exp_src = inspect.getsource(exp_params)
comm = MPI.COMM_WORLD
my_rank, size = (comm.Get_rank(), comm.Get_size())

if my_rank == 0:
    src.tools.ensure_dir_exists(exp_params.cov_output_dir)
    src.tools.ensure_dir_exists(os.path.join(exp_params.cov_output_dir,
                                             'data'))
    src.tools.ensure_dir_exists(os.path.join(exp_params.cov_output_dir,
                                             'param'))
    cov_id = 0
    while os.path.isfile(os.path.join(
            exp_params.cov_output_dir, 'param',
            'experiment_params_id{0:d}.py'.format(cov_id))):
        cov_id += 1
    exp_params_fp = os.path.join(
        exp_params.cov_output_dir, 'param',
        'experiment_params_id{0:d}.py'.format(cov_id))
    data_fp = os.path.join(
        exp_params.cov_output_dir, 'data',
        'data_id{0:d}.h5'.format(cov_id))
    with open(exp_params_fp, 'w') as exp_file:
        exp_file.write(exp_src)

src.tools.make_picklable((exp_params,))

model, observables, extra_observables, small_map, full_map = \
    src.tools.set_up_cov(exp_params)


def get_data(exp_params, model, observables,
             extra_observables, small_map,
             full_map, halos):
    model_params = exp_params.model_params_cov[exp_params.cov_model]

    if exp_params.map_smoothing:
        if exp_params.FWHM_nu is None:
            full_map.map, full_map.lum_func = src.tools.create_smoothed_map(
                model, model_params, halos=halos
            )
        else:
            full_map.map, full_map.lum_func = src.tools.create_smoothed_map_3d(
                model, model_params, halos=halos
            )
    else:
        full_map.map, full_map.lum_func = model.generate_map(
            model_params, halos=halos)
    full_map.map += full_map.generate_noise_map()

    n_maps = full_map.n_maps_x * full_map.n_maps_y
    n_T = len(exp_params.vid_Tbins) - 1
    n_k = len(exp_params.ps_kbins) - 1
    n_data = n_T + n_k
    B_i = np.zeros((n_T, n_maps))
    ps = np.zeros((n_k, n_maps))
    data = np.zeros((n_data, n_maps))
    for i in range(full_map.n_maps_x):
        for j in range(full_map.n_maps_y):
            index = full_map.n_maps_x * i + j
            small_map.map = full_map.map[
                i * small_map.n_pix_x:(i + 1) * small_map.n_pix_x,
                j * small_map.n_pix_y:(j + 1) * small_map.n_pix_y]
            small_map.map -= small_map.map.mean()

            B_i[:, index], _ = \
                src.tools.calculate_vid(small_map, exp_params.vid_Tbins)
            ps[:, index], _, _ = \
                src.tools.calculate_power_spec_3d(small_map,
                                                  exp_params.ps_kbins)

    data[:n_k] = ps
    data[n_k:n_data] = B_i
    return data, full_map.lum_func

halo_fp_list = os.listdir(exp_params.cov_catalogue_folder)

n_catalogues = len(halo_fp_list)

my_indices = src.tools.distribute_indices(n_catalogues, size, my_rank)
n_catalogues_local = len(my_indices)
n_T = len(exp_params.vid_Tbins) - 1
n_k = len(exp_params.ps_kbins) - 1
n_data = n_T + n_k
data = np.zeros((n_data, full_map.n_maps_x * full_map.n_maps_y,
                 n_catalogues_local))
lum_hist = np.zeros((len(exp_params.lumfunc_bins) - 1,
                     n_catalogues_local))
model.all_halos = []
print(n_catalogues_local)
for i in range(n_catalogues_local):
    halo_fp = os.path.join(exp_params.cov_catalogue_folder,
                           halo_fp_list[my_indices[i]])
    print(halo_fp)
    halos, _ = src.tools.load_peakpatch_catalogue(halo_fp)
    halos = src.tools.cull_peakpatch_catalogue(
        halos, exp_params.min_mass, full_map)
    data[:, :, i], lum_hist[:, i] = get_data(
        exp_params, model, observables,
        extra_observables, small_map,
        full_map, halos)

gathered_data = comm.gather(data, root=0)
gathered_lum_hist = comm.gather(lum_hist, root=0)

if my_rank == 0:
    all_data = np.zeros((
        n_data,
        full_map.n_maps_x * full_map.n_maps_y,
        n_catalogues))

    all_lum_hist = np.zeros((len(exp_params.luminosity),
                             n_catalogues))
    for i in range(size):
        index = src.tools.distribute_indices(n_catalogues, size, i)
        all_data[:, :, index] = gathered_data[i]
        all_lum_hist[:, index] = gathered_lum_hist[i]
    all_data = all_data.reshape((
        n_data, full_map.n_maps_x * full_map.n_maps_y * n_catalogues))

    lum_hist_avg = all_lum_hist.mean(1)
    cov = np.cov(all_data)
    data_avg = all_data.mean(1)
    var_indep = np.zeros_like(data_avg)

    _, k, n_modes = src.tools.calculate_power_spec_3d(small_map,
                                                      exp_params.ps_kbins)
    var_indep[:n_k] = data_avg[:n_k] ** 2 / n_modes
    var_indep[n_k:n_data] = data_avg[n_k:n_data]
    observable_order = np.array(['ps', 'vid'])
    hf = h5py.File(data_fp, 'w')
    hf.create_dataset('lum_func_avg', data=lum_hist_avg)
    hf.create_dataset('n_modes', data=n_modes)
    hf.create_dataset('all_data', data=all_data)
    hf.create_dataset('cov_mat', data=cov)
    # hf.create_dataset('obs_order', data=observable_order)
    hf.create_dataset('var_indep_0', data=var_indep)
    hf['obs_order'] = observable_order.astype('S')
    hf.close()

    cov_divisor = np.sqrt(np.outer(var_indep, var_indep))
    plt.imshow(cov / cov_divisor, interpolation='none', vmax=1.2)
    plt.colorbar()

    plt.figure()
    plt.loglog(exp_params.luminosity,
               exp_params.luminosity * lum_hist_avg)
    plt.axis([1e4, 1e7, 1e-6, 1e-1])
    plt.show()
