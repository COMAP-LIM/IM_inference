import numpy as np
import numpy.fft as fft
import scipy
from scipy import signal
import sys
import copyreg
import copy
import datetime
import os
import socket
import errno
import shutil
import src.MapObj
import src.Model
import src.Observable


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
    if (mcmc_params.model == 'simp_Li'):
        model = src.Model.Simplified_Li(exp_params, map_obj)

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


def get_data(mcmc_params, exp_params, model, observables, map_obj):
    maps = np.zeros((
        mcmc_params.n_patches,
        map_obj.n_x,
        map_obj.n_y,
        map_obj.n_z
    ))
    if not mcmc_params.generate_file:
        print('opening map data file')
        maps = np.load(mcmc_params.map_filename)
    else:
        model_params = mcmc_params.model_params_true[model.label]
        for i in range(mcmc_params.n_patches):
            if exp_params.map_smoothing:
                maps[i] = create_smoothed_map(
                    model, model_params
                ) + map_obj.generate_noise_map()
            else:
                maps[i] = model.generate_map(
                    model_params) + map_obj.generate_noise_map()
    if mcmc_params.save_file:
        print('saving map to file')
        np.save(mcmc_params.map_filename, maps)

    data = dict()
    for i in range(mcmc_params.n_patches):
        map_obj.map = maps[i]
        map_obj.calculate_observables(observables)

        for observable in observables:
            if i == 0:
                data[observable.label] = (
                    observable.values / mcmc_params.n_patches
                )
            else:
                data[observable.label] += (
                    observable.values / mcmc_params.n_patches
                )

            if 0 in observable.values:
                print('some of data values are equal to 0')

    insert_data(data, observables)
    return data


def calculate_power_spec_3d(map_obj, k_bin=None):

    kx = np.fft.fftfreq(map_obj.n_x, d=map_obj.dx) * 2 * np.pi
    ky = np.fft.fftfreq(map_obj.n_y, d=map_obj.dy) * 2 * np.pi
    kz = np.fft.fftfreq(map_obj.n_z, d=map_obj.dz) * 2 * np.pi
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, kz,
                                                    indexing='ij')))

    if k_bin is None:
        dk = max(np.diff(kx)[0], np.diff(ky)[0], np.diff(kz)[0])
        kmax_dk = int(np.ceil(max(np.amax(kx), np.amax(ky), np.amax(kz)) / dk))
        k_bin = np.linspace(0, kmax_dk, kmax_dk + 1)

    fft_map = fft.fftn(map_obj.map) / (map_obj.n_x * map_obj.n_y * map_obj.n_z)

    ps = np.abs(fft_map)**2 * map_obj.volume

    Pk_modes = np.histogram(
        kgrid[kgrid > 0], bins=k_bin, weights=ps[kgrid > 0])[0]
    nmodes, k_binedges = np.histogram(kgrid[kgrid > 0], bins=k_bin)

    Pk = Pk_modes
    Pk[np.where(nmodes > 0)] = Pk_modes[np.where(
        nmodes > 0)] / nmodes[np.where(nmodes > 0)]
    k_bincents = (k_binedges[1:] + k_binedges[:-1]) / 2.

    return Pk, k_bincents, nmodes / 2.0


def calculate_vid(map_obj, T_bin):
    T_array = (T_bin[1:] + T_bin[:-1]) / 2.
    if not np.isfinite(map_obj.map.flatten()).all():
        B_val = np.zeros_like(T_array) + np.nan
    else:
        B_val, _ = np.histogram(map_obj.map.flatten(), bins=T_bin)
        B_val = B_val.astype(float)
    return B_val, T_array


def degrade(largemap, factor, axes=[0, 1]):
    dims = np.array(largemap.shape)[axes]
    rows, cols = dims // factor
    if any(dims % factor > 0):
        print("Invalid degration factor!")
        sys.exit()
    if len(largemap.shape) == 3:
        return largemap.reshape(
            rows, largemap.shape[0] // rows,
            cols, largemap.shape[1] // cols,
            largemap.shape[2]).mean(axis=1).mean(axis=2)
    else:
        return largemap.reshape(
            rows, largemap.shape[0] // rows,
            cols, largemap.shape[1] // cols).mean(axis=1).mean(axis=2)


def gaussian_kernel(sigma_x, sigma_y, n_sigma=5.0):
    size_y = int(n_sigma * sigma_y)
    size_x = int(n_sigma * sigma_x)
    y, x = scipy.mgrid[-size_y:size_y + 1, -size_x:size_x + 1]
    g = np.exp(-(x**2 / (2. * sigma_x**2) + y**2 / (2. * sigma_y**2)))
    return g / g.sum()


def gaussian_smooth(mymap, sigma_x, sigma_y, n_sigma=5.0):
    kernel = gaussian_kernel(sigma_y, sigma_x, n_sigma=n_sigma)
    smoothed_map = signal.fftconvolve(mymap, kernel[:, :, None], mode='same')
    return smoothed_map


def create_smoothed_map(model, model_params):
    exp_params = model.exp_params
    factor = exp_params.resolution_factor

    # make high-resolution grid to make map
    model.map_obj.Ompix /= factor * factor

    model.map_obj.pix_binedges_x = np.linspace(
        model.map_obj.pix_binedges_x[0], model.map_obj.pix_binedges_x[-1],
        factor * (len(model.map_obj.pix_binedges_x) - 1) + 1)

    model.map_obj.pix_binedges_y = np.linspace(
        model.map_obj.pix_binedges_y[0], model.map_obj.pix_binedges_y[-1],
        factor * (len(model.map_obj.pix_binedges_y) - 1) + 1)
    # make high-resolution map
    model.map_obj.map = model.generate_map(model_params)

    pixwidth = (
        model.map_obj.pix_binedges_x[1] - model.map_obj.pix_binedges_x[0]
    ) * 60

    sigma = exp_params.FWHM / pixwidth / np.sqrt(8 * np.log(2))
    # convolve map with gaussian beam
    model.map_obj.map = gaussian_smooth(model.map_obj.map, sigma, sigma)

    # plt.figure()
    # plt.imshow(filteredmap[:, :, 0], interpolation='none')
    # plt.show()
    
    # change grid to low resolution again
    model.map_obj.Ompix *= factor * factor

    model.map_obj.pix_binedges_x = np.linspace(
        model.map_obj.pix_binedges_x[0], model.map_obj.pix_binedges_x[-1],
        (len(model.map_obj.pix_binedges_x) - 1) / factor + 1)

    model.map_obj.pix_binedges_y = np.linspace(
        model.map_obj.pix_binedges_y[0], model.map_obj.pix_binedges_y[-1],
        (len(model.map_obj.pix_binedges_y) - 1) / factor + 1)
    
    # degrade map back to low-res grid
    model.map_obj.map = degrade(model.map_obj.map, factor)
    return model.map_obj.map


# From Tony Li

def ensure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def set_up_log(output_dir, mcmc_params, n_pool):
    ensure_dir_exists(output_dir + '/params')
    ensure_dir_exists(output_dir + '/chains')
    ensure_dir_exists(output_dir + '/log_files')
    ensure_dir_exists(output_dir + '/samples')
    ensure_dir_exists(output_dir + '/blob')
    runid = 0
    while os.path.isfile(os.path.join(
            output_dir, 'params',
            'mcmc_params_run{0:d}.py'.format(runid))):
        runid += 1
    print('Current run_id: %i' % runid)

    if os.path.isfile(mcmc_params.run_log_file):
        with open(mcmc_params.run_log_file, 'r') as run_file:
            save = run_file.read()
            run_file.close()
    else:
        save = ''
    with open(mcmc_params.run_log_file, 'w') as run_file:
        run_file.write('starting run: %i on %s with %i procs, %s \n' % (
            runid, socket.gethostname(), n_pool, datetime.datetime.now()
        ))
        run_file.close()
    with open(mcmc_params.run_log_file, 'a') as run_file:
        run_file.write(save)
        run_file.close()
    
    mcmc_params_fp = os.path.join(output_dir, 'params',
                                  'mcmc_params_run{0:d}.py'.format(runid))
    exp_params_fp = os.path.join(output_dir, 'params',
                                 'experiment_params_run{0:d}.py'.format(runid))
    if len(sys.argv) < 2:
        shutil.copy2('mcmc_params.py', mcmc_params_fp)
        shutil.copy2('experiment_params.py', exp_params_fp)
    else:
        shutil.copy2(sys.argv[1], mcmc_params_fp)
        shutil.copy2(sys.argv[2], exp_params_fp)

    mcmc_chains_fp = os.path.join(output_dir, 'chains',
                                  'mcmc_chains_run{0:d}.dat'.format(runid))
    mcmc_log_fp = os.path.join(output_dir, 'log_files',
                               'mcmc_log_run{0:d}.txt'.format(runid))
    samples_log_fp = os.path.join(output_dir, 'samples',
                                  'samples_log_run{0:d}.npy'.format(runid))
    blob_fp = os.path.join(output_dir, 'blob', 'blob_')

    return mcmc_chains_fp, mcmc_log_fp, samples_log_fp, blob_fp, runid


def write_state_to_file(
        pos, blobs, mcmc_chains_fp,
        blob_fp, mcmc_params, runid):
    with open(mcmc_chains_fp, 'a') as chains_file:
        for j in range(mcmc_params.n_walkers):
            chains_file.write('{0:4d} {1:s}\n'.format(
                j, ' '.join([str(p) for p in pos[j]])))
    for obs in mcmc_params.observables:
        with open(blob_fp + obs + '_' + str(runid) + '.dat', 'a') as blob_file:
            for j in range(mcmc_params.n_walkers):
                for o in blobs[j]:
                    if o.label == obs:
                        obs_data = o.mean
                        break
                blob_file.write('{0:4d} {1:s}\n'.format(
                    j, ' '.join([str(d) for d in obs_data])))
    sys.stdout.flush()


def write_log_file(mcmc_log_fp, samples_log_fp, start_time, samples, n_pool):
    if len(sys.argv) < 2:
        mcmc_file = 'mcmc_params.py'
    else:
        mcmc_file = sys.argv[1] 
    with open(mcmc_log_fp, 'w') as log_file, \
            open(mcmc_file, 'r') as param_file:
        log_file.write('Time start of run     : %s \n' % (start_time))
        log_file.write('Time end of run       : %s \n' %
                       (datetime.datetime.now()))
        tot_time = (datetime.datetime.now() - start_time)
        log_file.write('Total execution time  : %s \n' %
                       tot_time)

        np.save(samples_log_fp, samples)
        n_steps, n_walkers, n_params = samples.shape
        samples = samples.reshape(n_steps * n_walkers, n_params)
        n_par = len(samples[0, :])
        percentiles = [15.87, 68.27 + 15.87]  # fix at some point
        n_cut = samples.shape[0] // 2
        log_file.write('\nPosterior parameter constraints: \n')
        for i in range(n_par):
            median = np.median(samples[n_cut:, i])
            constraints = np.percentile(
                samples[n_cut:, i], percentiles) - median
            log_file.write('Parameter %i: %.3f +%.3f %.3f \n' % (
                i, median, constraints[1], constraints[0]
            ))
        log_file.write('Run on %s with %i processes' % (
            socket.gethostname(), n_pool
        ))
        log_file.write('\nmcmc_params.py: \n' + param_file.read())

        print('Execution time:', datetime.datetime.now() - start_time)


class empty_table():
    """
    simple Class creating an empty table
    used for halo catalogue and map instances
    """

    def __init__(self):
        pass

    def copy(self):
        """@brief Creates a copy of the table."""
        return copy.copy(self)


def load_peakpatch_catalogue(filein):
    """
    Load peak patch halo catalogue into halos
    class and cosmology into cosmo class

    Returns
    -------
    halos : class
        Contains all halo information (position, redshift, etc..)
    """
    # creates empty class to put any halo info into
    halos = empty_table()
    # creates empty class to put any cosmology info into
    cosmo = empty_table()

    halo_info = np.load(filein)

    # get cosmology from halo catalogue
    params_dict = halo_info['cosmo_header'][()]
    cosmo.Omega_M = params_dict.get('Omega_M')
    cosmo.Omega_B = params_dict.get('Omega_B')
    cosmo.Omega_L = params_dict.get('Omega_L')
    cosmo.h = params_dict.get('h')
    cosmo.ns = params_dict.get('ns')
    cosmo.sigma8 = params_dict.get('sigma8')

    # if the halo catalogue is not centered along the z axis
    cen_x_fov = params_dict.get('cen_x_fov', 0.)
    # if the halo catalogue is not centered along the z axis
    cen_y_fov = params_dict.get('cen_y_fov', 0.)

    halos.M = halo_info['M']     # halo mass in Msun
    halos.x_pos = halo_info['x']     # halo x position in comoving Mpc
    halos.y_pos = halo_info['y']     # halo y position in comoving Mpc
    halos.z_pos = halo_info['z']     # halo z position in comoving Mpc
    halos.vx = halo_info['vx']    # halo x velocity in km/s
    halos.vy = halo_info['vy']    # halo y velocity in km/s
    halos.vz = halo_info['vz']    # halo z velocity in km/s
    halos.redshift = halo_info['zhalo']  # observed redshift incl velocities
    halos.zformation = halo_info['zform']  # formation redshift of halo

    halos.nhalo = len(halos.M)

    halos.chi = np.sqrt(halos.x_pos**2 + halos.y_pos**2 + halos.z_pos**2)
    halos.ra = np.arctan2(-halos.x_pos, halos.z_pos) * 180. / np.pi - cen_x_fov
    halos.dec = np.arcsin(halos.y_pos / halos.chi) * 180. / np.pi - cen_y_fov

    assert np.max(halos.M) < 1.e17, "Halos seem too massive"

    return halos, cosmo


def cull_peakpatch_catalogue(halos, min_mass, map_obj):
    """
    crops the halo catalogue to only include desired halos
    """
    dm = [(halos.M > min_mass) * (halos.redshift >= map_obj.z_i)
                               * (np.abs(halos.ra) <= map_obj.fov_x / 2)
                               * (np.abs(halos.dec) <= map_obj.fov_y / 2)
                               * (halos.redshift <= map_obj.z_f)]

    for i in dir(halos):
        if i[0] == '_':
            continue
        try:
            setattr(halos, i, getattr(halos, i)[dm])
        except TypeError:
            pass
    halos.nhalo = len(halos.M)

    return halos


def reduce_mod(m):
    assert sys.modules[m.__name__] is m
    return rebuild_mod, (m.__name__,)


def rebuild_mod(name):
    __import__(name)
    return sys.modules[name]


def make_picklable(exp_params, mcmc_params):
    # make parameter files pickable
    copyreg.pickle(type(exp_params), reduce_mod)
    copyreg.pickle(type(mcmc_params), reduce_mod)
