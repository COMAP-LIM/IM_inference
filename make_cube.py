import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.optimize as opt
from scipy.stats import norm
import src.MapObj
import src.tools
import src.Model
import src.Observable

from astropy import units as u
import os
import sys
import importlib
import h5py
import time


import warnings

# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Make_Cube:
    def __init__(self):
        """Initialize class object"""
        self.initial_time = time.perf_counter()
        self.read_params()
        self.define_experimental_parameters()

    def read_params(self):
        """Method to read parameters from command line or parameter file.

        Raises:
            ValueError: If no output path is provided an error is raised
        """

        # NOTE: for now this is hardcoded, but ideally we want to integrate this into the global COMAP pipeline framework
        sys.path.append("/mn/stornext/d5/data/nilsoles/nils/pipeline")

        from l2gen_argparser import parser

        params = parser.parse_args()

        if not params.signal_path:
            raise ValueError("An output name for the signal cube must be provided.")

        self.params = params

    def define_experimental_parameters(self):
        """Method that defines the set of experimental parameters needed to generate a simulation cube."""
        if self.params.verbose:
            print(
                f"Defining experimental parameters: {time.perf_counter() - self.initial_time} sec"
            )

        # Define MCMC parameters and experimental parameters used to generate simulation cube
        self.mcmc_params = importlib.import_module("mc_cube")
        self.exp_params = importlib.import_module(self.params.exp_params[:-3])
        
        # Load model parameters from experimental parameters
        src.tools.make_picklable((self.exp_params, self.mcmc_params))

        self.model_params = self.exp_params.model_params_cov[self.params.model_name]
        self.mcmc_params.mcmc_model = self.params.model_name
        self.exp_params.cov_model = self.params.model_name
        print("MCMC model:", self.mcmc_params.mcmc_model)
        print("Experimental model:", self.exp_params.cov_model)
        
        self.mcmc_params.observables = ("ps", "vid")

        self.model, self.observables, _, self.map_obj = src.tools.set_up_mcmc(
            self.mcmc_params, self.exp_params
        )

        # # Optionally upgrade/downgrade from 4 times the COMAP standard geometry
        # self.exp_params.n_pix_x *= self.params.res_factor
        # self.exp_params.n_pix_y *= self.params.res_factor

    def make_sim_cube(self):
        """Method that makes simulation cube, given a model and a set of experimental parameters."""

        if self.params.verbose:
            print(
                f"Generating simulation cube: {time.perf_counter() - self.initial_time} sec"
            )

        # Setting seed for reproducibility sake
        if self.params.seed is None:
            self.seed = int(round(time.time()))
        else:
            self.seed = self.params.seed

        np.random.seed(self.seed)

        # Generate map object and luminocity function
        self.map_obj.map, self.map_obj.lum_func = src.tools.create_smoothed_map(
            self.model, self.model_params
        )
        

        map = self.map_obj.map

        self.map_x, self.map_y = (
            self.map_obj.pix_binedges_x,
            self.map_obj.pix_binedges_y,
        )
        
        self.map_x_centers, self.map_y_centers = (
            self.map_obj.pix_bincents_x,
            self.map_obj.pix_bincents_y,
        )
        
        self.map_nu_centers, self.map_nu_edges = (
            self.map_obj.nu_bincents,
            self.map_obj.nu_binedges,
        )
        
        print("Computing map power spectrum before beam smoothing is applied:")
        for observable in self.map_obj.observables:
            if isinstance(observable, src.Observable.Power_Spectrum):
                self.power_spectrum_unsmoothed, self.k = observable.values, observable.k


        simulation = map.reshape(map.shape[0], map.shape[1], 4, 1024)
        self.simulation = simulation.transpose(2, 3, 0, 1)
        
        
        unsmoothed_simulation = self.map_obj.map_unsmoothed.astype(np.float32)
        unsmoothed_simulation = unsmoothed_simulation.reshape(map.shape[0], map.shape[1], 4, 1024)
        self.unsmoothed_simulation = unsmoothed_simulation.transpose(2, 3, 0, 1)
        
        # self.map_obj.map[:50, :, :] = 0
        # self.map_obj.map[-50:, :, :] = 0
        # self.map_obj.map[:, :50, :] = 0
        # self.map_obj.map[:, -50:, :] = 0
        # print(self.map_obj.map[..., 0])
        
        ps_kbins = self.exp_params.ps_kbins
        # ps_kbins = np.logspace(-1.5, 0.3, 21)
        # ps_kbins = np.logspace(-2.0, np.log10(1.5), 21)
        # ps_kbins = np.logspace(-2.0, np.log10(1.5), 200)

        self.power_spectrum_smoothed, k, _ = src.tools.calculate_power_spec_3d(self.map_obj, ps_kbins)

        fig, ax = plt.subplots(figsize = (15, 5))
        ax.plot(k, k * self.power_spectrum_smoothed, label = "after")
        ax.plot(self.k, self.k * self.power_spectrum_unsmoothed, label = "before")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 1)
        ax.set_ylim(1, 3e4)
        ax.set_xlabel(r"$k$ [1/Mpc]")
        ax.set_ylabel(r"$kP(k)[\mathrm{\mu K^2 Mpc^2}]$")
        ax.legend()
        fig.savefig(f"test_cube_fig_{self.params.seed}_no_linewidth.png")
        
    def write(self):
        if self.params.verbose:
            print(
                f"Writing simulation cube to {self.params.signal_path}:\n{time.perf_counter() - self.initial_time} sec"
            )

        with h5py.File(self.params.signal_path, "w") as outfile:
            outfile.create_dataset("simulation", data=self.simulation.astype(np.float32))
            outfile.create_dataset("x_bin_edges", data=self.map_x)
            outfile.create_dataset("y_bin_edges", data=self.map_y)
            outfile.create_dataset("x_bin_centers", data=self.map_x_centers)
            outfile.create_dataset("y_bin_centers", data=self.map_y_centers)
            outfile.create_dataset("frequency_bin_edges", data=self.map_nu_edges)
            outfile.create_dataset("frequency_bin_centers", data=self.map_nu_centers)
            outfile.create_dataset("seed", data=self.seed)
            outfile.create_dataset("model_name", data = self.params.model_name)
            outfile.create_dataset("model_params", data = np.array(self.model_params))
            try:
                outfile.create_dataset("power_spectrum_smoothed", data=self.power_spectrum_smoothed)
                outfile.create_dataset("power_spectrum_unsmoothed", data=self.power_spectrum_unsmoothed)
                outfile.create_dataset("k", data=self.k)
                outfile.create_dataset("simulation_no_beamsmoothing", data=self.unsmoothed_simulation)
            except:
                return
        


if __name__ == "__main__":
    make_cube = Make_Cube()
    make_cube.make_sim_cube()
    make_cube.write()
