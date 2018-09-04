import numpy as np
import src.tools


class Observable:
    """
    Parent class for observables that can be calculated from
    intensity maps. Examples of observables can be power spectrum
    voxel intensity distribution etc.
    """

    def __init__(self):
        self.sum = None

    def calculate_observable(self, map):
        pass

    def independent_variance(self):
        pass

    def add_observable_to_sum(self):
        if self.sum is None:
            self.sum = self.values
        else:
            self.sum += self.values

    def calculate_mean(self, n):
        self.mean = self.sum / n
        self.sum = None
        self.independent_variance()


class Power_Spectrum(Observable):
    def __init__(self, mcmc_params):
        self.sum = None
        self.label = 'ps'
        self.mcmc_params = mcmc_params

    def calculate_observable(self, map_obj):
        self.values, self.k, self.n_modes = \
            src.tools.calculate_power_spec_3d(map_obj,
                                              self.mcmc_params.ps_kbins)

    def independent_variance(self):
        self.independent_var = self.mean**2 / self.n_modes


class Voxel_Intensity_Distribution(Observable):
    def __init__(self, mcmc_params):
        self.sum = None
        self.label = 'vid'
        self.mcmc_params = mcmc_params

    def calculate_observable(self, map_obj):
        self.values, self.dk = \
            src.tools.calculate_vid(map_obj, self.mcmc_params.vid_Tbins)

    def independent_variance(self):
        self.independent_var = self.mean  # *np.sqrt(1- self.mean/n_vox)
