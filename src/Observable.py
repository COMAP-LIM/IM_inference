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
    def __init__(self):
        self.sum = None
        self.label = 'ps'

    def calculate_observable(self, map):
        self.values, self.dk, self.n_modes = \
            src.tools.calculate_power_spec_3d(map)

    def independent_variance(self):
        self.independent_var = self.mean / np.sqrt(self.n_modes)
