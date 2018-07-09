import numpy as np
from scipy.stats import norm
import src.MapObj
import src.tools
import sys

class Model:
    """
    Parent class for Intensity mapping models. A model is here
    some procedure to simulate intensity maps. Each
    model has a set of model parameters that can be
    constrained from a given data-set using the mcmc-tool.
    """

    def __init__(self, exp_params):
        self.exp_params = exp_params

    def generate_map(self):
        pass

    def set_up(self):
        pass


class WhiteNoisePowerSpectrum(Model):
    """
    White noise (constant) power spectrum.
    """

    def __init__(self, exp_params):
        self.exp_params = exp_params
        self.n_params = 1
        self.label = 'wn_ps'

    def generate_map(self, model_params):
        sigma_T = model_params[0]
        n_x = len(self.exp_params.x) - 1
        n_y = len(self.exp_params.y) - 1
        n_z = len(self.exp_params.z) - 1
        mymap = np.random.randn(n_x, n_y, n_z) * sigma_T
        if self.exp_params.map_smoothing:
            return src.tools.gaussian_smooth(mymap, self.exp_params.sigma_x, self.exp_params.sigma_y, n_sigma=5.0)
        return mymap

    def mcmc_walker_initial_positions(self, prior_params, n_walkers):
        p_par = np.transpose(prior_params)
        mean, sigma = p_par[0], p_par[1]
        return mean + sigma*np.random.randn(n_walkers, len(mean))

    def ln_prior(self, model_params, prior_params):
        ln_prior = 0.0
        if (model_params[0] < 0.0):
            return - np.infty
        for m_par, p_par in zip(model_params, prior_params):
            ln_prior += norm.logpdf(m_par,
                                    loc=p_par[0],
                                    scale=p_par[1])
        return ln_prior

class PowerLawPowerSpectrum(Model):
    """
    Power law power spectrum P(k)=Ak^alpha.
    """

    def __init__(self, exp_params, map_obj):
        self.exp_params = exp_params
        self.map_obj = map_obj
        self.n_params = 2
        self.label = 'pl_ps'

    @staticmethod
    def power_spect_pl(k, model_params):
        A, alpha = model_params
        return A*k**alpha

    def generate_map(self, model_params):
        #A, alpha = model_param
        n_x = self.map_obj.n_x
        n_y = self.map_obj.n_y
        n_z = self.map_obj.n_z
        

        kx = np.fft.fftfreq(n_x, d = self.map_obj.dx)*2*np.pi
        ky = np.fft.fftfreq(n_y, d = self.map_obj.dy)*2*np.pi
        kz = np.fft.fftfreq(n_z, d = self.map_obj.dz)*2*np.pi
        kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, kz, indexing ='ij')))

        f_k_real= np.random.randn(n_x, n_y, n_z)*np.sqrt( self.power_spect_pl(kgrid,model_params)/self.map_obj.volume )        
        f_k=f_k_real + 1j*f_k_real
        if_k = np.fft.ifftn(f_k) * (n_x * n_y * n_z)
        #print(self.power_spect_pl(kgrid,model_params)/volume)
        if self.exp_params.map_smoothing:
            return src.tools.gaussian_smooth(if_k.real, self.exp_params.sigma_x, self.exp_params.sigma_y, n_sigma=5.0)
        if np.any(np.isnan(if_k.real)):
            print('hello')
            sys.exit()
        return if_k.real

    def mcmc_walker_initial_positions(self, prior_params, n_walkers):
        p_par = np.transpose(prior_params)
        mean, sigma = p_par[0], p_par[1] 
        return mean + sigma*np.random.randn(n_walkers, len(mean))
        

    def ln_prior(self, model_params, prior_params):
        ln_prior = 0.0
        if ((model_params[0] < 0.0) or (model_params[1]<0.0)):
            return - np.infty
        for m_par, p_par in zip(model_params, prior_params):
            ln_prior += norm.logpdf(m_par,
                                    loc=p_par[0],
                                    scale=p_par[1])
        return ln_prior
