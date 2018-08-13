import numpy as np
from scipy.stats import norm
import src.tools
import sys
import scipy as sp
import os
import scipy.interpolate


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
            return src.tools.gaussian_smooth(mymap, self.exp_params.sigma_x,
                                             self.exp_params.sigma_y,
                                             n_sigma=5.0)
        return mymap

    def mcmc_walker_initial_positions(self, prior_params, n_walkers):
        p_par = np.transpose(prior_params)
        mean, sigma = p_par[0], p_par[1]
        return mean + sigma * np.random.randn(n_walkers, len(mean))

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
        return A * k**alpha

    def generate_map(self, model_params):
        n_x = self.map_obj.n_x
        n_y = self.map_obj.n_y
        n_z = self.map_obj.n_z

        kx = np.fft.fftfreq(n_x, d=self.map_obj.dx) * 2*np.pi
        ky = np.fft.fftfreq(n_y, d=self.map_obj.dy) * 2*np.pi
        kz = np.fft.fftfreq(n_z, d=self.map_obj.dz) * 2*np.pi
        kgrid = np.sqrt(
            sum(ki**2 for ki in np.meshgrid(kx, ky, kz, indexing='ij')))

        f_k_real = (np.random.randn(n_x, n_y, n_z)
                    * np.sqrt(self.power_spect_pl(kgrid, model_params)
                    / self.map_obj.volume))
        f_k = f_k_real + 1j * f_k_real
        if_k = np.fft.ifftn(f_k) * (n_x * n_y * n_z)
        # print(self.power_spect_pl(kgrid,model_params)/volume)

        if self.exp_params.map_smoothing:
            return src.tools.gaussian_smooth(
                            if_k.real, self.exp_params.sigma_x,
                            self.exp_params.sigma_y, n_sigma=5.0)

        if np.any(np.isnan(if_k.real)):
            print('hello')
            sys.exit()

        return if_k.real

    def mcmc_walker_initial_positions(self, prior_params, n_walkers):
        p_par = np.transpose(prior_params)
        mean, sigma = p_par[0], p_par[1]
        return mean + sigma * np.random.randn(n_walkers, len(mean))

    def ln_prior(self, model_params, prior_params):
        ln_prior = 0.0
        if ((model_params[0] < 0.0) or (model_params[1] < 0.0)):
            return - np.infty
        for m_par, p_par in zip(model_params, prior_params):
            ln_prior += norm.logpdf(m_par,
                                    loc=p_par[0],
                                    scale=p_par[1])
        return ln_prior


class Mhalo_to_Lco(Model):
    def __init__(self, exp_params, halos, map_obj):
        self.exp_params = exp_params
        self.halos = halos
        self.map_obj = map_obj


    def T_line(self):  # map, halos
        """
        The line Temperature in Rayleigh-Jeans limit
        T_line = c^2/2/kb/nuobs^2 * I_line

         where the Intensity I_line = L_line/4/pi/D_L^2/dnu
            D_L = D_p*(1+z), I_line units of L_sun/Mpc^2/Hz

         T_line units of [L_sun/Mpc^2/GHz] * [(km/s)^2 / (J/K) / (GHz)^2]*1/sr
            = [ 3.48e26 W/Mpc^2/GHz ] * [ 6.50966e21 s^2/K/kg ]
            = 2.63083e-6 K = 2.63083 muK
        """
        halos = self.halos
        map_obj = self.map_obj
        convfac = 2.63083

        Tco = 1./2 * convfac/halos.nu**2 * halos.Lco / 4/np.pi / \
            halos.chi**2/(1 + halos.redshift)**2/map_obj.dnu/map_obj.Ompix

        return Tco

    def generate_map(self, model_params):  # Lco_to_map
        # generate map
        # Calculate line freq from redshift
        map_obj = self.map_obj
        halos = self.halos

        halos.nu = map_obj.nu_rest / (halos.redshift + 1)
        halos.Lco = self.calculate_Lco(model_params)
        # Transform from Luminosity to Temperature
        halos.Tco = self.T_line()

        # flip frequency bins because np.histogram needs increasing bins
        bins3D = [map_obj.pix_binedges_x,
                  map_obj.pix_binedges_y, map_obj.nu_binedges[::-1]]

        # bin in RA, DEC, NU_obs
        maps, edges = np.histogramdd(np.c_[halos.ra, halos.dec, halos.nu],
                                     bins=bins3D,
                                     weights=halos.Tco)

        # flip back frequency bins
        if self.exp_params.map_smoothing:
            return src.tools.gaussian_smooth(
                            maps[:, :, ::-1], map_obj.sigma_x,
                            map_obj.sigma_y, n_sigma=5.0)

        return maps[:, :, ::-1]


class Mhalo_to_Lco_Pullen(Mhalo_to_Lco):

    def __init__(self, exp_params, halos, map_obj):
        self.label = 'Lco_Pullen'
        self.n_params = 1
        super().__init__(exp_params, halos, map_obj)

    def mcmc_walker_initial_positions(self, prior_params, n_walkers):
        p_par = np.transpose(prior_params)
        mean, sigma = p_par[0], p_par[1]
        initial_pos = mean + sigma * np.random.randn(n_walkers, len(mean))
        return initial_pos

    def ln_prior(self, model_params, prior_params):
        ln_prior = 0.0
        for m_par, p_par in zip(model_params, prior_params):
            ln_prior += norm.logpdf(m_par,
                                    loc=p_par[0],
                                    scale=p_par[1])
        return ln_prior

    def calculate_Lco(self, model_param=None):  # halos, coeffs
        """
        Luminosity in units of L_sun, halos mass in units of M_sun
        Coefficients in units of L_sun/M_sun
        Default coefficients taken from Pullen et al. 2013
        """
        model_param = model_param[0]
        if model_param is None:
            model_param = np.log10(1e6/5e11)

        Lco = 10**model_param * self.halos.M
        return Lco


sfr_interp_tab = None


class Mhalo_to_Lco_Li(Mhalo_to_Lco):

    def __init__(self, exp_params, halos, map_obj):
        self.label = 'Lco_Li'
        self.n_params = 5
        super().__init__(exp_params, halos, map_obj)

    def mcmc_walker_initial_positions(self, prior_params, n_walkers):
        p_par = np.transpose(prior_params)
        mean, sigma = p_par[0], p_par[1]
        initial_pos = mean + sigma * np.random.randn(n_walkers, len(mean))
        initial_pos[:, 3:] = np.abs(initial_pos[:, 3:])
        return initial_pos

    def ln_prior(self, model_params, prior_params):
        ln_prior = 0.0
        if (model_params[3] < 0) or (model_params[4] < 0):
            return - np.infty
        for m_par, p_par in zip(model_params, prior_params):
            print('par', p_par[0], p_par[1])
            ln_prior += norm.logpdf(m_par,
                                    loc=p_par[0],
                                    scale=p_par[1])
        return ln_prior

    def calculate_Lco(self, model_params=None):
        global sfr_interp_tab
        halos = self.halos

        if model_params is None:
            # Power law parameters from paper
            log_delta_mf, alpha, beta, sigma_sfr, sigma_lco = [
                0.0, 1.37, -1.74, 0.3, 0.3]
        else:
            log_delta_mf, alpha, beta, sigma_sfr, sigma_lco = model_params
        delta_mf = 10**log_delta_mf

        # Get Star formation rate
        if sfr_interp_tab is None:
            sfr_interp_tab = self.get_sfr_table()
        sfr = sfr_interp_tab.ev(np.log10(halos.M), np.log10(halos.redshift+1))
        sfr = self.add_log_normal_scatter(sfr, sigma_sfr)
        # infrared luminosity
        lir = sfr * 1e10 / delta_mf
        alphainv = 1./alpha
        # Lco' (observers units
        #overflow in either of these
        lir_ =lir**alphainv
        beta_ = 10**(-beta * alphainv)

        Lcop = lir_ * beta_
        Lco = 4.9e-5 * Lcop
        Lco = self.add_log_normal_scatter(Lco, sigma_lco)

        return Lco

    @staticmethod
    def get_sfr_table():
        """
        Load SFR Table
        Columns are: z+1, logmass, logsfr, logstellarmass
        Intermediate processing of tabulated data
        """

        tablepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        tablepath += '/tables/sfr_behroozi_release.dat'
        dat_zp1, dat_logm, dat_logsfr, _ = np.loadtxt(tablepath, unpack=True)

        dat_logzp1 = np.log10(dat_zp1)
        dat_sfr = 10.**dat_logsfr

        # Reshape arrays
        dat_logzp1 = np.unique(dat_logzp1)    # log(z), 1D
        dat_logm = np.unique(dat_logm)    # log(Mhalo), 1D
        dat_sfr = np.reshape(dat_sfr, (dat_logm.size, dat_logzp1.size))

        # Get interpolated SFR value(s)
        sfr_interp_tab = sp.interpolate.RectBivariateSpline(dat_logm, dat_logzp1, dat_sfr,
                                                            kx=1, ky=1)
        return sfr_interp_tab

    @staticmethod
    def add_log_normal_scatter(data, dex):
        """
        Return array x, randomly scattered by a log-normal distribution with sigma=dexscatter.
        [via @tonyyli - https://github.com/dongwooc/imapper2]
        Note: scatter maintains mean in linear space (not log space).
        """
        if (dex <= 0):
            return data
        # Calculate random scalings
        sigma = dex * 2.302585  # Stdev in log space (DIFFERENT from stdev in linear space), note: ln(10)=2.302585
        mu = -0.5*sigma**2

        randscaling = np.random.lognormal(mu, sigma, data.shape)
        xscattered = np.where(data > 0, data*randscaling, data)
        return xscattered
