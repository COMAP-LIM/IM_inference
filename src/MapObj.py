import numpy as np
import numpy.fft as fft
import src.tools
import astropy.cosmology

class MapObj:
    """
    Class for the map-objects which carry the info about the maps and
    also the maps themselves.
    Should in the future also contain info about pixel size, FWHM,
    frequencies etc.
    """
    def __init__(self, exp_params):
        self.exp_params = exp_params
        cosmo = getattr(astropy.cosmology, exp_params.cosmology)
        # cosmo = FlatwCDM(name=exp_params.cosmology) # could use this?
        self.fov_x = exp_params.fov_x
        self.fov_y = exp_params.fov_y
        self.nu_i    = float(exp_params.nu_i)
        self.nu_f    = float(exp_params.nu_f)
        self.nu_rest = float(exp_params.nu_rest)
        self.n_nu_bins = exp_params.n_nu_bins
        self.n_pix_x = exp_params.n_pix_x
        self.n_pix_y = exp_params.n_pix_y

        self.z_i     = self.nu_rest/self.nu_i - 1
        self.z_f     = self.nu_rest/self.nu_f - 1
        z_array = np.linspace(self.z_i, self.z_f, self.n_nu_bins+1)


        # instrumental beam
        exp_params.sigma_x = exp_params.FWHM/60*np.pi/180. / np.sqrt(8 * np.log(2))  # for instrumental beam
        exp_params.sigma_y = exp_params.FWHM/60*np.pi/180. / np.sqrt(8 * np.log(2))
        self.sigma_x = exp_params.sigma_x
        self.sigma_y = exp_params.sigma_y

        self.pix_size_x = self.fov_x/self.n_pix_x
        self.pix_size_y = self.fov_y/self.n_pix_y

        self.Ompix = (self.pix_size_x*np.pi/180)*(self.pix_size_y*np.pi/180)

        self.pix_binedges_x = np.arange(-self.fov_x/2,
                                        self.fov_x/2 + self.pix_size_x,
                                        self.pix_size_x)
        self.pix_binedges_y = np.arange(-self.fov_y/2,
                                        self.fov_y/2 + self.pix_size_y,
                                        self.pix_size_y)

        self.pix_bincents_x = 0.5*(self.pix_binedges_x[:-1] + self.pix_binedges_x[:-1])
        self.pix_bincents_y = 0.5*(self.pix_binedges_y[:-1] + self.pix_binedges_y[:-1])

        # comoving distances
        self.z = cosmo.comoving_distance(z_array)
        self.x = self.pix_binedges_x*np.pi/180*cosmo.comoving_transverse_distance(np.mean(z_array))
        self.y = self.pix_binedges_y*np.pi/180*cosmo.comoving_transverse_distance(np.mean(z_array))

        self.n_x = len(self.x) - 1
        self.n_y = len(self.y) - 1
        self.n_z = len(self.z) - 1
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.volume = ((self.x[-1] - self.x[0])
                       * (self.y[-1] - self.y[0])
                       * (self.z[-1] - self.z[0])
                       )
        self.vox_vol = self.volume / (self.n_x * self.n_y * self.n_z)

        # map frequency dimension
        # negative steps as larger observed frequency means lower redshift
        self.dnu         = (self.nu_i - self.nu_f)/(self.n_nu_bins)
        self.nu_binedges = np.arange(self.nu_i, self.nu_f-self.dnu, -self.dnu)
        self.nu_bincents = self.nu_binedges[:-1] - self.dnu/2

        #self.fx = fft.fftshift(fft.fftfreq(self.n_x, self.dx))
        #self.fy = fft.fftshift(fft.fftfreq(self.n_y, self.dy))
        #self.fz = fft.fftshift(fft.fftfreq(self.n_z, self.dz))
        self.map = None

    def calculate_observables(self, Observables):
        for observable in Observables:
            observable.calculate_observable(self)

    def generate_noise_map(self):
        return self.exp_params.sigma_T*np.random.randn(self.n_x,
                                                       self.n_y, self.n_z)
