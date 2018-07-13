import numpy as np
import numpy.fft as fft


class MapObj:
    """
    Class for the map-objects which carry the info about the maps and
    also the maps themselves.
    Should in the future also contain info about pixel size, FWHM,
    frequencies etc.
    """
    def __init__(self, exp_params):
        self.exp_params = exp_params
        self.x = exp_params.x
        self.y = exp_params.y
        self.z = exp_params.z
        self.fov_y = self.x[-1]#exp_params.fov_x
        self.fov_x = self.y[-1]#exp_params.fov_y
        self.nu_i    = float(exp_params.nu_i)
        self.nu_f    = float(exp_params.nu_f)
        self.nu_rest = float(exp_params.nu_rest)
        self.z_i     = self.nu_rest/self.nu_i - 1
        self.z_f     = self.nu_rest/self.nu_f - 1

        self.nmaps = exp_params.nmaps
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
        self.pix_size_x = (self.x[-1] - self.x[0])/self.n_x
        self.pix_size_y = (self.y[-1] - self.y[0])/self.n_y

        self.Ompix = (self.pix_size_x*np.pi/180)*(self.pix_size_y*np.pi/180)

        self.pix_binedges_x = np.arange(-self.fov_x/2,
                                        self.fov_x/2 + self.pix_size_x,
                                        self.pix_size_x)
        self.pix_binedges_y = np.arange(-self.fov_y/2,
                                        self.fov_y/2 + self.pix_size_y,
                                        self.pix_size_y)

        self.pix_bincents_x = 0.5*(self.pix_binedges_x[:-1] + self.pix_binedges_x[:-1])
        self.pix_bincents_y = 0.5*(self.pix_binedges_y[:-1] + self.pix_binedges_y[:-1])

        # map frequency dimension
        # negative steps as larger observed frequency means lower redshift
        self.dnu         = (self.nu_i - self.nu_f)/(self.nmaps)
        self.nu_binedges = np.arange(self.nu_i,self.nu_f-self.dnu,-self.dnu) 
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
