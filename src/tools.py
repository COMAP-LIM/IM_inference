import numpy as np
import numpy.fft as fft
import scipy
from scipy import signal
import sys
# Calculates the angular average of any map.
def angular_average_3d(inmap, x, y, z, dr, x0=0, y0=0, z0=0):
    x_ind, y_ind, z_ind = np.indices(inmap.shape)

    r = np.sqrt((x[x_ind] - x0) ** 2 +
                (y[y_ind] - y0) ** 2 +
                (z[z_ind] - z0) ** 2)

    # np.hypot(x[x_ind] - x0, y[y_ind] - y0, z[z_ind] - z0)
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind] / dr
    map_sorted = inmap.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    delta_r = r_int[1:] - r_int[:-1]  # Assumes all dr intervals represented

    rind = np.where(delta_r)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(map_sorted, dtype=float)
    sum_rbin = csim[rind[1:]] - csim[rind[:-1]]

    return sum_rbin / nr, (r_int[rind[1:]] + 0.5) * dr, nr  # average value of
    # function in each radial bin of length dr


def calculate_power_spec_3d(map_obj, k_bin=None):

    # just something to get reasonable values for dk, not very good

    #dk = (np.sqrt(np.sqrt(map_obj.dx * map_obj.dy * map_obj.dz))
    #      / np.sqrt(map_obj.volume))

    kx = np.fft.fftfreq(map_obj.n_x, d=map_obj.dx)*2*np.pi
    ky = np.fft.fftfreq(map_obj.n_y, d=map_obj.dy)*2*np.pi
    kz = np.fft.fftfreq(map_obj.n_z, d=map_obj.dz)*2*np.pi
    kgrid = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx, ky, kz, indexing ='ij')))

    if k_bin is None:

        dk = max(np.diff(kx)[0], np.diff(ky)[0], np.diff(kz)[0])
        kmax_dk = int(np.ceil(max(np.amax(kx),np.amax(ky), np.amax(kz))/dk))
        k_bin = np.linspace(0, kmax_dk, kmax_dk+1)

    fft_map = fft.fftn(map_obj.map) / (map_obj.n_x * map_obj.n_y * map_obj.n_z)
    #fft_map = fft.fftshift(fft_map)
    ps = np.abs(fft_map) ** 2 * map_obj.volume
    Pk_modes = np.histogram(kgrid[kgrid>0], bins=k_bin, weights=ps[kgrid>0])[0]
    nmodes, k_edges = np.histogram(kgrid[kgrid>0], bins=k_bin)

    Pk = Pk_modes
    Pk[np.where(nmodes>0)] = Pk_modes[np.where(nmodes>0)]/nmodes[np.where(nmodes>0)]
    k_array = (k_edges[1:] + k_edges[:-1])/2.

    return Pk, k_array, nmodes#angular_average_3d(ps, map_obj.fx, map_obj.fy, map_obj.fz, dk)


def calculate_vid(map_obj, T_bin=None):

    if T_bin is None:
        Tx = np.fft.fftfreq(map_obj.n_x, d=map_obj.dx)*2*np.pi
        Ty = np.fft.fftfreq(map_obj.n_y, d=map_obj.dy)*2*np.pi
        dT = max(np.diff(Tx)[0], np.diff(Ty)[0])
        Tmax_dT = int(np.ceil(max(np.amax(Tx),np.amax(Ty), np.amax(Tz))/dT))
        T_bin = np.linspace(0, kmax_dk, kmax_dk+1)
    try:
        B_val, T_edges = np.histogram(map_obj.map.flatten(), bins=T_bin)
        T_array = (T_edges[1:] + T_edges[:-1])/2.
        return B_val, T_array
    except ValueError:
        print('wrong')
        sys.exit()
        #print(map_obj.map)


def gaussian_kernel(sigma_x, sigma_y, n_sigma=5.0):
     size_y = int(n_sigma * sigma_y)
     size_x = int(n_sigma * sigma_x)
     y, x = scipy.mgrid[-size_y:size_y + 1, -size_x:size_x + 1]
     g = np.exp(-(x ** 2 / (2. * sigma_x ** 2) + y ** 2 / (2. * sigma_y 
** 2)))
     return g / g.sum()


def gaussian_smooth(mymap, sigma_x, sigma_y, n_sigma=5.0):
     kernel = gaussian_kernel(sigma_y, sigma_x, n_sigma=n_sigma)
     smoothed_map = signal.fftconvolve(mymap, kernel[:, :, None], mode='same')
     return smoothed_map
