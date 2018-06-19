import numpy as np
import numpy.fft as fft


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


def calculate_power_spec_3d(map_obj):

    # just something to get reasonable values for dk, not very good

    dk = (np.sqrt(np.sqrt(map_obj.dx * map_obj.dy * map_obj.dz))
          / np.sqrt(map_obj.volume))

    fft_map = fft.fftn(map_obj.map) / (map_obj.n_x * map_obj.n_y * map_obj.n_z)
    fft_map = fft.fftshift(fft_map)
    ps = np.abs(fft_map) ** 2 * map_obj.volume
    return angular_average_3d(ps, map_obj.fx, map_obj.fy, map_obj.fz, dk)
