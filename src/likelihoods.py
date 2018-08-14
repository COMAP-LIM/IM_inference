import numpy as np


def ln_chi_squared(data, avg, sigma):
    return -1.0 / 2 * np.sum(((data - avg) / sigma)**2)
