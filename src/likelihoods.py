import numpy as np


def ln_chi_squared(data, avg, variance):
    loglike = - 0.5 * np.sum((data - avg)**2 / variance + np.log(variance))
    if np.isfinite(loglike):
        return loglike
    else:
        return -np.infty
