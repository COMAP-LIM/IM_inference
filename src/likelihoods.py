import numpy as np


def ln_chi_squared(data, avg, variance):
    # chi2 = np.sum((data - avg)**2 / variance) / len(data)
    # print(chi2)  # print('chi2 :', chi2)
    return - 0.5 * np.nansum((data - avg)**2 / variance + np.log(variance))
