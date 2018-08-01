import numpy as np
import sys

def ln_chi_squared(data, avg, sigma):
    #if (0-sigma) < 1e-7:
    #print('sigma', sigma)
    #sys.exit()

    like = -1.0 / 2 * np.sum((data - avg) ** 2 / (2 * sigma ** 2))
    return like
