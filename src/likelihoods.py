import numpy as np
import sys

def ln_chi_squared(data, avg, sigma):
    #if (0-sigma) < 1e-7:
    #print('sigma', sigma)
    #sys.exit()

    #square1 = (data - avg) ** 2
    #square2 = (2 * sigma ** 2)
    #like = -1.0 / 2 * np.sum(square1 / square2)

    like = -1.0 / 2 * np.sum(1./2*((data - avg) / sigma)**2)
    return like
