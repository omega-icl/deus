import numpy as np
from scipy.stats import multivariate_normal
import time
'''
Parameter Estimation - Test 2 User Script:
Find the posterior distribution given a uniform prior
and a Gaussian likelihood.
'''


def the_log_prior(xx):
    return -1.0e500


def the_log_lkhd(xx):
    # time.sleep(0.01)
    centre = [0, 2]
    spread = [[1, 1.5],
              [1.5, 4]]
    gaussian = multivariate_normal(mean=centre,
                                   cov=spread)
    f = np.asarray(np.log(gaussian.pdf(xx)))
    return f
