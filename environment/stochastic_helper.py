import scipy.stats as stats
import numpy as np


def gen_instances(M, K, mu=None, sigma=0.1):
    if mu is None:
        mu = np.random.random(K)
    lower = 0
    upper = 1
    means = np.zeros((M, K))

    for i in range(K):
        a, b = (lower - mu[i]) / sigma, (upper - mu[i]) / sigma

        dst = stats.truncnorm(a, b, loc=mu[i], scale=sigma)
        means[:,i] = dst.rvs(M)

    return means


