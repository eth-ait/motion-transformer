import numpy as np


def kld_normal_isotropic(mu1, sigma1, mu2, sigma2, reduce_sum=False):
    """
    Kullback-Leibler divergence between two isotropic Gaussian distributions.

    Args:
        mu1:
        sigma1: standard deviation.
        mu2:
        sigma2: standard deviation.
        reduce_sum:

    Returns:
    """
    result = (2*np.log(np.maximum(1e-9, sigma2)) - 2*np.log(np.maximum(1e-9, sigma1)) + ((np.square(sigma1) + np.square(mu1 - mu2))/np.maximum(1e-9, (np.square(sigma2)))) - 1)
    result = np.sum(result, axis=-1, keepdims=True)

    if reduce_sum is False:
        return result
    else:
        return np.sum(result, axis=reduce_sum)
