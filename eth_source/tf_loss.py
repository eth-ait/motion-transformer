import tensorflow as tf
import numpy as np


def logli_normal_bivariate(x, mu, sigma, rho, reduce_sum=False):
    """
    Bivariate Gaussian log-likelihood. Rank of arguments is expected to be 3.

    Args:
        x: data samples with shape (batch_size, seq_len, feature_size).
        mu:
        sigma: standard deviation.
        rho:
        reduce_sum: False, None or list of axes.
    Returns:

    """
    last_axis = tf.rank(x)-1
    x1, x2 = tf.split(x, 2, axis=last_axis)
    mu1, mu2 = tf.split(mu, 2, axis=last_axis)
    sigma1, sigma2 = tf.split(sigma, 2, axis=last_axis)

    with tf.name_scope('logli_normal_bivariate'):
        x_mu1 = tf.subtract(x1, mu1)
        x_mu2 = tf.subtract(x2, mu2)
        z_denom = tf.square(tf.div(x_mu1, tf.maximum(1e-9, sigma1))) + \
                  tf.square(tf.div(x_mu2, tf.maximum(1e-9, sigma2))) - \
                  2*tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.maximum(1e-9, tf.multiply(sigma1, sigma2)))

        rho_square_term = tf.maximum(1e-9, 1-tf.square(rho))
        log_regularize_term = tf.log(tf.maximum(1e-9, 2*np.pi*tf.multiply(tf.multiply(sigma1, sigma2), tf.sqrt(rho_square_term))))
        log_power_e = tf.div(z_denom, 2*rho_square_term)
        result = -(log_regularize_term + log_power_e)

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)


def logli_normal_isotropic(x, mu, sigma, reduce_sum=False):
    """
    Isotropic Gaussian log-likelihood.

    Args:
        x:
        mu:
        sigma: standard deviation.
        reduce_sum:

    Returns:

    """
    with tf.name_scope('logli_normal_isotropic'):
        var = tf.maximum(1e-6, tf.square(sigma))
        result = -0.5 * (tf.log(2*np.pi*var) + tf.div(tf.square(x-mu), var))

        return tf.reduce_sum(result, -1, keepdims=True)


def logli_bernoulli(x, theta, reduce_sum=False):
    """
    Bernoulli log-likelihood.

    Args:
        x:
        theta:
        reduce_sum:

    Returns:

    """
    with tf.name_scope('logli_bernoulli'):
        result = tf.multiply(x, tf.log(tf.maximum(1e-5, theta))) + tf.multiply((1 - x), tf.log(tf.maximum(1e-5, 1 - theta)))

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)


def logli_gmm_logsumexp(x, mu, sigma, coefficient):
    """
    Gaussian mixture model (with Gaussian components with diagonal covariance matrix) log-likelihood.

    Args:
        x: (batch_size, seq_len, feature_size)
        mu: (batch_size, seq_len, feature_size*num_gmm_components)
        sigma: standard deviation (batch_size, seq_len, feature_size*num_gmm_components)
        coefficient: (batch_size, seq_len, num_gmm_components)

    Returns:
    """
    with tf.name_scope('logli_gmm_logsumexp'):
        batch_size, seq_len, feature_gmm_components = mu.shape.as_list()
        _, _, num_gmm_components = coefficient.shape.as_list()
        feature_size = int(feature_gmm_components/num_gmm_components)
        seq_len = tf.shape(mu)[1] if seq_len is None else seq_len  # Variable-length sequences.
        batch_size = tf.shape(mu)[0] if batch_size is None else batch_size

        mu_ = tf.reshape(mu, (batch_size, seq_len, feature_size, num_gmm_components))
        sigma_ = tf.reshape(sigma, (batch_size, seq_len, feature_size, num_gmm_components))
        x_ = tf.expand_dims(x, axis=-1)
        log_coeff = tf.log(tf.maximum(1e-6, coefficient))

        var = tf.maximum(1e-6, tf.square(sigma_))
        log_normal = -0.5*tf.reduce_sum((tf.log(2*np.pi*var) + tf.div(tf.square(x_ - mu_), var)), axis=2)
        return tf.reduce_logsumexp(log_coeff + log_normal, axis=-1, keepdims=True)


def logli_gmm(x, mu, sigma, coefficient):
    """
    Gaussian mixture model (with Gaussian components with diagonal covariance matrix) log-likelihood.

    Args:
        x: (batch_size, seq_len, feature_size)
        mu: (batch_size, seq_len, feature_size*num_gmm_components)
        sigma: standard deviation (batch_size, seq_len, feature_size*num_gmm_components)
        coefficient: (batch_size, seq_len, num_gmm_components)

    Returns:
    """
    with tf.name_scope('logli_gmm'):
        batch_size, seq_len, feature_gmm_components = mu.shape.as_list()
        _, _, num_gmm_components = coefficient.shape.as_list()
        feature_size = int(feature_gmm_components/num_gmm_components)
        seq_len = tf.shape(mu)[1] if seq_len is None else seq_len  # Variable-length sequences.
        batch_size = tf.shape(mu)[0] if batch_size is None else batch_size

        mu_ = tf.reshape(mu, (batch_size, seq_len, feature_size, num_gmm_components))
        sigma_ = tf.reshape(sigma, (batch_size, seq_len, feature_size, num_gmm_components))
        x_ = tf.expand_dims(x, axis=-1)
        coefficient_ = tf.expand_dims(coefficient, axis=2)

        var = tf.maximum(1e-6, tf.square(sigma_))
        z_term = tf.div(1.0, tf.sqrt(2*np.pi*var))
        exp_term = tf.exp(tf.div(-tf.square(x_-mu_), 2*var))
        gaussian_likelihood = z_term * exp_term
        gmm_likelihood = tf.reduce_sum(tf.multiply(gaussian_likelihood, coefficient_), axis=-1)
        gmm_loglikelihood = tf.log(tf.maximum(1e-6, gmm_likelihood))

        return tf.reduce_sum(gmm_loglikelihood, -1, keepdims=True)


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
    with tf.name_scope("kld_normal_isotropic"):
        result = tf.reduce_sum(tf.log(tf.maximum(1e-6, sigma2)) - tf.log(tf.maximum(1e-6, sigma1)) + (tf.square(sigma1) + tf.square(mu1 - mu2)) / (2*tf.maximum(1e-6, (tf.square(sigma2)))) - 0.5, keepdims=True, axis=-1)

        if reduce_sum is False:
            return result
        else:
            return tf.reduce_sum(result, reduce_sum)


def kld_bernoulli(probs1, probs2):
    return tf.reduce_sum(probs1*(tf.log(tf.maximum(probs1, 1e-6)) - tf.log(tf.maximum(probs2, 1e-6))) + (1-probs1)*(tf.log(tf.maximum((1-probs1), 1e-6)) - tf.log(tf.maximum((1-probs2), 1e-6))), axis=-1, keepdims=True)


def entropy(probs):
    """
    Calculates entropy given probabilities.
    Args:
        probs: has shape of (batch_size, K) where K corresponds to number of categories. Entropy is calculated for each
            (1,K) dimensional row vector.
    Returns:
        (batch_size, 1)
    """
    return -tf.reduce_sum(probs*tf.log(tf.maximum(probs, 1e-6)), axis=1, keepdims=True)
