import numpy as np


def power_spectrum(seq):
    """
    
    Args:
      seq: (batch_size, n_joints, seq_len, feature_size)
  
    Returns:
        (n_joints, seq_len, feature_size)
    """
    seq_fft = np.fft.fft(seq, axis=2)
    seq_ps = np.abs(seq_fft)**2
    
    seq_ps_global = seq_ps.sum(axis=0) + 1e-8
    seq_ps_global /= seq_ps_global.sum(axis=1, keepdims=True)
    return seq_ps_global


def ps_entropy(seq_ps):
    """
    
    Args:
        seq_ps: (n_joints, seq_len, feature_size)

    Returns:
    """
    return -np.sum(seq_ps * np.log(seq_ps), axis=1)


def ps_kld(seq_ps_from, seq_ps_to):
    """ Calculates KL(seq_ps_from, seq_ps_to).
    Args:
        seq_ps_from:
        seq_ps_to:

    Returns:
    """
    return np.sum(seq_ps_from * np.log(seq_ps_from / seq_ps_to), axis=1)
