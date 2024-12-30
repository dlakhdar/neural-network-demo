import numpy as np


def TAP(J, m, β):
    """ """
    d = np.diag(J**2 * (1 - m**2))
    return np.tanh(J @ m - β * d * m)


def linearized_TAP(J, m, β):
    """ """
    D = np.diag(1 - m**2)
    d = np.diag(J**2 * (1 - m**2))
    return 1 - D @ J + β * D @ d
