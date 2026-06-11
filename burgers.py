import numpy as np

from spectral import spectral_derivatives


def burgers_rhs(u, t, nu, k):
    ux, uxx = spectral_derivatives(u, k)

    return -u * ux + nu * uxx