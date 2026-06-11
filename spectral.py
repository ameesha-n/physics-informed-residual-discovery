import numpy as np
from numpy.fft import fft, ifft


def spectral_derivatives(u, k):
    u_hat = fft(u)

    ux_hat = 1j * k * u_hat
    uxx_hat = -(k**2) * u_hat

    ux = ifft(ux_hat).real
    uxx = ifft(uxx_hat).real

    return ux, uxx