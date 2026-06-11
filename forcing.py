import numpy as np


def sinusoidal_forcing(
    x,
    t,
    A=1.0,
    mode=2,
    omega=3
):
    """
    f(x,t) = A sin(mode*x - omega*t)
    """

    return A * np.sin(
        mode * x - omega * t
    )