import numpy as np

from spectral import spectral_derivatives
from residual import hidden_residual


def forced_burgers_rhs(
    u,
    t,
    nu,
    k,
    x,
    forcing_fn,
    alpha=1.0
):

    ux, uxx = spectral_derivatives(
        u,
        k
    )

    forcing = forcing_fn(
        x,
        t
    )

    return (
    -u * ux
    + nu * uxx
    + alpha * forcing
    + hidden_residual(u)
)