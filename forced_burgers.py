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
    alpha=1.0,
    include_hidden_residual=False
):

    ux, uxx = spectral_derivatives(
        u,
        k
    )

    forcing = forcing_fn(
        x,
        t
    )

    rhs = (
        -u * ux
        + nu * uxx
        + alpha * forcing
    )

    if include_hidden_residual:

        rhs += hidden_residual(
            u
        )

    return rhs