from spectral import spectral_derivatives


def forced_burgers_rhs(
    u,
    t,
    nu,
    k,
    x,
    forcing_fn
):

    ux, uxx = spectral_derivatives(
        u,
        k
    )

    f = forcing_fn(
        x,
        t
    )

    return (
        -u * ux
        + nu * uxx
        + f
    )