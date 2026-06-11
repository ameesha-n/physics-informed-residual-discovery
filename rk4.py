import numpy as np


def rk4_step(u, t, dt, rhs_func):

    k1 = rhs_func(u, t)

    k2 = rhs_func(
        u + 0.5 * dt * k1,
        t + 0.5 * dt
    )

    k3 = rhs_func(
        u + 0.5 * dt * k2,
        t + 0.5 * dt
    )

    k4 = rhs_func(
        u + dt * k3,
        t + dt
    )

    return u + (dt / 6.0) * (
        k1 + 2*k2 + 2*k3 + k4
    )