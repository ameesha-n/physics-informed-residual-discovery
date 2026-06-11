import numpy as np
from numpy.fft import fftfreq

from spectral import spectral_derivatives
from forcing import sinusoidal_forcing
from forced_burgers import forced_burgers_rhs
from rk4 import rk4_step


def generate_trajectory(
    A,
    mode,
    omega,
    alpha=1.0,
    Nx=256,
    T=2.0,
    dt=0.001,
    nu=0.01
):

    # ==========================================
    # Domain
    # ==========================================

    L = 2 * np.pi

    Nt = int(T / dt)

    x = np.linspace(
        0,
        L,
        Nx,
        endpoint=False
    )

    dx = L / Nx

    k = fftfreq(
        Nx,
        d=dx
    ) * 2 * np.pi

    time_grid = np.linspace(
        0,
        T,
        Nt + 1
    )

    # ==========================================
    # Initial Condition
    # ==========================================

    u = np.sin(x)

    # ==========================================
    # Storage
    # ==========================================

    solution = np.zeros((Nt + 1, Nx))

    ux_storage = np.zeros((Nt + 1, Nx))
    uxx_storage = np.zeros((Nt + 1, Nx))
    ut_storage = np.zeros((Nt + 1, Nx))

    forcing_storage = np.zeros((Nt + 1, Nx))

    solution[0] = u

    # ==========================================
    # RHS
    # ==========================================

    def rhs(u, t):

        return forced_burgers_rhs(
            u=u,
            t=t,
            nu=nu,
            k=k,
            x=x,
            forcing_fn=lambda x_, t_: sinusoidal_forcing(
                x_,
                t_,
                A=A,
                mode=mode,
                omega=omega
            ),
            alpha=alpha
        )

    # ==========================================
    # Time Integration
    # ==========================================

    t = 0.0

    for n in range(Nt):

        ux, uxx = spectral_derivatives(
            u,
            k
        )

        ut = rhs(
            u,
            t
        )

        f = sinusoidal_forcing(
            x,
            t,
            A=A,
            mode=mode,
            omega=omega
        )

        ux_storage[n] = ux
        uxx_storage[n] = uxx
        ut_storage[n] = ut

        forcing_storage[n] = f

        # RK4 Step

        u = rk4_step(
            u,
            t,
            dt,
            rhs
        )

        # ======================================
        # Stability Checks
        # ======================================

        if np.isnan(u).any():
            raise ValueError(
                f"NaN detected at step={n}, t={t:.6f}"
            )

        if np.isinf(u).any():
            raise ValueError(
                f"Inf detected at step={n}, t={t:.6f}"
            )

        max_u = np.max(np.abs(u))

        if max_u > 100:
            raise ValueError(
                f"Explosion detected at step={n}, "
                f"t={t:.6f}, "
                f"max_u={max_u:.6f}"
            )

        solution[n + 1] = u

        t += dt

    # ==========================================
    # Store Final State
    # ==========================================

    ux, uxx = spectral_derivatives(
        u,
        k
    )

    ut = rhs(
        u,
        t
    )

    f = sinusoidal_forcing(
        x,
        t,
        A=A,
        mode=mode,
        omega=omega
    )

    ux_storage[-1] = ux
    uxx_storage[-1] = uxx
    ut_storage[-1] = ut

    forcing_storage[-1] = f

    # ==========================================
    # Return Dataset
    # ==========================================

    return {

        "x": x,
        "t": time_grid,

        "u": solution,

        "ux": ux_storage,
        "uxx": uxx_storage,
        "ut": ut_storage,

        "forcing": forcing_storage,

        "A": A,
        "mode": mode,
        "omega": omega,

        "nu": nu,

        "alpha": alpha
    }