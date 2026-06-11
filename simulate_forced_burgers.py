import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq

from spectral import spectral_derivatives
from forcing import sinusoidal_forcing
from forced_burgers import forced_burgers_rhs
from rk4 import rk4_step

# =====================================================
# Parameters
# =====================================================

Nx = 256
L = 2 * np.pi

nu = 0.01

dt = 0.001
T = 2.0

Nt = int(T / dt)

# Forcing parameters

A = 1.0
mode = 2
omega = 3

# =====================================================
# Grid
# =====================================================

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

# =====================================================
# Initial Condition
# =====================================================

u0 = np.sin(x)

u = u0.copy()

# =====================================================
# Storage
# =====================================================

solution = np.zeros((Nt + 1, Nx))

ux_storage = np.zeros((Nt + 1, Nx))
uxx_storage = np.zeros((Nt + 1, Nx))
ut_storage = np.zeros((Nt + 1, Nx))

forcing_storage = np.zeros((Nt + 1, Nx))

# =====================================================
# Initial State
# =====================================================

solution[0] = u

ux, uxx = spectral_derivatives(
    u,
    k
)

ut = forced_burgers_rhs(
    u=u,
    t=0.0,
    nu=nu,
    k=k,
    x=x,
    forcing_fn=lambda x_, t_: sinusoidal_forcing(
        x_,
        t_,
        A=A,
        mode=mode,
        omega=omega
    )
)

f = sinusoidal_forcing(
    x,
    0.0,
    A=A,
    mode=mode,
    omega=omega
)

ux_storage[0] = ux
uxx_storage[0] = uxx
ut_storage[0] = ut

forcing_storage[0] = f

# =====================================================
# RHS Wrapper
# =====================================================

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
        )
    )

# =====================================================
# Time Integration
# =====================================================

t = 0.0

for n in range(Nt):

    u = rk4_step(
        u,
        t,
        dt,
        rhs
    )

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

    solution[n + 1] = u

    ux_storage[n + 1] = ux
    uxx_storage[n + 1] = uxx
    ut_storage[n + 1] = ut

    forcing_storage[n + 1] = f

    t += dt

# =====================================================
# Validation
# =====================================================

print("solution:", solution.shape)
print("ux:", ux_storage.shape)
print("uxx:", uxx_storage.shape)
print("ut:", ut_storage.shape)
print("forcing:", forcing_storage.shape)

# =====================================================
# Save Dataset
# =====================================================

np.savez(
    "burgers_forced_001.npz",

    x=x,
    t=time_grid,

    u=solution,

    ux=ux_storage,
    uxx=uxx_storage,
    ut=ut_storage,

    forcing=forcing_storage,

    nu=nu,

    forcing_type="sinusoidal",

    A=A,
    mode=mode,
    omega=omega
)

print("\nDataset saved:")
print("burgers_forced_001.npz")

# =====================================================
# Heatmap
# =====================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    solution,
    aspect="auto",
    origin="lower",
    extent=[0, L, 0, T]
)

plt.colorbar(label="u")

plt.xlabel("x")
plt.ylabel("t")

plt.title(
    f"Forced Burgers (A={A}, mode={mode}, omega={omega})"
)

plt.tight_layout()

plt.show()