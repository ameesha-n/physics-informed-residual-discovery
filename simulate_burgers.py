import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftfreq

from spectral import spectral_derivatives
from burgers import burgers_rhs
from rk4 import rk4_step

# -------------------------
# Parameters
# -------------------------

Nx = 256
L = 2 * np.pi

nu = 0.01

dt = 0.001
T = 2.0

Nt = int(T / dt)

# -------------------------
# Grid
# -------------------------

x = np.linspace(0, L, Nx, endpoint=False)

dx = L / Nx

k = fftfreq(Nx, d=dx) * 2 * np.pi

time_grid = np.linspace(
    0,
    T,
    Nt + 1
)

# -------------------------
# Initial Condition
# -------------------------

u0 = np.sin(x)

u = u0.copy()

# -------------------------
# Storage
# -------------------------

solution = np.zeros((Nt + 1, Nx))

ux_storage = np.zeros((Nt + 1, Nx))
uxx_storage = np.zeros((Nt + 1, Nx))
ut_storage = np.zeros((Nt + 1, Nx))

# Store initial state

solution[0] = u

ux, uxx = spectral_derivatives(u, k)

ut = burgers_rhs(
    u=u,
    t=0.0,
    nu=nu,
    k=k
)

ux_storage[0] = ux
uxx_storage[0] = uxx
ut_storage[0] = ut

# -------------------------
# RHS Wrapper
# -------------------------

def rhs(u, t):
    return burgers_rhs(
        u=u,
        t=t,
        nu=nu,
        k=k
    )

# -------------------------
# Time Loop
# -------------------------

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

    ut = burgers_rhs(
        u=u,
        t=t,
        nu=nu,
        k=k
    )

    solution[n + 1] = u

    ux_storage[n + 1] = ux
    uxx_storage[n + 1] = uxx
    ut_storage[n + 1] = ut

    t += dt

# -------------------------
# Validation
# -------------------------

print("solution:", solution.shape)
print("ux:", ux_storage.shape)
print("uxx:", uxx_storage.shape)
print("ut:", ut_storage.shape)

# -------------------------
# Save Dataset
# -------------------------

np.savez(
    "burgers_unforced.npz",

    x=x,
    t=time_grid,

    u=solution,
    ux=ux_storage,
    uxx=uxx_storage,
    ut=ut_storage,

    nu=nu
)

print("\nDataset saved:")
print("burgers_unforced.npz")

# -------------------------
# Heatmap
# -------------------------

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

plt.title("Burgers Solution Evolution")

plt.tight_layout()

plt.show()