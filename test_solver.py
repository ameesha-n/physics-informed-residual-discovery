import numpy as np
from numpy.fft import fftfreq

from spectral import spectral_derivatives
from burgers import burgers_rhs

# Domain setup
Nx = 256
L = 2 * np.pi

x = np.linspace(0, L, Nx, endpoint=False)

dx = L / Nx

# Fourier frequencies
k = fftfreq(Nx, d=dx) * 2 * np.pi

# Test function
u = np.sin(x)

# Compute derivatives
ux, uxx = spectral_derivatives(u, k)

# Analytical derivative
true_ux = np.cos(x)

# Error check
error = np.max(np.abs(ux - true_ux))

print(f"Max Error: {error:.2e}")

# Burgers RHS test
nu = 0.01

rhs = burgers_rhs(
    u=u,
    t=0.0,
    nu=nu,
    k=k
)

print("RHS Shape:", rhs.shape)

# Optional sanity checks
print("u Shape:", u.shape)
print("ux Shape:", ux.shape)
print("uxx Shape:", uxx.shape)

print("\nFirst 5 RHS values:")
print(rhs[:5])