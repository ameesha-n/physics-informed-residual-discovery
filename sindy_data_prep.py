import numpy as np

# Load trajectory

data = np.load(
    "datasets/trajectory_001.npz"
)

# Extract fields

u = data["u"]

ux = data["ux"]

uxx = data["uxx"]

ut = data["ut"]

forcing = data["forcing"]

# Flatten everything

u_flat = u.reshape(-1)

ux_flat = ux.reshape(-1)

uxx_flat = uxx.reshape(-1)

ut_flat = ut.reshape(-1)

f_flat = forcing.reshape(-1)

# Build candidate library

Theta = np.column_stack([

    np.ones_like(u_flat),     # 1

    u_flat,                   # u

    u_flat**2,                # u^2

    u_flat**3,                # u^3

    ux_flat,                  # ux

    uxx_flat,                 # uxx

    u_flat * ux_flat,         # u*ux

    u_flat * uxx_flat,        # u*uxx

    f_flat                    # forcing

])

print("Theta shape:", Theta.shape)

print("ut shape:", ut_flat.shape)

print()

print("Candidate Terms:")

terms = [
    "1",
    "u",
    "u^2",
    "u^3",
    "ux",
    "uxx",
    "u*ux",
    "u*uxx",
    "forcing"
]

for i, term in enumerate(terms):
    print(f"{i}: {term}")