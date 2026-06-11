import numpy as np

from generator import generate_trajectory

data = generate_trajectory(
    A=0.3,
    mode=2,
    omega=2.0,
    alpha=1.0
)

np.savez(
    "hidden_residual.npz",
    **data
)

print("Saved")