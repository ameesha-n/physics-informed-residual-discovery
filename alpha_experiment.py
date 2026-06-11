import numpy as np

from generator import generate_trajectory

alphas = [
    0.3,
    0.7,
    1.0,
    1.5
]

for alpha in alphas:

    print("\n===================")
    print("Alpha =", alpha)

    data = generate_trajectory(
        A=0.3,
        mode=2,
        omega=2.0,
        alpha=alpha
    )

    filename = (
        f"alpha_{alpha:.1f}.npz"
    )

    np.savez(
        filename,
        **data
    )

    print(
        "Saved",
        filename
    )