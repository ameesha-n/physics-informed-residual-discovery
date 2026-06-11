import os
import numpy as np

from generator import generate_trajectory

os.makedirs(
    "datasets",
    exist_ok=True
)

TARGET = 20
saved = 0

while saved < TARGET:

    A = np.random.uniform(
        0.1,
        0.5
    )

    mode = np.random.randint(
        1,
        4
    )

    omega = np.random.uniform(
        1.0,
        4.0
    )

    try:

        data = generate_trajectory(
            A=A,
            mode=mode,
            omega=omega
        )

        filename = (
            f"datasets/"
            f"trajectory_{saved+1:03d}.npz"
        )

        np.savez(
            filename,
            **data
        )

        saved += 1

        print(
            f"[{saved}/{TARGET}]",
            filename
        )

    except ValueError as e:

        print("Rejected:", e)