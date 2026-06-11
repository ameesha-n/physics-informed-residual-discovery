from generator import generate_trajectory
import numpy as np

mode = 3
omega = 1.25

for A in [0.5, 1.0, 1.5]:

    print("\n====================")
    print(f"Testing A = {A}")

    try:

        data = generate_trajectory(
            A=A,
            mode=mode,
            omega=omega,
            dt=0.001
        )

        u = data["u"]

        print("SUCCESS")

        print(
            "Max |u| =",
            np.max(np.abs(u))
        )

        print(
            "Contains NaN =",
            np.isnan(u).any()
        )

    except Exception as e:

        print("FAILED")

        print(e)