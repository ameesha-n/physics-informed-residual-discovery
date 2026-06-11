# inspect_bad_files.py

import numpy as np

bad_files = [
    "trajectory_002.npz",
    "trajectory_007.npz",
    "trajectory_009.npz",
    "trajectory_015.npz",
    "trajectory_020.npz"
]

for file in bad_files:

    data = np.load(f"datasets/{file}")

    print(
        file,
        "| A =", float(data["A"]),
        "| mode =", int(data["mode"]),
        "| omega =", float(data["omega"])
    )