# check_dataset.py

import os
import numpy as np

folder = "datasets"

bad = []

for file in sorted(os.listdir(folder)):

    path = os.path.join(folder, file)

    data = np.load(path)

    u = data["u"]

    if np.isnan(u).any():
        bad.append(file)

    elif np.isinf(u).any():
        bad.append(file)

print("Bad files:")
print(bad)

print("\nCount:", len(bad))