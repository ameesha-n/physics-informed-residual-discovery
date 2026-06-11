import numpy as np
import torch

# ==========================================
# DATASET
# ==========================================

filename = "alpha_1.5.npz"

# try later:
# alpha_0.7.npz
# alpha_1.0.npz
# alpha_1.5.npz

data = np.load(filename)

u = torch.tensor(
    data["u"],
    dtype=torch.float32
)

ux = torch.tensor(
    data["ux"],
    dtype=torch.float32
)

uxx = torch.tensor(
    data["uxx"],
    dtype=torch.float32
)

ut = torch.tensor(
    data["ut"],
    dtype=torch.float32
)

forcing = torch.tensor(
    data["forcing"],
    dtype=torch.float32
)

nu = float(
    data["nu"]
)

true_alpha = float(
    data["alpha"]
)

# ==========================================
# LEARNABLE ALPHA
# ==========================================

alpha = torch.nn.Parameter(
    torch.tensor(1.0)
)

optimizer = torch.optim.Adam(
    [alpha],
    lr=1e-2
)

# ==========================================
# TRAIN
# ==========================================

for epoch in range(5000):

    optimizer.zero_grad()

    prediction = (
        -u * ux
        + nu * uxx
        + alpha * forcing
    )

    loss = torch.mean(
        (prediction - ut) ** 2
    )

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d}",
            f"Loss={loss.item():.8e}",
            f"Alpha={alpha.item():.6f}"
        )

# ==========================================
# RESULTS
# ==========================================

print("\n===================")

print(
    "True Alpha:",
    true_alpha
)

print(
    "Recovered Alpha:",
    alpha.item()
)