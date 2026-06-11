import torch
import numpy as np

from model import PINN
from dataset import load_dataset

model = PINN()

model.load_state_dict(
    torch.load("pinn.pt")
)

model.eval()

x, t, u_true = load_dataset()

with torch.no_grad():

    u_pred = model(
        x,
        t
    )

mse = torch.mean(
    (u_pred - u_true) ** 2
)

rmse = torch.sqrt(mse)

relative_error = (
    torch.norm(u_pred - u_true)
    /
    torch.norm(u_true)
)

print("\nResults")

print(
    "RMSE:",
    rmse.item()
)

print(
    "Relative Error:",
    relative_error.item()
)