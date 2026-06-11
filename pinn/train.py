import torch
import torch.nn as nn

from model import PINN
from dataset import load_dataset
from losses import burgers_residual


model = PINN()

alpha = torch.nn.Parameter(
    torch.tensor(0.0)
)

raw_data = torch.nn.Parameter(
    torch.tensor(0.0)
)

raw_physics = torch.nn.Parameter(
    torch.tensor(0.0)
)

optimizer = torch.optim.Adam(
    list(model.parameters())
    + [
        alpha,
        raw_data,
        raw_physics
    ],
    lr=1e-3
)

x, t, u_true, forcing = load_dataset()

for epoch in range(5000):

    optimizer.zero_grad()

    u_pred = model(
        x,
        t
    )

    data_loss = nn.MSELoss()(
        u_pred,
        u_true
    )

    physics_loss = torch.mean(

        burgers_residual(
            model,
            x,
            t,
            forcing,
            alpha
        ) ** 2

    )

    lambda_data = (
        0.1
        + 9.9
        * torch.sigmoid(raw_data)
    )

    lambda_physics = (
        0.1
        + 9.9
        * torch.sigmoid(raw_physics)
    )

    loss = (

        lambda_data
        * data_loss

        +

        lambda_physics
        * physics_loss

    )

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch:5d}",
            f"Loss={loss.item():.8f}",
            f"Data={data_loss.item():.8f}",
            f"Physics={physics_loss.item():.8f}",
            f"Alpha={alpha.item():.6f}",
            f"LambdaData={lambda_data.item():.6f}",
            f"LambdaPhysics={lambda_physics.item():.6f}"
        )

torch.save(
    {
        "model": model.state_dict(),
        "alpha": alpha.detach()
    },
    "pinn_alpha_stable.pt"
)

print("\n===================")

print(
    "Final Data Loss:",
    data_loss.item()
)

print(
    "Final Physics Loss:",
    physics_loss.item()
)

print(
    "Recovered Alpha:",
    alpha.item()
)

print(
    "Final Lambda Data:",
    lambda_data.item()
)

print(
    "Final Lambda Physics:",
    lambda_physics.item()
)