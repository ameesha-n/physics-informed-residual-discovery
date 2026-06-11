import torch
import torch.nn as nn

from model import PINN
from dataset import load_dataset
from losses import burgers_residual


model = PINN()

# Learnable PDE coefficient
alpha = torch.nn.Parameter(
    torch.tensor(0.0)
)

# Learnable loss weights
s_data = torch.nn.Parameter(
    torch.tensor(0.0)
)

s_physics = torch.nn.Parameter(
    torch.tensor(0.0)
)

optimizer = torch.optim.Adam(
    list(model.parameters())
    + [
        alpha,
        s_data,
        s_physics
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

    loss = (

        torch.exp(-s_data)
        * data_loss

        + s_data

        +

        torch.exp(-s_physics)
        * physics_loss

        + s_physics

    )

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:

        lambda_data = torch.exp(
            -s_data
        ).item()

        lambda_physics = torch.exp(
            -s_physics
        ).item()

        print(
            f"Epoch {epoch:5d}",
            f"Loss={loss.item():.8f}",
            f"Data={data_loss.item():.8f}",
            f"Physics={physics_loss.item():.8f}",
            f"Alpha={alpha.item():.6f}",
            f"LambdaData={lambda_data:.6f}",
            f"LambdaPhysics={lambda_physics:.6f}"
        )

torch.save(
    {
        "model": model.state_dict(),
        "alpha": alpha.detach(),
        "s_data": s_data.detach(),
        "s_physics": s_physics.detach()
    },
    "pinn_alpha_adaptive.pt"
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
    torch.exp(-s_data).item()
)

print(
    "Final Lambda Physics:",
    torch.exp(-s_physics).item()
)