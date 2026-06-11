import torch
import torch.nn as nn

from model import PINN
from dataset import load_dataset
from losses import burgers_residual


model = PINN()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

x, t, u_true = load_dataset()

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
            t
        ) ** 2
    )

    loss = (
        data_loss
        + physics_loss
    )

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:

        print(
            epoch,
            loss.item(),
            data_loss.item(),
            physics_loss.item()
        )

torch.save(
    model.state_dict(),
    "pinn.pt"
)

print("\nModel Saved")