import numpy as np
import torch

from pathlib import Path


def load_dataset():

    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "alpha_0.3.npz"
    )

    data = np.load(dataset_path)

    x = data["x"]
    t = data["t"]

    u = data["u"]
    forcing = data["forcing"]

    X, T = np.meshgrid(
        x,
        t
    )

    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)

    U = u.reshape(-1, 1)
    F = forcing.reshape(-1, 1)

    N_samples = 10000

    idx = np.random.choice(
        len(X),
        N_samples,
        replace=False
    )

    X = X[idx]
    T = T[idx]

    U = U[idx]
    F = F[idx]

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    T = torch.tensor(
        T,
        dtype=torch.float32
    )

    U = torch.tensor(
        U,
        dtype=torch.float32
    )

    F = torch.tensor(
        F,
        dtype=torch.float32
    )

    return (
        X,
        T,
        U,
        F
    )


if __name__ == "__main__":

    x, t, u, f = load_dataset()

    print(x.shape)
    print(t.shape)
    print(u.shape)
    print(f.shape)