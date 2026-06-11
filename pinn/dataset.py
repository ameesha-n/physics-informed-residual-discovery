import numpy as np
import torch

from pathlib import Path


def load_dataset():

    # ==========================================
    # Locate Dataset
    # ==========================================

    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "datasets"
        / "trajectory_001.npz"
    )

    # ==========================================
    # Load Data
    # ==========================================

    data = np.load(dataset_path)

    x = data["x"]
    t = data["t"]
    u = data["u"]

    # ==========================================
    # Create Space-Time Grid
    # ==========================================

    X, T = np.meshgrid(
        x,
        t
    )

    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)

    U = u.reshape(-1, 1)

    # ==========================================
    # Random Subsampling
    # ==========================================

    N_samples = 10000

    idx = np.random.choice(
        len(X),
        N_samples,
        replace=False
    )

    X = X[idx]
    T = T[idx]
    U = U[idx]

    # ==========================================
    # Convert To Torch
    # ==========================================

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

    return (
        X,
        T,
        U
    )


if __name__ == "__main__":

    x, t, u = load_dataset()

    print("x:", x.shape)
    print("t:", t.shape)
    print("u:", u.shape)