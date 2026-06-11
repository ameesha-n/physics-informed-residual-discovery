import numpy as np

from sklearn.linear_model import Lasso

# ==========================================
# Load Data
# ==========================================

data = np.load(
    "datasets/trajectory_001.npz"
)

u = data["u"]
ux = data["ux"]
uxx = data["uxx"]
ut = data["ut"]
forcing = data["forcing"]

# ==========================================
# Flatten
# ==========================================

u_flat = u.reshape(-1)

ux_flat = ux.reshape(-1)

uxx_flat = uxx.reshape(-1)

ut_flat = ut.reshape(-1)

f_flat = forcing.reshape(-1)

# ==========================================
# Candidate Library
# ==========================================

Theta = np.column_stack([

    np.ones_like(u_flat),

    u_flat,

    u_flat**2,

    u_flat**3,

    ux_flat,

    uxx_flat,

    u_flat * ux_flat,

    u_flat * uxx_flat,

    f_flat
])

terms = [
    "1",
    "u",
    "u^2",
    "u^3",
    "ux",
    "uxx",
    "u*ux",
    "u*uxx",
    "forcing"
]

print("Theta shape:", Theta.shape)
print("ut shape:", ut_flat.shape)

# ==========================================
# Sparse Regression
# ==========================================

model = Lasso(
    alpha=1e-5,
    fit_intercept=False,
    max_iter=10000
)

model.fit(
    Theta,
    ut_flat
)

# ==========================================
# Results
# ==========================================

print("\nRecovered Coefficients:\n")

for term, coef in zip(
    terms,
    model.coef_
):

    print(
        f"{term:10s} : {coef: .8f}"
    )