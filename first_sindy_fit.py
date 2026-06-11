import numpy as np
from sklearn.linear_model import Lasso

# ==========================================
# SELECT DATASET
# ==========================================

filename = "alpha_1.5.npz"

# Try:
# filename = "alpha_0.7.npz"
# filename = "alpha_1.0.npz"
# filename = "alpha_1.5.npz"

# ==========================================
# LOAD DATA
# ==========================================

data = np.load(filename)

u = data["u"]
ux = data["ux"]
uxx = data["uxx"]
ut = data["ut"]
forcing = data["forcing"]

# ==========================================
# FLATTEN ARRAYS
# ==========================================

u_flat = u.reshape(-1)

ux_flat = ux.reshape(-1)

uxx_flat = uxx.reshape(-1)

ut_flat = ut.reshape(-1)

f_flat = forcing.reshape(-1)

# ==========================================
# BUILD SINDY LIBRARY
# ==========================================

Theta = np.column_stack([

    np.ones_like(u_flat),     # 1

    u_flat,                   # u

    u_flat**2,                # u^2

    u_flat**3,                # u^3

    ux_flat,                  # ux

    uxx_flat,                 # uxx

    u_flat * ux_flat,         # u*ux

    u_flat * uxx_flat,        # u*uxx

    f_flat                    # forcing
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

print("=" * 50)
print("Dataset:", filename)
print("=" * 50)

print("Theta shape:", Theta.shape)
print("ut shape:", ut_flat.shape)

# ==========================================
# SPARSE REGRESSION
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
# RESULTS
# ==========================================

print("\nRecovered Coefficients\n")

for term, coef in zip(
    terms,
    model.coef_
):

    print(
        f"{term:10s} : {coef: .8f}"
    )

# ==========================================
# TRUE PARAMETERS
# ==========================================

print("\nTrue Parameters\n")

print(
    "alpha =",
    float(data["alpha"])
)

print(
    "nu =",
    float(data["nu"])
)

# ==========================================
# RECONSTRUCTED PDE
# ==========================================

print("\nRecovered PDE\n")

print(
    f"u_t = "
    f"{model.coef_[6]:.6f}(u*ux) + "
    f"{model.coef_[5]:.6f}(uxx) + "
    f"{model.coef_[8]:.6f}(forcing)"
)