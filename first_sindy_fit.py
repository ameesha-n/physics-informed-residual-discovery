import numpy as np
from sklearn.linear_model import Lasso

# ==========================================
# DATASET
# ==========================================

filename = "hidden_u4.npz"

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
# FLATTEN
# ==========================================

u_flat = u.reshape(-1)
ux_flat = ux.reshape(-1)
uxx_flat = uxx.reshape(-1)
ut_flat = ut.reshape(-1)
f_flat = forcing.reshape(-1)

# ==========================================
# LIBRARY
# ==========================================

Theta = np.column_stack([

    np.ones_like(u_flat),

    u_flat,

    u_flat**2,

    u_flat**3,

    u_flat**4,

    ux_flat,

    uxx_flat,

    u_flat * ux_flat,

    f_flat
])

terms = [

    "1",

    "u",

    "u^2",

    "u^3",

    "u^4",

    "ux",

    "uxx",

    "u*ux",

    "forcing"
]

# ==========================================
# INFO
# ==========================================

print("=" * 50)
print("Dataset:", filename)
print("=" * 50)

print("Theta shape:", Theta.shape)
print("ut shape:", ut_flat.shape)

# ==========================================
# FIT
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
        f"{term:10s} : {coef:.8f}"
    )

print("\nImportant Terms\n")

for term, coef in zip(
    terms,
    model.coef_
):
    if abs(coef) > 1e-3:

        print(
            f"{term:10s} : {coef:.6f}"
        )

print("\nTrue Parameters\n")

print(
    "alpha =",
    float(data["alpha"])
)

print(
    "nu =",
    float(data["nu"])
)

print("\nTop Terms\n")

pairs = list(
    zip(
        terms,
        model.coef_
    )
)

pairs.sort(
    key=lambda x: abs(x[1]),
    reverse=True
)

for term, coef in pairs[:5]:

    print(
        f"{term:10s} : {coef:.6f}"
    )