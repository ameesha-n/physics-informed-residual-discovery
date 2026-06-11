import numpy as np
import matplotlib.pyplot as plt

true_alpha = np.array([
    0.3,
    0.7,
    1.0,
    1.5
])

learned_alpha = np.array([
    0.234,
    0.527,
    0.855,
    1.226
])

m, b = np.polyfit(
    true_alpha,
    learned_alpha,
    1
)

print("Slope =", m)
print("Intercept =", b)

plt.figure(figsize=(7, 5))

plt.scatter(
    true_alpha,
    learned_alpha,
    s=80
)

plt.plot(
    true_alpha,
    true_alpha,
    "--",
    label="Ideal"
)

plt.plot(
    true_alpha,
    m * true_alpha + b,
    label=f"Fit: y={m:.3f}x+{b:.3f}"
)

plt.xlabel("True Alpha")
plt.ylabel("Recovered Alpha")

plt.title(
    "PINN Alpha Recovery"
)

plt.legend()

plt.grid(True)

plt.show()