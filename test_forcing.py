import numpy as np
import matplotlib.pyplot as plt

from forcing import sinusoidal_forcing

x = np.linspace(
    0,
    2*np.pi,
    256,
    endpoint=False
)

t = 0.0

f = sinusoidal_forcing(
    x,
    t,
    A=1.0,
    mode=2,
    omega=3
)

plt.plot(x, f)

plt.title("Sinusoidal Forcing")

plt.show()