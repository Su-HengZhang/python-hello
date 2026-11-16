import numpy as np
import matplotlib.pyplot as plt

data = np.load("data.npy")
ind = np.arange(len(data))

positions = np.load("positions.npy")


data_inter = np.interp(positions, ind, data)


# ------------------------------------------------------
# 绘图
# ------------------------------------------------------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4.5))

ax.plot(data, lw=0.7, label="Original data")

ax.scatter(
    positions,
    data_inter,
    s=14,
    c="tab:red",
    zorder=3,
    label="Data @ peaks (fitted)",
)

ax.set_xlabel("Sample index")
ax.set_ylabel("Signal amplitude")
ax.grid(alpha=0.2)

fig.tight_layout()
# fig.savefig("example_fit_sawtooth.png", dpi=150)
plt.show()
