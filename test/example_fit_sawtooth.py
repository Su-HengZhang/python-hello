import numpy as np
import matplotlib.pyplot as plt

from utils.fit_sawtooth3 import fit_sawtooth_extrema

from utils.datalc import load_bin

# ------------------------------------------------------
# 1. 生成合成数据
# ------------------------------------------------------

BIN_FILE = "./data/2026-11-07-data-frame1/acq_20261107_180021.bin"
# BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"


seg_len = 12000
active_start = 520
active_len = 4800

data_arr = load_bin(BIN_FILE)
data_arr = data_arr.reshape(-1, seg_len, 2)
data_arr = data_arr[:, active_start : active_start + active_len, :]
# print(data_arr[0].shape)
# np.save("data.npy", data_arr[n, :, 0])

n = 0

# 读取数据
data = data_arr[n, :, 0]

# ------------------------------------------------------
# 2. 调用函数
# ------------------------------------------------------
positions, envelope_values = fit_sawtooth_extrema(
    data,
    extrema_count=191,
    envelope_degree=6,
)

np.save("positions.npy", positions)
np.save("data.npy", data_arr[n, :, 1])

print("positions 形状:      ", positions.shape)
print("envelope_values 形状:", envelope_values.shape)
print("前 5 个位置:  ", np.round(positions[:5], 1))
print("前 5 个包络值:", np.round(envelope_values[:5], 4))

# 仅为作图区分波峰/波谷：波峰处信号接近包络，波谷处远低于包络
sig_at_pos = np.interp(positions, np.arange(len(data)), data)
is_peak = sig_at_pos > 0.5 * (envelope_values + data.min())

# ------------------------------------------------------
# 3. 绘图
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.5))

ax.plot(data, lw=0.7, label="Original data")
ax.plot(positions, envelope_values, "r-", lw=1.2, alpha=0.6, label="Envelope (line)")
ax.scatter(
    positions[is_peak],
    envelope_values[is_peak],
    s=14,
    c="tab:red",
    zorder=3,
    label="Envelope @ peaks (fitted)",
)
ax.scatter(
    positions[~is_peak],
    envelope_values[~is_peak],
    s=14,
    c="tab:orange",
    zorder=3,
    label="Envelope @ valleys (from adjacent peaks)",
)
ax.set_xlabel("Sample index")
ax.set_ylabel("Signal amplitude")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("example_fit_sawtooth.png", dpi=150)
plt.show()
