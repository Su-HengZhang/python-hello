import numpy as np

from matplotlib import pyplot as plt
from utils.datalc import load_bin
from utils.fit_sawtooth import fit_sawtooth_extrema

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

# 执行整体拟合
result = fit_sawtooth_extrema(
    data,
    extrema_count=191,
    smooth_window=9,
    poly_degree=3,
    prominence_ratio=0.05,
)

global_fit_positions = result


# 绘图
plt.figure(figsize=(15, 5))
plt.plot(data, linewidth=0.8, label="Original data")
plt.scatter(
    global_fit_positions,
    np.interp(global_fit_positions, np.arange(len(data)), data),
    color="blue",
    s=18,
    label="Fitted global positions",
    zorder=3,
)

plt.xlabel("Sample index")
plt.ylabel("Signal amplitude")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
