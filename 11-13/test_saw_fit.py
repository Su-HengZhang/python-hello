import numpy as np

from matplotlib import pyplot as plt
from utils.datalc import load_bin, fit_saw_extrema
from utils.fit_extrema import fit_sawtooth_extrema

# BIN_FILE = "./data/2026-11-07-data-frame1/acq_20261107_180021.bin"
BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"


seg_len = 12000
active_start = 450
active_len = 5000

data_arr = load_bin(BIN_FILE)
data_arr = data_arr.reshape(-1, seg_len, 2)
data_arr = data_arr[:, active_start : active_start + active_len, :]
# print(data_arr[0].shape)

n = 1002
# np.save("data.npy", data_arr[n, :, 0])

# 读取数据
data = data_arr[n, :, 0]

# 执行整体拟合
result = fit_saw_extrema(
    data,
    extrema_count=191,
    smooth_window=9,
    poly_degree=3,
    prominence_ratio=0.05,
    refine_radius=3,
)

peaks = result["peak_positions"]
valleys = result["valley_positions"]
global_fit_positions = result["global_fit_positions"]

# print("波峰数量：", len(peaks))
# print("波谷数量：", len(valleys))

# print("前10个波峰位置：")
# print(peaks[:10])

# print("前10个波谷位置：")
# print(valleys[:10])

all_positions = fit_sawtooth_extrema(data)


# 绘图
plt.figure(figsize=(15, 5))

plt.plot(data, linewidth=0.8, label="Original data")

# plt.scatter(
#     peaks,
#     np.interp(peaks, np.arange(len(data)), data),
#     color="red",
#     s=18,
#     label="Fitted peaks",
#     zorder=3,
# )

# plt.scatter(
#     valleys,
#     np.interp(valleys, np.arange(len(data)), data),
#     color="green",
#     s=18,
#     label="Fitted valleys",
#     zorder=3,
# )

plt.scatter(
    global_fit_positions,
    np.interp(global_fit_positions, np.arange(len(data)), data),
    color="blue",
    s=18,
    label="Fitted global positions",
    zorder=3,
)

plt.scatter(
    all_positions,
    np.interp(all_positions, np.arange(len(data)), data),
    color="red",
    s=18,
    label="Fitted all positions",
    zorder=3,
)

plt.xlabel("Sample index")
plt.ylabel("Signal amplitude")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
