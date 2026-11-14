import numpy as np

from matplotlib import pyplot as plt
from utils.datalc import load_bin
from utils.fit_sawtooth import fit_sawtooth_extrema

# BIN_FILE = "./data/2026-11-07-data-frame1/acq_20261107_180021.bin"
BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"


seg_len = 12000
active_start = 500
active_len = 4800

prominence_ratio_init = 0.3
prominence_ratio_list = [0.25, 0.2, 0.15, 0.1, 0.05]

data_arr = load_bin(BIN_FILE)
data_arr = data_arr.reshape(-1, seg_len, 2)
data_arr = data_arr[:, active_start : active_start + active_len, :]
total_segs = data_arr.shape[0]

peak_position_list = []
failed_list = []
for n in range(total_segs):
    # 读取数据
    data = data_arr[n, :, 0]
    try:
        # 执行拟合
        peak_positions = fit_sawtooth_extrema(
            data,
            extrema_count=191,
            smooth_window=9,
            poly_degree=3,
            prominence_ratio=prominence_ratio_init,
            refine_radius=3,
        )
        peak_position_list.append(peak_positions)
    except RuntimeError:
        print(f"段 {n} 拟合失败")
        failed_list.append(n)
        peak_position_list.append(None)
        continue

for k, prominence_ratio in enumerate(prominence_ratio_list, start=2):
    # 如果没有失败段，结束循环
    if len(failed_list) == 0:
        break
    print(
        f"对失败段{failed_list}以 prominence_ratio={prominence_ratio} 执行第{k}轮拟合"
    )
    # 尝试拟合
    for n in failed_list:
        try:
            peak_positions = fit_sawtooth_extrema(
                data_arr[n, :, 0],
                extrema_count=191,
                smooth_window=9,
                poly_degree=3,
                prominence_ratio=prominence_ratio,
                refine_radius=3,
            )
            peak_position_list[n] = peak_positions
            failed_list.remove(n)
        except RuntimeError:
            print(f"段 {n} 拟合失败，prominence_ratio={prominence_ratio}")
            continue
for n in range(total_segs):
    print(peak_position_list[n])

# data = data_arr[113, :, 0]
# data = data_arr[726, :, 0]
# result = fit_sawtooth_extrema(
#     data,
#     extrema_count=191,
#     smooth_window=9,
#     poly_degree=3,
#     prominence_ratio=0.2,
#     refine_radius=3,
# )
# global_fit_positions = result["global_fit_positions"]

# 绘图
# plt.figure(figsize=(15, 5))
# plt.plot(data, linewidth=0.8, label="Original data")

# plt.scatter(
#     global_fit_positions,
#     np.interp(global_fit_positions, np.arange(len(data)), data),
#     color="blue",
#     s=18,
#     label="Fitted global positions",
#     zorder=3,
# )

# plt.xlabel("Sample index")
# plt.ylabel("Signal amplitude")
# plt.legend()
# plt.grid(alpha=0.2)
# plt.tight_layout()
# plt.show()
