import numpy as np
from matplotlib import pyplot as plt

from utils.smask import g_matrix
from utils.datalc import load_bin
from utils.fit_sawtooth import fit_sawtooth_extrema

################################################################################
# 循环矩阵参数
################################################################################
p, q = 41, 43
N = p * q
frame_len = 24

g = g_matrix(p, q)

################################################################################
# 读取数据, 进行分段, 然后提取有效数据
################################################################################
# BIN_FILE = "./data/acq_20261107_180021.bin"
BIN_FILE = "./data/acq_20261109_162051.bin"
seg_len = 12000
active_start = 500
active_len = 4800

# 读取数据
data_arr = load_bin(BIN_FILE)
data_arr = data_arr.reshape(-1, seg_len, 2)
data_arr = data_arr[:, active_start : active_start + active_len, :]

total_segs = data_arr.shape[0]  # 总数据段数
print(f"总数据段数: {total_segs}")

frame_num = total_segs // frame_len
print(f"Image frames: {frame_num}")


################################################################################
# 拟合通道1数据峰谷位置
################################################################################
prominence_ratio_init = 0.3
prominence_ratio_list = [0.25, 0.2, 0.15, 0.1, 0.05]

peak_position_list = []
failed_list = []
for n in range(total_segs):
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
        peak_position_list.append(peak_positions["global_fit_positions"])
    except RuntimeError:
        print(f"段 {n} 拟合峰谷位置失败")
        failed_list.append(n)
        peak_position_list.append(None)
        continue

for k, prominence_ratio in enumerate(prominence_ratio_list, start=2):
    # 如果没有失败段，结束循环
    if len(failed_list) == 0:
        break
    print(
        f"尝试对失败段{failed_list}以 prominence_ratio={prominence_ratio} 执行第{k}轮峰谷位置拟合"
    )
    # 尝试拟合
    still_failed = []
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
            peak_position_list[n] = peak_positions["global_fit_positions"]
        except RuntimeError:
            print(f"段 {n} 拟合失败，prominence_ratio={prominence_ratio}")
            still_failed.append(n)
    failed_list = still_failed

if failed_list:
    raise RuntimeError(f"以下数据段在所有轮次中均拟合失败: {failed_list}")
print("所有数据段都成功拟合")


################################################################################
# 按照通道1数据的峰谷位置, 提取通道2对应位置的数据点，
################################################################################
pos_data = []
neg_data = []
ind = np.arange(active_len)
for n in range(total_segs):
    positions = peak_position_list[n]
    data = np.interp(positions, ind, data_arr[n, :, 1])
    # 补充最后一个数据点
    data = np.append(data, data[-1])
    # 添加到列表
    if n % 2 == 0:
        pos_data.append(data)
    else:
        neg_data.append(data)

################################################################################
# 由通道2数据重建图像
################################################################################
img_list = []
for n in range(frame_num):
    img_pos_data = pos_data[n * frame_len : (n + 1) * frame_len]
    img_neg_data = neg_data[n * frame_len : (n + 1) * frame_len]
    img_pos_data = np.concatenate(img_pos_data)
    img_neg_data = np.concatenate(img_neg_data)
    img_data = img_pos_data - img_neg_data
    img_data = img_data[:N]

    img_g = np.zeros((p, q), dtype=np.float64)
    for i in range(N):
        mask = np.reshape(g[i], (p, q))
        img_g += mask * img_data[i]
    img_g = img_g / N
    # img_g = img_g / (np.mean(img_data) * np.mean(g, axis=0).reshape((p, q)))
    img_list.append(img_g)

################################################################################
# 显示图像
################################################################################
plt.imshow(img_list[0])
plt.show()
# fig, axs = plt.subplots(1,frame_num,figsize=(10,3))
# for n in range(frame_num):
#     axs[n].imshow(img_list[n])
#     axs[n].axis("off") 
# plt.show()

