import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.smask import s_matrix, g_matrix
from utils.datalc import load_bin
from utils.fit_sawtooth_extrema import fit_sawtooth_extrema
from utils.estimate_signal_fast import estimate_signal
from utils.reconstruction import direct_reconstruct, reconstruct_with_envelope

################################################################################
# 循环矩阵参数
################################################################################
p, q = 41, 43
N = p * q
frame_segs = 24

smatrix = s_matrix(p, q)
gmatrix = g_matrix(p, q)

################################################################################
# 读取数据, 进行分段, 然后提取有效数据
################################################################################
BIN_FILE = "./data/2026-11-07-data-frame1/acq_20261107_180021.bin"
# BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"
seg_len = 12000
active_start = 500
active_len = 4800

# 读取数据
data_arr = load_bin(BIN_FILE)
data_arr = data_arr.reshape(-1, seg_len, 2)
data_arr = data_arr[:, active_start : active_start + active_len, :]

total_segs = data_arr.shape[0]  # 总数据段数
print(f"总数据段数: {total_segs}")

frame_num = total_segs // frame_segs
print(f"Image frames: {frame_num}")


################################################################################
# 拟合通道1数据峰谷位置
################################################################################
def try_segment_fit(data, ratios):
    """辅助函数：对单个数据段依次尝试不同的 prominence_ratio，直到成功"""
    for ratio in ratios:
        try:
            result = fit_sawtooth_extrema(
                data,
                extrema_count=191,
                smooth_window=9,
                poly_degree=3,
                prominence_ratio=ratio,
            )
            return result
        except RuntimeError:
            continue  # 当前 ratio 失败，尝试下一个更宽松的 ratio
    return None  # 所有 ratio 均尝试失败


# 参数队列
prominence_ratios = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

peak_position_list = []
peak_val_list = []
envelope_list = []
failed_list = []

for n in range(total_segs):
    data = data_arr[n, :, 0]
    positions, peak_val, envelope = try_segment_fit(data, prominence_ratios)

    if positions is not None:
        peak_position_list.append(positions)
        peak_val_list.append(peak_val)
        envelope_list.append(envelope)
    else:
        print(f"段 {n} 在所有 prominence_ratio 条件下拟合均失败")
        failed_list.append(n)
        peak_position_list.append(None)
        peak_val_list.append(None)
        envelope_list.append(None)

if failed_list:
    raise RuntimeError(f"以下数据段在所有轮次中均拟合失败: {failed_list}")

print("所有数据段都成功拟合！")

envelope_stacked = np.stack(envelope_list, axis=0)

envelope_max = np.max(envelope_stacked, axis=1)
envelope_normalized = envelope_stacked / envelope_max[:, np.newaxis]
envelope_avg = np.mean(envelope_normalized, axis=0)


# np.save(f"./results/envelope_stacked.npy", envelope_stacked)

peak_val_array = np.array(peak_val_list)
peak_val_array = peak_val_array / peak_val_array.mean()
################################################################################
# 按照通道1数据的峰谷位置, 提取通道2对应位置的数据点，
################################################################################
pos_data = []
neg_data = []
ind = np.arange(active_len)
for n in range(total_segs):
    positions = peak_position_list[n]
    # 插值
    # data = np.interp(positions, ind, data_arr[n, :, 1])
    # 最小二乘估计
    data, _, _ = estimate_signal(data_arr[n, :, 1], positions)
    # 修正转镜反射率不同引起的信号起伏
    # data = data / peak_val_array[n]
    # 补充最后一个数据点
    data = np.append(data, data[-1])
    # 去掉前q-1个无效数据点
    data = data[q - 1 :]
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
    img_pos_data = pos_data[n * frame_segs // 2 : (n + 1) * frame_segs // 2]
    img_neg_data = neg_data[n * frame_segs // 2 : (n + 1) * frame_segs // 2]
    img_pos_data = np.concatenate(img_pos_data)
    img_neg_data = np.concatenate(img_neg_data)
    # np.save(f"./results/pos_data_{n}.npy", img_pos_data)
    # np.save(f"./results/neg_data_{n}.npy", img_neg_data)
    img_data = img_neg_data - img_pos_data
    img_data = img_data[:N]
    reimg = direct_reconstruct(p, q, img_data)
    # reimg = reconstruct_with_envelope(p, q, img_data, envelope_avg)
    img_list.append(reimg)

################################################################################
# 显示图像
################################################################################
fig, axs = plt.subplots(figsize=(10, 5))
axs.imshow(img_list[0])
axs.axis("off")
plt.show()

# img_even = img_list[::2]
# img_odd = img_list[1::2]
# img_even_num = len(img_even)
# print(img_even_num)

# rows = 6
# cols = img_even_num // rows
# fig, axs = plt.subplots(rows, cols, figsize=(10, 5))
# for n in range(img_even_num):
#     axs[n // cols, n % cols].imshow(img_even[n])
#     axs[n // cols, n % cols].axis("off")
# plt.show()

################################################################################
# 将所有帧写入视频
################################################################################
# VIDEO_FILE = "./results/recovered_frames.mp4"
# fps = 10
# scale = 8  # 每帧放大倍数，便于观看

# img_all = np.stack(img_list)  # (frame_num, p, q)
# vmin, vmax = np.percentile(img_all, [1, 99])
# frames = (img_all - vmin) / (vmax - vmin + 1e-12)
# frames = np.clip(frames, 0, 1)
# frames = (frames * 255).astype(np.uint8)

# h, w = p * scale, q * scale
# writer = cv2.VideoWriter(VIDEO_FILE, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
# for n in range(frame_num):
#     frame = cv2.resize(frames[n], (w, h), interpolation=cv2.INTER_NEAREST)
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     cv2.putText(
#         frame,
#         f"frame {n + 1}/{frame_num}",
#         (5, 20),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 0, 255),
#         1,
#     )
#     writer.write(frame)
# writer.release()
# print(f"视频已保存: {VIDEO_FILE}")
