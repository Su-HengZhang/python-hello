import numpy as np

from matplotlib import pyplot as plt
from utils.datalc import load_bin, extract

# BIN_FILE = "./data/2026-11-07-data-frame1/acq_20261107_180021.bin"
BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"


seg_len = 12000
active_start = 450
active_len = 5000

data_arr = load_bin(BIN_FILE)
data_arr = data_arr.reshape(-1, seg_len, 2)
data_arr = data_arr[:, active_start : active_start + active_len, :]
print(data_arr[0].shape)

n = 1002
np.save("data.npy", data_arr[n, :, 0])

plt.plot(data_arr[n, :, 0])
# plt.plot(data_arr[n, :, 1])
plt.show()

# data_seg = data_arr[n * seg_len : (n + 1) * seg_len]
# print(data_seg.shape)
# plt.plot(data_seg[active_start : active_start + active_len, 0])
# plt.plot(data_seg[active_start : active_start + active_len, 1])
# plt.show()


# df = extract(data_arr, env_thr=0.12)

# print(df.groupby(level=0).size().describe())

# 筛选出每个 packet 中，idx 对应的行数（即长度）正好等于 191 的 packet
# filtered_df = df.groupby(level="packet").filter(lambda group: len(group) == 191)
# print(len(filtered_df))

# print(filtered_df.groupby(level=0).size().describe())
# filtered_df_group = filtered_df.groupby(level="packet")
# for i, group in filtered_df_group:
#     print(i)
#     print(group)


# num_frames = 10


# plt.plot(mdata[:, 0])
# plt.show()
