import numpy as np
from matplotlib import pyplot as plt

from utils.smask import twin_prime_s_row
from utils.reconstruction import direct_reconstruct, reconstruct_with_envelope

p, q = 41, 43
N = p * q

s = twin_prime_s_row(p, q)  # s 循环矩阵的第1行
s = np.roll(s, 1)  # 通过循环右移1列，得到 s 循环矩阵的最后1行
g = np.ones_like(s) - 2 * s  # 由s循环矩阵最后1行，计算出g循环矩阵最后1行

# 加载测量数据
img_pos_data = np.load("./results/pos_data_0.npy")
img_neg_data = np.load("./results/neg_data_0.npy")
img_pos_data = img_pos_data[:N]
img_neg_data = img_neg_data[:N]
img_data = img_neg_data - img_pos_data  # 计算差分信号

# 滤波重建
re_img1 = direct_reconstruct(p, q, img_data)

# 考虑Envelope 精确重建
envelope = np.load("./results/page_envelope.npy")  # 加载Envelope
re_img2 = reconstruct_with_envelope(p, q, img_data, envelope)


# 滤波重建
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
ax[0].imshow(re_img1)
ax[1].imshow(re_img2)
plt.show()
