import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

from utils.smask import twin_prime_s_row

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
img_data = img_pos_data - img_neg_data  # 计算差分信号


# 滤波重建
g_fft = fft.fft(g)
img_data_fft = fft.fft(img_data)

img_fft = img_data_fft / g_fft

img_recover = fft.ifft(img_fft)
img_recover = img_recover.reshape(p, q)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
ax[0].plot(np.abs(img_data_fft))
ax[1].plot(np.abs(g_fft))
ax[2].plot(np.abs(img_fft))
ax[3].imshow(np.real(img_recover))
plt.show()
