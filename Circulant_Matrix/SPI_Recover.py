import numpy as np
from numpy import fft
from PIL import Image
from matplotlib import pyplot as plt

from twin_prime_s_row import s_matrix, g_matrix


# 前50对孪生素数：
# (3, 5) (5, 7) (11, 13) (17, 19) (29, 31) (41, 43) (59, 61) (71, 73)
# (101, 103) (107, 109) (137, 139) (149, 151) (179, 181) (191, 193)
# (197, 199) (227, 229) (239, 241) (269, 271) (281, 283) (311, 313)
# (347, 349) (419, 421) (431, 433) (461, 463) (521, 523) (569, 571)
# (599, 601) (617, 619) (641, 643) (659, 661) (809, 811) (821, 823)
# (827, 829) (857, 859) (881, 883) (1019, 1021) (1031, 1033) (1049, 1051)
# (1061, 1063) (1091, 1093) (1151, 1153) (1229, 1231) (1277, 1279)
# (1289, 1291) (1301, 1303) (1319, 1321) (1427, 1429) (1451, 1453)
# (1481, 1483) (1487, 1489)

# 矩阵大小
p, q = 59, 61  # 孪生素数对
N = p * q

# 读取物体图像
obj = Image.open("hbar.tif")
# obj = Image.open("./lena.bmp")
obj = obj.resize((q, p), resample=Image.Resampling.BICUBIC)
# obj = obj.resize((q, p), resample=Image.Resampling.NEAREST)
obj = np.array(obj).astype(np.float64)
obj = obj / 255
obj_vector = obj.reshape(N, -1)

obj_mean = obj_vector.mean()

# 纯白物体
# obj_vector = np.ones_like(obj_vector)

# 生成 S 循环矩阵 和 其互补矩阵, 以及 相应的 G 矩阵
s = s_matrix(p, q).astype(np.float64)
s_pos = s
s_neg = 1 - s_pos

g = g_matrix(p, q)

# S, G 循环矩阵最后一列的频谱
s_last_row = s[:, -1]
s_spec = fft.fft(s_last_row)

g_last_row = g[:, -1]
g_spec = fft.fft(g_last_row)

# 线性测量
y_pos = s_pos @ obj_vector / ((N + 1) / 2)
y_neg = s_neg @ obj_vector / ((N + 1) / 2)

y_pos_spec = fft.fft(y_pos)

print(y_pos.max(), y_pos.min(), y_pos.mean())
print(y_neg.max(), y_neg.min(), y_neg.mean())

# 添加噪声
rng1 = np.random.default_rng(seed=17)
rng2 = np.random.default_rng(seed=4)
sigma = 0.1 * obj_mean

noise_pos = rng1.normal(loc=0, scale=sigma, size=(N, 1))
noise_neg = rng2.normal(loc=0, scale=sigma, size=(N, 1))

y_pos_noise = y_pos + noise_pos
y_neg_noise = y_neg + noise_neg

y_pos_noise_spec = fft.fft(y_pos_noise)

# 差分
y = y_pos_noise - y_neg_noise

y_spec = fft.fft(y)

# 由 S 关联重建
img_s = np.zeros((p, q), dtype=np.float64)

for i in range(N):
    mask = np.reshape(s_pos[i], (p, q))
    img_s += mask * y_pos_noise[i]
img_s = img_s / N

img_s = img_s / (np.mean(y_pos_noise) * np.mean(s_pos, axis=0).reshape((p, q)))

# 由 G 关联重建
img_g = np.zeros((p, q), dtype=np.float64)

for i in range(N):
    mask = np.reshape(g[i], (p, q))
    img_g += mask * y[i]
img_g = img_g / N

img_g = img_g / (np.mean(y) * np.mean(g, axis=0).reshape((p, q)))

fig = plt.figure(figsize=(12, 4))
# 创建2行4列的GridSpec
gs = fig.add_gridspec(2, 4)

# 左上：第一行，0~2列（3列合并）
ax_top = fig.add_subplot(gs[0, 0:3])
# 左下：第二行，0~2列（3列合并）
ax_bot = fig.add_subplot(gs[1, 0:3])
# 右侧：两行合并，第3列
ax_im_s = fig.add_subplot(gs[0, 3])
ax_im_g = fig.add_subplot(gs[1, 3])

ax_top.plot(y_pos_noise, label="y_pos")
ax_bot.plot(y_neg_noise, label="y_neg")
ax_im_s.imshow(img_s)
ax_im_g.imshow(img_g)


fig1, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
axs[0].plot(np.real(y_pos_spec), label="y_pos_spec")
axs[1].plot(np.real(y_pos_noise_spec), label="y_pos_spec")
axs[2].plot(np.real(s_spec), label="s_spec")
axs[3].plot(np.real(g_spec), label="g_spec")


plt.show()


# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))

# axs[0].plot(y_pos, label="y_pos")
# axs[1].plot(y_neg, label="y_neg")
# axs[2].imshow(img_s)

# plt.show()


# # 验证 S 循环矩阵的性质 S @ S.T = (N / 4) * (I + J)
# s = s_matrix(p, q)
# s = s.astype(np.int64)
# if np.array_equal(s @ s.T, (N / 4) * (np.eye(M) + np.ones((M, M)))):
#     print(f"S matrix is valid.\n S @ S.T = \n{s @ s.T}")

# # 验证 由 S 循环矩阵生成的Hadamard矩阵的性质 H @ H.T = N * I
# H = hadamard_matrix(s)
# H = H.astype(np.int64)
# if np.array_equal(H @ H.T, N * np.eye(N)):
#     print(f"Hadamard matrix is valid.\n H @ H.T = \n{H @ H.T}")

# # 验证 S 循环矩阵的快速傅里变换重建性质
# first_row = twin_prime_s_row(p, q)
# print(f"First row of S matrix:\n {first_row.reshape((p, q))}")
# # 左移 M-1 位 <=> 右移1位   得到最后一行
# # last_row = np.roll(first_row, -(M - 1))
# last_row = np.roll(first_row, 1)
# print(f"Last row of S matrix:\n {last_row.reshape((p, q))}")
# # 左移 M-1 位 <=> 右移1位   得到最后一行

# # 计算 S 循环矩阵最后一行的离散傅里叶变换
# spectrum_last_row = fft.fft(last_row)

# # 目标物体
# f = np.arange(M) + 1j * np.random.randint(0, 10, M)

# # 线性测量
# g = s @ f

# # 测量的线傅里叶变换
# spectrum_g = fft.fft(g)

# # 重建
# f_reverse = fft.ifft(spectrum_g / spectrum_last_row)
# f_r = f_reverse[::-1]  # 反转序列

# if np.allclose(f, f_r):
#     print("FFT reconstruction is valid.")
#     residual = np.linalg.norm(f - f_r)
#     print(residual)
