"""
fit_sawtooth_extrema 使用示例（仅拟合上包络）

返回：
  positions        全部峰谷的位置
  envelope_values  对应位置的包络值（波峰处为上包络，波谷处由相邻波峰插值）
"""

import numpy as np
import matplotlib.pyplot as plt

from fit_sawtooth import fit_sawtooth_extrema


# ------------------------------------------------------
# 1. 生成合成数据（钟形包络 × 周期条纹 + 噪声，两端为平坦基线）
# ------------------------------------------------------
def make_demo_data(n=4800, n_extrema=191, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n)

    start, stop = 80, 4720
    n_periods = (n_extrema - 1) / 2  # 191 个极值 -> 95 个周期
    period = (stop - start) / n_periods

    env = np.exp(-(((x - 2400) / 1800) ** 2))

    y = np.full(n, 0.04)
    m = (x >= start) & (x <= stop)
    phase = 2 * np.pi * (x[m] - start) / period
    # 两端为波峰（共 96 峰 + 95 谷 = 191 个极值），谷值约 0.05
    y[m] = 0.05 + 0.22 * env[m] * (0.5 + 0.5 * np.cos(phase))

    y += rng.normal(0, 0.001, n)
    return y


data = make_demo_data()

# ------------------------------------------------------
# 2. 调用函数
# ------------------------------------------------------
positions, envelope_values = fit_sawtooth_extrema(
    data,
    extrema_count=191,
    envelope_degree=6,
)

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
    positions[is_peak], envelope_values[is_peak],
    s=14, c="tab:red", zorder=3, label="Envelope @ peaks (fitted)",
)
ax.scatter(
    positions[~is_peak], envelope_values[~is_peak],
    s=14, c="tab:orange", zorder=3, label="Envelope @ valleys (from adjacent peaks)",
)
ax.set_xlabel("Sample index")
ax.set_ylabel("Signal amplitude")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("example_fit_sawtooth.png", dpi=150)
plt.show()
