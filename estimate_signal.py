import numpy as np

import matplotlib.pyplot as plt


def estimate_signal(data, positions, x=None):
    """
    由扫描测量数据估计所需位置处的信号（线性叠加模型 + 最小二乘）。

    模型：位于 p_k 与 p_{k+1} 之间的测量值
        d(x) = (1-w) * s_k + w * s_{k+1},  w = (x-p_k)/(p_{k+1}-p_k)

    参数
    ----
    data      : (M,) 测量数据
    positions : (N,) 所需信号的位置（需单调递增，单位与 x 相同）
    x         : (M,) 测量点的坐标；默认为采样点序号 0..M-1

    返回
    ----
    s     : (N,) 所需位置处的信号估计
    s_err : (N,) 估计的标准误差（假设噪声独立同分布）
    resid : (K,) 拟合残差（仅含位于 [positions[0], positions[-1]] 内的测量点）
    """
    d = np.asarray(data, dtype=float)
    p = np.asarray(positions, dtype=float)
    x = np.arange(len(d), dtype=float) if x is None else np.asarray(x, dtype=float)
    N = len(p)

    # 只使用位于两端所需位置之间的测量点
    inside = (x >= p[0]) & (x <= p[-1])
    xi, di = x[inside], d[inside]

    # 构造稀疏结构的设计矩阵 A：每行只有相邻两个非零权重
    k = np.clip(np.searchsorted(p, xi, side="right") - 1, 0, N - 2)
    w = (xi - p[k]) / (p[k + 1] - p[k])
    A = np.zeros((len(xi), N))
    rows = np.arange(len(xi))
    A[rows, k] = 1 - w
    A[rows, k + 1] = w

    # 最小二乘求解
    s, *_ = np.linalg.lstsq(A, di, rcond=None)
    resid = di - A @ s

    # 标准误差：sigma * sqrt(diag((A^T A)^-1))
    sigma = resid.std(ddof=N)
    s_err = sigma * np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    return s, s_err, resid


if __name__ == "__main__":
    data = np.load("./data/data.npy")
    positions = np.load("./data/positions.npy")
    x = np.arange(len(data))

    s_ls, s_err, resid = estimate_signal(data, positions)
    s_interp = np.interp(positions, x, data)  # 直接插值

    print(f"残差 RMS = {resid.std():.4g}, 平均标准误差 = {s_err.mean():.4g}")
    print(f"最小二乘与直接插值差异 RMS = {np.std(s_ls - s_interp):.4g}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(x, data, color="0.7", lw=0.8, label="Raw data")
    ax.plot(
        positions,
        s_interp,
        "s--",
        ms=3,
        lw=0.8,
        color="tab:orange",
        label="Direct interpolation",
    )
    ax.errorbar(
        positions,
        s_ls,
        yerr=s_err,
        fmt="o-",
        ms=3,
        lw=1,
        color="tab:blue",
        capsize=1.5,
        label="Least-squares estimate (±1σ)",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Signal")
    ax.set_title("Raw data vs. estimated signals")
    ax.legend()

    # 局部放大
    ax = axes[1]
    lo, hi = 1500, 2300
    m = (x >= lo) & (x <= hi)
    mp = (positions >= lo) & (positions <= hi)
    ax.plot(x[m], data[m], ".", color="0.6", ms=3, label="Raw data")
    ax.plot(
        positions[mp],
        s_interp[mp],
        "s--",
        ms=5,
        lw=1,
        color="tab:orange",
        label="Direct interpolation",
    )
    ax.errorbar(
        positions[mp],
        s_ls[mp],
        yerr=s_err[mp],
        fmt="o-",
        ms=5,
        lw=1.2,
        color="tab:blue",
        capsize=2,
        label="Least-squares estimate (±1σ)",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Signal")
    ax.set_title(f"Zoom: {lo}–{hi}")
    ax.legend()

    plt.tight_layout()
    # plt.savefig("comparison.png", dpi=150)
    # np.save("signal_estimate.npy", np.vstack([positions, s_ls, s_err]))
    plt.show()
