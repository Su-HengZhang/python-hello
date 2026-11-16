import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr


def estimate_signals(data, positions):
    """
    根据线性叠加模型，联合估计指定位置的真实信号。

    参数
    ----
    data : ndarray
        实验测量数据，一维数组。
    positions : ndarray
        实际所需信号的位置，可以是非整数。

    返回
    ----
    signals : ndarray
        估计得到的目标信号。
    fitted : ndarray
        模型在参与拟合的测量位置上的预测值。
    residuals : ndarray
        测量值与预测值之间的残差。
    sample_indices : ndarray
        参与拟合的原始数据索引。
    """

    data = np.asarray(data, dtype=float)
    positions = np.asarray(positions, dtype=float)

    if data.ndim != 1 or positions.ndim != 1:
        raise ValueError("输入必须是一维数组")

    if len(positions) < 2:
        raise ValueError("至少需要两个目标位置")

    if not np.all(np.diff(positions) > 0):
        raise ValueError("positions 必须严格递增")

    if positions[0] < 0 or positions[-1] >= len(data):
        raise ValueError("目标位置超出数据范围")

    # 只使用第一个与最后一个目标位置之间的测量数据
    sample_indices = np.arange(
        int(np.ceil(positions[0])), int(np.floor(positions[-1])) + 1
    )

    n_samples = len(sample_indices)
    n_signals = len(positions)

    # 构建稀疏矩阵 A
    A = lil_matrix((n_samples, n_signals), dtype=float)

    for row, k in enumerate(sample_indices):

        # 找到测量点 k 所属的目标区间
        i = np.searchsorted(positions, k, side="right") - 1

        # 计算线性叠加系数
        alpha = (k - positions[i]) / (positions[i + 1] - positions[i])

        A[row, i] = 1.0 - alpha
        A[row, i + 1] = alpha

    A = A.tocsr()

    # 全局最小二乘求解
    result = lsqr(A, data[sample_indices], atol=1e-13, btol=1e-13, iter_lim=10000)

    signals = result[0]

    # 计算拟合值和残差
    fitted = A @ signals
    residuals = data[sample_indices] - fitted

    return signals, fitted, residuals, sample_indices


if __name__ == "__main__":

    # 读取实验数据
    data = np.load("./data/data.npy")
    positions = np.load("./data/positions.npy")

    # 联合估计
    signals, fitted, residuals, indices = estimate_signals(data, positions)

    # 保存结果
    # np.save("estimated_signals.npy", signals)
    # np.save("fitted_data.npy", fitted)
    # np.save("residuals.npy", residuals)

    # 输出误差
    rmse = np.sqrt(np.mean(residuals**2))

    print("目标信号数量:", len(signals))
    print("RMSE:", rmse)
    print("信号范围:", signals.min(), signals.max())

    # 绘制估计结果
    plt.figure(figsize=(12, 5))

    plt.plot(
        np.arange(len(data)),
        data,
        color="gray",
        linewidth=0.7,
        alpha=0.6,
        label="Measured data",
    )

    plt.plot(
        positions, signals, "ro-", markersize=3, linewidth=1, label="Estimated signals"
    )

    plt.xlabel("Position")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 绘制拟合残差
    plt.figure(figsize=(12, 3))

    plt.plot(indices, residuals, linewidth=0.7)

    plt.xlabel("Position")
    plt.ylabel("Residual")
    plt.title("Fitting residuals")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
