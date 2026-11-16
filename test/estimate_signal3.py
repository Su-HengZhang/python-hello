import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def estimate_signals(data, positions):
    """
    利用向量化稀疏矩阵构建和 LSQR，
    联合估计所有目标位置的信号。

    Parameters
    ----------
    data : ndarray
        实验测量数据。
    positions : ndarray
        实际目标位置，可以是非整数。

    Returns
    -------
    signals : ndarray
        目标位置估计信号。
    fitted : ndarray
        模型预测的测量数据。
    residuals : ndarray
        拟合残差。
    sample_indices : ndarray
        参与拟合的原始数据索引。
    """

    data = np.asarray(data, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    if data.ndim != 1 or positions.ndim != 1:
        raise ValueError("输入必须是一维数组")

    if len(positions) < 2:
        raise ValueError("至少需要两个目标位置")

    if not np.all(np.isfinite(data)):
        raise ValueError("data 中存在 NaN 或 Inf")

    if not np.all(np.isfinite(positions)):
        raise ValueError("positions 中存在 NaN 或 Inf")

    if np.any(np.diff(positions) <= 0):
        raise ValueError("positions 必须严格递增")

    if positions[0] < 0 or positions[-1] >= len(data):
        raise ValueError("目标位置超出数据范围")

    # 1. 确定参与计算的测量数据
    sample_indices = np.arange(
        int(np.ceil(positions[0])), int(np.floor(positions[-1])) + 1
    )

    # 2. 向量化确定区间编号
    interval = np.searchsorted(positions, sample_indices, side="right") - 1

    # 3. 向量化计算线性叠加权重
    alpha = (sample_indices - positions[interval]) / (
        positions[interval + 1] - positions[interval]
    )

    # 4. 向量化构建 COO 稀疏矩阵
    n_samples = len(sample_indices)
    n_signals = len(positions)

    rows = np.repeat(np.arange(n_samples), 2)

    cols = np.column_stack((interval, interval + 1)).ravel()

    values = np.column_stack((1.0 - alpha, alpha)).ravel()

    A = coo_matrix((values, (rows, cols)), shape=(n_samples, n_signals)).tocsr()

    # 5. 全局最小二乘求解
    result = lsqr(A, data[sample_indices], atol=1e-13, btol=1e-13, iter_lim=10000)

    signals = result[0]

    # 6. 计算拟合值与残差
    fitted = A @ signals
    residuals = data[sample_indices] - fitted

    return signals, fitted, residuals, sample_indices
