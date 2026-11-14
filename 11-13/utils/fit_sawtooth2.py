import numpy as np

from scipy.interpolate import PchipInterpolator


def regularized_spline_fit(
    initial_positions,
    lam=10.0,
    robust=True,
    max_iter=20,
    tolerance=1e-6,
):
    """
    二阶差分正则化的极值位置平滑拟合。

    Parameters
    ----------
    initial_positions : array_like
        按时间顺序排列的初始峰谷位置。

    lam : float
        正则化参数。
        越大，拟合结果越平滑。

    robust : bool
        是否启用鲁棒迭代加权，降低异常极值的影响。

    max_iter : int
        鲁棒迭代最大次数。

    tolerance : float
        迭代收敛阈值。

    Returns
    -------
    dict
        smooth_positions : 平滑后的极值位置
        intervals        : 相邻极值间距
        spline           : 连续位置插值函数
        weights          : 最终权重
    """

    x0 = np.asarray(initial_positions, dtype=float).ravel()

    N = len(x0)

    if N < 3 or not np.all(np.isfinite(x0)):
        raise ValueError("至少需要3个有效极值位置")

    if lam < 0:
        raise ValueError("lam不能小于0")

    if np.any(np.diff(x0) <= 0):
        raise ValueError("极值位置必须严格递增")

    # --------------------------------------------------
    # 1. 构造二阶差分矩阵
    # --------------------------------------------------

    D2 = np.zeros((N - 2, N))

    for i in range(N - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1

    # 正则化矩阵
    P = D2.T @ D2

    # 初始权重
    weights = np.ones(N)

    # --------------------------------------------------
    # 2. 正则化拟合及鲁棒迭代
    # --------------------------------------------------

    x = x0.copy()

    for iteration in range(max_iter if robust else 1):
        W = np.diag(weights)

        A = W + lam * P

        b = weights * x0

        # 求解线性方程
        x_new = np.linalg.solve(A, b)

        if not robust:
            x = x_new
            break

        # 计算拟合残差
        residual = x0 - x_new

        # MAD估计残差尺度
        median_residual = np.median(residual)

        mad = np.median(np.abs(residual - median_residual))

        sigma = 1.4826 * mad

        if sigma < 1e-12:
            x = x_new
            break

        # Huber权重
        u = np.abs(residual) / (1.345 * sigma)

        weights_new = np.ones(N)

        mask = u > 1

        weights_new[mask] = 1.0 / u[mask]

        # 检查收敛
        error = np.max(np.abs(x_new - x))

        x = x_new
        weights = weights_new

        if error < tolerance:
            break

    # --------------------------------------------------
    # 3. 保证极值位置严格递增
    # --------------------------------------------------

    if np.any(np.diff(x) <= 0):
        raise RuntimeError("平滑结果出现极值位置交叉，请减小lam或检查初始极值位置")

    # --------------------------------------------------
    # 4. 构造连续位置函数
    # --------------------------------------------------

    k = np.arange(N, dtype=float)

    spline = PchipInterpolator(k, x, extrapolate=False)

    # --------------------------------------------------
    # 5. 计算相邻极值间距
    # --------------------------------------------------

    intervals = np.diff(x)

    return {
        "smooth_positions": x,
        "intervals": intervals,
        "spline": spline,
        "weights": weights,
        "iterations": iteration + 1,
    }
