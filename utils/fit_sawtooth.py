import numpy as np

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares


def fit_sawtooth_extrema(
    data,
    extrema_count=191,
    smooth_window=9,
    poly_degree=3,
    prominence_ratio=0.05,
):
    """
    对周期性锯齿波进行全局峰谷位置拟合。

    Parameters
    ----------
    data : ndarray
        原始一维数据。

    extrema_count : int
        峰谷总数，当前数据为191。

    smooth_window : int
        Savitzky-Golay平滑窗口，必须为奇数。

    poly_degree : int
        全局位置拟合多项式阶数，默认3。

    prominence_ratio : float
        极值显著性阈值与信号幅度范围的比例。


    Returns
    -------
    result : ndarray
        global_fit_positions 全局模型预测位置
    """

    y = np.asarray(data, dtype=float).ravel()

    if y.size < 10 or not np.all(np.isfinite(y)):
        raise ValueError("输入数据必须是一维且不含 NaN/Inf")

    # --------------------------------------------------
    # 1. Savitzky-Golay 平滑
    # --------------------------------------------------

    if smooth_window % 2 == 0:
        smooth_window += 1

    smooth_window = min(smooth_window, y.size if y.size % 2 else y.size - 1)

    if smooth_window < 5:
        raise ValueError("数据长度不足")

    ys = savgol_filter(y, smooth_window, 3)

    # --------------------------------------------------
    # 2. 检测初始峰谷位置
    # --------------------------------------------------

    # 根据预期极值数量, 估算最小间隔
    distance = max(3, int(len(y) / (extrema_count + 1) * 0.9))

    prominence = prominence_ratio * np.ptp(ys)

    peaks, _ = find_peaks(ys, distance=distance, prominence=prominence)

    valleys, _ = find_peaks(-ys, distance=distance, prominence=prominence)

    # 合并峰谷，按照位置排序
    initial_positions = np.sort(np.concatenate((peaks, valleys))).astype(float)

    # 检查是否检测到预期数量的极值
    if len(initial_positions) != extrema_count:
        raise RuntimeError(
            f"检测到 {len(initial_positions)} 个极值，"
            f"其中波峰 {len(peaks)} 个，"
            f"波谷 {len(valleys)} 个；"
            f"预期 {extrema_count} 个。"
            "请调整 prominence_ratio。"
        )

    # --------------------------------------------------
    # 3. 全局多项式拟合
    # --------------------------------------------------

    k = np.arange(extrema_count, dtype=float)

    # 归一化，改善多项式拟合的数值稳定性
    z = (k - k.mean()) / np.ptp(k) * 2

    X = np.polynomial.polynomial.polyvander(z, poly_degree)

    # 初始最小二乘拟合
    coef0 = np.linalg.lstsq(X, initial_positions, rcond=None)[0]

    # 鲁棒拟合，抑制异常点
    fit = least_squares(
        lambda coef: X @ coef - initial_positions, coef0, loss="soft_l1", f_scale=1.0
    )

    global_positions = X @ fit.x

    return global_positions
