import numpy as np

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares
from numpy.polynomial import chebyshev as C


def _robust_poly_envelope(x, v, x_eval, degree):
    """
    对一组 (x, v) 做鲁棒切比雪夫多项式拟合，并在 x_eval 上求值。

    x_eval 中落在 [x.min(), x.max()] 之外的点返回 NaN（不外推）。

    Returns
    -------
    env : ndarray, 与 x_eval 等长
    coef : ndarray, 切比雪夫系数（自变量已归一化到 [-1, 1]）
    """
    if x.size <= degree:
        raise ValueError(f"包络拟合点数 ({x.size}) 不足以拟合 {degree} 阶多项式")

    x0, x1 = x.min(), x.max()

    def to_t(xx):
        return (xx - x0) / (x1 - x0) * 2 - 1

    V = C.chebvander(to_t(x), degree)

    # 初始最小二乘
    coef0 = np.linalg.lstsq(V, v, rcond=None)[0]

    # 用 MAD 估计噪声尺度，作为 soft_l1 的 f_scale
    res0 = V @ coef0 - v
    sigma = 1.4826 * np.median(np.abs(res0 - np.median(res0)))
    f_scale = max(sigma, 1e-12)

    # 鲁棒拟合，抑制端部畸变点或噪声尖峰
    fit = least_squares(
        lambda c: V @ c - v, coef0, loss="soft_l1", f_scale=f_scale
    )

    env = np.full(x_eval.shape, np.nan)
    inside = (x_eval >= x0) & (x_eval <= x1)
    env[inside] = C.chebvander(to_t(x_eval[inside]), degree) @ fit.x

    return env, fit.x


def fit_sawtooth_extrema(
    data,
    extrema_count=191,
    smooth_window=9,
    poly_degree=3,
    prominence_ratio=0.05,
    fit_envelope=False,
    envelope_degree=6,
):
    """
    对周期性锯齿波进行全局峰谷位置拟合，并可选地拟合上/下包络。

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

    fit_envelope : bool
        是否拟合包络。False 时保持原有行为，只返回 global_positions。

    envelope_degree : int
        包络（切比雪夫多项式）阶数，默认6。

    Returns
    -------
    fit_envelope=False:
        global_positions : ndarray
            全局模型预测的峰谷位置（浮点样本索引）。

    fit_envelope=True:
        result : dict
            positions      全局模型预测的峰谷位置
            is_peak        bool 数组，True 为波峰、False 为波谷
            extrema_values 各极值处的（平滑后）幅值
            x              样本索引 0..N-1
            upper / lower  上/下包络，长度 N；
                           超出对应极值范围的位置为 NaN
            center         (upper + lower) / 2，中心线（直流分量）
            amplitude      (upper - lower) / 2，半幅值
            visibility     (upper - lower) / (upper + lower)，条纹可见度
            upper_coef / lower_coef
                           包络的切比雪夫系数
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

    # 合并峰谷，按照位置排序，同时记录每个极值是峰还是谷
    all_idx = np.concatenate((peaks, valleys))
    all_is_peak = np.concatenate(
        (np.ones(len(peaks), dtype=bool), np.zeros(len(valleys), dtype=bool))
    )
    order = np.argsort(all_idx)
    extrema_idx = all_idx[order]
    is_peak = all_is_peak[order]

    initial_positions = extrema_idx.astype(float)

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

    if not fit_envelope:
        return global_positions

    # --------------------------------------------------
    # 4. 包络拟合
    #    上包络：所有波峰的 (位置, 幅值)
    #    下包络：所有波谷的 (位置, 幅值)
    #    位置采用第3步的全局拟合位置（已去噪、等间隔性更好），
    #    幅值取平滑信号在检测到的极值点处的值。
    # --------------------------------------------------

    extrema_values = ys[extrema_idx]

    x_all = np.arange(y.size, dtype=float)

    upper, upper_coef = _robust_poly_envelope(
        global_positions[is_peak],
        extrema_values[is_peak],
        x_all,
        envelope_degree,
    )
    lower, lower_coef = _robust_poly_envelope(
        global_positions[~is_peak],
        extrema_values[~is_peak],
        x_all,
        envelope_degree,
    )

    center = 0.5 * (upper + lower)
    amplitude = 0.5 * (upper - lower)

    with np.errstate(divide="ignore", invalid="ignore"):
        visibility = (upper - lower) / (upper + lower)

    return {
        "positions": global_positions,
        "is_peak": is_peak,
        "extrema_values": extrema_values,
        "x": x_all,
        "upper": upper,
        "lower": lower,
        "center": center,
        "amplitude": amplitude,
        "visibility": visibility,
        "upper_coef": upper_coef,
        "lower_coef": lower_coef,
    }
