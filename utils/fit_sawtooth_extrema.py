import numpy as np

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import least_squares
from numpy.polynomial import chebyshev as C


def _robust_cheb_fit(x, v, degree):
    """
    对 (x, v) 做鲁棒切比雪夫多项式拟合。

    Returns
    -------
    evaluate : callable
        evaluate(x_new) -> 拟合曲线在 x_new 处的值
    """
    if x.size <= degree:
        raise ValueError(f"包络拟合点数 ({x.size}) 不足以拟合 {degree} 阶多项式")

    x0, x1 = x.min(), x.max()

    def to_t(xx):
        # 归一化到 [-1, 1]，改善数值稳定性
        return (np.asarray(xx, dtype=float) - x0) / (x1 - x0) * 2 - 1

    V = C.chebvander(to_t(x), degree)

    # 初始最小二乘
    coef0 = np.linalg.lstsq(V, v, rcond=None)[0]

    # 用 MAD 估计噪声尺度，作为 soft_l1 的 f_scale
    res0 = V @ coef0 - v
    sigma = 1.4826 * np.median(np.abs(res0 - np.median(res0)))
    f_scale = max(sigma, 1e-12)

    # 鲁棒拟合，抑制端部畸变点或噪声尖峰
    fit = least_squares(lambda c: V @ c - v, coef0, loss="soft_l1", f_scale=f_scale)

    return lambda xn: C.chebvander(to_t(xn), degree) @ fit.x


def fit_sawtooth_extrema(
    data,
    extrema_count=191,
    smooth_window=9,
    poly_degree=3,
    prominence_ratio=0.05,
    envelope_degree=6,
):
    """
    对周期性锯齿波进行全局峰谷位置拟合，并拟合上包络。

    只拟合波峰构成的上包络；波谷位置处的包络值，由相邻波峰处的
    包络值线性插值得到（两端的波谷取最近波峰的值）。

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

    envelope_degree : int
        上包络（切比雪夫多项式）阶数，默认6。

    Returns
    -------
    positions : ndarray, shape (extrema_count,)
        全部峰谷的全局拟合位置（浮点样本索引，按位置升序）。

    envelope_values : ndarray, shape (extrema_count,)
        与 positions 一一对应的包络值。
        波峰处为上包络拟合值，波谷处由相邻波峰的包络值插值得到。
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
    # 3. 全局多项式拟合峰谷位置
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

    # --------------------------------------------------
    # 4. 上包络拟合（仅用波峰）
    #    x: 波峰的全局拟合位置；y: 平滑信号在波峰处的值
    # --------------------------------------------------

    peak_pos = global_positions[is_peak]
    peak_val = ys[extrema_idx[is_peak]]
    peak_val_mean = peak_val.mean()  # 波峰值的均值，用于反映该段数据对应转镜的反射率

    envelope = _robust_cheb_fit(peak_pos, peak_val, envelope_degree)

    # --------------------------------------------------
    # 5. 各峰谷位置处的包络值
    #    波峰：上包络拟合值
    #    波谷：由相邻波峰处的包络值，按位置线性插值（端点取最近波峰）
    # --------------------------------------------------

    envelope_values = np.empty(extrema_count, dtype=float)

    env_at_peaks = envelope(peak_pos)
    envelope_values[is_peak] = env_at_peaks
    envelope_values[~is_peak] = np.interp(
        global_positions[~is_peak], peak_pos, env_at_peaks
    )

    return global_positions, peak_val_mean, envelope_values
