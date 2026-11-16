import numpy as np
from numpy.polynomial import legendre as L
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares


def fit_sawtooth_extrema(y, n_extrema=191, phase_order=3, env_order=5,
                         n_harm=4, prominence_frac=0.15, return_model=False):
    """
    整体拟合周期(锯齿/准正弦)波,返回所有峰谷的亚像素位置。

    模型:  y(x) = B(u) + A(u) * [cos(phi) + sum_{h=2..H} (a_h cos(h phi) + b_h sin(h phi))]
      phi(u) = 2*pi*(u_idx/T + k0 + 慢变多项式修正)   (周期固定为主,允许缓慢渐变)
      A(u), B(u): Legendre 多项式包络/基线
    用 soft_l1 鲁棒损失抑制噪声与毛刺(如 x~4200 处的尖峰)。
    拟合完成后在无噪的模型曲线上求局部极值,得到峰谷位置。

    参数
    ----
    y : 1D array          原始数据
    n_extrema : int       峰谷总数(默认191)
    phase_order : int     相位慢变多项式阶数(0=严格固定周期,2~3允许缓慢渐变)
    env_order : int       幅度/基线包络多项式阶数
    n_harm : int          波形谐波数(越大越能描述锯齿的不对称)
    return_model : bool   是否同时返回 (x_roi, y_fit)

    返回
    ----
    positions : ndarray (n_extrema,)  峰谷位置(升序,浮点)
    kinds     : ndarray (n_extrema,)  +1=峰, -1=谷
    [x_roi, y_fit]  (若 return_model)
    """
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)

    # ---------- 1. 粗估:有效区间、周期、初相 ----------
    ys = uniform_filter1d(y, 9)
    pk, _ = find_peaks(ys, prominence=prominence_frac * np.ptp(y))
    if len(pk) < 5:
        raise ValueError("粗略峰检测失败,请调整 prominence_frac")
    k = np.arange(len(pk))
    slope, icpt = np.polyfit(k, pk, 1)           # 峰位 ≈ icpt + slope*k
    T0 = slope
    # 期望: n_extrema 个极值 => (n_extrema-1)/2 个周期; 以此约束有效区间
    span = (n_extrema - 1) / 2 * T0
    first = pk[0]
    lo = max(0, int(first - 0.5 * T0))
    hi = min(len(y), int(first + span + 0.5 * T0) + 1)
    xr, yr = x[lo:hi], y[lo:hi]
    xc, xh = 0.5 * (xr[0] + xr[-1]), 0.5 * (xr[-1] - xr[0])
    u = (xr - xc) / xh

    # ---------- 2. 整体非线性最小二乘 ----------
    nA, nB, nP = env_order + 1, env_order + 1, phase_order  # 相位修正从2阶起(1阶并入T)
    def unpack(p):
        i = 0
        A = p[i:i+nA]; i += nA
        B = p[i:i+nB]; i += nB
        ah = p[i:i+n_harm-1]; i += n_harm-1
        bh = p[i:i+n_harm-1]; i += n_harm-1
        T, ph0 = p[i], p[i+1]; i += 2
        c = p[i:i+nP]
        return A, B, ah, bh, T, ph0, c

    def phase(p, uu, xx):
        A, B, ah, bh, T, ph0, c = unpack(p)
        ph = 2*np.pi*(xx - xc)/T + ph0
        if nP > 0:   # c_j * P_{j+1}(u), j>=1 -> 慢变相位
            ph = ph + L.legval(uu, np.r_[0, 0, c]) if nP > 0 else ph
        return ph

    def model(p, uu, xx):
        A, B, ah, bh, T, ph0, c = unpack(p)
        ph = phase(p, uu, xx)
        w = np.cos(ph)
        for h in range(2, n_harm+1):
            w = w + ah[h-2]*np.cos(h*ph) + bh[h-2]*np.sin(h*ph)
        return L.legval(uu, B) + L.legval(uu, A)*w

    # 初值: 峰位 ph = 0 (mod 2pi)
    A0 = np.zeros(nA); A0[0] = 0.5*np.ptp(yr)*0.5
    B0 = np.zeros(nB); B0[0] = 0.5*(yr.max()+yr.min())*0.9
    ph00 = -2*np.pi*(first - xc)/T0
    p0 = np.r_[A0, B0, np.zeros(2*(n_harm-1)), T0, ph00, np.zeros(nP)]

    # 先固定相位(只拟线性部分+T,ph0)以稳健收敛,再放开全部
    res = least_squares(lambda p: model(p, u, xr) - yr, p0,
                        loss="soft_l1", f_scale=0.01 * np.ptp(yr) / 0.3 * 0.3,
                        x_scale="jac", max_nfev=400)
    p = res.x

    # ---------- 3. 在无噪模型上求极值 ----------
    xf = np.arange(xr[0], xr[-1], 0.02)
    uf = (xf - xc) / xh
    yf = model(p, uf, xf)
    d = np.diff(yf)
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    kinds = np.where(d[idx-1] > 0, 1, -1)
    pos = xf[idx]
    # 去除边缘处因多项式包络产生的伪极值: 只保留最接近 n_extrema 的连续峰谷序列
    if len(pos) != n_extrema:
        # 以拟合相位为准重新定位: 峰 phi_fund=2*pi*m, 谷 phi_fund=2*pi*m+pi 附近取极值
        keep = np.ones(len(pos), bool)
        # 选取幅度最大的 n_extrema 个(伪极值的局部起伏很小)
        if len(pos) > n_extrema:
            Aenv = np.abs(L.legval((pos-xc)/xh, unpack(p)[0]))
            order = np.argsort(-Aenv)[:n_extrema]
            keep[:] = False; keep[order] = True
        pos, kinds = pos[keep], kinds[keep]
    if len(pos) != n_extrema:
        print(f"警告: 找到 {len(pos)} 个极值, 期望 {n_extrema}; 可调 env_order/n_harm/phase_order")

    if return_model:
        return pos, kinds, xf, yf
    return pos, kinds
