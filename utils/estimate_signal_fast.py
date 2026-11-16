"""
由扫描测量数据估计"所需位置"处的信号（线性叠加模型 + 最小二乘）。

【物理/数学模型】
    设所需位置 p_0 < p_1 < ... < p_{N-1} 处的真实信号为 s_0, ..., s_{N-1}。
    位于 p_k 与 p_{k+1} 之间的测量点 x_i，其测量值是两端信号的线性叠加：

        d_i = (1 - w_i) * s_k + w_i * s_{k+1},     w_i = (x_i - p_k) / (p_{k+1} - p_k)

    写成矩阵形式 d = A s。A 是 M×N 矩阵（M 为测量点数），每一行只有相邻两列
    （k 和 k+1）非零，值分别为 (1-w_i) 和 w_i。

【求解思路】
    最小二乘 min ||d - A s||^2 的法方程为 (A^T A) s = A^T d。
    由于 A 每行只涉及相邻两个未知量，A^T A 是 N×N 的【对称三对角矩阵】：
        对角线      : 第 j 个位置被所有"左端点是 j"或"右端点是 j"的测量点共同约束
        次对角线    : 第 j 与第 j+1 个位置仅通过它们之间的测量点耦合
    因此不需要构造稠密的 A（M×N，可能非常大），而是：
        1. 用 np.bincount 直接累加出三对角法方程          —— O(M)
        2. 用 scipy.linalg.solveh_banded 解三对角方程组     —— O(N)
        3. 用 LDL^T + Takahashi 递推求逆矩阵对角线（标准误差）—— O(N)
"""

import numpy as np
from scipy.linalg import solveh_banded, LinAlgError


def _inv_diag_tridiag(a, b):
    """
    求对称正定三对角矩阵的逆矩阵的【对角线元素】，复杂度 O(N)。

    参数
    ----
    a : (N,)   主对角线
    b : (N-1,) 次对角线（上下对称，所以只需给一条）

    原理
    ----
    对矩阵 T = L D L^T 做 LDL^T 分解，其中 L 是单位下双对角矩阵
    （对角线为 1，L[i+1, i] = l[i]），D = diag(d)。
    设 Z = T^{-1}，Takahashi 方程给出自下而上的递推：
        Z[N-1, N-1] = 1 / d[N-1]
        Z[i, i]     = 1 / d[i] + l[i]^2 * Z[i+1, i+1]
    我们只需要对角线（即每个估计量的方差因子），不需要整个逆矩阵。
    """
    N = len(a)
    d = np.empty(N)  # LDL^T 分解中的对角矩阵 D
    l = np.empty(N - 1)  # L 的次对角线

    # ---- 前向：LDL^T 分解 ----
    d[0] = a[0]
    for i in range(N - 1):
        l[i] = b[i] / d[i]  # L[i+1, i]
        d[i + 1] = a[i + 1] - l[i] * b[i]  # 消元后的主元

    # ---- 后向：Takahashi 递推，得到 inv(T) 的对角线 ----
    z = np.empty(N)
    z[-1] = 1.0 / d[-1]
    for i in range(N - 2, -1, -1):
        z[i] = 1.0 / d[i] + l[i] ** 2 * z[i + 1]
    return z


def estimate_signal(
    data,
    positions,
    x=None,
    weights=None,
    robust=False,
    edge="ignore",
    compute_err=False,
    huber_c=1.345,
    max_iter=30,
    tol=1e-8,
):
    """
    线性叠加模型 d(x) = (1-w) s_k + w s_{k+1} 的（加权 / 稳健）最小二乘估计。

    参数
    ----
    data        : (M,) 测量数据。
    positions   : (N,) 所需信号的位置，必须严格递增，单位与 x 一致。
    x           : (M,) 测量点的坐标；默认取采样点序号 0..M-1。可以不排序。
    weights     : (M,) 每个测量点的权重（通常取 1/σ_i^2）；默认等权。
    robust      : True 时用 Huber 损失的迭代重加权最小二乘（IRLS），抑制离群点/尖峰。
    edge        : 对 [positions[0], positions[-1]] 之外的测量点如何处理：
                  'ignore'：丢弃（默认，这些点不是"两个所需位置之间"的叠加）；
                  'extend'：保留，并假设它们只取决于最近的端点信号。
    compute_err : 是否计算标准误差（O(N)，开销很小）。
    huber_c     : Huber 阈值，单位为噪声尺度；1.345 对高斯噪声约保留 95% 效率。
    max_iter    : IRLS 最大迭代次数。
    tol         : IRLS 收敛阈值（相对于噪声尺度）。

    返回
    ----
    s     : (N,) 所需位置处的信号估计。
    s_err : (N,) 标准误差；compute_err=False 时为 None。
            robust=True 时只是近似值（见函数末尾注释）。
    res   : (K,) 拟合残差，对应实际被使用的那 K 个测量点。
    """
    # =====================================================================
    # 1. 输入整理与校验
    # =====================================================================
    d = np.asarray(data, float).ravel()
    p = np.asarray(positions, float).ravel()
    # 没给 x 时，默认测量点坐标就是采样序号 0,1,2,...
    x = np.arange(d.size, dtype=float) if x is None else np.asarray(x, float).ravel()
    # 没给权重时，所有点权重为 1（普通最小二乘）
    r0 = np.ones_like(d) if weights is None else np.asarray(weights, float).ravel()

    if not (d.size == x.size == r0.size):
        raise ValueError("data, x, weights 长度必须一致")
    # 区间 [p_k, p_{k+1}] 要有正的宽度，否则权重 w 的分母为 0
    if p.size < 2 or not np.all(np.diff(p) > 0):
        raise ValueError("positions 至少 2 个且必须严格递增")
    if edge not in ("ignore", "extend"):
        raise ValueError("edge 必须是 'ignore' 或 'extend'")
    N = p.size

    # =====================================================================
    # 2. 筛选可用的测量点
    # =====================================================================
    # 剔除 NaN/Inf 以及非正权重的点（否则一个 NaN 会污染整个解）
    ok = np.isfinite(d) & np.isfinite(x) & np.isfinite(r0) & (r0 > 0)
    if edge == "ignore":
        # 只保留位于首尾所需位置之间的点：只有这些点才满足"两端信号的线性叠加"
        ok &= (x >= p[0]) & (x <= p[-1])
    d, x, r0 = d[ok], x[ok], r0[ok]

    # =====================================================================
    # 3. 计算每个测量点所属区间 k 以及插值权重 w
    # =====================================================================
    # searchsorted(side='right') - 1：找到满足 p[k] <= x < p[k+1] 的 k。
    # clip 到 [0, N-2]：
    #   - x == p[-1] 时会得到 k = N-1，需要拉回最后一个区间（此时 w = 1）；
    #   - edge='extend' 下，x < p[0] 的点得到 k = -1，拉回第一个区间。
    k = np.clip(np.searchsorted(p, x, side="right") - 1, 0, N - 2)
    # 线性插值权重 w ∈ [0, 1]：w=0 表示完全等于 s_k，w=1 表示完全等于 s_{k+1}
    w = (x - p[k]) / (p[k + 1] - p[k])
    if edge == "extend":
        # 区间外的点 w 会 <0 或 >1；截断后它们等价于 d = s_0 或 d = s_{N-1}
        w = np.clip(w, 0.0, 1.0)
    # A 矩阵第 i 行的两个非零元：列 k[i] 处为 a0[i]，列 k[i]+1 处为 a1[i]
    a0, a1 = 1.0 - w, w

    # =====================================================================
    # 4. 内部函数：给定每点权重 r，组装并求解三对角法方程 (A^T R A) s = A^T R d
    # =====================================================================
    def solve(r):
        # A^T R A 的主对角线第 j 项 = Σ_i r_i * A[i,j]^2。
        #   - 左端点为 j 的测量点（k==j）贡献 r*(1-w)^2
        #   - 右端点为 j 的测量点（k+1==j）贡献 r*w^2
        # bincount(索引, 值, 长度) 等价于按索引分组求和，避免了稠密矩阵。
        diag = np.bincount(k, r * a0 * a0, N) + np.bincount(k + 1, r * a1 * a1, N)
        # 次对角线第 j 项 = Σ_i r_i * A[i,j] * A[i,j+1]：只有落在区间 j 内的点贡献 r*(1-w)*w
        off = np.bincount(k, r * a0 * a1, N - 1)
        # 右端项 A^T R d 的第 j 项 = Σ_i r_i * A[i,j] * d_i
        rhs = np.bincount(k, r * a0 * d, N) + np.bincount(k + 1, r * a1 * d, N)

        # 覆盖检查：若某个位置 j 的对角元（≈相邻区间的数据量）几乎为 0，
        # 说明它两侧没有足够数据，该信号在模型下不可估计，直接报错而不是返回垃圾值。
        if diag.min() <= 1e-12 * diag.max():
            bad = np.flatnonzero(diag <= 1e-12 * diag.max())
            raise ValueError(
                f"位置索引 {bad[:10].tolist()} 附近没有足够的测量数据，无法估计"
            )

        # solveh_banded 的"上三角带状存储"格式：
        #   ab[0, 1:] 为上次对角线（ab[0,0] 不使用，补 0），ab[1, :] 为主对角线。
        ab = np.vstack([np.r_[0.0, off], diag])
        try:
            s = solveh_banded(ab, rhs, lower=False)  # O(N) 的带状 Cholesky 求解
        except LinAlgError as e:
            raise ValueError("法方程奇异：测量点覆盖不足") from e
        return s, diag, off

    # =====================================================================
    # 5. 初次求解（普通/加权最小二乘）
    # =====================================================================
    r = r0.copy()
    s, diag, off = solve(r)
    # 残差 = 测量值 - 模型预测值，其中预测值 = (1-w) s_k + w s_{k+1}
    res = d - (a0 * s[k] + a1 * s[k + 1])
    # 噪声尺度（标准差）估计：sqrt(加权残差平方和 / 自由度)，自由度 = 点数 - 未知数个数
    # 除以平均权重是为了在 weights 不为 1 时仍给出"原始数据单位"的标准差。
    scale = np.sqrt(np.sum(r * res**2) / max(res.size - N, 1) / r.mean())

    # =====================================================================
    # 6. 可选：Huber 稳健回归（IRLS，迭代重加权最小二乘）
    # =====================================================================
    # Huber 损失：|残差| 小于阈值时像最小二乘，大于阈值时只按线性增长，
    # 因此离群点对结果的影响有上限。IRLS 通过反复调整每点权重来实现它。
    if robust:
        for _ in range(max_iter):
            # 稳健噪声尺度：MAD（中位数绝对偏差）×1.4826，对高斯噪声等价于标准差，
            # 但不会被离群点拉大。加 1e-300 防止尺度为 0 时除零。
            scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-300
            u = np.abs(res) / (huber_c * scale)  # 残差相对阈值的倍数
            # Huber 权：阈值内权重 1，阈值外按 1/u 衰减（残差越大权重越小）
            r = r0 * np.where(u <= 1, 1.0, 1.0 / u)
            s_new, diag, off = solve(r)
            res = d - (a0 * s_new[k] + a1 * s_new[k + 1])
            # 收敛判据：估计值最大变化量小于噪声尺度的 tol 倍
            done = np.max(np.abs(s_new - s)) < tol * scale
            s = s_new
            if done:
                break
        # 最终的稳健噪声尺度，用于标准误差
        scale = 1.4826 * np.median(np.abs(res - np.median(res)))

    # =====================================================================
    # 7. 标准误差
    # =====================================================================
    s_err = None
    if compute_err:
        # 估计量协方差 ≈ σ^2 (A^T R A)^{-1}，所以 Var(s_j) = σ^2 * [逆矩阵]_{jj}。
        # 三对角求逆对角线只需 O(N)；这里的 diag/off 正是最后一次迭代中的法方程矩阵。
        # 若用户给了 weights，法方程矩阵被权重缩放，需乘上平均权重把单位换回来。
        # 注意：robust=True 时这只是近似（把 Huber 权当作固定权重、忽略权重对残差的依赖），
        #       噪声相关时也会被低估，应视为量级参考而非严格置信区间。
        s_err = scale * np.sqrt(
            _inv_diag_tridiag(diag, off) * (r0.mean() if weights is not None else 1.0)
        )
    return s, s_err, res
