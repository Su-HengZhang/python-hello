import numpy as np
from numpy import fft

import scipy.fft as sfft
from numpy.lib.stride_tricks import sliding_window_view
from scipy.sparse.linalg import LinearOperator, gmres

from .smask import twin_prime_s_row


def direct_reconstruct(p, q, diff_data):
    """直接利用差分信号重建图像（不考虑 Envelope）"""

    N = p * q
    d = diff_data[:N]
    # 由孪生素数构造 S 循环矩阵的第1行，然后循环右移1位，得到 S 循环矩阵的最后1行
    s_row = np.roll(twin_prime_s_row(p, q).astype(float), 1)
    g_row = 1 - 2 * s_row  # 构造 G 循环矩阵的最后1行

    # 直接滤波重建
    g_row_fft = fft.fft(g_row)
    diff_data_fft = fft.fft(diff_data)
    re_img_fft = diff_data_fft / g_row_fft
    re_img_data = fft.ifft(re_img_fft)
    re_img_data = np.flip(re_img_data)

    re_img = np.real(re_img_data.reshape(p, q))
    return np.real(re_img)


def reconstruct_with_envelope(
    p,
    q,
    diff_data,
    envelope,
    rank_tol=1e-12,
    tol=1e-10,
    maxiter=200,
    return_info=False,
):
    """已知 Envelope 的精确重建（无需构造 N×N 矩阵，适用于 p*q ~ 1e5）。

    测量模型（pos: page*env, neg: (1-page)*env, **diff = neg - pos**）：
        diff[k] = sum_n x[n] * g[(n+k) % N] * env[j_k + (n % q)]
        g = 1 - 2 s,  j_k = k % step,  step = page_width - (q-1)
    令 c = roll(g, 1) 为 G 循环矩阵的最后一行，则 g[(n+k)%N] = c[(k - n') % N]，
    n' = N-1-n。因此 S[k,n] = g[(n+k)%N] 作用于 x，等价于 c 与 x 的逆序做循环卷积：
        S x = c ⊛ flip(x)
    （直接用最后一行滤波只能得到物体的 N 点逆序，即倒立的镜像，需整体 flip 一次。）

    A = sum_l diag(a_l[k%step]) · S · diag(b_l[n%q])，其中 a_l, b_l 来自 env[j+m] 的
    Hankel 矩阵的 SVD 低秩分解，光滑 Envelope 的秩很小，每次 A·x 只需 L 次 FFT。
    以 S 的精确逆（最后一行的 FFT 逆滤波 + 整体翻转）作预条件，用 GMRES 求解。

    Parameters
    ----------
    p, q      : 物体行数、列数（孪生素数，q = p + 2）
    diff_data : (>=N,) 差分信号 neg - pos，按窗口 k 顺序，多出的末页无效窗口会被截掉
    envelope  : (page_width,) 每页的固定包络
    rank_tol  : Envelope 低秩分解的相对奇异值截断阈值（越小越精确，秩越大）
    tol       : GMRES 相对残差容限
    Returns
    -------
    img : (p, q)，与送入仿真的 object_array 方向一致（若仿真前做过 np.flip，需自行翻回）
    """
    N = p * q
    # if envelope.shape != (page_width,):
    #     raise ValueError(f"envelope 长度 {envelope.shape} 与页宽 {page_width} 不符")
    page_width = len(envelope)
    step = page_width - (q - 1)
    # d = np.asarray(diff_data, dtype=float)
    d = diff_data[:N]

    # ---- Envelope 的 Hankel 矩阵 H[j, m] = env[j+m] 的低秩分解 ----
    H = sliding_window_view(envelope, q)[:step]  # (step, q)
    U, sv, Vt = np.linalg.svd(H, full_matrices=False)
    L = int((sv > rank_tol * sv[0]).sum())
    a = U[:, :L] * sv[:L]  # (step, L)   页内偏移方向
    b = Vt[:L]  # (L, q)      窗口列方向
    A_k = a.T[:, np.arange(N) % step]  # (L, N) : a_l[k % step]
    B_n = b[:, np.arange(N) % q]  # (L, N) : b_l[n % q]

    # ---- 用 G 循环矩阵的最后一行 c 作滤波器 ----
    g = 1.0 - 2.0 * twin_prime_s_row(p, q).astype(float)
    c_r = sfft.rfft(np.roll(g, 1))  # 只算一次，供 S_mul 与 P_mul 复用

    def S_mul(X):  # (S x)[k] = sum_n g[(n+k)%N] x[n] = (c ⊛ flip(x))[k]；X: (..., N)
        return sfft.irfft(c_r * sfft.rfft(X[..., ::-1], axis=-1), n=N, axis=-1)

    def A_mul(x):
        return np.sum(A_k * S_mul(B_n * x), axis=0)

    # ---- 预条件：S 的精确逆（env 恒为 1 时的滤波重建）----
    def P_mul(y):
        return sfft.irfft(sfft.rfft(y) / c_r, n=N)[::-1]

    A_op = LinearOperator((N, N), matvec=A_mul, dtype=float)
    M_op = LinearOperator((N, N), matvec=P_mul, dtype=float)

    x0 = P_mul(d)  # 忽略 Envelope 的初值
    it = [0]
    x, flag = gmres(
        A_op,
        d,
        x0=x0,
        M=M_op,
        rtol=tol,
        atol=0.0,
        restart=min(N, 50),
        maxiter=maxiter,
        callback=lambda _: it.__setitem__(0, it[0] + 1),
        callback_type="pr_norm",
    )
    if flag != 0:
        print(f"警告: GMRES 未收敛 (flag={flag})，可增大 maxiter 或放宽 tol")

    img = x.reshape(p, q)
    if return_info:
        res = np.linalg.norm(A_mul(x) - d) / np.linalg.norm(d)
        return img, {"rank": L, "gmres_iters": it[0], "rel_residual": res}
    return img
