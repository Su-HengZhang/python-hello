import numpy as np


def legendre_table(m: int) -> np.ndarray:
    """
    生成模 m 的 Legendre 符号查找表。

    tbl[n]：
        0   -> n ≡ 0 (mod m)
        1   -> n 是模 m 的二次剩余
       -1   -> n 是模 m 的二次非剩余
    """

    # 只需要 -1、0、1，因此使用 int8 节省内存。
    tbl = np.full(m, -1, dtype=np.int8)

    # 只计算一半的平方数，因为：
    # r² ≡ (-r)² (mod m)
    r = np.arange(1, (m + 1) // 2, dtype=np.int64)

    # 标记所有二次剩余。
    tbl[(r * r) % m] = 1

    # 0 的 Legendre 符号为 0。
    tbl[0] = 0

    return tbl


def twin_prime_s_row(p: int, q: int) -> np.ndarray:
    """
    根据孪生素数 p、q 构造长度 p*q 的 0/1 序列。

    要求：
        p < q
        q = p + 2
        p、q 一对孪生素数
    """

    # ------------------------------------------------------------
    # Legendre 查找表
    # ------------------------------------------------------------
    lp = legendre_table(p)
    lq = legendre_table(q)

    # ------------------------------------------------------------
    # 利用周期结构直接构造
    #
    # 对于 n = 0, 1, ..., p*q-1：
    #
    #     chi_p(n) = lp[n % p]
    #     chi_q(n) = lq[n % q]
    #
    # 只利用两个周期 p 和 q。
    # ------------------------------------------------------------

    # 每个 p 周期的 Legendre 符号，重复 q 次
    chi_p = np.tile(lp, q)

    # 每个 q 周期的 Legendre 符号，重复 p 次
    chi_q = np.tile(lq, p)

    # ------------------------------------------------------------
    # 构造最终二值序列
    result = np.ones(p * q, dtype=np.int8)

    # chi_p * chi_q == 1 的位置最终为 0。
    result[chi_p * chi_q == 1] = 0

    # q 的倍数最终也为 0。
    #
    # 因为 chi_q == 0 恰好对应：
    #
    #     n = 0, q, 2q, 3q, ...
    #
    result[::q] = 0

    return result


def s_matrix(p: int, q: int) -> np.ndarray:
    """
    根据孪生素数 p、q 构造 S 矩阵。

    """
    N = p * q
    s_matrix = np.empty((N, N), dtype=np.int8)

    first_row = twin_prime_s_row(p, q)

    for k in range(N):
        s_matrix[k] = np.roll(first_row, -k)

    return s_matrix


def hadamard_matrix(s_matrix: np.ndarray) -> np.ndarray:
    """
    根据 S 矩阵构造 Hadamard 矩阵。

    """
    N = s_matrix.shape[0]
    J = np.ones((N, N), dtype=np.int8)
    G = J - 2 * s_matrix
    H = np.ones((N + 1, N + 1), dtype=np.int8)
    H[1:, 1:] = G

    return H


if __name__ == "__main__":

    from numpy import fft

    # 前50对孪生素数：
    # (3, 5) (5, 7) (11, 13) (17, 19) (29, 31) (41, 43) (59, 61) (71, 73)
    # (101, 103) (107, 109) (137, 139) (149, 151) (179, 181) (191, 193)
    # (197, 199) (227, 229) (239, 241) (269, 271) (281, 283) (311, 313)
    # (347, 349) (419, 421) (431, 433) (461, 463) (521, 523) (569, 571)
    # (599, 601) (617, 619) (641, 643) (659, 661) (809, 811) (821, 823)
    # (827, 829) (857, 859) (881, 883) (1019, 1021) (1031, 1033) (1049, 1051)
    # (1061, 1063) (1091, 1093) (1151, 1153) (1229, 1231) (1277, 1279)
    # (1289, 1291) (1301, 1303) (1319, 1321) (1427, 1429) (1451, 1453)
    # (1481, 1483) (1487, 1489)

    # 测试
    p, q = 11, 13  # 孪生素数对

    M = p * q  # S 循环矩阵的阶数
    print(f"S matrix order: {M}")
    N = M + 1  # Hadamard 矩阵的阶数
    print(f"Hadamard matrix order: {N}")

    # 验证 S 循环矩阵的性质 S @ S.T = (N / 4) * (I + J)
    s = s_matrix(p, q)
    s = s.astype(np.int64)
    if np.array_equal(s @ s.T, (N / 4) * (np.eye(M) + np.ones((M, M)))):
        print(f"S matrix is valid.\n S @ S.T = \n{s @ s.T}")

    # 验证 由 S 循环矩阵生成的Hadamard矩阵的性质 H @ H.T = N * I
    H = hadamard_matrix(s)
    H = H.astype(np.int64)
    if np.array_equal(H @ H.T, N * np.eye(N)):
        print(f"Hadamard matrix is valid.\n H @ H.T = \n{H @ H.T}")

    # 验证 S 循环矩阵的快速傅里变换重建性质
    first_row = twin_prime_s_row(p, q)
    # 左移 M-1 位 <=> 右移1位   得到最后一行
    # last_row = np.roll(first_row, -(M - 1))
    last_row = np.roll(first_row, 1)

    # 计算 S 循环矩阵最后一行的离散傅里叶变换
    spectrum_last_row = fft.fft(last_row)

    # 目标物体
    f = np.arange(M) + 1j * np.random.randint(0, 10, M)

    # 线性测量
    g = s @ f

    # 测量的线傅里叶变换
    spectrum_g = fft.fft(g)

    # 重建
    f_reverse = fft.ifft(spectrum_g / spectrum_last_row)
    f_r = f_reverse[::-1]  # 反转序列

    if np.allclose(f, f_r):
        print("FFT reconstruction is valid.")
        residual = np.linalg.norm(f - f_r)
        print(residual)
