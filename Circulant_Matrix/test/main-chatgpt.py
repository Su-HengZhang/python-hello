import numpy as np


def is_prime(n):
    """判断 n 是否为素数"""
    if n < 2:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False

    return True


def quadratic_character(i, p):
    """
    计算原书中 f(i) / g(i) 所定义的二次特征：

        1   : i mod p 是二次剩余
        0   : p | i
        -1  : 其他情况
    """

    r = i % p

    # p 整除 i
    if r == 0:
        return 0

    # Euler criterion
    value = pow(r, (p - 1) // 2, p)

    if value == 1:
        return 1
    else:
        return -1


def twin_prime_s_matrix(p, q=None):
    """
    根据《Hadamard Transform Optics》
    Appendix A.2.4 的 Twin Prime Construction
    构造 0/1 型 S-cyclic matrix。

    参数
    ----
    p : int
        较小的孪生素数

    q : int, optional
        较大的孪生素数，默认为 p + 2

    返回
    ----
    S : ndarray, shape (n, n)
        0/1 型 S 循环矩阵
    """

    # --------------------------------------------------
    # 1. 确定孪生素数
    # --------------------------------------------------

    if q is None:
        q = p + 2

    if q != p + 2:
        raise ValueError("必须满足 q = p + 2")

    if not is_prime(p):
        raise ValueError(f"p = {p} 不是素数")

    if not is_prime(q):
        raise ValueError(f"q = {q} 不是素数")

    # --------------------------------------------------
    # 2. S-matrix 的阶数
    # --------------------------------------------------

    n = p * q

    # --------------------------------------------------
    # 3. 找到所有满足 f(i) = g(i) 的 a_i
    #
    #    1 <= i <= pq - 1
    # --------------------------------------------------

    zero_positions = []

    for i in range(1, n):

        f = quadratic_character(i, p)
        g = quadratic_character(i, q)

        if f == g:
            zero_positions.append(i)

    # --------------------------------------------------
    # 4. 加入
    #
    #    0, q, 2q, ..., (p-1)q
    # --------------------------------------------------

    zero_positions.extend(j * q for j in range(p))

    # 去重，防止潜在重复
    zero_positions = sorted(set(zero_positions))

    # --------------------------------------------------
    # 5. 构造第一行
    #
    #    原书：
    #
    #    s_ai = 0
    #    其他 s_i = 1
    # --------------------------------------------------

    first_row = np.ones(n, dtype=np.uint8)

    first_row[zero_positions] = 0

    # return first_row
    # --------------------------------------------------
    # 6. 构造 S-cyclic matrix
    #
    #    每一行是前一行的循环移位
    # --------------------------------------------------

    S = np.empty((n, n), dtype=np.uint8)

    for i in range(n):
        S[i] = np.roll(first_row, -i)

    return S


if __name__ == "__main__":
    from numpy import fft

    p = 3
    q = 5
    # first_row = twin_prime_s_matrix(p, q)
    # print(first_row)
    s = twin_prime_s_matrix(p, q)

    f = np.arange(p * q) + 1j * np.random.randint(0, 10, p * q)
    print(f)

    g = s @ f

    # 计算 s 最后一行 的离散傅里叶变换
    S_last_row = fft.fft(s[-1])
    # 计算 g 的离散傅里叶变换
    G = fft.fft(g)
    # 重建f序列的频谱
    Fr = G / S_last_row
    fr = fft.ifft(Fr)
    fr = fr[::-1]
    print(np.real(fr))
    print(np.imag(fr))
    # 计算重建误差的范数
    residual = np.linalg.norm(fr - f)
    print(residual)
