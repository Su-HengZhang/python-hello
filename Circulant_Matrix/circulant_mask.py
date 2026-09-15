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


# def dmd_mask((p: int, q: int), (dmd_row: int = 1080, dmd_col: int = 1920)) -> np.ndarray:
#     pass


if __name__ == "__main__":
    """
    前20对孪生素数：
    (3, 5) (5, 7) (11, 13) (17, 19) (29, 31) (41, 43) (59, 61) (71, 73) (101, 103)
    (107, 109) (137, 139) (149, 151) (179, 181) (191, 193) (197, 199) (227, 229)
    (239, 241) (269, 271) (281, 283) (311, 313)
    """

    # 测试
    p = 3
    q = 5

    s = twin_prime_s_row(p, q)
    mask0 = s.reshape((p, q))

    masks = []
    for i in range(p):
        mask = np.roll(mask0, -i, axis=0)
        masks.append(mask)
    masks.append(mask0)
    pattern = np.hstack(masks)
    print(pattern)
