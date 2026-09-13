#!/usr/bin/env python3
"""
twin_prime_s_matrix.py
=======================
利用孪生素数法 (Whiteman twin-prime difference set) 生成循环 S 矩阵。

原理简述
--------
给定孪生素数对 (p, q) 且 q = p + 2, 令 N = p*q。通过中国剩余定理把
Z_N 上的每个位置 i 映射为 (a, b) = (i mod p, i mod q)，再按下述规则
判定第一行序列 s_i ∈ {0,1}：

    s_0            = 1
    a != 0, b == 0 -> 1   (q 的非零倍数，全部收入)
    a == 0, b != 0 -> 0   (p 的非零倍数，全部排除)
    a != 0, b != 0 -> 1   当且仅当 legendre(a,p)*legendre(b,q) == +1

以 s 为首行做循环移位，得到 N×N 循环矩阵 S，满足

    S @ S.T == ((N+1)/4) * I + ((N-3)/4) * J

即 S 矩阵的定义式 (J 为全 1 矩阵; 这等价于 (k-lambda)*I + lambda*J，
其中 k=(N-1)/2, lambda=(N-3)/4 是对应差集的参数)。由 S 可进一步构造 (N+1) 阶
对称 Hadamard 矩阵: H = [[1, 1^T], [1, 2S - J]]。

用法
----
直接修改文件底部 "参数设置" 部分的 P, Q 等变量, 然后运行:
    python3 twin_prime_s_matrix.py

或在其他脚本里调用:
    from twin_prime_s_matrix import main
    main(p=11, q=13, save_path="matrix.npy")
"""

import sys
import numpy as np


# --------------------------------------------------------------------------
# 数论工具
# --------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    """朴素但足够快的素性测试 (对本脚本涉及的规模足够)。"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def legendre_table(p: int) -> np.ndarray:
    """预计算模 p 的勒让德符号表, 长度为 p, 下标即 a。"""
    tbl = np.zeros(p, dtype=np.int64)
    qr = {(k * k) % p for k in range(1, p)}
    for a in range(1, p):
        tbl[a] = 1 if a in qr else -1
    tbl[0] = 0
    return tbl


# --------------------------------------------------------------------------
# 核心构造
# --------------------------------------------------------------------------
def twin_prime_first_row(p: int, q: int) -> np.ndarray:
    """
    生成循环 S 矩阵的第一行 (长度 N = p*q)。
    要求 p < q, 且 (p, q) 为孪生素数对 (q = p + 2)，两者均为素数。
    """
    if q != p + 2:
        raise ValueError(f"(p, q) = ({p}, {q}) 不是孪生素数对 (要求 q = p + 2)")
    if not (is_prime(p) and is_prime(q)):
        raise ValueError(f"p={p}, q={q} 必须均为素数")
    if p >= q:
        raise ValueError("必须满足 p < q")

    N = p * q
    idx = np.arange(N)
    a = idx % p
    b = idx % q

    chi_p = legendre_table(p)[a]
    chi_q = legendre_table(q)[b]

    s = np.zeros(N, dtype=np.int64)
    s[0] = 1
    s[(a != 0) & (b == 0)] = 1                     # q 的非零倍数
    core = (a != 0) & (b != 0)
    s[core] = (chi_p[core] * chi_q[core] == 1).astype(np.int64)
    # a == 0 且 b != 0 (p 的非零倍数) 保持为 0, 无需处理
    return s


def circulant_matrix(first_row: np.ndarray, symmetric: bool = False) -> np.ndarray:
    """
    由第一行生成循环矩阵。

    symmetric=False (默认): 右移版本, 第 i 行第 j 列 = s[(j-i) mod N]
        (标准卷积意义下的循环矩阵, 一般不对称)
    symmetric=True: 左移版本, 第 i 行第 j 列 = s[(i+j) mod N]
        (元素只依赖 i+j, 天然满足 S = S.T, 常用于需要对称S矩阵的场合,
        例如Hadamard变换光学中编码/解码共用同一块掩模)
    """
    N = len(first_row)
    S = np.empty((N, N), dtype=first_row.dtype)
    for k in range(N):
        S[k] = np.roll(first_row, -k if symmetric else k)
    return S


def build_s_matrix(p: int, q: int, symmetric: bool = False):
    """返回 (S矩阵, 第一行, N)。symmetric 参数含义见 circulant_matrix。"""
    s = twin_prime_first_row(p, q)
    S = circulant_matrix(s, symmetric=symmetric)
    return S, s, p * q


def to_hadamard(S: np.ndarray) -> np.ndarray:
    """由 N 阶 S 矩阵构造 (N+1) 阶对称 Hadamard 矩阵。"""
    N = S.shape[0]
    J = np.ones((N, N), dtype=np.int64)
    core = 2 * S - J                       # N阶, 元素为 ±1
    H = np.ones((N + 1, N + 1), dtype=np.int64)
    H[1:, 1:] = core
    return H


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def verify_s_matrix(S: np.ndarray) -> dict:
    """验证 S @ S.T == ((N+1)/4) I + ((N-3)/4) J  (等价于 (k-lambda)I + lambda*J)"""
    N = S.shape[0]
    lhs = S @ S.T
    I = np.eye(N, dtype=np.int64)
    J = np.ones((N, N), dtype=np.int64)
    rhs = (N + 1) / 4 * I + (N - 3) / 4 * J
    ok = np.allclose(lhs, rhs)
    return {"pass": bool(ok), "max_abs_diff": float(np.max(np.abs(lhs - rhs)))}


def verify_hadamard(H: np.ndarray) -> dict:
    """验证 H @ H.T == n * I"""
    n = H.shape[0]
    ok = np.array_equal(H @ H.T, n * np.eye(n, dtype=np.int64))
    return {"pass": bool(ok), "order": n}


# --------------------------------------------------------------------------
# 参数设置 (在此直接修改数值, 无需命令行参数)
# --------------------------------------------------------------------------
P = 3            # 较小的孪生素数 p
Q = 5            # 较大的孪生素数 q = p + 2
SYMMETRIC = True         # False=右移(标准循环矩阵,一般不对称); True=左移(保证 S=S.T)
BUILD_HADAMARD = True     # 是否同时构造 (N+1) 阶 Hadamard 矩阵
SAVE_PATH = None          # 若不为 None, 将 S 矩阵保存为该 .npy 路径
DO_PRINT = True           # 是否在终端打印矩阵内容 (矩阵较大时建议设为 False)


def main(p=P, q=Q, symmetric=SYMMETRIC, build_hadamard=BUILD_HADAMARD,
         save_path=SAVE_PATH, do_print=DO_PRINT):
    try:
        S, s, N = build_s_matrix(p, q, symmetric=symmetric)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    report = verify_s_matrix(S)
    print(f"孪生素数对: p={p}, q={q}  ->  N = p*q = {N}  "
          f"(移位方式: {'左移/对称' if symmetric else '右移/标准'})")
    print(f"第一行权重 k = {s.sum()}  (理论值 (N-1)/2 = {(N - 1) // 2})")
    print(f"S 是否对称 (S == S.T): {np.array_equal(S, S.T)}")
    print(f"S@S.T == ((N+1)/4)I + ((N-3)/4)J : {'通过' if report['pass'] else '未通过'} "
          f"(最大误差 {report['max_abs_diff']:.2e})")

    if do_print:
        print("\n第一行 s =")
        print(s)
        if N <= 40:
            print("\nS 矩阵 =")
            print(S)
        else:
            print(f"\n(N={N} 较大, 已跳过完整矩阵打印; 可设置 SAVE_PATH 导出)")

    if build_hadamard:
        H = to_hadamard(S)
        hrep = verify_hadamard(H)
        print(f"\n由 S 构造的 Hadamard 矩阵阶数 = {hrep['order']}, "
              f"H@H.T == n*I : {'通过' if hrep['pass'] else '未通过'}")
        if do_print and H.shape[0] <= 40:
            print("\nH 矩阵 =")
            print(H)

    if save_path:
        np.save(save_path, S)
        print(f"\n已保存 S 矩阵到: {save_path}")


if __name__ == "__main__":
    main()
