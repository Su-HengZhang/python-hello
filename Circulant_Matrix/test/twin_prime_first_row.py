# import numpy as np


# def legendre_table(m: int) -> np.ndarray:
#     """
#     预计算模 m 的勒让德符号 (二次特征) 表, 下标即自变量 a, 取值范围 0..m-1。

#     勒让德符号定义:
#         chi(a) =  0   若 a ≡ 0 (mod m)
#         chi(a) = +1   若 a 是模 m 的二次剩余 (存在 x 使 x^2 ≡ a mod m)
#         chi(a) = -1   若 a 是模 m 的二次非剩余

#     实现技巧: 与其对 0..m-1 每个数分别做一次 O(log m) 的模幂运算
#     pow(a, (m-1)//2, m) 来判断是否为二次剩余, 不如直接对
#     1..(m-1)//2 做平方取模 —— 因为模素数 m 下, 非零二次剩余恰好有
#     (m-1)/2 个, 且每个剩余都能由某个 x∈[1,(m-1)/2] 的平方得到
#     (x 和 m-x 平方相同, 只需算一半即可覆盖所有二次剩余)。
#     这样只需要 O(m) 次乘法/取模, 且可以整体向量化, 避免逐元素调用
#     Python 级别的 pow(), 速度快很多。
#     """
#     tbl = np.full(m, -1, dtype=np.int64)  # 默认先全部标为"非剩余"(-1)
#     r = np.arange(1, (m - 1) // 2 + 1)  # 只需要 1..(m-1)/2, 覆盖所有非零二次剩余
#     tbl[(r * r) % m] = 1  # 这些位置对应的自变量是二次剩余, 标记为 +1
#     tbl[0] = 0  # 单独修正: a=0 时勒让德符号规定为 0
#     return tbl


# def twin_prime_first_row_fast(p: int, q: int) -> np.ndarray:
#     """
#     利用孪生素数法 (Whiteman 差集构造), 生成循环 S 矩阵第一行的
#     0/1 反转版本 (即互补差集, 权重为 (N+1)/2)。

#     数学原理
#     --------
#     要求 p < q 且 q = p + 2 (孪生素数对), 令 N = p*q。
#     由中国剩余定理 (CRT), 因为 gcd(p,q)=1, 每个位置 i ∈ {0,...,N-1}
#     唯一对应一对 (a, b) = (i mod p, i mod q), 且这个对应是双射。

#     原始 (未反转) 差集的判定规则:
#         s[0]                          = 1   (i=0 单独定义为属于差集)
#         a != 0, b == 0 (q 的非零倍数)  -> 1   (全部收入)
#         a == 0, b != 0 (p 的非零倍数)  -> 0   (全部排除)
#         a != 0, b != 0 (核心区)       -> 1 当且仅当
#                                           legendre(a,p) * legendre(b,q) == +1
#                                           (两个勒让德符号同号)

#     这样构造出的集合 D 满足 |D| = (N-1)/2, 且是一个 (N,k,λ) 循环差集
#     (k=(N-1)/2, λ=(N-3)/4), 对应循环矩阵满足
#         S @ S.T == ((N+1)/4)*I + ((N-3)/4)*J

#     本函数最后返回 1-s, 即上述差集的**补集**(0/1 全部反转):
#     权重变为 (N+1)/2, 满足 S'@S'.T == ((N+1)/4)*(I+J)。
#     两种版本互为补集, 都是合法的循环 S 矩阵, 但只有"原始版"能直接
#     用简单的"全 +1 边框"公式扩展成 Hadamard 矩阵。

#     向量化实现技巧
#     --------------
#     因为 N = p*q, 序列 a_i = i mod p (i=0..N-1) 本身就是
#     [0,1,...,p-1] 这个长度为 p 的模式**重复 q 遍**; 同理
#     b_i = i mod q 是 [0,1,...,q-1] 这个模式**重复 p 遍**。
#     所以不需要对长度 N 的大数组做逐元素取模 (代价较高的整数除法)
#     再花式索引取表值 (gather, 内存访问不连续); 只需要先在"小规模"
#     (长度 p 或 q) 上算好勒让德符号表, 再用 np.tile 直接平铺复制
#     成长度 N 的数组即可 (纯内存搬运, 连续访问, 非常快)。

#     另外, 勒让德符号的定义决定了 chi(a)=0 当且仅当 a≡0 (mod 对应素数)。
#     也就是说 chi_q == 0 这一条件本身就精确等价于 "b == 0"
#     (即 i 是 q 的倍数, 包含 i=0 这个特殊情形), 不需要额外算出
#     原始的 b 数组再判断 b==0, 直接复用 chi_q 的 0 值即可, 省一步。

#     参数
#     ----
#     p, q : 孪生素数对, 要求 p < q 且 q = p + 2, 且两者均为素数
#            (本函数不做检查, 调用方需自行保证)

#     返回
#     ----
#     长度为 N=p*q 的 0/1 numpy 数组 (int64), 即互补版本的第一行 s'。
#     """
#     # 第一步: 在"小规模"(长度分别为 p, q) 上算出两张勒让德符号表,
#     # 再用 tile 平铺成长度 N 的完整序列, 分别对应每个位置 i 的
#     # a=i mod p 和 b=i mod q 所对应的勒让德符号值 (无需真正算出 a, b)。
#     chi_p = np.tile(legendre_table(p), q)  # 长度 N, 模式 [chi_p(0..p-1)] 重复 q 遍
#     chi_q = np.tile(legendre_table(q), p)  # 长度 N, 模式 [chi_q(0..q-1)] 重复 p 遍

#     # 第二步: "核心区"规则 —— 两个勒让德符号乘积为 +1 时, 原始差集
#     # 该位置为 1。注意: 若 a=0 (chi_p=0) 或 b=0 (chi_q=0), 乘积自动为 0,
#     # 不会等于 1, 因此这一行同时也隐式地把"p 的非零倍数"位置正确置为 0,
#     # 不需要再单独处理这一条边界规则。
#     s = (chi_p * chi_q == 1).astype(np.int64)

#     # 第三步: 处理"q 的倍数(含 i=0)"这一条边界规则 —— 这些位置原始差集
#     # 应统一为 1。因为 chi_q==0 精确对应 "b==0" (即 i 是 q 的倍数,
#     # 自然包含 i=0), 直接用这个条件一次性覆盖, 不需要分别处理
#     # "i=0" 和 "q 的非零倍数"两种情形。
#     s[chi_q == 0] = 1

#     # 第四步: 整体 0/1 反转, 得到互补差集对应的第一行 (权重 (N+1)/2)。
#     return 1 - s


# if __name__ == "__main__":
#     p = 3
#     q = 5
#     s = twin_prime_first_row_fast(p, q)
#     print(s)
import numpy as np


def legendre_table(m):
    tbl = np.full(m, -1, dtype=np.int64)
    r = np.arange(1, (m - 1) // 2 + 1)
    tbl[(r * r) % m] = 1
    tbl[0] = 0
    return tbl


def twin_prime_first_row_fast(p: int, q: int) -> np.ndarray:
    """
    要求: p < q, q = p + 2, 且 p, q 均为素数。
    """
    chi_p = np.tile(legendre_table(p), q)  # 长度 N=p*q, 无需 idx%p
    chi_q = np.tile(legendre_table(q), p)  # 长度 N=p*q, 无需 idx%q

    s = (chi_p * chi_q == 1).astype(np.int64)
    s[chi_q == 0] = 1  # q的倍数(含0)统一置1, 一步到位
    return 1 - s


if __name__ == "__main__":
    p = 3
    q = 5
    s = twin_prime_first_row_fast(p, q)
    print(s)

# import numpy as np


# def legendre_table(m):
#     tbl = np.full(m, -1, dtype=np.int64)
#     r = np.arange(1, (m - 1) // 2 + 1)
#     tbl[(r * r) % m] = 1  # 二次剩余标记为 +1
#     tbl[0] = 0
#     return tbl


# def twin_prime_first_row_fast(p: int, q: int) -> np.ndarray:
#     """
#     高效版: 用向量化的"平方筛"代替逐元素勒让德符号 pow() 计算。
#     要求: p < q, q = p + 2, 且 p, q 均为素数。
#     """

#     N = p * q
#     idx = np.arange(N)
#     a, b = idx % p, idx % q

#     chi_p = legendre_table(p)[a]
#     chi_q = legendre_table(q)[b]

#     s = np.zeros(N, dtype=np.int64)
#     s[0] = 1
#     s[(a != 0) & (b == 0)] = 1
#     core = (a != 0) & (b != 0)
#     s[core] = (chi_p[core] * chi_q[core] == 1).astype(np.int64)
#     return 1 - s


# if __name__ == "__main__":
#     p = 3
#     q = 5
#     s = twin_prime_first_row_fast(p, q)
#     print(s)
