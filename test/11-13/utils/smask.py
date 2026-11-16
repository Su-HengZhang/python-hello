import math
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


def g_matrix(p: int, q: int) -> np.ndarray:
    """
    根据孪生素数 p、q 构造 G 矩阵。
    """
    N = p * q
    s = s_matrix(p, q)

    J = np.ones((N, N), dtype=np.int8)
    G = J - 2 * s

    return G


def mask_tape(p: int, q: int) -> np.ndarray:
    """
    根据孪生素数 p、q 构造行数为 p，列数为 (p+1)*q-1 的循环掩模带，
    若物体为 p 行，q 列，当物体从左向右在该掩模带上滑动时，每向前移动一列，
    就会对应循环矩阵的新一行。

    要求：
        p、q 一对孪生素数
        q = p + 2
    """

    s = twin_prime_s_row(p, q)
    mask0 = s.reshape((p, q))

    masks = []
    for i in range(p):
        mask = np.roll(mask0, -i, axis=0)
        masks.append(mask)
    masks.append(mask0[:, :-1])
    tape = np.hstack(masks)

    return tape.astype(np.uint8)


def tape_cutting(
    p: int, q: int, page_width: int, overlap_flag: bool = True
) -> tuple[list[np.ndarray], int, int]:
    """按 page_width 将 mask tape 切页。

    overlap_flag=True 时相邻页重叠 q-1 列,保证跨页边界的窗口完整;
    False 时页间不重叠。末页不足 page_width 的部分补零。
    每页均为独立拷贝,可安全原地修改。

    Returns:
        page_list:  所有页组成的列表,每页形状为 (height, page_width)
        page_nums:  总页数
        last_width: 末页中来自 tape 的有效列数(1 ~ page_width),
                    其余 page_width - last_width 列为补零
    """
    # 相邻页重叠的列数:开启重叠时为 q-1,否则为 0
    overlap = q - 1 if overlap_flag else 0

    # 每页必须比重叠区更宽,否则步长 <= 0,无法向前推进
    if page_width <= overlap:
        raise ValueError(f"page_width ({page_width}) 必须大于 overlap ({overlap})")

    # 相邻两页起点之间的步长(每页真正"新增"的列数)
    step = page_width - overlap

    # 生成完整的 mask tape,形状为 (height, width)
    tape = mask_tape(p, q)
    height, width = tape.shape

    # 总页数:第一页覆盖 page_width 列,之后每页新增 step 列
    # 若 tape 一页就放得下,则只有 1 页
    page_nums = 1 if width <= page_width else 1 + math.ceil((width - page_width) / step)

    page_list = []
    last_width = page_width  # 循环结束后即为末页有效宽度
    for n in range(page_nums):
        # 第 n 页在 tape 上的截取区间 [n*step, n*step + page_width)
        # 末页切片会被 tape 右边界截断,宽度可能小于 page_width
        seg = tape[:, n * step : n * step + page_width]
        last_width = seg.shape[1]

        # 先建全零页(dtype 与 tape 一致),再写入有效数据
        # 完整页与末页走同一路径,末页不足的部分保持为 0
        page = np.zeros((height, page_width), dtype=tape.dtype)
        page[:, :last_width] = seg
        page_list.append(page)

    return page_list, page_nums, last_width


def dmd_images(
    p: int,
    q: int,
    page_width: int,
    nbin: int = 10,
    diff: bool = True,
    top_sep: int = 0,
    left_sep: int = 0,
    grat_shift: int = 0,
    grat_sep: int = 100,
    dmd_row: int = 1080,
    dmd_col: int = 1920,
):
    """逐页生成 DMD 图像:上方为 tape 页(nbin×nbin 放大),下方为参考光栅。

    grat_shift: 光栅相对页面的水平位移,单位为 DMD 像素(已放大后);
                    >0 右移,<0 左移,左移量不能超过 left_sep。
    """

    # 页面区域 [rb, re) × [cb, ce)(放大 nbin 倍后)
    page_rb = top_sep
    page_re = top_sep + p * nbin

    page_cb = left_sep
    page_ce = left_sep + page_width * nbin

    # 光栅区域: 页面下方留 grat_sep 间隔后,一直铺到 DMD 底部
    grating_rb = page_re + grat_sep
    # 光栅微调，向左位移最大受限于left_sep
    if grat_shift > 0:
        row_shift = grat_shift
        grating_cb = page_cb + row_shift
        grating_ce = page_ce
    else:
        row_shift = 0 if left_sep <= np.abs(grat_shift) else grat_shift
        grating_cb = page_cb + row_shift
        grating_ce = page_ce + row_shift

    # 边界检查，明确错误信息
    if min(top_sep, left_sep, grat_sep) < 0:
        raise ValueError("top_sep / left_sep / grat_sep 不能为负")
    if page_ce > dmd_col:
        raise ValueError(f"页面右边界 {page_ce} 超出 DMD 列数 {dmd_col}")
    if grating_rb >= dmd_row:
        raise ValueError(
            f"光栅起始行 {grating_rb} 超出 DMD 行数 {dmd_row},无空间放置光栅"
        )

    # 初始化全0的dmd图像
    dmd_img = np.zeros((dmd_row, dmd_col), dtype=np.uint8)

    # 生成一维光栅
    grating = np.zeros(page_width, dtype=np.uint8)  # 光栅与页面宽度相同
    grating[::2] = 1  # 光栅的周期为2
    grating = grating.repeat(nbin)  # 逐像素放大 nbin 倍

    grating = grating[:-grat_shift] if grat_shift > 0 else grating
    dmd_img[grating_rb:, grating_cb:grating_ce] = grating

    page_list, page_nums, last_width = tape_cutting(p, q, page_width)
    for n in range(page_nums):
        page = page_list[n]
        page_binning = page.repeat(nbin, axis=0).repeat(nbin, axis=1)
        dmd_img[page_rb:page_re, page_cb:page_ce] = page_binning  # positive page
        yield dmd_img
        if diff == True:
            dmd_img[page_rb:page_re, page_cb:page_ce] = 1 - page_binning  # negtive page
            if n == page_nums - 1:
                # 末页不足的部分保持为 0
                # 末页有效列数为 last_width, 补零列数为 page_width - last_width
                # 因此, 末页的 negtive page 为 (page_width - last_width) * nbin 列
                last_zero_width = (page_width - last_width) * nbin
                if last_zero_width > 0:
                    dmd_img[page_rb:page_re, page_ce - last_zero_width : page_ce] = 0
            yield dmd_img


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
    smatrix_first_row = twin_prime_s_row(p, q)
    print(f"smatrix_first_row for p={p},q={q} is:\n {smatrix_first_row}")
