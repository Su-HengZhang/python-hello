import os
import numpy as np
from PIL import Image


def legendre_table(m: int) -> np.ndarray:
    """
    生成模 m (m为奇素数) 的 Legendre 符号查找表。

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


def mask_tape(p: int, q: int) -> np.ndarray:
    """
    根据孪生素数 p、q 构造长度 (p+1)*q 的 0/1 序列。

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

    return tape


def dmd_images(
    p: int,
    q: int,
    nbin: int = 2,
    dir_path: str = "dmd_images",
    dmd_row: int = 1080,
    dmd_col: int = 1920,
):
    # 创建图片目录
    full_dir_path = f"{dir_path}_p{p}_q{q}_nbin{nbin}"
    try:
        os.makedirs(full_dir_path, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {e}")
        return
    # 生成掩模带, 然后将每个像素扩展为 nbin x nbin 的微镜块
    tape = mask_tape(p, q)
    tape = tape.repeat(nbin, axis=0).repeat(nbin, axis=1)

    # 整条掩模带的行数与列数
    tape_row, tape_col = tape.shape

    # 将掩模带放置在 DMD 的中央区域(行)
    dmd_center_row = dmd_row // 2  # DMD 图像中心行
    row_beg = dmd_center_row - tape_row // 2  # 起始行
    row_end = row_beg + tape_row  # 结束行

    # 将掩模带分割为若干个 pattern，每个pattern的列数为 pat_col
    pat_col = (dmd_col // nbin) * nbin  # 每个 pattern 的列数必须是 nbin 的整数倍

    # 将掩模带分割为若干个完整的 pattern 后， 剩余的列数
    rem_cols = tape_col % pat_col
    # 存储全部 pattern (包括最后一个不完整的 pattern) 的DMD图像数量
    img_num = tape_col // pat_col if rem_cols == 0 else tape_col // pat_col + 1

    # 初始化 pattern 和 img 为全0数组
    pattern = np.zeros((tape_row, pat_col), dtype=np.uint8)
    img = np.zeros((dmd_row, dmd_col), dtype=np.uint8)
    for i in range(img_num):
        # 对于最后一个不完整的 pattern， 只取tape最后剩余的列数
        if (rem_cols != 0) and (i == img_num - 1):
            pattern[:, :] = 0  # 清零
            pattern[:, 0:rem_cols] = tape[:, i * pat_col : (i * pat_col + rem_cols)]
        else:
            pattern[:, :] = tape[:, i * pat_col : (i + 1) * pat_col]

        # 将 pattern 放置在 DMD 图像的中央区域(行)
        img[row_beg:row_end, 0:pat_col] = pattern

        # 将图像保存为二值 BMP 文件
        filename = os.path.join(full_dir_path, f"{i:05}.bmp")
        im = Image.fromarray(img.astype(bool))
        im.save(filename)


def dmd_images_diff(
    p: int,
    q: int,
    nbin: int = 2,
    diff: bool = True,
    dir_path: str = "dmd_images",
    dmd_row: int = 1080,
    dmd_col: int = 1920,
):
    # 创建图片目录
    full_dir_path = f"{dir_path}_p{p}_q{q}_nbin{nbin}"
    if diff == True:
        full_dir_path = full_dir_path + "_diff"
    # 创建目录
    try:
        os.makedirs(full_dir_path, exist_ok=True)
    except Exception as e:
        print(f"创建目录失败: {e}")
        return

    # 生成掩模带, 然后将每个像素扩展为 nbin x nbin 的微镜块
    tape = mask_tape(p, q)
    tape = tape.repeat(nbin, axis=0).repeat(nbin, axis=1)

    # 整条掩模带的行数与列数
    tape_row, tape_col = tape.shape

    # 将掩模带放置在 DMD 的中央区域(行)
    dmd_center_row = dmd_row // 2  # DMD 图像中心行
    row_beg = dmd_center_row - tape_row // 2  # 起始行
    row_end = row_beg + tape_row  # 结束行

    # 将掩模带分割为若干个 pattern，每个pattern的列数为 pat_col
    pat_col = (dmd_col // nbin) * nbin  # 每个 pattern 的列数必须是 nbin 的整数倍

    # 将掩模带分割为若干个完整的 pattern 后， 剩余的列数
    rem_cols = tape_col % pat_col
    # 存储全部 pattern (包括最后一个不完整的 pattern) 的DMD图像数量
    img_num = tape_col // pat_col if rem_cols == 0 else tape_col // pat_col + 1

    # 初始化 pattern 和 img 为全0数组
    pattern = np.zeros((tape_row, pat_col), dtype=np.uint8)
    img = np.zeros((dmd_row, dmd_col), dtype=np.bool)
    for i in range(img_num):
        # 对于最后一个不完整的 pattern， 只取tape最后剩余的列数
        if (rem_cols != 0) and (i == img_num - 1):
            pattern[:, :] = 0  # 清零
            pattern[:, 0:rem_cols] = tape[:, i * pat_col : (i * pat_col + rem_cols)]
        else:
            pattern[:, :] = tape[:, i * pat_col : (i + 1) * pat_col]

        if diff == False:
            # 若 diff 为 False，则只生成 pattern_pos 的图像
            pattern_pos = pattern.copy().astype(bool)
            # 将 pattern_pos 放置在 DMD 图像的中央区域(行)
            img[row_beg:row_end, 0:pat_col] = pattern_pos
            # 将图像保存为二值 BMP 文件
            filename = os.path.join(full_dir_path, f"{i:05}.bmp")
            im = Image.fromarray(img)
            im.save(filename)
        else:
            # 若 diff 为 True，则生成 pattern_pos 和 pattern_neg 的图像
            pattern_pos = pattern.copy().astype(bool)
            pattern_neg = np.logical_not(pattern_pos)
            # 将 pattern_pos 放置在 DMD 图像的中央区域(行)
            img[row_beg:row_end, 0:pat_col] = pattern_pos
            # 将图像保存为二值 BMP 文件
            filename = os.path.join(full_dir_path, f"{2 * i:05}.bmp")
            im = Image.fromarray(img)
            im.save(filename)

            # 将 pattern_neg 放置在 DMD 图像的中央区域(行)
            img[row_beg:row_end, 0:pat_col] = pattern_neg
            # 将图像保存为二值 BMP 文件
            filename = os.path.join(full_dir_path, f"{2 * i + 1:05}.bmp")
            im = Image.fromarray(img)
            im.save(filename)

    # 解决最后一张差分负图案，因直接取反，引起其尾部，无掩模区域，0/1反转问题
    # 重新生成正确图案，将上面生成的错误图案覆写
    if (rem_cols != 0) and (diff == True):
        pattern[:, :] = 0  # 清零
        n = img_num - 1
        pattern[:, 0:rem_cols] = 1 - tape[:, n * pat_col : (n * pat_col + rem_cols)]
        pattern_neg = pattern.copy().astype(bool)

        # 将 pattern_neg 放置在 DMD 图像的中央区域(行)
        img[row_beg:row_end, 0:pat_col] = pattern_neg
        # 将图像保存为二值 BMP 文件
        filename = os.path.join(full_dir_path, f"{2 * n + 1:05}.bmp")
        im = Image.fromarray(img)
        im.save(filename)


if __name__ == "__main__":
    """
    前20对孪生素数：
    (3, 5) (5, 7) (11, 13) (17, 19) (29, 31) (41, 43) (59, 61) (71, 73) (101, 103)
    (107, 109) (137, 139) (149, 151) (179, 181) (191, 193) (197, 199) (227, 229)
    (239, 241) (269, 271) (281, 283) (311, 313)
    """

    # 测试
    p = 239
    q = 241
    nbin = 4
    dmd_images_diff(p, q, nbin)
