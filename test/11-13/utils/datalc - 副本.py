import numpy as np
import pandas as pd

from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks

# load_bin 函数的配置参数
SIGN_MODE = "twos_comp"  # 指定符号模式，twos_comp或offset_binary，默认twos_comp
FS_VPP = 10  # ADC 满量程峰峰值电压，单位 V; 若FS_VPP = 10, 则对应 ±5 V 满量程

# 包络判定find_envelopes函数的配置参数
ENV_WIN = 120  # 包络判定窗口(点), 需 > 1个锯齿周期(~49点)
ENV_THR = 0.12  # 窗口内起伏(V) 超过此值视为"包络内"

# 峰谷中心定位find_peaks函数的配置参数
LOC_HALF = 25  # 计算局部高低电平的半窗口(点), 约 1 个周期
FRAC = 0.5  # cross 的幅度比例(0.5=半高)
MAX_STEP = 40  # cross 左右搜索的最大步数, 通常取到「峰半高宽样本数」的 1.5 倍
SINE_W = 24  # sine 方法拟合半窗口(点)
PERIOD = 49.0  # sine 方法的锯齿周期(点)
METHOD = "argmax"  # 峰谷中心定位: 'cross50'(半幅交点中线, 默认) / 'sine'(局部正弦相位拟合) / 'argmax'(旧方法)

SMOOTH_N = 8  # 仅用于"定位"峰谷的平滑点数, 不影响提取的原始数据
PROMINENCE = 0.1  # 峰/谷显著度(V)
MIN_DIST = 46  # 相邻峰(或谷)最小间隔(点), 需 < 1个周期(~49点)

HALF = 1  # 中心左右各取几点平均

BASE = 0  # 索引起始编号 (1 或 0)


################################################################################
def load_bin(bin_file_path, num_chs=2, sign_mode=SIGN_MODE, fs_vpp=FS_VPP):
    """
    从取二进制文件中读取数据，返回 N x num_chs 数组，其中 N 为数据点数量。
    每个元素为 16 bit 有符号整数，其中低 14 位为有效位，需要根据 sign_mode 处理符号。
    最终返回电压数组，单位 V。
    """
    # 读取二进制文件，其中：
    # "<"：小端字节序（Little Endian）。
    # "u"：无符号整数。
    # "2"：16 bit，即每个元素占两个字节。
    raw = np.fromfile(bin_file_path, dtype="<u2")

    # 确保元素数量为 num_chs 的倍数，否则进行截断处理，然后重塑为 num_chs 列
    # 列0=ch1, 列1=ch2, ..., 列num_chs-1=chnum_chs
    raw = raw[: raw.size // num_chs * num_chs].reshape(-1, num_chs)

    # 把 raw 中的每个元素先转换成 32 位有符号整数，然后与十六进制掩码 0x3FFF 做逐元素
    # 按位与运算最终只保留每个数的低 14 位
    # 注意到 0x3FFF 的二进制形式是: 0011 1111 1111 1111
    c = raw.astype(np.int32) & 0x3FFF

    # 处理符号
    if sign_mode == "twos_comp":
        # 14 bit二进制补码转换
        # 原始数字小于8192时，表示正数；大于或等于8192时，表示负数
        c = np.where(c >= 8192, c - 16384, c)
    elif sign_mode == "offset_binary":
        # 14 bit偏移二进制转换
        # 原始数字减去8192，即可得到有符号整数
        c = c - 8192
    else:
        c = raw.astype(np.int32)
    return c.astype(np.float64) * (fs_vpp / 2.0 / 8192.0)  # 转换为电压，单位 V


################################################################################
def find_envelopes(s, env_win=ENV_WIN, env_thr=ENV_THR):
    """用滑动窗口内的 max-min 判定哪些区间有锯齿(包络内)

    原理：
        1. 对每个位置计算滑动窗口内 max-min（局部峰峰值），作为"波动幅度"
        2. 峰峰值 > 阈值的连续段即为激活区间
        3. 把这些连续段提取为 [(start, end), ...] 闭区间列表

    参数:
        s:       一维输入信号
        env_win: 滑动窗口长度（用于计算局部 max/min）
        env_thr: 峰峰值阈值，超过则判定为激活（有锯齿）

    返回:
        [(start, end), ...] 闭区间列表，标注所有激活段
    """
    # 1) 逐点计算局部峰峰值：滑动窗口内的 max - min
    rng = maximum_filter1d(s, env_win) - minimum_filter1d(s, env_win)

    # 2) 峰峰值超过阈值的位置标记为 True（激活 / 有锯齿）
    act = rng > env_thr

    # 3) 找布尔数组中状态跳变的位置：
    #    np.diff(act) 在位置 i 处 = act[i+1] - act[i]
    #    非零 => act[i] 与 act[i+1] 不同
    #      +1 表示上升沿 (0→1)，-1 表示下降沿 (1→0)
    #    flatnonzero 只取位置索引，正负号信息丢失，
    #    但下面利用"上升沿/下降沿交替出现"的规律配对即可
    edge = np.flatnonzero(np.diff(act.astype(np.int8)))

    # 4) 边界处理：若起点就是激活态，补一个虚拟起点 -1，
    #    使后续 edge[0::2] + 1 = 0 能正确表示区间从索引 0 开始
    if act[0]:
        edge = np.r_[-1, edge]

    # 5) 边界处理：若末尾仍是激活态，补一个虚拟终点 size-1，
    #    使后续 edge[1::2] = size-1 能正确表示区间以最后一个元素结束
    if act[-1]:
        edge = np.r_[edge, act.size - 1]

    # 6) edge 中交替存放 [上升沿位置, 下降沿位置, 上升沿位置, ...]
    #    - 上升沿位置 i：跳变发生在 i 和 i+1 之间 => 实际起点为 i+1
    #    - 下降沿位置 i：i 是最后一个激活元素     => 实际终点为 i
    #    两组配对即得到每个激活区间的闭区间 [start, end]
    return list(zip(edge[0::2] + 1, edge[1::2]))  # [(start, end), ...]


################################################################################
def _cross_center(y, c, is_peak, loc_half=LOC_HALF, frac=FRAC, max_step=MAX_STEP):
    """求峰(或谷)在"局部半幅电平"处的左右两个交点, 返回两交点的中点(亚像素位置)

    用途:
        直接用离散极值点 c 作为中心, 精度和抗噪性都受限;
        这里改为找 c 附近信号"穿越某条电平线"的左右两个交点,
        取二者平均作为中心, 从而得到更鲁棒、更精确的亚像素位置。

    做法:
        1. 取 c 附近的局部窗口 [c-loc_half, c+loc_half], 求局部 max/min
        2. 计算电平 lvl:
             - 峰: lvl = lo + frac*(hi-lo)   (从谷底往上 frac 处)
             - 谷: lvl = hi - frac*(hi-lo)   (从峰顶往下 frac 处)
           典型 frac=0.5, 即"半高"电平
        3. 从 c 向左右各自扩张, 走到第一个跨越 lvl 的位置, 得到 l、r
        4. 在 [l-1, l] 与 [r, r+1] 两小段上对 lvl 线性插值, 求交点 xl、xr
        5. 返回 (xl + xr) / 2

    参数:
        y:        一维信号
        c:        初始中心(峰/谷的整数索引), 作为左右搜索起点
        is_peak:  True=峰(内部判定 y>=lvl), False=谷(内部判定 y<=lvl)
        loc_half: 局部窗口半宽, 用于估计 c 附近的 max/min
        frac:     电平高度比例, 0.5 即半高电平
        max_step: 左右搜索的最大步数上限, 防止在长平台 / 缓坡上无界搜索。
                  应按信号的最宽峰宽(样本数)来设定; 过小会截断交点,
                  过大则在平台区退化时耗费计算。默认 40。

    返回:
        亚像素精度的中心位置 (float)

    前提/假设:
        - 期望 y[c] 满足 inside 条件(即 c 处于该峰/谷内部), 否则交点会落在 c 附近而失真
        - 左右搜索最多各 max_step 步, 防止在平台区无限扩张
        - 相邻两样本不相等(y[l]!=y[l-1] 且 y[r]!=y[r+1]), 否则线性插值会除零
    """
    # 1) 取局部窗口内的极值, 作为电平高度的参考
    loc = y[max(0, c - loc_half) : c + loc_half + 1]
    hi, lo = loc.max(), loc.min()

    # 2) 计算电平 lvl:
    #    峰 -> 从谷底往上 frac;  谷 -> 从峰顶往下 frac
    lvl = lo + frac * (hi - lo) if is_peak else hi - frac * (hi - lo)

    # 判定"仍在内部": 峰要求 y>=lvl, 谷要求 y<=lvl
    inside = (lambda v: v >= lvl) if is_peak else (lambda v: v <= lvl)

    # 3) 从 c 向左扩张, 直到下一个样本不再满足 inside
    #    循环结束时: l 是最左的 inside 样本, y[l-1] 已在 outside
    #    限制条件: l>1(保证 l-1 合法) 且最多走 max_step 步
    l = c
    while l > 1 and inside(y[l - 1]) and c - l < max_step:
        l -= 1

    # 4) 从 c 向右扩张, 直到下一个样本不再满足 inside
    #    循环结束时: r 是最右的 inside 样本, y[r+1] 已在 outside
    #    限制条件: r<y.size-2(保证 r+1 合法) 且最多走 max_step 步
    r = c
    while r < y.size - 2 and inside(y[r + 1]) and r - c < max_step:
        r += 1

    # 5) 左交点: 在 [l-1, l] 之间对 lvl 线性插值
    xl = (l - 1) + (lvl - y[l - 1]) / (y[l] - y[l - 1])

    # 6) 右交点: 在 [r, r+1] 之间对 lvl 线性插值
    xr = r + (lvl - y[r]) / (y[r + 1] - y[r])

    # 7) 左右交点的中点即为该峰/谷的亚像素中心
    return 0.5 * (xl + xr)


def _sine_center(y, c, is_peak):
    """以 c 为中心的窗口内, 固定周期的 [1, cos, sin] 最小二乘, 由相位得到峰(或谷)位置"""
    x = np.arange(max(0, c - SINE_W), min(y.size, c + SINE_W + 1))
    w = 2 * np.pi / PERIOD
    A = np.c_[np.ones(x.size), np.cos(w * (x - c)), np.sin(w * (x - c))]
    a = np.linalg.lstsq(A, y[x], rcond=None)[0]
    ph = np.arctan2(a[2], a[1]) - (0.0 if is_peak else np.pi)
    ph = (ph + np.pi) % (2 * np.pi) - np.pi
    return c + ph / w


def refine(y_s, y_raw, c, is_peak):
    if METHOD == "argmax":
        return float(c)
    try:
        if METHOD == "sine":
            return _sine_center(y_raw, c, is_peak)
        return _cross_center(y_s, c, is_peak)
    except Exception:  # 边界异常回退到粗定位
        return float(c)


def extract(data, smooth_n=SMOOTH_N, env_thr=ENV_THR, prominence=PROMINENCE):
    ch1, ch2 = data[:, 0], data[:, 1]
    s = uniform_filter1d(ch1, smooth_n)
    rows = []
    for k, (a, b) in enumerate(find_envelopes(s, env_thr=env_thr), start=BASE):
        seg = s[a : b + 1]
        pk, _ = find_peaks(seg, prominence=prominence, distance=MIN_DIST)
        vl, _ = find_peaks(-seg, prominence=prominence, distance=MIN_DIST)
        pts = sorted(
            [(int(p) + a, "peak") for p in pk] + [(int(v) + a, "valley") for v in vl]
        )
        for j, (c0, kind) in enumerate(pts, start=BASE):
            cf = refine(s, ch1, c0, kind == "peak")  # 亚采样点中心(小数)
            pos = int(round(cf))  # 取值用最近的整数点 ±HALF
            lo, hi = pos - HALF, pos + HALF + 1
            if lo < 0 or hi > ch2.size:  # 越界保护
                continue
            rows.append(
                dict(
                    packet=k,
                    idx=j,
                    sample=pos,
                    center=cf,
                    kind=kind,
                    ch1=ch1[lo:hi].mean(),
                    ch2=ch2[lo:hi].mean(),
                )
            )
    df = pd.DataFrame(rows).set_index(["packet", "idx"])
    return df
