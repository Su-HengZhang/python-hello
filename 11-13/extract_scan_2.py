#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_scan.py — 以通道1(标识通道)锯齿的峰/谷中心为基准, 提取通道2(扫描数据)
  - 通道2取值: 中心点及其左右各1点(共3点)取平均
  - 输出: 双层索引 DataFrame, 索引 = (包序号 packet, 包内数据序号 idx), 均从1开始
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d

from utils.smask import g_matrix

# ============ 配置 ============
# BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"
# BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"
BIN_FILE = "./data/2026-11-07-data-frame1/acq_20261107_180021.bin"
SIGN_MODE = "twos_comp"  # 与 data_show.py 一致
FS_VPP = 10  # 与 data_show.py 一致 (±8192码 <-> ±FS_VPP/2)
SMOOTH_N = 5  # 仅用于"定位"峰谷的平滑点数, 不影响提取的原始数据
ENV_WIN = 120  # 包络判定窗口(点), 需 > 1个锯齿周期(~49点)
ENV_THR = 0.06  # 窗口内起伏(V) 超过此值视为"包络内"
PROMINENCE = 0.05  # 峰/谷显著度(V)
MIN_DIST = 30  # 相邻峰(或谷)最小间隔(点), 需 < 1个周期(~49点)
HALF = 1  # 中心左右各取几点平均
METHOD = "sine"  # 峰谷中心定位: 'cross50'(半幅交点中线, 默认) / 'sine'(局部正弦相位拟合) / 'argmax'(旧方法)
LOC_HALF = 25  # 计算局部高低电平的半窗口(点), 约 1 个周期
FRAC = 0.5  # cross50 的幅度比例(0.5=半高)
SINE_W = 24  # sine 方法拟合半窗口(点)
PERIOD = 49.0  # sine 方法的锯齿周期(点)

BASE = 0  # 索引起始编号 (1 或 0)
# ==============================


def load(bin_file):
    raw = np.fromfile(bin_file, dtype="<u2")
    raw = raw[: raw.size // 2 * 2].reshape(-1, 2)
    c = raw.astype(np.int32) & 0x3FFF
    if SIGN_MODE == "twos_comp":
        c = np.where(c >= 8192, c - 16384, c)
    elif SIGN_MODE == "offset_binary":
        c = c - 8192
    else:
        c = raw.astype(np.int32)
    return c.astype(np.float64) * (FS_VPP / 2.0 / 8192.0)  # 列0=ch1, 列1=ch2


def find_envelopes(s):
    """用滑动窗口内的 max-min 判定哪些区间有锯齿(包络内)"""
    rng = maximum_filter1d(s, ENV_WIN) - minimum_filter1d(s, ENV_WIN)
    act = rng > ENV_THR
    edge = np.flatnonzero(np.diff(act.astype(np.int8)))
    if act[0]:
        edge = np.r_[-1, edge]
    if act[-1]:
        edge = np.r_[edge, act.size - 1]
    return list(zip(edge[0::2] + 1, edge[1::2]))  # [(start, end), ...]


def _cross_center(y, c, is_peak):
    """局部半幅电平处, 左右两个交点(线性插值)的中点, 返回小数位置"""
    loc = y[max(0, c - LOC_HALF) : c + LOC_HALF + 1]
    hi, lo = loc.max(), loc.min()
    lvl = lo + FRAC * (hi - lo) if is_peak else hi - FRAC * (hi - lo)
    inside = (lambda v: v >= lvl) if is_peak else (lambda v: v <= lvl)
    l = c
    while l > 1 and inside(y[l - 1]) and c - l < 40:
        l -= 1
    r = c
    while r < y.size - 2 and inside(y[r + 1]) and r - c < 40:
        r += 1
    xl = (l - 1) + (lvl - y[l - 1]) / (y[l] - y[l - 1])
    xr = r + (lvl - y[r]) / (y[r + 1] - y[r])
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


def extract(data):
    ch1, ch2 = data[:, 0], data[:, 1]
    s = uniform_filter1d(ch1, SMOOTH_N)
    rows = []
    for k, (a, b) in enumerate(find_envelopes(s), start=BASE):
        seg = s[a : b + 1]
        pk, _ = find_peaks(seg, prominence=PROMINENCE, distance=MIN_DIST)
        vl, _ = find_peaks(-seg, prominence=PROMINENCE, distance=MIN_DIST)
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


if __name__ == "__main__":
    p, q = 41, 43
    N = p * q

    df = extract(load(BIN_FILE))
    # 获取 ch2 的 numpy 数组， 先 unstack，形状：(idx数, packet数)
    ch2_matrix = df["ch2"].unstack(level="packet").values

    # 偶数列与奇数列相减
    diff_matrix = ch2_matrix[:, 0::2] - ch2_matrix[:, 1::2]

    # 增加最后一行
    diff_matrix = np.vstack((diff_matrix, diff_matrix[-1]))

    # 去掉每个packet的前q-1个点
    diff_matrix = diff_matrix[q - 1 :, :]

    # 展平为一维数组
    mdata = diff_matrix.ravel(order="F")

    # 只取前 N 个点
    mdata = mdata[:N]

    # 由 G 关联重建
    img_g = np.zeros((p, q), dtype=np.float64)
    g = g_matrix(p, q)

    for i in range(N):
        mask = np.reshape(g[i], (p, q))
        img_g += mask * mdata[i]
    img_g = img_g / N

    img_g = img_g / (np.mean(mdata) * np.mean(g, axis=0).reshape((p, q)))

    plt.imshow(img_g)
    plt.show()
