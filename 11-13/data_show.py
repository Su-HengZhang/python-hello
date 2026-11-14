#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fmc_ad9648_data_plot.py — 读取FMC AD9648采集bin文件并用matplotlib显示波形
(全局变量配置版: 直接点"运行"即可, 不需要命令行参数)

bin格式(v5, FMC AD9648 128bit帧, 小端):
  双通道(chsel=0): 每4B = [ch1_lo,ch1_hi,ch2_lo,ch2_hi] 交错, 时间递增
                   即 numpy: np.memmap(f, dtype='<u2').reshape(-1,2), 列0=ch1 列1=ch2
  单通道(chsel=1/2): 纯 u16le 采样流(每2B一个点);
                   --layout 2ch 拆出的 .ch1.bin/.ch2.bin 也是这个格式, 直接用单通道模式读

码制(重要!):
  ad_14bit_to_16bit 是【零扩展】: 有效数据在低14bit, 高2bit恒0。
  lut_config 默认序列 0x14=0x05 (bit0=1) -> AD9648 输出【14bit补码】,
  零扩展后符号位丢失, 需要 bit13 符号扩展还原负压 -> SIGN_MODE='twos_comp'。
  若波形像"从中间撕开"或整体偏上, 改 'offset_binary'(码-8192) 或 'raw' 对照。

依赖: pip install numpy matplotlib
"""

import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# ==================== 配置(改这里) ====================
# BIN_FILE   = 'acq_20261107_180021.bin'              # bin文件路径; 留空 '' = 自动使用脚本同目录下最新修改的 .bin
BIN_FILE = "./data/2026-11-09-data-frame120/acq_20261109_162051.bin"  # bin文件路径; 留空 '' = 自动使用脚本同目录下最新修改的 .bin
CH_MODE = (
    "dual"  # 'dual'=双通道交错bin(chsel=0采的) / 'ch1','ch2'=单通道流(或.ch1/.ch2.bin)
)
CHANNELS = "all"  # 仅dual有效: 'all' 或 [1] 只看ch1 / [2] 只看ch2
START = 0  # 起始采样点号(每通道计)
LENGTH = 0  # 显示采样点数; 0 = 到文件尾
RATE = 50e6  # 每通道采样率Hz(横轴换算; 100MS/s填100e6)
MAX_POINTS = 200_000  # 显示抽稀上限(超过自动均匀抽稀; 看细节用START/LENGTH放大)
OUT_PNG = ""  # 图片保存路径; 留空 '' = 弹窗显示不保存
SIGN_MODE = "twos_comp"  # 'twos_comp'=14bit补码(默认) / 'offset_binary'=偏移二进制 / 'raw'=原码0..16383
SHOW_VOLT = True  # True=纵轴换算电压; False=显示码值(LSB)
FS_VPP = 10  # AD9648满量程: 默认2Vpp差分(±8192码 ↔ ±1V)
# ======================================================


def to_signed(arr_u16):
    """u16零扩展码 -> 有符号码值(低14bit有效)"""
    c = arr_u16.astype(np.int32) & 0x3FFF
    if SIGN_MODE == "twos_comp":  # 14bit补码: bit13=符号
        return np.where(c >= 8192, c - 16384, c)
    if SIGN_MODE == "offset_binary":  # 偏移二进制: 0码=-满幅
        return c - 8192
    return arr_u16.astype(np.int32)  # raw


def main():
    # ---- 找bin文件 ----
    bin_file = BIN_FILE
    if not bin_file:
        cands = glob.glob(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.bin")
        )
        if not cands:
            cands = glob.glob("*.bin")
        if not cands:
            sys.exit("未指定BIN_FILE且当前目录没有.bin文件")
        bin_file = max(cands, key=os.path.getmtime)
        print(f"自动选择最新bin: {bin_file}")
    if not os.path.isfile(bin_file):
        sys.exit(f"文件不存在: {bin_file}")

    dual = CH_MODE == "dual"
    n_ch = 2 if dual else 1

    # ---- 读文件(memmap, 大文件不占内存); 零扩展码用 '<u2' ----
    raw = np.memmap(bin_file, dtype="<u2", mode="r")
    if raw.size % n_ch:
        print(
            f"[WARN] 文件长度{raw.size * 2}B不是{n_ch * 2}的倍数, 末尾{raw.size % n_ch}个半字忽略"
        )
        raw = raw[: raw.size // n_ch * n_ch]
    n_total = raw.size // n_ch
    if n_total == 0:
        sys.exit("文件为空")
    print(
        f"文件: {bin_file}  模式: {CH_MODE}  共 {n_total} 采样点/通道 "
        f"(时长 {n_total / RATE * 1e3:.3f}ms @ {RATE / 1e6:.3f}MS/s)"
    )

    # ---- 截取区间 ----
    start = max(0, min(START, n_total - 1))
    stop = n_total if LENGTH <= 0 else min(n_total, start + LENGTH)
    data = np.asarray(raw[start * n_ch : stop * n_ch]).reshape(-1, n_ch)
    n = data.shape[0]
    print(f"显示区间: 点 {start} ~ {stop - 1}  (共 {n} 点)")

    # ---- 抽稀(仅显示用) ----
    step = max(1, n // MAX_POINTS)
    # if step > 1:
    #     data = data[::step]
    #     print(f'抽稀: 每 {step} 点取 1 (显示 {data.shape[0]} 点; 细节请调START/LENGTH放大)')

    # t = np.arange(start, stop, step) / RATE * 1e6  # 横轴: 微秒
    # t = np.arange(start, stop, 1) / RATE * 1e6  # 横轴: 微秒
    t = np.arange(start, stop, 1)  # 横轴: 微秒

    # ---- 码制还原 + 电压换算(±8192码 ↔ ±FS_VPP/2) ----
    data = to_signed(data)
    if SHOW_VOLT:
        data = data.astype(np.float64) * (FS_VPP / 2.0 / 8192.0)
        unit = "V"
    else:
        unit = "LSB"

    # ---- 通道选择 ----
    if dual:
        chs = [1, 2] if str(CHANNELS).strip().lower() == "all" else list(CHANNELS)
        for c in chs:
            if c not in (1, 2):
                sys.exit(f"通道号非法: {c} (双通道只有1/2)")
    else:
        chs = [int(CH_MODE[-1])]  # 单通道流: 标题标来源通道

    # ---- 画图 ----
    fig, axes = plt.subplots(
        len(chs), 1, figsize=(12, 3.0 * len(chs)), sharex=True, squeeze=False
    )
    for i, (ax, c) in enumerate(zip(axes[:, 0], chs)):
        v = data[:, c - 1] if dual else data[:, 0]
        ax.plot(t, v, lw=0.6)
        ax.set_ylabel(f"ch{c} ({unit})")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        stats = f"min={v.min():.4g}  max={v.max():.4g}  mean={v.mean():.4g}"
        ax.set_title(f"ch{c}   {stats}", loc="right", fontsize=9)
        print(f"ch{c}: {stats} {unit}")
    axes[-1, 0].set_xlabel("时间 (µs)")
    fig.suptitle(f"{os.path.basename(bin_file)}  [{CH_MODE}, {SIGN_MODE}]")
    fig.tight_layout()

    if OUT_PNG:
        fig.savefig(OUT_PNG, dpi=150)
        print(f"已保存: {OUT_PNG}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
