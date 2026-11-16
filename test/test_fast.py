import time, numpy as np
import utils.estimate_signal as old  # 朴素版
import utils.estimate_signal_fast as new

# 1) 与朴素版在真实数据上一致
d = np.load("./data/data.npy")
p = np.load("./data/positions.npy")
s0, e0, _ = old.estimate_signal(d, p)
s1, e1, _ = new.estimate_signal(d, p)
print(
    "real data: max|ds| = %.2e, max|d err| = %.2e"
    % (np.abs(s0 - s1).max(), np.abs(e0 - e1).max())
)
t = time.perf_counter()
[old.estimate_signal(d, p) for _ in range(20)]
t0 = (time.perf_counter() - t) / 20
t = time.perf_counter()
[new.estimate_signal(d, p) for _ in range(20)]
t1 = (time.perf_counter() - t) / 20
print(
    "real data time: old %.2f ms, new %.2f ms (%.0fx)" % (t0 * 1e3, t1 * 1e3, t0 / t1)
)

# 2) 规模扩展
rng = np.random.default_rng(0)


def synth(M, N, noise=0.003, spikes=0.0):
    step = (M - 1) / (N - 1)
    pp = np.arange(N) * step + rng.uniform(-0.3, 0.3, N) * step  # 带抖动的准均匀网格
    pp[0] = 0
    pp[-1] = M - 1
    truth = np.cumsum(rng.normal(0, 0.01, pp.size))
    xx = np.arange(M, dtype=float)
    dd = np.interp(xx, pp, truth) + rng.normal(0, noise, M)
    m = rng.random(M) < spikes
    dd[m] += rng.normal(0, 0.2, m.sum())
    return xx, dd, pp, truth


for M, N, run_old in [
    (20000, 500, True),
    (200000, 5000, False),
    (2000000, 50000, False),
]:
    xx, dd, pp, truth = synth(M, N)
    t = time.perf_counter()
    s, e, _ = new.estimate_signal(dd, pp)
    tn = time.perf_counter() - t
    msg = f"M={M:>8}, N={pp.size:>6}: new {tn*1e3:8.1f} ms"
    if run_old:
        t = time.perf_counter()
        so, eo, _ = old.estimate_signal(dd, pp)
        to = time.perf_counter() - t
        msg += f", old {to*1e3:8.1f} ms, max|ds|={np.abs(s-so).max():.1e}"
    else:
        msg += ", old: 稠密矩阵 %.1f GB，不可行" % (M * pp.size * 8 / 1e9)
    print(
        msg, "| rms err vs truth %.4f, mean s_err %.4f" % (np.std(s - truth), e.mean())
    )

# 3) 离群点稳健性
xx, dd, pp, truth = synth(20000, 500, spikes=0.01)
for rb in (False, True):
    s, e, _ = new.estimate_signal(dd, pp, robust=rb)
    print("1%% spikes, robust=%s: rms err vs truth = %.4f" % (rb, np.std(s - truth)))

# 4) 边界/异常输入
xx, dd, pp, truth = synth(20000, 500)
dd[[5, 100]] = np.nan
s, _, _ = new.estimate_signal(dd, pp)
print("NaN 自动剔除: OK, nan in s =", np.isnan(s).any())
s, _, _ = new.estimate_signal(dd, pp, edge="extend")
print("edge='extend' 使用点数更多: OK")
try:
    dd2 = dd.copy()
    mask = (xx > pp[10]) & (xx < pp[12])
    dd2[mask] = np.nan  # 挖掉 p10~p12 之间所有数据
    new.estimate_signal(dd2, pp)
except ValueError as ex:
    print("覆盖不足 ->", ex)
try:
    new.estimate_signal(dd, pp[::-1])
except ValueError as ex:
    print("非递增 positions ->", ex)
