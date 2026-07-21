import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def analyze_signal_levels_pandas(
    data,
    threshold_high=0.35,
    threshold_low=0.05,
    frame_length=24,
):
    """
    使用 Pandas 优化的信号电平分析函数
    """
    # 转换为 DataFrame
    df = pd.DataFrame(data, columns=["ch0", "ch1"])
    # 通过Flag信号的状态，筛选基底播放区间的测量数据
    df = df.query("ch1 > @threshold_high or ch1 < @threshold_low")

    # Flag信号二值化
    threshold = (threshold_low + threshold_high) / 2
    df["ch1"] = (df["ch1"] > threshold).astype(int)  # 1=High, 0=Low

    df["ch1"] = df["ch1"].diff().ne(0).cumsum().astype(int)

    df_clean = df.groupby("ch1").head(frame_length)

    return df_clean["ch0"].values


# ====================== 使用示例 ======================

if __name__ == "__main__":
    # ================== 生成测试数据 ==================
    data = np.load("./acquired_results.npy")
    data = np.array(data.reshape(-1, 2))

    # ================== 分析 ==================
    mdata = analyze_signal_levels_pandas(data)

    # ====================== 可视化 ======================
    plt.figure(figsize=(15, 6))
    plt.plot(mdata, "*-", label="原始电压信号", linewidth=1.5)

    plt.xlabel("采样点序号")
    plt.ylabel("电压")
    plt.title("光信号")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
