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


def analyze_signal_levels_pandas(data, threshold=1.5, min_run_length=1):
    """
    使用 Pandas 优化的信号电平分析函数
    """
    # 转换为 DataFrame
    df = pd.DataFrame({"voltage": data})
    df["pos"] = range(len(df))

    df["high_level"] = (df["voltage"] > threshold_high).astype(int)  # 1=High, 0=Low
    df["low_level"] = (df["voltage"] < threshold_low).astype(int)  # 1=High, 0=Low
    df["normal_level"] = (df["voltage"] > threshold_low) & (
        df["voltage"] < threshold_high
    ).astype(
        int
    )  # 1=Normal, 0=High or Low

    # 二值化
    df["level"] = (df["voltage"] > threshold).astype(int)  # 1=High, 0=Low

    # 检测电平变化（翻转点）
    df["change"] = df["level"].diff().ne(0)  # True 表示发生变化
    df["segment_id"] = df["change"].cumsum().fillna(0).astype(int)

    # 计算每段统计信息
    segments = (
        df.groupby("segment_id")
        .agg(
            start=("pos", "min"),
            end=("pos", "max"),
            length=("pos", "count"),
            level=("level", "first"),
            mean_voltage=("voltage", "mean"),
            std_voltage=("voltage", "std"),
            min_voltage=("voltage", "min"),
            max_voltage=("voltage", "max"),
        )
        .reset_index(drop=True)
    )

    # 过滤过短的段（噪声）
    segments = segments[segments["length"] >= min_run_length].reset_index(drop=True)

    # 重新分配连续的段编号（过滤后）
    segments["segment_id"] = range(len(segments))

    # 生成每点对应的 segment_id（过滤后）
    df["segment_id"] = -1
    for idx, row in segments.iterrows():
        df.loc[row["start"] : row["end"], "segment_id"] = idx

    flip_count = max(0, len(segments) - 1)

    # 增加 human-readable 的 level 列
    segments["level_str"] = segments["level"].map({1: "HIGH", 0: "LOW"})

    return df, segments, flip_count


# ====================== 使用示例 ======================

if __name__ == "__main__":
    # ================== 生成测试数据 ==================
    data = np.load("./acquired_results.npy")
    data = np.array(data.ravel())

    # ================== 分析 ==================
    df, segments, flip_count = analyze_signal_levels_pandas(
        data,
        threshold=0.0,
        min_run_length=5,
    )

    print(f"总采样点数: {len(data)}")
    print(f"电平翻转次数: {flip_count} 次")
    print(f"有效电平段数量: {len(segments)}\n")

    # 显示段信息（精简版）
    print(
        segments[
            ["segment_id", "level_str", "start", "end", "length", "mean_voltage"]
        ].round(3)
    )

    # ====================== 可视化 ======================
    plt.figure(figsize=(15, 6))
    plt.plot(df["voltage"], label="原始电压信号", linewidth=1.5)
    plt.plot(df["segment_id"] * 2, "r--", label="段编号 (台阶)", linewidth=2)

    for _, row in segments.iterrows():
        plt.axvline(row["start"], color="red", linestyle=":", alpha=0.6)

    plt.xlabel("采样点序号")
    plt.ylabel("电压 / 段编号")
    plt.title("Pandas优化版 - 光信号电平段识别")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 保存结果
    segments.to_csv("signal_segments.csv", index=False)
    df[["voltage", "level", "segment_id"]].to_csv("full_labeled_data.csv", index=False)

    print("\n文件已保存：")
    print("   • signal_segments.csv     ← 每段汇总信息（推荐使用）")
    print("   • full_labeled_data.csv   ← 完整逐点标注数据")
