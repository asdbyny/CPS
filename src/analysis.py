import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict


def analyze_model_performance(evaluation_results: List[Dict]) -> None:
    """生成模型性能统计（复现表IV）"""
    df = pd.DataFrame(evaluation_results)

    # 按模型分组计算均值（保留3位小数，贴合文档风格）
    model_stats = df.groupby("model_name").agg({
        "correctness": lambda x: round(x.mean() * 100, 1),  # 正确率（%）
        "novelty_score": lambda x: round(x.mean(), 3),
        "process_score": lambda x: round(x.mean(), 3),
        "cps": lambda x: round(x.mean(), 3)
    }).reset_index()

    # 按CPS排序（类似中Gemini 1.5 Pro排第一）
    model_stats = model_stats.sort_values("cps", ascending=False)
    print("模型性能统计（对应表IV）：")
    print(model_stats.to_string(index=False))

    # 保存为CSV（用于论文表格）
    model_stats.to_csv("output/analysis/model_performance.csv", index=False)


def analyze_dwe_impact(evaluation_results: List[Dict]) -> None:
    """分析DWE对CPS的提升（复现表V）"""
    df = pd.DataFrame(evaluation_results)

    # 计算固定α=0.5时的CPS（用于对比）
    df["cps_fixed_alpha"] = 0.5 * df["novelty_score"] + 0.5 * df["process_score"]
    df["cps_fixed_alpha"] = df["cps_fixed_alpha"].round(3)

    # 按模型分组计算均值
    dwe_stats = df.groupby("model_name").agg({
        "cps_fixed_alpha": "mean",
        "cps": "mean"
    }).reset_index()

    # 计算提升率（%）
    dwe_stats["improvement_rate"] = round(
        (dwe_stats["cps"] - dwe_stats["cps_fixed_alpha"]) / dwe_stats["cps_fixed_alpha"] * 100, 1
    )

    print("\nDWE对CPS的提升（对应表V）：")
    print(dwe_stats.to_string(index=False))
    dwe_stats.to_csv("output/analysis/dwe_impact.csv", index=False)


def plot_creativity_rigor_tradeoff(evaluation_results: List[Dict]) -> None:
    """绘制创造力-严谨性权衡图（类似图1）"""
    df = pd.DataFrame(evaluation_results)
    plt.scatter(df["novelty_score"], df["process_score"], c=df["cps"], cmap="viridis")
    plt.xlabel("Novelty Score (创造力)")
    plt.ylabel("Process Score (严谨性)")
    plt.colorbar(label="CPS")
    plt.title("创造力-严谨性权衡（对应图1）")
    plt.savefig("output/analysis/tradeoff.png")
    plt.close()