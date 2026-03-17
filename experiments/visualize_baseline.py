from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


CORE_METRICS = [
    ("avg_delay", "Average Delay", False),
    ("avg_total_cost", "Average Total Cost", False),
    ("avg_failed_allocations", "Average Failed Allocations", False),
    ("avg_migrations", "Average Migrations", False),
]


def _normalize_for_comparison(series: pd.Series, lower_is_better: bool) -> pd.Series:
    min_v = series.min()
    max_v = series.max()

    if max_v - min_v < 1e-9:
        return pd.Series([1.0] * len(series), index=series.index)

    if lower_is_better:
        return (max_v - series) / (max_v - min_v)
    return (series - min_v) / (max_v - min_v)


def visualize_baseline_results(csv_file: str | Path, output_dir: str | Path = "experiments/results/figures") -> None:
    csv_path = Path(csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    total_users = 15
    if "avg_failed_allocations" in df.columns:
        df["allocation_success_rate"] = 1 - (df["avg_failed_allocations"] / total_users)
        df["allocation_success_rate"] = df["allocation_success_rate"].clip(lower=0, upper=1)

    metrics = [
        ("avg_delay", "Average Delay"),
        ("avg_total_cost", "Average Total Cost"),
        ("avg_migrations", "Average Migrations"),
        ("avg_failed_allocations", "Average Failed Allocations"),
        ("allocation_success_rate", "Allocation Success Rate"),
        ("avg_load_ratio", "Average Load Ratio"),
    ]

    for col, title in metrics:
        if col not in df.columns:
            continue
        plt.figure(figsize=(7, 4.5))
        plt.bar(df["Policy"], df[col])
        plt.title(title)
        plt.xlabel("Policy")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(output_path / f"{col}.png", dpi=160, bbox_inches="tight")
        plt.close()

    # 核心指标归一化后统一成“越高越好”，用于更公平对比
    score_df = df[["Policy"]].copy()
    for col, _, higher_is_better in CORE_METRICS:
        if col not in df.columns:
            continue
        score_df[col] = _normalize_for_comparison(df[col], lower_is_better=not higher_is_better)

    # 归一化热力图（策略 × 核心指标）
    heatmap_df = score_df.set_index("Policy")
    plt.figure(figsize=(8.5, 4.8))
    plt.imshow(heatmap_df.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=20, ha="right")
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.colorbar(label="Normalized score (higher is better)")
    plt.title("Policy Scorecard (Normalized Core Metrics)")
    plt.tight_layout()
    plt.savefig(output_path / "policy_scorecard_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 综合得分（按业务常见关注度加权）
    metric_weights = {
        "avg_delay": 0.35,
        "avg_total_cost": 0.35,
        "avg_failed_allocations": 0.20,
        "avg_migrations": 0.10,
    }
    available_weights = {k: v for k, v in metric_weights.items() if k in score_df.columns}
    weight_sum = sum(available_weights.values())
    normalized_weights = {k: v / weight_sum for k, v in available_weights.items()}

    score_df["overall_score"] = 0.0
    for metric, weight in normalized_weights.items():
        score_df["overall_score"] += score_df[metric] * weight

    ranked_df = score_df.sort_values("overall_score", ascending=False)
    plt.figure(figsize=(7.2, 4.6))
    bars = plt.bar(ranked_df["Policy"], ranked_df["overall_score"])
    for bar, value in zip(bars, ranked_df["overall_score"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    plt.ylim(0, 1.05)
    plt.xlabel("Policy")
    plt.ylabel("Weighted score")
    plt.title("Overall Policy Ranking (Delay/Cost/Failures/Migrations)")
    plt.tight_layout()
    plt.savefig(output_path / "overall_policy_ranking.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"柱状图已保存到: {output_path}")


def visualize_step_curves(step_csv_file: str | Path, output_dir: str | Path = "experiments/results/figures") -> None:
    step_csv_path = Path(step_csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not step_csv_path.exists():
        raise FileNotFoundError(f"Step CSV file not found: {step_csv_path}")

    df = pd.read_csv(step_csv_path)

    metrics = [
        ("avg_delay", "Average Delay over Time"),
        ("avg_total_cost", "Average Total Cost over Time"),
        ("avg_migrations", "Migrations over Time"),
        ("avg_failed_allocations", "Failed Allocations over Time"),
        ("avg_load_ratio", "Average Load Ratio over Time"),
    ]

    policies = df["Policy"].unique()

    for col, title in metrics:
        if col not in df.columns:
            continue
        plt.figure(figsize=(8, 4.8))

        for policy in policies:
            sub_df = df[df["Policy"] == policy].sort_values("step")
            plt.plot(sub_df["step"], sub_df[col], linewidth=1.8, alpha=0.35)

            smooth = sub_df[col].rolling(window=5, min_periods=1).mean()
            plt.plot(sub_df["step"], smooth, linewidth=2.4, label=f"{policy} (MA-5)")

        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"{col}_curve.png", dpi=160, bbox_inches="tight")
        plt.close()

    # 累计迁移和累计失败分配，更能体现长期代价
    cumulative_metrics = [
        ("avg_migrations", "Cumulative Migrations"),
        ("avg_failed_allocations", "Cumulative Failed Allocations"),
    ]
    for col, title in cumulative_metrics:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8, 4.8))
        for policy in policies:
            sub_df = df[df["Policy"] == policy].sort_values("step")
            cumulative = sub_df[col].cumsum()
            plt.plot(sub_df["step"], cumulative, linewidth=2.2, label=policy)

        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(f"cumulative_{col}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"cumulative_{col}.png", dpi=160, bbox_inches="tight")
        plt.close()

    print(f"折线图已保存到: {output_path}")


if __name__ == "__main__":
    visualize_baseline_results("experiments/results/baseline_results.csv")
    visualize_step_curves("experiments/results/baseline_step_results.csv")
