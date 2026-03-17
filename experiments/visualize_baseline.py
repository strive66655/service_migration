from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def visualize_baseline_results(csv_file: str | Path, output_dir: str | Path = "experiments/results/figures") -> None:
    csv_path = Path(csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    metrics = [
        ("avg_delay", "Average Delay"),
        ("avg_total_cost", "Average Total Cost"),
        ("avg_migrations", "Average Migrations"),
        ("avg_failed_allocations", "Average Failed Allocations"),
        ("avg_load_ratio", "Average Load Ratio"),
    ]

    for col, title in metrics:
        plt.figure(figsize=(7, 4.5))
        plt.bar(df["Policy"], df[col])
        plt.title(title)
        plt.xlabel("Policy")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(output_path / f"{col}.png", dpi=160, bbox_inches="tight")
        plt.close()

    norm_df = df.copy()
    main_cols = [
        "avg_delay",
        "avg_total_cost",
        "avg_migrations",
        "avg_failed_allocations",
    ]

    for col in main_cols:
        max_val = norm_df[col].max()
        if max_val > 0:
            norm_df[col] = norm_df[col] / max_val

    plt.figure(figsize=(9, 5))
    x = list(range(len(norm_df["Policy"])))
    width = 0.2

    for i, col in enumerate(main_cols):
        offsets = [p + (i - 1.5) * width for p in x]
        plt.bar(offsets, norm_df[col], width=width, label=col)

    plt.xticks(x, norm_df["Policy"])
    plt.xlabel("Policy")
    plt.ylabel("Normalized value")
    plt.title("Baseline Comparison (Normalized Main Metrics)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "baseline_comparison_normalized.png", dpi=160, bbox_inches="tight")
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
        plt.figure(figsize=(8, 4.8))

        for policy in policies:
            sub_df = df[df["Policy"] == policy].sort_values("step")
            plt.plot(sub_df["step"], sub_df[col], marker="o", linewidth=1.8, label=policy)

        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"{col}_curve.png", dpi=160, bbox_inches="tight")
        plt.close()

    print(f"折线图已保存到: {output_path}")


if __name__ == "__main__":
    visualize_baseline_results("experiments/results/baseline_results.csv")
    visualize_step_curves("experiments/results/baseline_step_results.csv")