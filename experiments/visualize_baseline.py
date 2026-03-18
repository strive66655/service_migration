from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_NAMES = {
    "never_migrate": "Never",
    "nearest": "Nearest",
    "cost_aware": "Cost-aware",
    "llm_policy": "LLM+Solver",
}

CORE_METRICS = [
    ("avg_delay", "Average Delay"),
    ("avg_total_cost", "Average Total Cost"),
    ("avg_migrations", "Average Migrations"),
    ("avg_failed_allocations", "Failed Allocations"),
]

TIME_SERIES_METRICS = [
    ("avg_total_cost", "Average Total Cost over Time"),
    ("avg_delay", "Average Delay over Time"),
]

LLM_PARAM_METRICS = [
    ("llm_lambda_delay", "Lambda Delay"),
    ("llm_lambda_migration", "Lambda Migration"),
    ("llm_lambda_resource", "Lambda Resource"),
    ("llm_lambda_balance", "Lambda Balance"),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _prepare_policy_labels(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["PolicyLabel"] = plot_df["Policy"].map(DISPLAY_NAMES).fillna(plot_df["Policy"])
    return plot_df


def visualize_baseline_results(
    csv_file: str | Path,
    output_dir: str | Path = "experiments/results/figures",
) -> None:
    csv_path = Path(csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = _prepare_policy_labels(pd.read_csv(csv_path))

    available_metrics = [(col, title) for col, title in CORE_METRICS if col in df.columns]
    if not available_metrics:
        raise ValueError("No supported metrics found in baseline results.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, available_metrics):
        ax.bar(df["PolicyLabel"], df[col], color=COLORS[: len(df)], width=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Policy")
        ax.set_ylabel(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    for ax in axes[len(available_metrics) :]:
        ax.axis("off")

    fig.suptitle("Baseline Policy Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path / "baseline_policy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved summary figure to {output_path}")


def _plot_standard_curves(df: pd.DataFrame, output_path: Path) -> None:
    policies = list(df["PolicyLabel"].drop_duplicates())

    for col, title in TIME_SERIES_METRICS:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8.2, 4.8))
        for idx, policy in enumerate(policies):
            sub_df = df[df["PolicyLabel"] == policy].sort_values("step")
            plt.plot(
                sub_df["step"],
                sub_df[col],
                linewidth=2.2,
                label=policy,
                color=COLORS[idx % len(COLORS)],
            )

        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(title.replace(" over Time", ""))
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path / f"{col}_curve.png", dpi=180, bbox_inches="tight")
        plt.close()


def _plot_llm_parameter_curves(df: pd.DataFrame, output_path: Path) -> None:
    llm_df = df[df["Policy"] == "llm_policy"].sort_values("step")
    if llm_df.empty:
        return

    available_metrics = [(col, title) for col, title in LLM_PARAM_METRICS if col in llm_df.columns]
    if not available_metrics:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, available_metrics):
        ax.plot(llm_df["step"], llm_df[col], linewidth=2.2, color="#C44E52")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.25)

    for ax in axes[len(available_metrics) :]:
        ax.axis("off")

    fig.suptitle("LLM Dynamic Parameter Curves", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path / "llm_parameter_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_llm_mode_distribution(df: pd.DataFrame, output_path: Path) -> None:
    llm_df = df[df["Policy"] == "llm_policy"].copy()
    if llm_df.empty or "llm_mode" not in llm_df.columns:
        return

    mode_counts = llm_df["llm_mode"].fillna("unknown").value_counts()
    if mode_counts.empty:
        return

    plt.figure(figsize=(6.6, 4.6))
    plt.bar(mode_counts.index.astype(str), mode_counts.values, color=["#4C72B0", "#C44E52", "#55A868"])
    plt.title("LLM Mode Distribution")
    plt.xlabel("Mode")
    plt.ylabel("Step Count")
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path / "llm_mode_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()


def visualize_step_curves(
    step_csv_file: str | Path,
    output_dir: str | Path = "experiments/results/figures",
) -> None:
    step_csv_path = Path(step_csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not step_csv_path.exists():
        raise FileNotFoundError(f"Step CSV file not found: {step_csv_path}")

    df = _prepare_policy_labels(pd.read_csv(step_csv_path))

    _plot_standard_curves(df, output_path)
    _plot_llm_parameter_curves(df, output_path)
    _plot_llm_mode_distribution(df, output_path)

    print(f"Saved time-series figures to {output_path}")


if __name__ == "__main__":
    visualize_baseline_results("experiments/results/baseline_results.csv")
    visualize_step_curves("experiments/results/baseline_step_results.csv")
