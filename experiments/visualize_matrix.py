from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _setup_matplotlib_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D0D5DD",
            "axes.labelcolor": "#344054",
            "axes.titleweight": "semibold",
            "axes.titlecolor": "#101828",
            "axes.facecolor": "#FCFCFD",
            "figure.facecolor": "#FFFFFF",
            "grid.color": "#E4E7EC",
            "grid.alpha": 0.8,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
        }
    )


def _load_clean_csv(csv_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df.columns = [str(column).strip() for column in df.columns]
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return df


def _policy_color(policy: str) -> str:
    if policy.startswith("llm_cost_aware_"):
        return "#DD6B20"
    colors = {
        "never_migrate": "#C44E52",
        "nearest": "#4C72B0",
        "myopic": "#8172B3",
        "cost_aware": "#55A868",
    }
    return colors.get(policy, "#667085")


def _policy_rank(policy: str) -> int:
    order = {
        "never_migrate": 0,
        "nearest": 1,
        "myopic": 2,
        "cost_aware": 3,
        "llm_cost_aware_openrouter": 4,
        "llm_cost_aware_mock": 4,
    }
    if policy.startswith("llm_cost_aware_"):
        return 4
    return order.get(policy, 99)


def visualize_matrix_main_comparison(
    aggregate_csv: str | Path,
    output_dir: str | Path,
    scenario_name: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib_style()

    df = _load_clean_csv(aggregate_csv)
    scenario_df = df[df["scenario_name"] == scenario_name].copy()
    metrics = [
        ("avg_total_cost_mean", "Average Total Cost"),
        ("avg_delay_mean", "Average Delay"),
        ("avg_failed_allocations_mean", "Failed Allocations"),
        ("qos_index_mean", "QoS Index"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        sub_df = scenario_df.copy()
        sub_df["_policy_rank"] = sub_df["Policy"].map(_policy_rank)
        sub_df = sub_df.sort_values(
            ["_policy_rank", metric],
            ascending=[True, metric != "qos_index_mean"],
        )
        colors = [_policy_color(policy) for policy in sub_df["Policy"]]
        ax.barh(sub_df["Policy"], sub_df[metric], color=colors)
        ax.set_title(f"{title} ({scenario_name})", loc="left", pad=10)
        ax.invert_yaxis()
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")

    fig.tight_layout()
    output_file = output_path / "paper_main_comparison.png"
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_file


def visualize_scenario_cost_comparison(
    aggregate_csv: str | Path,
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib_style()

    df = _load_clean_csv(aggregate_csv)
    pivot_df = df.pivot(
        index="scenario_name",
        columns="Policy",
        values="avg_total_cost_mean",
    )
    pivot_df = pivot_df[[column for column in sorted(pivot_df.columns, key=_policy_rank)]]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    for policy in pivot_df.columns:
        ax.plot(
            pivot_df.index,
            pivot_df[policy],
            marker="o",
            linewidth=2.2,
            color=_policy_color(policy),
            label=policy,
        )

    ax.set_title("Scenario-wise Average Total Cost", loc="left", pad=10)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average Total Cost")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    output_file = output_path / "scenario_cost_comparison.png"
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_file


def visualize_llm_parameter_trends(
    llm_csv: str | Path,
    output_dir: str | Path,
    scenario_name: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _setup_matplotlib_style()

    df = _load_clean_csv(llm_csv)
    scenario_df = df[df["scenario_name"] == scenario_name].sort_values("step")
    metrics = [
        ("lambda_delay", "lambda_delay"),
        ("lambda_resource", "lambda_resource"),
        ("lambda_balance", "lambda_balance"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    phase_colors = {
        "default_instruction": "#4C72B0",
        "updated_instruction": "#DD6B20",
    }

    for ax, (metric, title) in zip(axes, metrics):
        for phase, phase_df in scenario_df.groupby("phase"):
            ax.plot(
                phase_df["step"],
                phase_df[metric],
                marker="o",
                linewidth=2,
                color=phase_colors.get(phase, "#667085"),
                label=phase,
            )
        ax.set_ylabel(title)
        ax.set_title(f"{title} trend ({scenario_name})", loc="left", pad=10)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

    axes[-1].set_xlabel("Step")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout()

    output_file = output_path / "llm_parameter_trends.png"
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_file
