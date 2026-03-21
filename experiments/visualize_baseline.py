from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


CORE_METRICS = [
    ("avg_total_cost", "Average Total Cost"),
    ("avg_delay", "Average Delay"),
    ("avg_failed_allocations", "Failed Allocations"),
]

POLICY_COLORS = {
    "never_migrate": "#C44E52",
    "nearest": "#4C72B0",
    "cost_aware": "#55A868",
}

LEGACY_FIGURES = [
    "avg_delay.png",
    "avg_total_cost.png",
    "avg_migrations.png",
    "avg_failed_allocations.png",
    "avg_load_ratio.png",
    "baseline_comparison_normalized.png",
    "avg_delay_curve.png",
    "avg_total_cost_curve.png",
    "avg_migrations_curve.png",
    "avg_failed_allocations_curve.png",
    "avg_load_ratio_curve.png",
]


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


def _policy_color(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#667085")


def _format_value(metric: str, value: float) -> str:
    if metric == "avg_failed_allocations":
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"
    return f"{value:.2f}"


def _clean_output_dir(output_path: Path) -> None:
    for filename in LEGACY_FIGURES + ["baseline_overview.png", "cost_trend.png"]:
        file_path = output_path / filename
        if file_path.exists():
            file_path.unlink()


def _draw_metric_bar(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str) -> None:
    metric_df = df[["Policy", metric]].sort_values(metric, ascending=True)
    colors = [_policy_color(policy) for policy in metric_df["Policy"]]
    bars = ax.barh(metric_df["Policy"], metric_df[metric], color=colors, height=0.58)

    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(title)
    ax.set_ylabel("")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    max_val = float(metric_df[metric].max()) if not metric_df.empty else 0.0
    label_offset = max(max_val * 0.02, 0.02)

    for bar, value in zip(bars, metric_df[metric]):
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            _format_value(metric, float(value)),
            va="center",
            ha="left",
            color="#344054",
            fontsize=9,
        )

    right_limit = max_val + label_offset * 6 if max_val > 0 else 1.0
    ax.set_xlim(0, right_limit)


def _spread_end_labels(values: list[float], min_gap: float) -> list[float]:
    if not values:
        return []

    adjusted = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    prev_y: float | None = None

    for index, value in adjusted:
        target = value if prev_y is None else max(value, prev_y + min_gap)
        output[index] = target
        prev_y = target

    return output


def visualize_baseline_results(
    csv_file: str | Path, output_dir: str | Path = "experiments/results/figures"
) -> None:
    csv_path = Path(csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    _setup_matplotlib_style()
    _clean_output_dir(output_path)

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    axes = axes.flatten()

    for axis, (metric, title) in zip(axes[:3], CORE_METRICS):
        _draw_metric_bar(axis, df, metric, title)

    norm_df = df.copy()
    for metric, _ in CORE_METRICS:
        max_val = norm_df[metric].max()
        norm_df[metric] = norm_df[metric] / max_val if max_val > 0 else 0.0

    norm_df = norm_df.sort_values("avg_total_cost", ascending=True)
    x_positions = list(range(len(norm_df)))
    width = 0.22
    hatches = ["", "//", "xx"]

    comparison_ax = axes[3]
    for idx, (metric, title) in enumerate(CORE_METRICS):
        offsets = [x + (idx - 1) * width for x in x_positions]
        comparison_ax.bar(
            offsets,
            norm_df[metric],
            width=width,
            color=[_policy_color(policy) for policy in norm_df["Policy"]],
            alpha=0.9,
            edgecolor="#FFFFFF",
            linewidth=0.6,
            hatch=hatches[idx],
        )

    comparison_ax.set_title("Normalized Core Metrics", loc="left", pad=10)
    comparison_ax.set_ylabel("Relative value")
    comparison_ax.set_xlabel("Policy")
    comparison_ax.set_xticks(x_positions)
    comparison_ax.set_xticklabels(norm_df["Policy"])
    comparison_ax.set_ylim(0, 1.12)
    comparison_ax.yaxis.grid(True)
    comparison_ax.xaxis.grid(False)

    metric_handles = [
        Patch(facecolor="#D0D5DD", edgecolor="#98A2B3", hatch=hatches[idx], label=title)
        for idx, (_, title) in enumerate(CORE_METRICS)
    ]
    policy_handles = [
        Patch(facecolor=_policy_color(policy), edgecolor="none", label=policy)
        for policy in norm_df["Policy"]
    ]

    fig.suptitle("Baseline Policy Overview", fontsize=15, fontweight="semibold", y=0.98)
    fig.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.35, 0.01),
        ncol=len(metric_handles),
        frameon=False,
    )
    fig.legend(
        handles=policy_handles,
        loc="lower center",
        bbox_to_anchor=(0.8, 0.01),
        ncol=len(policy_handles),
        frameon=False,
        title="Policy",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.96), w_pad=2.4, h_pad=2.2)
    fig.savefig(output_path / "baseline_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved overview figure to: {output_path / 'baseline_overview.png'}")


def visualize_step_curves(
    step_csv_file: str | Path, output_dir: str | Path = "experiments/results/figures"
) -> None:
    step_csv_path = Path(step_csv_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not step_csv_path.exists():
        raise FileNotFoundError(f"Step CSV file not found: {step_csv_path}")

    _setup_matplotlib_style()

    df = pd.read_csv(step_csv_path)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    policy_frames: list[tuple[str, float, float, str]] = []

    for policy in df["Policy"].unique():
        sub_df = df[df["Policy"] == policy].sort_values("step")
        color = _policy_color(policy)
        ax.plot(
            sub_df["step"],
            sub_df["avg_total_cost"],
            linewidth=2.3,
            color=color,
            label=policy,
        )

        last_row = sub_df.iloc[-1]
        policy_frames.append(
            (
                policy,
                float(last_row["step"]),
                float(last_row["avg_total_cost"]),
                color,
            )
        )

    y_values = [frame[2] for frame in policy_frames]
    y_span = max(y_values) - min(y_values) if len(y_values) > 1 else 0.0
    min_gap = max(y_span * 0.08, 10.0)
    adjusted_y = _spread_end_labels(y_values, min_gap)

    for (policy, step_value, y_value, color), label_y in zip(policy_frames, adjusted_y):
        ax.text(
            step_value + 0.4,
            label_y,
            policy,
            color=color,
            va="center",
            fontsize=9,
        )

    ax.set_title("Average Total Cost over Time", loc="left", pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Total Cost")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.margins(x=0.1)

    y_min, y_max = ax.get_ylim()
    if adjusted_y:
        ax.set_ylim(y_min, max(y_max, max(adjusted_y) + min_gap * 0.35))

    fig.tight_layout()
    fig.savefig(output_path / "cost_trend.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved trend figure to: {output_path / 'cost_trend.png'}")


if __name__ == "__main__":
    visualize_baseline_results("experiments/results/baseline_results.csv")
    visualize_step_curves("experiments/results/baseline_step_results.csv")
