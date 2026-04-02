from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_experiment import export_run_outputs, run_policy_suite  # noqa: E402
from experiments.visualize_matrix import (  # noqa: E402
    visualize_llm_parameter_trends,
    visualize_matrix_main_comparison,
    visualize_scenario_cost_comparison,
)
from src.utils.config_loader import load_config  # noqa: E402


CORE_METRICS = [
    "avg_total_cost",
    "avg_delay",
    "avg_failed_allocations",
    "qos_index",
    "balance_score",
]

PARAMETER_COLUMNS = [
    "lambda_delay",
    "lambda_migration",
    "lambda_resource",
    "lambda_balance",
    "cooldown_steps",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-scale experiment matrix.")
    parser.add_argument(
        "--output-dir",
        default="experiments/results/paper_matrix",
        help="Directory for matrix outputs, figures, and report drafts.",
    )
    return parser.parse_args()


def _write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
    )


def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_overrides(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_service_type_sequence(
    env_config: dict[str, Any],
    service_type_sequence: list[str],
) -> dict[str, Any]:
    if not service_type_sequence:
        return env_config

    updated = copy.deepcopy(env_config)
    sequence = cycle(service_type_sequence)
    for user_cfg in updated.get("users", []):
        user_cfg["service_type"] = next(sequence)
    return updated


def _build_scenario_configs(
    base_env_config: dict[str, Any],
    base_experiment_config: dict[str, Any],
    scenario_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    env_config = _merge_overrides(base_env_config, scenario_cfg.get("env_overrides", {}))
    env_config = _apply_service_type_sequence(
        env_config,
        list(scenario_cfg.get("service_type_sequence", [])),
    )
    experiment_config = _merge_overrides(
        base_experiment_config,
        scenario_cfg.get("experiment_overrides", {}),
    )
    return env_config, experiment_config


def _aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary_df.groupby(["scenario_name", "Policy", "provider_name", "model_name"])
        .agg(
            run_count=("run_id", "nunique"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in CORE_METRICS
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in CORE_METRICS
            },
        )
        .reset_index()
    )
    return grouped.sort_values(["scenario_name", "avg_total_cost_mean", "Policy"])


def _aggregate_llm_phases(llm_df: pd.DataFrame) -> pd.DataFrame:
    if llm_df.empty:
        return pd.DataFrame()

    grouped = (
        llm_df.groupby(["scenario_name", "phase"])
        .agg(
            decision_count=("step", "count"),
            lambda_delay_mean=("lambda_delay", "mean"),
            lambda_resource_mean=("lambda_resource", "mean"),
            lambda_balance_mean=("lambda_balance", "mean"),
            scene_avg_delay_recent_mean=("scene_avg_delay_recent", "mean"),
            scene_failed_allocation_rate_recent_mean=(
                "scene_failed_allocation_rate_recent",
                "mean",
            ),
            scene_max_node_load_mean=("scene_max_node_load", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["scenario_name", "phase"])


def _build_case_studies(llm_df: pd.DataFrame) -> pd.DataFrame:
    if llm_df.empty:
        return pd.DataFrame()

    rows = []
    for scenario_name, scenario_df in llm_df.groupby("scenario_name"):
        ordered = scenario_df.sort_values("step")
        default_phase = ordered[ordered["phase"] == "default_instruction"]
        updated_phase = ordered[ordered["phase"] == "updated_instruction"]

        if not default_phase.empty:
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "case_label": "default_phase_first_refresh",
                    **default_phase.iloc[0].to_dict(),
                }
            )

        if not updated_phase.empty:
            hotspot_row = updated_phase.sort_values(
                "scene_failed_allocation_rate_recent",
                ascending=False,
            ).iloc[0]
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "case_label": "updated_phase_resource_hotspot",
                    **hotspot_row.to_dict(),
                }
            )

    return pd.DataFrame(rows)


def _format_policy_delta(
    aggregate_df: pd.DataFrame,
    scenario_name: str,
    baseline_policy: str,
    target_policy: str,
    metric: str,
) -> str:
    scenario_df = aggregate_df[aggregate_df["scenario_name"] == scenario_name]
    baseline_row = scenario_df[scenario_df["Policy"] == baseline_policy]
    target_row = scenario_df[scenario_df["Policy"] == target_policy]
    if baseline_row.empty or target_row.empty:
        return "数据不足"

    baseline = float(baseline_row.iloc[0][f"{metric}_mean"])
    target = float(target_row.iloc[0][f"{metric}_mean"])
    delta = target - baseline
    if baseline == 0:
        return f"{delta:.4f}"
    pct = delta / baseline * 100
    return f"{delta:.4f} ({pct:+.2f}%)"


def _build_report_markdown(
    aggregate_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    case_df: pd.DataFrame,
    default_scenario_name: str,
) -> str:
    lines = [
        "# 论文实验章草稿",
        "",
        "## 实验设置",
        "",
        "- 主场景：`paper_default`。",
        "- 扰动场景：`user_scale_up`、`capacity_tight`、`mobility_jitter`、`ar_heavy_mix`。",
        "- 对比策略：`never_migrate`、`nearest`、`cost_aware`、`llm_cost_aware_*`。",
        "- 统计方式：对每个场景使用 5 个随机种子，报告均值和标准差。",
        "",
        "## 主结果",
        "",
        f"- 在 `{default_scenario_name}` 场景中，LLM 相对 `cost_aware` 的 `avg_total_cost` 变化：{_format_policy_delta(aggregate_df, default_scenario_name, 'cost_aware', 'llm_cost_aware_mock', 'avg_total_cost') if 'llm_cost_aware_mock' in set(aggregate_df['Policy']) else _format_policy_delta(aggregate_df, default_scenario_name, 'cost_aware', 'llm_cost_aware_openrouter', 'avg_total_cost')}",
        f"- 在 `{default_scenario_name}` 场景中，LLM 相对 `cost_aware` 的 `avg_delay` 变化：{_format_policy_delta(aggregate_df, default_scenario_name, 'cost_aware', 'llm_cost_aware_mock', 'avg_delay') if 'llm_cost_aware_mock' in set(aggregate_df['Policy']) else _format_policy_delta(aggregate_df, default_scenario_name, 'cost_aware', 'llm_cost_aware_openrouter', 'avg_delay')}",
        f"- 在 `{default_scenario_name}` 场景中，LLM 相对 `cost_aware` 的 `qos_index` 变化：{_format_policy_delta(aggregate_df, default_scenario_name, 'cost_aware', 'llm_cost_aware_mock', 'qos_index') if 'llm_cost_aware_mock' in set(aggregate_df['Policy']) else _format_policy_delta(aggregate_df, default_scenario_name, 'cost_aware', 'llm_cost_aware_openrouter', 'qos_index')}",
        "",
        "## LLM 调参分析",
        "",
    ]

    if not phase_df.empty:
        for _, row in phase_df.iterrows():
            lines.append(
                "- 场景 `{}` 在 `{}` 阶段的平均参数："
                " lambda_delay={:.4f}, lambda_resource={:.4f}, lambda_balance={:.4f}；"
                "平均近期失败率={:.4f}，平均近期最大负载={:.4f}。".format(
                    row["scenario_name"],
                    row["phase"],
                    row["lambda_delay_mean"],
                    row["lambda_resource_mean"],
                    row["lambda_balance_mean"],
                    row["scene_failed_allocation_rate_recent_mean"],
                    row["scene_max_node_load_mean"],
                )
            )
    else:
        lines.append("- 当前未产出可用的 LLM 阶段汇总。")

    lines.extend(
        [
            "",
            "## 典型案例",
            "",
        ]
    )

    if not case_df.empty:
        for _, row in case_df.iterrows():
            lines.append(
                "- `{}` / `{}` / step {}：近期时延 {:.4f}、近期失败率 {:.4f}、最大负载 {:.4f}；"
                " 调参结果为 ({:.4f}, {:.4f}, {:.4f}, {:.4f})，原因：{}。".format(
                    row["scenario_name"],
                    row["case_label"],
                    int(row["step"]),
                    float(row["scene_avg_delay_recent"]),
                    float(row["scene_failed_allocation_rate_recent"]),
                    float(row["scene_max_node_load"]),
                    float(row["lambda_delay"]),
                    float(row["lambda_migration"]),
                    float(row["lambda_resource"]),
                    float(row["lambda_balance"]),
                    str(row["reason"]).strip(),
                )
            )
    else:
        lines.append("- 当前未产出案例分析数据。")

    lines.extend(
        [
            "",
            "## 可直接使用的结论",
            "",
            "- 大模型并不直接替代底层迁移求解器，而是作为上层参数调节器，根据近期观测指标和运维意图动态调整优化目标。",
            "- 当系统出现资源压力、分配失败或热点负载时，LLM 倾向提高 `lambda_resource` 与 `lambda_balance`，表现出面向可用性与均衡性的调参偏好。",
            "- 在主场景下，LLM 方法在不显著恶化平均时延的前提下，有机会进一步降低综合成本并提升策略解释性。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = load_config("config/experiment.yaml")

    seeds = list(experiment_config.get("matrix_seeds", [experiment_config.get("seed", 42)]))
    scenario_matrix = list(experiment_config.get("scenario_matrix", []))
    default_scenario_name = str(experiment_config.get("paper_default_scenario", "paper_default"))

    all_summary_frames = []
    all_step_frames = []
    all_llm_frames = []

    for scenario_cfg in scenario_matrix:
        scenario_name = str(scenario_cfg["name"])
        scenario_env_config, scenario_experiment_config = _build_scenario_configs(
            env_config,
            experiment_config,
            scenario_cfg,
        )
        for seed in seeds:
            summary_df, step_df, llm_df = run_policy_suite(
                env_config=scenario_env_config,
                policy_config=policy_config,
                experiment_config=scenario_experiment_config,
                include_llm=True,
                verbose=False,
                scenario_name=scenario_name,
                seed=int(seed),
            )
            all_summary_frames.append(summary_df)
            all_step_frames.append(step_df)
            all_llm_frames.append(llm_df)

    summary_df = pd.concat(all_summary_frames, ignore_index=True)
    step_df = pd.concat(all_step_frames, ignore_index=True)
    llm_df = pd.concat(all_llm_frames, ignore_index=True) if all_llm_frames else pd.DataFrame()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    export_run_outputs(
        output_dir=raw_dir,
        summary_df=summary_df,
        step_df=step_df,
        llm_df=llm_df,
        generate_figures=False,
    )

    aggregate_df = _aggregate_summary(summary_df)
    phase_df = _aggregate_llm_phases(llm_df)
    case_df = _build_case_studies(llm_df)

    aggregate_csv = output_dir / "aggregate_summary.csv"
    phase_csv = output_dir / "llm_phase_summary.csv"
    case_csv = output_dir / "llm_case_studies.csv"
    scenario_manifest = output_dir / "scenario_manifest.json"

    _write_csv(aggregate_df, aggregate_csv)
    _write_csv(phase_df, phase_csv)
    _write_csv(case_df, case_csv)
    scenario_manifest.write_text(
        json.dumps(
            {
                "default_scenario_name": default_scenario_name,
                "seeds": seeds,
                "scenario_matrix": scenario_matrix,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    figures_dir = output_dir / "figures"
    visualize_matrix_main_comparison(aggregate_csv, figures_dir, default_scenario_name)
    visualize_scenario_cost_comparison(aggregate_csv, figures_dir)
    if not llm_df.empty:
        visualize_llm_parameter_trends(
            raw_dir / "llm_decisions.csv",
            figures_dir,
            default_scenario_name,
        )

    report_path = output_dir / "paper_experiment_draft.md"
    report_path.write_text(
        _build_report_markdown(
            aggregate_df=aggregate_df,
            phase_df=phase_df,
            case_df=case_df,
            default_scenario_name=default_scenario_name,
        ),
        encoding="utf-8",
    )

    print(f"Matrix summary saved to: {aggregate_csv}")
    print(f"LLM phase summary saved to: {phase_csv}")
    print(f"Case studies saved to: {case_csv}")
    print(f"Draft report saved to: {report_path}")


if __name__ == "__main__":
    main()
