from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.visualize_baseline import (  # noqa: E402
    visualize_baseline_results,
    visualize_step_curves,
)
from src.algorithms.cost_aware import CostAwarePolicy  # noqa: E402
from src.algorithms.llm_policy import LLMCostAwarePolicy  # noqa: E402
from src.algorithms.myopic import MyopicPolicy  # noqa: E402
from src.algorithms.nearest import NearestPolicy  # noqa: E402
from src.algorithms.never_migrate import NeverMigratePolicy  # noqa: E402
from src.algorithms.policy_params import PolicyParams  # noqa: E402
from src.env.env_builder import build_environment  # noqa: E402
from src.llm.controller import LLMPolicyController  # noqa: E402
from src.llm.providers.mock_provider import MockProvider  # noqa: E402
from src.llm.providers.openrouter_provider import OpenRouterProvider  # noqa: E402
from src.llm.providers.qwen_provider import QwenProvider  # noqa: E402
from src.runners.simulation_runner import SimulationRunner  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.metrics import SimulationMetrics  # noqa: E402


PolicyFactory = Callable[[], object]

RUN_METADATA_COLUMNS = [
    "run_id",
    "scenario_name",
    "seed",
    "total_steps",
    "observe_steps",
    "auto_generate_operator_instruction",
    "interactive_operator_input",
    "initial_instruction_empty",
    "instruction_updated",
]

SUMMARY_METRIC_COLUMNS = [
    "Policy",
    "provider_name",
    "model_name",
    "avg_delay",
    "avg_total_cost",
    "avg_migrations",
    "avg_failed_allocations",
    "avg_load_ratio",
    "avg_load_std",
    "qos_index",
    "delay_score",
    "failure_rate",
    "failure_rate_score",
    "load_health_score",
    "load_distribution_score",
    "balance_score",
]

STEP_METRIC_COLUMNS = [
    "Policy",
    "provider_name",
    "model_name",
    "step",
    "phase",
    "avg_delay",
    "avg_total_cost",
    "avg_migrations",
    "avg_failed_allocations",
    "avg_load_ratio",
    "avg_load_std",
    "qos_index",
    "delay_score",
    "failure_rate",
    "failure_rate_score",
    "load_health_score",
    "load_distribution_score",
    "balance_score",
]

LLM_DECISION_COLUMNS = [
    "run_id",
    "scenario_name",
    "seed",
    "Policy",
    "phase",
    "step",
    "provider",
    "model",
    "experiment_mode",
    "used_fallback",
    "scene_step",
    "scene_avg_node_load",
    "scene_max_node_load",
    "scene_failed_allocations_recent",
    "scene_migrations_recent",
    "scene_avg_delay_recent",
    "scene_migration_rate_recent",
    "scene_failed_allocation_rate_recent",
    "scene_users_in_cooldown_ratio",
    "scene_avg_migrations_per_user_recent",
    "scene_user_context_summary",
    "reason",
    "operator_instruction",
    "lambda_delay",
    "lambda_migration",
    "lambda_resource",
    "lambda_balance",
    "migrate_threshold",
    "cooldown_steps",
]


@dataclass(frozen=True)
class BuiltLLMPolicy:
    policy: LLMCostAwarePolicy
    provider_name: str
    model_name: str


POLICY_OVERRIDE_SECTIONS = {
    "cost_aware": "cost_aware",
}


def _policy_base_config(policy_config: dict) -> dict[str, Any]:
    excluded = {"llm", *POLICY_OVERRIDE_SECTIONS.values()}
    return {
        key: value
        for key, value in policy_config.items()
        if key not in excluded
    }


def resolve_policy_params(
    policy_config: dict,
    policy_name: str | None = None,
) -> PolicyParams:
    resolved = _policy_base_config(policy_config)
    if policy_name is not None:
        section_name = POLICY_OVERRIDE_SECTIONS.get(policy_name)
        if section_name:
            resolved.update(policy_config.get(section_name, {}))
    return PolicyParams.from_dict(resolved).normalized()


def build_llm_policy(
    policy_params: PolicyParams,
    policy_config: dict,
    experiment_config: dict,
) -> BuiltLLMPolicy:
    llm_cfg = policy_config.get("llm", {})
    provider_name = str(llm_cfg.get("provider", "mock")).lower()
    model_name = str(llm_cfg.get("model", "mock-model"))
    update_interval = int(llm_cfg.get("update_interval", 5))
    experiment_mode = str(
        experiment_config.get("llm_experiment_mode", "main")
    ).strip().lower()

    operator_instruction = str(
        experiment_config.get(
            "operator_instruction",
            "无显式运维指令，默认以网络稳定、资源效率与服务质量平衡为目标。",
        )
    )

    if provider_name == "qwen":
        try:
            provider = QwenProvider(
                model_name=model_name,
                temperature=float(llm_cfg.get("temperature", 0.1)),
            )
        except ValueError:
            provider = MockProvider()
            provider_name = "mock"
            model_name = "mock-model"
    elif provider_name == "openrouter":
        try:
            provider = OpenRouterProvider(
                model_name=model_name,
                temperature=float(llm_cfg.get("temperature", 0.1)),
                site_url=llm_cfg.get("site_url"),
                app_name=llm_cfg.get("app_name"),
            )
        except ValueError:
            provider = MockProvider()
            provider_name = "mock"
            model_name = "mock-model"
    else:
        provider = MockProvider()
        provider_name = "mock"
        model_name = "mock-model"

    controller = LLMPolicyController(
        provider=provider,
        provider_name=provider_name,
        model_name=model_name,
        default_params=policy_params,
        operator_instruction=operator_instruction,
        experiment_mode=experiment_mode,
    )

    return BuiltLLMPolicy(
        policy=LLMCostAwarePolicy(
            default_params=policy_params,
            controller=controller,
            update_interval=update_interval,
        ),
        provider_name=provider_name,
        model_name=model_name,
    )


def show_observation_metrics(summary: dict) -> None:
    print("\n========== 当前系统观测指标 ==========")
    for key, value in summary.items():
        if key in {"Policy", "provider_name", "model_name"}:
            continue
        try:
            print(f"{key}: {float(value):.4f}")
        except (TypeError, ValueError):
            print(f"{key}: {value}")
    print("=====================================\n")


def generate_operator_instruction_from_observation(
    observation_summary: dict,
    experiment_config: dict,
) -> str:
    avg_delay = float(observation_summary.get("avg_delay", 0.0))
    avg_failed = float(observation_summary.get("avg_failed_allocations", 0.0))
    avg_load_ratio = float(observation_summary.get("avg_load_ratio", 0.0))
    avg_load_std = float(observation_summary.get("avg_load_std", 0.0))
    avg_migrations = float(observation_summary.get("avg_migrations", 0.0))

    rule_cfg = experiment_config.get("operator_instruction_rules", {})

    failure_high = float(rule_cfg.get("failure_high", 2.0))
    load_ratio_high = float(rule_cfg.get("load_ratio_high", 0.75))
    load_std_high = float(rule_cfg.get("load_std_high", 0.20))
    delay_medium = float(rule_cfg.get("delay_medium", 32.0))
    delay_high = float(rule_cfg.get("delay_high", 38.0))
    migrations_high = float(rule_cfg.get("migrations_high", 2.0))
    migrations_low = float(rule_cfg.get("migrations_low", 0.3))

    main_focus = []
    extra_focus = []

    if avg_failed >= failure_high:
        main_focus.append(
            "当前阶段系统已经出现较明显的分配失败，说明资源分配压力较大。"
        )
        extra_focus.append(
            "接下来优先缓解资源紧张，尽量减少新的分配失败，先保证服务可用性。"
        )
    elif avg_load_ratio >= load_ratio_high:
        main_focus.append(
            "当前整体节点负载已经偏高，系统存在较明显的资源压力。"
        )
        extra_focus.append(
            "接下来优先缓解资源占用压力，避免部分节点持续过载。"
        )

    if avg_load_std >= load_std_high:
        main_focus.append("同时，节点间负载分布不够均衡，存在局部热点节点。")
        extra_focus.append("需要进一步关注负载均衡，避免热点节点长期承压。")

    if avg_delay >= delay_high:
        extra_focus.append(
            "当前业务时延已经偏高，后续在保证资源可用性的前提下，应优先改善响应速度。"
        )
    elif avg_delay >= delay_medium:
        extra_focus.append("时延目前处于中等偏高水平，需要尽量保持在可接受范围内。")

    if avg_migrations >= migrations_high:
        migration_guidance = "最近迁移偏频繁，后续应适当抑制不必要的切换，优先保持策略稳定。"
    elif avg_migrations <= migrations_low and avg_delay >= delay_medium:
        migration_guidance = "当前迁移并不活跃，不要因为过度保守而错失必要的调整机会。"
    else:
        migration_guidance = "迁移策略不宜过于激进，也不要过度抑制，应保留必要的调整空间。"

    if not main_focus and not extra_focus:
        return (
            "当前系统整体运行基本平稳。接下来继续保持时延、迁移开销、"
            "资源压力和负载均衡之间的综合平衡，不要进行过于激进的调整。"
        )

    parts = []
    parts.extend(main_focus)
    parts.extend(extra_focus)
    parts.append(migration_guidance)
    return "".join(parts)


def resolve_operator_instruction_from_observation(
    experiment_config: dict,
    observation_summary: dict,
) -> str:
    auto_generate = bool(
        experiment_config.get("auto_generate_operator_instruction", True)
    )

    generated_instruction = generate_operator_instruction_from_observation(
        observation_summary,
        experiment_config,
    )

    show_observation_metrics(observation_summary)

    if auto_generate:
        print("根据当前观测结果，系统自动生成的 operator_instruction 为：")
        print(f"\n{generated_instruction}\n")
        return generated_instruction

    default_instruction = str(
        experiment_config.get("operator_instruction", generated_instruction)
    ).strip() or generated_instruction

    print("请根据以上指标输入新的 operator_instruction。")
    print("直接回车：使用当前默认值。")

    try:
        user_input = input(f"\n默认值: {default_instruction}\n> ").strip()
    except EOFError:
        user_input = ""

    return user_input if user_input else default_instruction


def resolve_instruction_phase(
    step_idx: int,
    observe_steps: int,
    instruction_updated: bool,
) -> str:
    if step_idx <= observe_steps:
        return "default_instruction"
    if instruction_updated:
        return "updated_instruction"
    return "default_instruction"


def _run_id_for(scenario_name: str, seed: int | None) -> str:
    seed_label = "none" if seed is None else str(seed)
    return f"{scenario_name}__seed_{seed_label}"


def build_run_metadata(
    experiment_config: dict,
    scenario_name: str,
    seed: int | None,
) -> dict[str, Any]:
    steps = int(experiment_config.get("steps", 20))
    observe_steps = max(0, min(int(experiment_config.get("observe_steps", 20)), steps))
    return {
        "run_id": _run_id_for(scenario_name, seed),
        "scenario_name": scenario_name,
        "seed": seed,
        "total_steps": steps,
        "observe_steps": observe_steps,
        "auto_generate_operator_instruction": bool(
            experiment_config.get("auto_generate_operator_instruction", True)
        ),
        "interactive_operator_input": bool(
            experiment_config.get("interactive_operator_input", False)
        ),
        "initial_instruction_empty": bool(
            experiment_config.get("initial_instruction_empty", False)
        ),
        "instruction_updated": False,
    }


def _clean_tabular_frame(df: pd.DataFrame, preferred_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=preferred_columns)

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    ordered = [column for column in preferred_columns if column in cleaned.columns]
    remaining = [column for column in cleaned.columns if column not in ordered]
    return cleaned[ordered + remaining]


def _write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC,
    )


def build_step_row(
    metadata: dict[str, Any],
    policy_name: str,
    provider_name: str,
    model_name: str,
    step_idx: int,
    step_metric,
    phase: str = "full_run",
) -> dict[str, Any]:
    return {
        **metadata,
        "Policy": policy_name,
        "provider_name": provider_name,
        "model_name": model_name,
        "step": step_idx,
        "phase": phase,
        "avg_delay": step_metric.avg_delay,
        "avg_total_cost": step_metric.total_cost,
        "avg_migrations": step_metric.migration_count,
        "avg_failed_allocations": step_metric.failed_allocations,
        "avg_load_ratio": step_metric.avg_load_ratio,
        "avg_load_std": step_metric.load_std,
        **step_metric.qos_summary(),
    }


def summarize_simulation(
    step_metrics: list,
    metadata: dict[str, Any],
    policy_name: str,
    provider_name: str = "",
    model_name: str = "",
) -> dict[str, Any]:
    metrics = SimulationMetrics(step_metrics=step_metrics)
    summary = metrics.summary()
    summary.update(metadata)
    summary["Policy"] = policy_name
    summary["provider_name"] = provider_name
    summary["model_name"] = model_name
    return summary


def _policy_rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    return _clean_tabular_frame(pd.DataFrame(rows), columns)


def export_run_outputs(
    output_dir: str | Path,
    summary_df: pd.DataFrame,
    step_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    generate_figures: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_csv = output_path / "baseline_results.csv"
    step_csv = output_path / "baseline_step_results.csv"
    llm_csv = output_path / "llm_decisions.csv"

    _write_csv(summary_df, summary_csv)
    _write_csv(step_df, step_csv)
    if not llm_df.empty:
        _write_csv(llm_df, llm_csv)

    if generate_figures:
        figures_dir = output_path / "figures"
        visualize_baseline_results(summary_csv, figures_dir)
        visualize_step_curves(step_csv, figures_dir)

    return {
        "summary_csv": summary_csv,
        "step_csv": step_csv,
        "llm_csv": llm_csv,
    }


def run_policy_suite(
    env_config: dict,
    policy_config: dict,
    experiment_config: dict,
    include_llm: bool = True,
    verbose: bool = True,
    scenario_name: str = "paper_default",
    seed: int | None = None,
):
    shared_policy_params = resolve_policy_params(policy_config)
    cost_aware_params = resolve_policy_params(policy_config, "cost_aware")
    effective_seed = experiment_config.get("seed") if seed is None else seed
    steps = int(experiment_config.get("steps", 20))
    observe_steps = max(0, min(int(experiment_config.get("observe_steps", 20)), steps))
    interactive_operator_input = bool(
        experiment_config.get("interactive_operator_input", False)
    )
    initial_instruction_empty = bool(
        experiment_config.get("initial_instruction_empty", False)
    )

    policies: dict[str, tuple[PolicyFactory, PolicyParams]] = {
        "never_migrate": (lambda: NeverMigratePolicy(), shared_policy_params),
        "nearest": (lambda: NearestPolicy(), shared_policy_params),
        "myopic": (lambda: MyopicPolicy(shared_policy_params), shared_policy_params),
        "cost_aware": (lambda: CostAwarePolicy(cost_aware_params), cost_aware_params),
    }

    base_metadata = build_run_metadata(experiment_config, scenario_name, effective_seed)
    summary_results: list[dict[str, Any]] = []
    step_results: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []

    for name, (policy_factory, env_policy_params) in policies.items():
        env = build_environment(env_config, env_policy_params, seed=effective_seed)
        policy = policy_factory()
        runner = SimulationRunner(env, policy)
        metrics = runner.run(steps)

        summary = summarize_simulation(
            metrics.step_metrics,
            metadata=base_metadata,
            policy_name=name,
        )
        summary_results.append(summary)

        for step_idx, step_metric in enumerate(metrics.step_metrics, start=1):
            step_results.append(
                build_step_row(
                    metadata=base_metadata,
                    policy_name=name,
                    provider_name="",
                    model_name="",
                    step_idx=step_idx,
                    step_metric=step_metric,
                    phase="full_run",
                )
            )

        if verbose:
            print(f"\nPolicy: {name}")
            for key, value in summary.items():
                if key not in {"Policy", "provider_name", "model_name"}:
                    print(f"  {key}: {value}")

    if include_llm:
        initial_experiment_config = dict(experiment_config)
        if initial_instruction_empty:
            initial_experiment_config["operator_instruction"] = ""

        built_llm_policy = build_llm_policy(
            cost_aware_params,
            policy_config,
            initial_experiment_config,
        )
        llm_policy = built_llm_policy.policy
        llm_policy_name = f"llm_cost_aware_{built_llm_policy.provider_name}"
        env = build_environment(env_config, cost_aware_params, seed=effective_seed)
        runner = SimulationRunner(env, llm_policy)
        llm_step_metrics = []

        for _ in range(observe_steps):
            llm_step_metrics.append(runner.step())

        observation_summary = summarize_simulation(
            llm_step_metrics,
            metadata=base_metadata,
            policy_name=f"{llm_policy_name}_default_phase",
            provider_name=built_llm_policy.provider_name,
            model_name=built_llm_policy.model_name,
        )

        instruction_updated = False
        if interactive_operator_input and observe_steps < steps:
            new_instruction = resolve_operator_instruction_from_observation(
                experiment_config,
                observation_summary,
            )
            llm_policy.controller.operator_instruction = new_instruction.strip()
            instruction_updated = True

        llm_metadata = dict(base_metadata)
        llm_metadata["instruction_updated"] = instruction_updated

        remaining_steps = steps - observe_steps
        for _ in range(remaining_steps):
            llm_step_metrics.append(runner.step())

        summary = summarize_simulation(
            llm_step_metrics,
            metadata=llm_metadata,
            policy_name=llm_policy_name,
            provider_name=built_llm_policy.provider_name,
            model_name=built_llm_policy.model_name,
        )
        summary_results.append(summary)

        for step_idx, step_metric in enumerate(llm_step_metrics, start=1):
            step_results.append(
                build_step_row(
                    metadata=llm_metadata,
                    policy_name=llm_policy_name,
                    provider_name=built_llm_policy.provider_name,
                    model_name=built_llm_policy.model_name,
                    step_idx=step_idx,
                    step_metric=step_metric,
                    phase=resolve_instruction_phase(
                        step_idx,
                        observe_steps,
                        instruction_updated,
                    ),
                )
            )

        for row in llm_policy.decision_history:
            row_step = int(row.get("step", -1))
            llm_rows.append(
                {
                    "run_id": llm_metadata["run_id"],
                    "scenario_name": llm_metadata["scenario_name"],
                    "seed": llm_metadata["seed"],
                    "Policy": llm_policy_name,
                    "phase": resolve_instruction_phase(
                        row_step,
                        observe_steps,
                        instruction_updated,
                    ),
                    **row,
                }
            )

        if verbose:
            print(f"\nPolicy: {llm_policy_name}")
            for key, value in summary.items():
                if key not in {"Policy", "provider_name", "model_name"}:
                    print(f"  {key}: {value}")

    summary_df = _policy_rows_to_frame(
        summary_results,
        RUN_METADATA_COLUMNS + SUMMARY_METRIC_COLUMNS,
    )
    step_df = _policy_rows_to_frame(
        step_results,
        RUN_METADATA_COLUMNS + STEP_METRIC_COLUMNS,
    )
    llm_df = _policy_rows_to_frame(llm_rows, LLM_DECISION_COLUMNS)
    return summary_df, step_df, llm_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline + LLM experiment suite.")
    parser.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Directory for CSV and figure outputs.",
    )
    parser.add_argument(
        "--scenario-name",
        default="paper_default",
        help="Scenario name written into result metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation after exporting CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = load_config("config/experiment.yaml")

    summary_df, step_df, llm_df = run_policy_suite(
        env_config=env_config,
        policy_config=policy_config,
        experiment_config=experiment_config,
        include_llm=True,
        verbose=True,
        scenario_name=args.scenario_name,
        seed=args.seed,
    )

    outputs = export_run_outputs(
        output_dir=args.output_dir,
        summary_df=summary_df,
        step_df=step_df,
        llm_df=llm_df,
        generate_figures=not args.skip_figures,
    )

    print(f"\nSummary results saved to: {outputs['summary_csv']}")
    print(f"Step results saved to: {outputs['step_csv']}")
    if not llm_df.empty:
        print(f"LLM decisions saved to: {outputs['llm_csv']}")


if __name__ == "__main__":
    main()
