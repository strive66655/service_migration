from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd

from experiments.visualize_baseline import (
    visualize_baseline_results,
    visualize_step_curves,
)
from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.llm_policy import LLMCostAwarePolicy
from src.algorithms.nearest import NearestPolicy
from src.algorithms.never_migrate import NeverMigratePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.env_builder import build_environment
from src.llm.controller import LLMPolicyController
from src.llm.providers.mock_provider import MockProvider
from src.llm.providers.openrouter_provider import OpenRouterProvider
from src.llm.providers.qwen_provider import QwenProvider
from src.runners.simulation_runner import SimulationRunner
from src.utils.config_loader import load_config
from src.utils.metrics import SimulationMetrics

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PolicyFactory = Callable[[], object]
BASELINE_SUMMARY_COLUMNS = [
    "Policy",
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

BASELINE_STEP_COLUMNS = [
    "Policy",
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


@dataclass(frozen=True)
class BuiltLLMPolicy:
    policy: LLMCostAwarePolicy
    provider_name: str
    model_name: str


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
        if key == "Policy":
            continue
        try:
            print(f"{key}: {float(value):.4f}")
        except (TypeError, ValueError):
            print(f"{key}: {value}")
    print("=====================================\n")


def resolve_operator_instruction_from_observation(
    experiment_config: dict,
    observation_summary: dict,
) -> str:
    default_instruction = str(
        experiment_config.get(
            "operator_instruction",
            "无显式运维指令，默认以网络稳定、资源效率与服务质量平衡为目标。",
        )
    ).strip()

    show_observation_metrics(observation_summary)
    print("请根据以上指标输入新的 operator_instruction。")
    print("直接回车：使用当前默认值。")

    try:
        user_input = input(f"\n默认值: {default_instruction}\n> ").strip()
    except EOFError:
        user_input = ""

    return user_input if user_input else default_instruction


def build_step_row(
    policy_name: str,
    step_idx: int,
    step_metric,
    phase: str = "full_run",
) -> dict:
    return {
        "Policy": policy_name,
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


def summarize_simulation(step_metrics: list, policy_name: str) -> dict:
    metrics = SimulationMetrics(step_metrics=step_metrics)
    summary = metrics.summary()
    summary["Policy"] = policy_name
    return summary


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


def run_policy_suite(
    env_config: dict,
    policy_config: dict,
    experiment_config: dict,
    include_llm: bool = True,
    verbose: bool = True,
):
    policy_params = PolicyParams.from_dict(policy_config).normalized()
    steps = int(experiment_config.get("steps", 20))
    seed = experiment_config.get("seed")

    observe_steps = int(experiment_config.get("observe_steps", 20))
    observe_steps = max(0, min(observe_steps, steps))

    interactive_operator_input = bool(
        experiment_config.get("interactive_operator_input", False)
    )

    # 是否让前半段用空指令。False 表示沿用 YAML 默认指令。
    initial_instruction_empty = bool(
        experiment_config.get("initial_instruction_empty", False)
    )

    policies: dict[str, PolicyFactory] = {
        "never_migrate": lambda: NeverMigratePolicy(),
        "nearest": lambda: NearestPolicy(),
        "cost_aware": lambda: CostAwarePolicy(policy_params),
    }

    summary_results: list[dict] = []
    step_results: list[dict] = []
    llm_rows: list[dict] = []

    # baseline
    for name, policy_factory in policies.items():
        env = build_environment(env_config, policy_params, seed=seed)
        policy = policy_factory()
        runner = SimulationRunner(env, policy)
        metrics = runner.run(steps)

        summary = metrics.summary()
        summary["Policy"] = name
        summary_results.append(summary)

        for step_idx, step_metric in enumerate(metrics.step_metrics, start=1):
            step_results.append(
                build_step_row(
                    name,
                    step_idx,
                    step_metric,
                    phase="full_run",
                )
            )

        if hasattr(policy, "decision_history"):
            for row in policy.decision_history:
                llm_rows.append(
                    {
                        "Policy": name,
                        "phase": "full_run",
                        **row,
                    }
                )

        if verbose:
            print(f"\nPolicy: {name}")
            for key, value in summary.items():
                if key != "Policy":
                    print(f"  {key}: {value:.4f}")

    # llm, same policy object across both phases
    if include_llm:
        initial_experiment_config = dict(experiment_config)
        if initial_instruction_empty:
            initial_experiment_config["operator_instruction"] = ""

        built_llm_policy = build_llm_policy(
            policy_params,
            policy_config,
            initial_experiment_config,
        )
        llm_policy = built_llm_policy.policy
        llm_policy_name = f"llm_cost_aware_{built_llm_policy.provider_name}"

        env = build_environment(env_config, policy_params, seed=seed)
        runner = SimulationRunner(env, llm_policy)

        llm_step_metrics = []

        # phase 1: default/empty instruction
        for _ in range(observe_steps):
            llm_step_metrics.append(runner.step())

        observation_summary = summarize_simulation(
            llm_step_metrics,
            f"{llm_policy_name}_default_phase",
        )

        # phase 2: optionally update instruction, but DO NOT rebuild policy object
        instruction_updated = False
        if interactive_operator_input and observe_steps < steps:
            new_instruction = resolve_operator_instruction_from_observation(
                experiment_config,
                observation_summary,
            )
            llm_policy.controller.operator_instruction = new_instruction.strip()
            instruction_updated = True

        remaining_steps = steps - observe_steps
        for _ in range(remaining_steps):
            llm_step_metrics.append(runner.step())

        summary = summarize_simulation(llm_step_metrics, llm_policy_name)
        summary_results.append(summary)

        for step_idx, step_metric in enumerate(llm_step_metrics, start=1):
            step_results.append(
                build_step_row(
                    llm_policy_name,
                    step_idx,
                    step_metric,
                    phase=resolve_instruction_phase(
                        step_idx,
                        observe_steps,
                        instruction_updated,
                    ),
                )
            )

        # decision_history lives in the same policy object and is continuous
        for row in llm_policy.decision_history:
            row_step = int(row.get("step", -1))
            llm_rows.append(
                {
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
                if key != "Policy":
                    print(f"  {key}: {value:.4f}")

    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df[BASELINE_SUMMARY_COLUMNS]

    step_df = pd.DataFrame(step_results)
    step_df = step_df[BASELINE_STEP_COLUMNS]

    llm_df = pd.DataFrame(llm_rows)

    return summary_df, step_df, llm_df


def main() -> None:
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = load_config("config/experiment.yaml")

    summary_df, step_df, llm_df = run_policy_suite(
        env_config=env_config,
        policy_config=policy_config,
        experiment_config=experiment_config,
        include_llm=True,
        verbose=True,
    )

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "baseline_results.csv"
    step_csv = output_dir / "baseline_step_results.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    step_df.to_csv(step_csv, index=False, encoding="utf-8-sig")

    if not llm_df.empty:
        llm_csv = output_dir / "llm_decisions.csv"
        llm_df.to_csv(llm_csv, index=False, encoding="utf-8-sig")
        print(f"LLM decisions saved to: {llm_csv}")

    print(f"\nSummary results saved to: {summary_csv}")
    print(f"Step results saved to: {step_csv}")

    figures_dir = output_dir / "figures"
    visualize_baseline_results(summary_csv, figures_dir)
    visualize_step_curves(step_csv, figures_dir)


if __name__ == "__main__":
    main()
