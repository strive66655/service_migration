from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.llm_policy import LLMCostAwarePolicy
from src.algorithms.nearest import NearestPolicy
from src.algorithms.never_migrate import NeverMigratePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.env_builder import build_environment
from src.runners.simulation_runner import SimulationRunner
from src.utils.config_loader import load_config
from src.llm.controller import LLMPolicyController
from src.llm.providers.mock_provider import MockProvider
from src.llm.providers.qwen_provider import QwenProvider
from experiments.visualize_baseline import (
    visualize_baseline_results,
    visualize_step_curves,
)

def build_llm_policy(policy_params: PolicyParams, policy_config: dict):
    llm_cfg = policy_config.get("llm", {})
    provider_name = str(llm_cfg.get("provider", "mock")).lower()
    model_name = str(llm_cfg.get("model", "mock-model"))
    update_interval = int(llm_cfg.get("update_interval", 5))

    if provider_name == "qwen":
        provider = QwenProvider(
            model_name=model_name,
            temperature=float(llm_cfg.get("temperature", 0.1)),
        )
    else:
        provider = MockProvider()
        provider_name = "mock"

    controller = LLMPolicyController(
        provider=provider,
        provider_name=provider_name,
        model_name=model_name,
        default_params=policy_params,
    )

    return LLMCostAwarePolicy(
        default_params=policy_params,
        controller=controller,
        update_interval=update_interval,
    )

def main() -> None:
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = load_config("config/experiment.yaml")

    policy_params = PolicyParams.from_dict(policy_config)
    steps = int(experiment_config.get("steps", 20))

    policies = {
        "never_migrate": NeverMigratePolicy(),
        "nearest": NearestPolicy(),
        "cost_aware": CostAwarePolicy(policy_params),
        "llm_cost_aware_qwen": build_llm_policy(policy_params, policy_config),
    }

    summary_results = []
    step_results = []

    for name, policy in policies.items():
        env = build_environment(env_config, policy_params)
        runner = SimulationRunner(env, policy)
        metrics = runner.run(steps)

        summary = metrics.summary()
        summary["Policy"] = name
        summary_results.append(summary)

        for step_idx, step_metric in enumerate(metrics.step_metrics, start=1):
            step_results.append(
                {
                    "Policy": name,
                    "step": step_idx,
                    "avg_delay": step_metric.avg_delay,
                    "avg_total_cost": step_metric.total_cost,
                    "avg_migrations": step_metric.migration_count,
                    "avg_failed_allocations": step_metric.failed_allocations,
                    "avg_load_ratio": step_metric.avg_load_ratio,
                }
            )

        print(f"\nPolicy: {name}")
        for k, v in summary.items():
            if k != "Policy":
                print(f"  {k}: {v:.4f}")

    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "baseline_results.csv"
    step_csv = output_dir / "baseline_step_results.csv"

    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df[
        [
            "Policy",
            "avg_delay",
            "avg_total_cost",
            "avg_migrations",
            "avg_failed_allocations",
            "avg_load_ratio",
        ]
    ]
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    step_df = pd.DataFrame(step_results)
    step_df.to_csv(step_csv, index=False, encoding="utf-8-sig")

    print(f"\n汇总结果已保存到: {summary_csv}")
    print(f"逐步结果已保存到: {step_csv}")

    figures_dir = output_dir / "figures"
    visualize_baseline_results(summary_csv, figures_dir)
    visualize_step_curves(step_csv, figures_dir)


if __name__ == "__main__":
    main()