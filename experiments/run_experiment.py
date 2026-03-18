from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.llm_policy import LLMPolicy
from src.algorithms.nearest import NearestPolicy
from src.algorithms.never_migrate import NeverMigratePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.env_builder import build_environment
from src.runners.simulation_runner import SimulationRunner
from src.utils.config_loader import load_config
from experiments.visualize_baseline import (
    visualize_baseline_results,
    visualize_step_curves,
)


def build_policies(policy_params: PolicyParams, experiment_config: dict) -> dict:
    llm_refresh_interval = int(experiment_config.get("llm_refresh_interval", 1))
    llm_retry_remote_every = int(experiment_config.get("llm_retry_remote_every", 5))

    all_policies = {
        "never_migrate": NeverMigratePolicy(),
        "nearest": NearestPolicy(),
        "cost_aware": CostAwarePolicy(policy_params),
        "llm_policy": LLMPolicy(
            policy_params,
            refresh_interval=llm_refresh_interval,
            retry_remote_every=llm_retry_remote_every,
        ),
    }

    selected = experiment_config.get("policy_name", "all")
    if selected == "all":
        return all_policies

    selected_names = [name.strip() for name in str(selected).split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in all_policies]
    if unknown:
        raise ValueError(f"Unknown policy_name values: {unknown}")

    return {name: all_policies[name] for name in selected_names}


def main() -> None:
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = load_config("config/experiment.yaml")

    policy_params = PolicyParams.from_dict(policy_config)
    steps = int(experiment_config.get("steps", 20))
    seed = experiment_config.get("seed")
    seed = int(seed) if seed is not None else None
    policies = build_policies(policy_params, experiment_config)

    summary_results = []
    step_results = []

    for name, policy in policies.items():
        env = build_environment(env_config, policy_params, seed=seed)
        runner = SimulationRunner(env, policy)
        metrics = runner.run(steps)

        summary = metrics.summary()
        summary["Policy"] = name
        summary_results.append(summary)

        for step_idx, step_metric in enumerate(metrics.step_metrics, start=1):
            row = {
                "Policy": name,
                "step": step_idx,
                "avg_delay": step_metric.avg_delay,
                "avg_total_cost": step_metric.total_cost,
                "avg_migrations": step_metric.migration_count,
                "avg_failed_allocations": step_metric.failed_allocations,
                "avg_load_ratio": step_metric.avg_load_ratio,
            }
            row.update(step_metric.policy_debug)
            step_results.append(row)

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

    print(f"\nSaved summary results to: {summary_csv}")
    print(f"Saved step results to: {step_csv}")

    figures_dir = output_dir / "figures"
    visualize_baseline_results(summary_csv, figures_dir)
    visualize_step_curves(step_csv, figures_dir)


if __name__ == "__main__":
    main()
