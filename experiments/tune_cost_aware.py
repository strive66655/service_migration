from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import pandas as pd
from experiments.run_experiment import run_policy_suite
from src.algorithms.policy_params import PolicyParams
from src.utils.config_loader import load_config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEARCH_SPACE = {
    "lambda_delay": [0.24, 0.28, 0.32],
    "lambda_migration": [0.18, 0.22, 0.26],
    "lambda_resource": [0.28, 0.32, 0.36],
    "lambda_balance": [0.12, 0.16, 0.2],
    "migrate_threshold": [0.03, 0.06, 0.09],
    "cooldown_steps": [2, 3, 4],
}


def _score_candidate(policy_config: dict, env_config: dict, experiment_config: dict) -> list[dict]:
    scenarios = [
        {
            "label": "base_seed",
            "env_config": deepcopy(env_config),
            "experiment_config": deepcopy(experiment_config),
        },
        {
            "label": "perturbed_seed",
            "env_config": {
                **deepcopy(env_config),
                "user_position_jitter": max(float(env_config.get("user_position_jitter", 0.0)), 2.0),
                "user_velocity_jitter": max(float(env_config.get("user_velocity_jitter", 0.0)), 0.8),
            },
            "experiment_config": {
                **deepcopy(experiment_config),
                "seed": int(experiment_config.get("seed", 42)) + 17,
            },
        },
    ]

    rows: list[dict] = []
    for scenario in scenarios:
        summary_df, _, _ = run_policy_suite(
            env_config=scenario["env_config"],
            policy_config=policy_config,
            experiment_config=scenario["experiment_config"],
            include_llm=False,
            verbose=False,
        )
        summary_map = summary_df.set_index("Policy").to_dict("index")
        cost_aware = summary_map["cost_aware"]
        nearest = summary_map["nearest"]
        rows.append(
            {
                "scenario": scenario["label"],
                "cost_aware_total_cost": cost_aware["avg_total_cost"],
                "nearest_total_cost": nearest["avg_total_cost"],
                "cost_margin_vs_nearest": cost_aware["avg_total_cost"] - nearest["avg_total_cost"],
                "cost_aware_failed_allocations": cost_aware["avg_failed_allocations"],
                "nearest_failed_allocations": nearest["avg_failed_allocations"],
                "cost_aware_migrations": cost_aware["avg_migrations"],
                "nearest_migrations": nearest["avg_migrations"],
                "cost_aware_delay": cost_aware["avg_delay"],
                "nearest_delay": nearest["avg_delay"],
            }
        )
    return rows


def main() -> None:
    env_config = load_config("config/env.yaml")
    policy_config = load_config("config/policy.yaml")
    experiment_config = load_config("config/experiment.yaml")

    search_rows: list[dict] = []
    keys = list(SEARCH_SPACE.keys())

    for values in product(*(SEARCH_SPACE[key] for key in keys)):
        trial_overrides = dict(zip(keys, values))
        trial_config = deepcopy(policy_config)
        trial_config.update(trial_overrides)
        trial_params = PolicyParams.from_dict(trial_config).normalized()
        trial_config.update(trial_params.to_dict())

        scenario_rows = _score_candidate(trial_config, env_config, experiment_config)
        mean_margin = sum(row["cost_margin_vs_nearest"] for row in scenario_rows) / len(scenario_rows)
        mean_failed_gap = sum(
            row["cost_aware_failed_allocations"] - row["nearest_failed_allocations"]
            for row in scenario_rows
        ) / len(scenario_rows)
        mean_delay_gap = sum(
            row["cost_aware_delay"] - row["nearest_delay"]
            for row in scenario_rows
        ) / len(scenario_rows)

        search_rows.append(
            {
                **trial_params.to_dict(),
                "mean_cost_margin_vs_nearest": mean_margin,
                "mean_failed_gap_vs_nearest": mean_failed_gap,
                "mean_delay_gap_vs_nearest": mean_delay_gap,
            }
        )

    results_df = pd.DataFrame(search_rows).sort_values(
        ["mean_cost_margin_vs_nearest", "mean_failed_gap_vs_nearest", "mean_delay_gap_vs_nearest"],
        ascending=[True, True, True],
    )

    output_path = Path("experiments/results/cost_aware_tuning.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    best_row = results_df.iloc[0]
    print("Best parameters:")
    for key in keys:
        print(f"  {key}: {best_row[key]}")
    print(f"  mean_cost_margin_vs_nearest: {best_row['mean_cost_margin_vs_nearest']:.4f}")
    print(f"Saved tuning results to: {output_path}")


if __name__ == "__main__":
    main()
