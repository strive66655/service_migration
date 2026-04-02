from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from src.algorithms.base_policy import BasePolicy
from src.algorithms.cost_aware import CostAwarePolicy
from src.algorithms.policy_params import PolicyParams
from src.env.entities import User
from src.env.mec_env import MECEnvironment
from src.llm.controller import LLMPolicyController
from src.utils.metrics import StepMetrics


class LLMCostAwarePolicy(BasePolicy):
    def __init__(
        self,
        default_params: PolicyParams,
        controller: LLMPolicyController,
        update_interval: int = 5,
        history_size: int = 5,
    ) -> None:
        self.default_params = default_params
        self.controller = controller
        self.update_interval = max(1, update_interval)
        self.current_params = default_params
        self.inner_policy = CostAwarePolicy(self.current_params)

        self.recent_metrics: Deque[StepMetrics] = deque(maxlen=history_size)
        self.last_refresh_step: int = -1
        self.last_decision_meta: dict = {}
        self.decision_history = []

    def record_step_metrics(self, metrics: StepMetrics) -> None:
        self.recent_metrics.append(metrics)

    def before_step(self, env: MECEnvironment) -> None:
        self._maybe_refresh(env)

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        return self.inner_policy.select_node(env, user)
    
    def _maybe_refresh(self, env: MECEnvironment) -> None:
        current_step = env.time_step
        should_refresh = (
            self.last_refresh_step < 0
            or current_step - self.last_refresh_step >= self.update_interval
        )
        if not should_refresh:
            return
        
        decision = self.controller.suggest_params(
            env=env,
            current_params=self.current_params,
            recent_step_metrics=list(self.recent_metrics),
        )
        self.current_params = decision.params
        self.inner_policy = CostAwarePolicy(self.current_params)
        self.last_refresh_step = current_step
        self.last_decision_meta = {
            "step": current_step,
            "provider": decision.provider,
            "model": decision.model,
            "experiment_mode": self.controller.experiment_mode,
            "used_fallback": decision.used_fallback,
            "reason": decision.reason,
            "operator_instruction": self.controller.operator_instruction,
            "scene_step": self.controller.last_scene_summary.get("step"),
            "scene_avg_node_load": self.controller.last_scene_summary.get("avg_node_load"),
            "scene_max_node_load": self.controller.last_scene_summary.get("max_node_load"),
            "scene_failed_allocations_recent": self.controller.last_scene_summary.get(
                "failed_allocations_recent"
            ),
            "scene_migrations_recent": self.controller.last_scene_summary.get(
                "migrations_recent"
            ),
            "scene_avg_delay_recent": self.controller.last_scene_summary.get(
                "avg_delay_recent"
            ),
            "scene_migration_rate_recent": self.controller.last_scene_summary.get(
                "migration_rate_recent"
            ),
            "scene_failed_allocation_rate_recent": self.controller.last_scene_summary.get(
                "failed_allocation_rate_recent"
            ),
            "scene_users_in_cooldown_ratio": self.controller.last_scene_summary.get(
                "users_in_cooldown_ratio"
            ),
            "scene_avg_migrations_per_user_recent": self.controller.last_scene_summary.get(
                "avg_migrations_per_user_recent"
            ),
            "scene_user_context_summary": self.controller.last_scene_summary.get(
                "user_context_summary"
            ),
            "lambda_delay": self.current_params.lambda_delay,
            "lambda_migration": self.current_params.lambda_migration,
            "lambda_resource": self.current_params.lambda_resource,
            "lambda_balance": self.current_params.lambda_balance,
            "migrate_threshold": self.current_params.migrate_threshold,
            "cooldown_steps": self.current_params.cooldown_steps,
        }

        print("\n========== LLM Param Refresh ==========")
        print(f"step: {current_step}")
        print(
            "params: "
            f"delay={self.current_params.lambda_delay:.4f}, "
            f"migration={self.current_params.lambda_migration:.4f}, "
            f"resource={self.current_params.lambda_resource:.4f}, "
            f"balance={self.current_params.lambda_balance:.4f}, "
            f"threshold={self.current_params.migrate_threshold:.4f}, "
            f"cooldown={self.current_params.cooldown_steps}"
        )
        print(
            "runtime_source: "
            "policy.current_params -> inner_policy.params -> runner cost accounting"
        )
        print(f"reason: {decision.reason}")
        print("=======================================\n")

        self.decision_history.append(self.last_decision_meta.copy())

