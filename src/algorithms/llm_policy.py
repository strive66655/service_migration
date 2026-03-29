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
            "lambda_delay": self.current_params.lambda_delay,
            "lambda_migration": self.current_params.lambda_migration,
            "lambda_resource": self.current_params.lambda_resource,
            "lambda_balance": self.current_params.lambda_balance,
            "migrate_threshold": self.current_params.migrate_threshold,
            "cooldown_steps": self.current_params.cooldown_steps,
        }

        self.decision_history.append(self.last_decision_meta.copy())

