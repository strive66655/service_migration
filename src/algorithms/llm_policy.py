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
        self.last_decision_mata: dict = {}

    def record_step_metrics(self, metrics: StepMetrics) -> None:
        self.recent_metrics.append(metrics)

    def select_node(self, env: MECEnvironment, user: User) -> Optional[int]:
        self._maybe_refresh(env)
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
            recent_step_metrics=list(self.recent_metrics),
        )
        self.current_params = decision.params
        self.inner_policy = CostAwarePolicy(self.current_params)
        self.last_refresh_step = current_step
        self.last_decision_meta = {
            "reason": decision.reason,
            "provider": decision.provider,
            "model": decision.model,
            "used_fallback": decision.used_fallback,
        }

